// -*- c++ -*-
// (c) Tomohiro Arakawa (tarakawa@purdue.edu)

#include <algorithm>
#include <cmath>
#include <iostream>
#include "channelizer.h"

// Macro for showing error and abort (based on RBcudaCheckErrors in
// ringbuffer.cu)
#define showErrorAndAbort(msg) \
  { fprintf(stderr, "Fatal error: %s (%s:%d)\n", msg, __FILE__, __LINE__); }

namespace bam {
namespace dsp {

__device__ inline float2 ComplexScale(float2 a, float s) {
  float2 c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

__device__ inline float2 ComplexMul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__global__ void shiftFreq(unsigned int n, float *freq_list, float *phase_list,
                          float2 *in, float2 *out) {
  size_t const channel = blockIdx.x;
  size_t const offset = threadIdx.x;
  size_t const stride = blockDim.x;
  float const f_normalized = freq_list[channel];
  float const phase_offset = phase_list[channel];
  for (size_t i = offset; i < n; i += stride) {
    float const phase = phase_offset + 2.0 * f_normalized * i;
    float2 const shift_coefficient = make_float2(cospif(phase), sinpif(phase));
    out[channel * n + i] = ComplexMul(in[i], shift_coefficient);
  }
}

__global__ void calcCorrelationVal(size_t const max_n, float2 **inptr_list,
                                   size_t *out_nitems_list,
                                   size_t *dfactor_list, float **filt_ptr_list,
                                   size_t *filt_size_list, float2 *out) {
  size_t const channel = blockIdx.x;
  size_t const decimation = dfactor_list[channel];
  float2 const *in = inptr_list[channel];
  size_t const nout = out_nitems_list[channel];
  size_t const out_offset = threadIdx.x;
  size_t const out_stride = blockDim.x;
  size_t const filt_len = filt_size_list[channel];
  float const *filter = filt_ptr_list[channel];

  for (size_t w = out_offset; w < nout; w += out_stride) {
    size_t const r = w * decimation;
    float2 accum = make_float2(0, 0);
    for (size_t j = 0; j < filt_len; ++j) {
      float2 mult = ComplexScale(in[r + j], filter[j]);
      accum.x += mult.x;
      accum.y += mult.y;
    }
    out[max_n * channel + w] = accum;
  }
}

Channelizer::Channelizer(size_t num_chan, size_t max_in_samps)
    : _num_chan(num_chan), _max_in_samps(max_in_samps) {
  // Allocate device memory for input samples
  cudaMalloc(&_device_in_buf, max_in_samps * sizeof(float2));
  // Allocate device memory for frequency-shifted samples
  cudaMalloc(&_device_freqshifted_buf,
             max_in_samps * num_chan * sizeof(float2));
  // Allocate device memory for output (decimated) samples
  cudaMalloc(&_device_out_buf, max_in_samps * num_chan * sizeof(float2));
  // Allocate device memory for input ptr list
  cudaMallocManaged(&_device_inptr_list, num_chan * sizeof(float2 *));
  // Allocate managed memory for decimation factor list
  cudaMallocManaged(&_managed_decimation_factor_list,
                    num_chan * sizeof(size_t));
  // Allocate managed memory for filter pointer list
  cudaMallocManaged(&_managed_filt_ptr_list, num_chan * sizeof(float *));
  // Allocate device memory for filter size list
  cudaMallocManaged(&_managed_filt_size_list, num_chan * sizeof(size_t));
  // Allocate device memory for freq list
  cudaMallocManaged(&_managed_freq_list, num_chan * sizeof(float));
  // Allocate managed memory for phase list
  cudaMallocManaged(&_managed_phase_list, num_chan * sizeof(float));
  // Allocate managed memory for phase list
  cudaMallocManaged(&_managed_out_nitems_list, num_chan * sizeof(size_t));

  _offset_list.resize(num_chan);

  // Set initial parameters
  for (size_t i = 0; i < num_chan; ++i) {
    setDecimationFactor(i, 10);  // decimation factor 10
    setOffsetFreq(i, 0);         // offset frequency is 0
    // initialize with one-tap filter
    float *filt_ptr;
    cudaMalloc(&filt_ptr, sizeof(float));
    float const tap_val = 1.0f;
    cudaMemcpy(filt_ptr, &tap_val, sizeof(float), cudaMemcpyHostToDevice);
    _managed_filt_ptr_list[i] = filt_ptr;
    _managed_filt_size_list[i] = 1;
  }
}

Channelizer::~Channelizer() {
  // Release memory
  cudaFree(_device_in_buf);
  cudaFree(_device_freqshifted_buf);
  cudaFree(_device_out_buf);
  cudaFree(_managed_decimation_factor_list);
  cudaFree(_managed_filt_size_list);
  cudaFree(_managed_freq_list);
  cudaFree(_managed_phase_list);
  for (size_t i = 0; i < _num_chan; ++i) {
    cudaFree(*(_managed_filt_ptr_list + i));
  }
  cudaFree(_managed_filt_ptr_list);
}

void Channelizer::setTaps(size_t channel, std::vector<float> const &taps) {
  cudaFree(_managed_filt_ptr_list[channel]);
  float *device_filt;
  cudaMalloc(&device_filt, taps.size() * sizeof(float));
  cudaMemcpy(device_filt, taps.data(), taps.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  _managed_filt_ptr_list[channel] = device_filt;
  _managed_filt_size_list[channel] = taps.size();
}

size_t Channelizer::execute(size_t nsamps, fcomplex const *in_buf,
                            bool copy_buf) {
  // Check input data size
  if (nsamps > _max_in_samps) {
    showErrorAndAbort("Input data size cannot be larger than max_in_samps.");
  }

  // Calculate how many decimated samples we will generate in this iteration
  for (size_t ch = 0; ch < _num_chan; ++ch) {
    size_t const offset = _offset_list[ch];
    size_t const filt_len = _managed_filt_size_list[ch];
    size_t const decm_factor = _managed_decimation_factor_list[ch];
    size_t const n_full_corrs =
        1 + ((nsamps - offset - filt_len) / decm_factor);
    if (filt_len > nsamps) {
      showErrorAndAbort("Filter length cannot be larger than input data size.");
    }
    _managed_out_nitems_list[ch] = n_full_corrs;
  }

  // Generate input ptr list
  for (size_t ch = 0; ch < _num_chan; ++ch) {
    _device_inptr_list[ch] =
        (float2 *)_device_freqshifted_buf + ch * nsamps + _offset_list[ch];
  }

  // Copy buffer
  float2 *gpu_in_buf;
  if (copy_buf) {
    cudaMemcpy(_device_in_buf, in_buf, nsamps * sizeof(float2),
               cudaMemcpyHostToDevice);
    gpu_in_buf = (float2 *)_device_in_buf;
  } else {
    gpu_in_buf = (float2 *)in_buf;
  }

  // Frequency shift
  shiftFreq<<<_num_chan, 512>>>(nsamps, _managed_freq_list, _managed_phase_list,
                                gpu_in_buf, (float2 *)_device_freqshifted_buf);
  // Filter and Downsample
  calcCorrelationVal<<<_num_chan, 512>>>(
      _max_in_samps, (float2 **)_device_inptr_list, _managed_out_nitems_list,
      _managed_decimation_factor_list, _managed_filt_ptr_list,
      _managed_filt_size_list, (float2 *)_device_out_buf);

  // Wait for the kernels to finish
  cudaDeviceSynchronize();

  // Calculate how many samples we have consumed
  std::vector<size_t> rem_samples(_num_chan);
  for (size_t ch = 0; ch < _num_chan; ++ch) {
    size_t const offset = _offset_list[ch];
    size_t const decm_factor = _managed_decimation_factor_list[ch];
    size_t const n_full_corrs = _managed_out_nitems_list[ch];
    size_t const n_consumed = offset + (decm_factor * n_full_corrs);
    int const remainder = nsamps - n_consumed;
    if (remainder < 0)
      abort();
    rem_samples[ch] = remainder;
  }

  // Number of samples which we need to keep
  size_t nsamp_forward = *max_element(rem_samples.begin(), rem_samples.end());
  for (size_t i = 0; i < _num_chan; ++i) {
    _offset_list[i] = nsamp_forward - rem_samples[i];
  }

  // Compute number of samples we consumed
  size_t const nsamp_consumed = nsamps - nsamp_forward;

  // Update phase
  for (size_t i = 0; i < _num_chan; ++i) {
    float phase_now = _managed_phase_list[i];
    float phase_new =
        phase_now + (2.0f * _managed_freq_list[i] * nsamp_consumed);
    _managed_phase_list[i] = fmod(phase_new, 2.0f);
  }

  return nsamp_consumed;
}

size_t Channelizer::getOutput(size_t channel, fcomplex *out_buf) {
  // Error check
  if (channel >= _num_chan) {
    showErrorAndAbort("Invalid channel.");
  }

  size_t const out_nitems = _managed_out_nitems_list[channel];
  cudaMemcpy(out_buf, (float2 *)_device_out_buf + _max_in_samps * channel,
             out_nitems * sizeof(float2), cudaMemcpyDeviceToHost);

  return out_nitems;
}

void Channelizer::getNumOutput(std::vector<size_t> &vn) {
  for (size_t i = 0; i < _num_chan; ++i) {
    vn[i] = _managed_out_nitems_list[i];
  }
}

void Channelizer::setDecimationFactor(size_t channel, size_t factor) {
  _managed_decimation_factor_list[channel] = factor;
}

void Channelizer::setOffsetFreq(size_t channel, float freq_normalized) {
  // Error check
  if (channel >= _num_chan) {
    showErrorAndAbort("Invalid channel.");
  }
  if ((freq_normalized < -0.5) || (freq_normalized > 0.5)) {
    showErrorAndAbort("Offset frequency has to be normalized to -0.5 to 0.5.");
  }
  // Multiply by -1 since we need to move the signal BACK to the original freq
  _managed_phase_list[channel] = 0.0f;
  _managed_freq_list[channel] = -1.0f * freq_normalized;
}

}  // namespace dsp
}  // namespace bam
