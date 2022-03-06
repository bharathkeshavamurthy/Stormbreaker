// -*- c++ -*-
// Copyright (c) 2018 Dennis Ogbe

#include "sc_sync.h"

#define CUDA(expr)                                                             \
  do {                                                                         \
    expr;                                                                      \
    auto err = cudaGetLastError();                                             \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error((boost::format("CUDA Error (%1%:%2%): %3%") %   \
                                __FILE__ % __LINE__ % cudaGetErrorString(err)) \
                                   .str());                                    \
    }                                                                          \
  } while (false)

namespace bam {
namespace dsp {

//
// Public Interface
//

// tors
SCSync::SCSync(size_t L, size_t num_chan, size_t max_in_samps,
               std::vector<cudaStream_t> const &streams)
    : _L(L), _num_chan(num_chan), _max_in_samps(max_in_samps),
      _streams(streams) {
  assert(_streams.size() == num_chan);
  CUDA(cudaMalloc(&_d_samp_hist, _L * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMalloc(&_d_Pd_hist, _L * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMalloc(&_d_Rd2_hist, _L * _num_chan * sizeof(float)));

  CUDA(cudaMemset(_d_samp_hist, 0x00, _L * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMemset(_d_Pd_hist, 0x00, _L * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMemset(_d_Rd2_hist, 0x00, _L * _num_chan * sizeof(float)));

  CUDA(cudaMalloc(&_d_mul_conj_delay,
                  _max_in_samps * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMalloc(&_d_samp_energy,
                  _max_in_samps * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMalloc(&_d_Pd, _max_in_samps * _num_chan * sizeof(cuComplex)));
  CUDA(cudaMalloc(&_d_Rd2, _max_in_samps * _num_chan * sizeof(float)));
  CUDA(cudaMalloc(&_d_Md, _max_in_samps * _num_chan * sizeof(float)));
}

SCSync::~SCSync() {
  cudaFree(_d_samp_hist);
  cudaFree(_d_Pd_hist);
  cudaFree(_d_Rd2_hist);

  cudaFree(_d_mul_conj_delay);
  cudaFree(_d_samp_energy);
  cudaFree(_d_Pd);
  cudaFree(_d_Md);
}

void SCSync::execute(std::vector<size_t> const &nsamps, cuComplex const *d_in) {
  // if this happens, we are in trouble.
  assert(nsamps.size() == _num_chan);
  for (auto const &nsamp : nsamps) {
    if (nsamp > _max_in_samps) {
      abort();
    }
  }
  // execute
  _mul_conj_delay(nsamps, (cuComplex *)d_in);
  _samp_energy(nsamps, (cuComplex *)d_in);
  cudaDeviceSynchronize();
  _compute_Pd(nsamps);
  _compute_Rd2(nsamps);
  cudaDeviceSynchronize();
  _compute_Md(nsamps);
  _update_history(nsamps, (cuComplex *)d_in);
  cudaDeviceSynchronize();
}

void SCSync::write_output(std::vector<size_t> const &nsamps,
                          cuComplex const *d_in, std::vector<outbuf> &h_out) {
  // We assume that the following conditions hold when this function is called
  assert(h_out.size() == _num_chan);
  for (int channel = 0; channel < _num_chan; ++channel) {
    // wait until there is enough room to write
    auto const ns = nsamps[channel];
    while (true) {
      if ((ns <= h_out[channel]->samples->space_avail()) &&
          (ns <= h_out[channel]->Md->space_avail()) &&
          (ns <= h_out[channel]->Pd->space_avail())) {
        break;
      }
    }
  }

  // write the output to the output ring buffers
  for (int channel = 0; channel < _num_chan; ++channel) {
    auto const ns = nsamps[channel];
    auto samples = d_in + channel * _max_in_samps;
    auto Pd = _d_Pd + channel * _max_in_samps;
    auto Md = _d_Md + channel * _max_in_samps;

    cudaMemcpyAsync(h_out[channel]->samples->write_ptr(), samples,
                    ns * sizeof(fcomplex), cudaMemcpyDeviceToHost,
                    _streams[channel]);
    cudaMemcpyAsync(h_out[channel]->Pd->write_ptr(), Pd, ns * sizeof(fcomplex),
                    cudaMemcpyDeviceToHost, _streams[channel]);
    cudaMemcpyAsync(h_out[channel]->Md->write_ptr(), Md, ns * sizeof(float),
                    cudaMemcpyDeviceToHost, _streams[channel]);

    cudaDeviceSynchronize();
    h_out[channel]->produce_each(ns);
  }
}

//
// Signal Processing
//

__global__ void k_mul_conj_delay(cuComplex *in, cuComplex *out, cuComplex *hist,
                                 size_t n, size_t L) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  // n.b. this loop can be manually unrolled to get rid of the
  // conditional... any reasonably smart compiler should be able to do that by
  // itself though... TODO: verify this claim or unroll by hand
  for (int i = offset; i < n; i += stride) {
    int a_idx = i - (int)L;
    cuComplex a, b;
    if (a_idx < 0) {
      a = hist[L + a_idx]; // need to get from history
    } else {
      a = in[a_idx];
    }
    b = in[i];
    // start after the history
    out[i] = cuCmulf(cuConjf(a), b);
  }
}

__global__ void k_samp_energy(cuComplex *in, float *out, size_t n, size_t L) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (int i = offset; i < n; i += stride) {
    auto const absval = cuCabsf(in[i]);
    out[i] = absval * absval;
  }
}

__global__ void k_compute_Pd(cuComplex *in, cuComplex *Pd, cuComplex *hist,
                             size_t n, size_t L) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (int i = offset; i < n; i += stride) {
    int start_idx = i - (L - 1);
    cuComplex accum{0.0, 0.0};
    for (int j = start_idx; j < start_idx + (int)L; ++j) {
      if (j < 0) {
        accum = cuCaddf(accum, hist[L + j]);
      } else {
        accum = cuCaddf(accum, in[j]);
      }
    }
    Pd[i] = accum;
  }
}

__global__ void k_compute_Rd2(float *in, float *Rd2, float *hist, size_t n,
                              size_t L) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (int i = offset; i < n; i += stride) {
    int start_idx = i - (L - 1);
    float accum = 0.0f;
    for (int j = start_idx; j < start_idx + (int)L; ++j) {
      if (j < 0) {
        accum += hist[L + j];
      } else {
        accum += in[j];
      }
    }
    Rd2[i] = accum * accum;
  }
}

__global__ void k_compute_Md(cuComplex *Pd, float *Rd2, float *out, size_t n,
                             size_t L) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (int i = offset; i < n; i += stride) {
    auto absPd = cuCabsf(Pd[i]);
    out[i] = absPd * absPd / Rd2[i];
  }
}

void SCSync::_mul_conj_delay(std::vector<size_t> const &nsamps,
                             cuComplex *d_in) {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = d_in + i * _max_in_samps;
    auto op = _d_mul_conj_delay + i * _max_in_samps;
    auto hist = _d_samp_hist + i * _L;
    k_mul_conj_delay<<<_ngrid, _tpb, 0, _streams[i]>>>(ip, op, hist, nsamps[i],
                                                       _L);
  }
}

void SCSync::_samp_energy(std::vector<size_t> const &nsamps, cuComplex *d_in) {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = d_in + i * _max_in_samps;
    auto op = _d_samp_energy + i * _max_in_samps;
    k_samp_energy<<<_ngrid, _tpb, 0, _streams[i]>>>(ip, op, nsamps[i], _L);
  }
}

void SCSync::_compute_Pd(std::vector<size_t> const &nsamps) {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = _d_mul_conj_delay + i * _max_in_samps;
    auto op = _d_Pd + i * _max_in_samps;
    auto hist = _d_Pd_hist + i * _L;
    k_compute_Pd<<<_ngrid, _tpb, 0, _streams[i]>>>(ip, op, hist, nsamps[i], _L);
  }
}

void SCSync::_compute_Rd2(std::vector<size_t> const &nsamps) {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = _d_samp_energy + i * _max_in_samps;
    auto op = _d_Rd2 + i * _max_in_samps;
    auto hist = _d_Rd2_hist + i * _L;
    k_compute_Rd2<<<_ngrid, _tpb, 0, _streams[i]>>>(ip, op, hist, nsamps[i],
                                                    _L);
  }
}

void SCSync::_compute_Md(std::vector<size_t> const &nsamps) {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto Pd = _d_Pd + i * _max_in_samps;
    auto Rd2 = _d_Rd2 + i * _max_in_samps;
    auto op = _d_Md + i * _max_in_samps;
    k_compute_Md<<<_ngrid, _tpb, 0, _streams[i]>>>(Pd, Rd2, op, nsamps[i], _L);
  }
}

void SCSync::_update_history(std::vector<size_t> const &nsamps,
                             cuComplex *d_in) {
  // Re: C-style casts: Trust me, I'm an engineer™
  auto do_update = [&, this](auto hist_base, auto samp_base) {
    for (size_t channel = 0; channel < _num_chan; ++channel) {
      auto const ns = nsamps[channel];
      auto const hist = hist_base + channel * _L;
      auto const samp = samp_base + channel * _max_in_samps;
      if (ns >= _L) {
        auto const samp_src = samp + (ns - _L);
        auto const samp_dst = hist;
        cudaMemcpyAsync((void *)samp_dst, (void *)samp_src, _L * sizeof(*samp),
                        cudaMemcpyDeviceToDevice, _streams[channel]);
      } else { // ns < _L
        auto const nhist = _L - ns;
        auto const samp_src = samp;
        auto const samp_dst = hist + nhist;
        auto const hist_src = hist + ns;
        auto const hist_dst = hist;
        cudaMemcpyAsync((void *)samp_dst, (void *)samp_src, ns * sizeof(*samp),
                        cudaMemcpyDeviceToDevice, _streams[channel]);
        cudaMemcpyAsync((void *)hist_dst, (void *)hist_src,
                        nhist * sizeof(*hist), cudaMemcpyDeviceToDevice,
                        _streams[channel]);
      }
    }
  };
  do_update(_d_samp_hist, d_in);
  do_update(_d_Pd_hist, _d_mul_conj_delay);
  do_update(_d_Rd2_hist, _d_samp_energy);
}

} // namespace dsp
} // namespace bam
