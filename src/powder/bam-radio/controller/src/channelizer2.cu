// -*- c++ -*-
// Copyright (c) 2018 Dennis Ogbe

#include "channelizer2.h"

#include <boost/format.hpp>

// sanity macros
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
#define CUFFT(expr)                                                            \
  do {                                                                         \
    auto err = expr;                                                           \
    if (err != CUFFT_SUCCESS) {                                                \
      throw std::runtime_error((boost::format("CUFFT Error (%1%:%2%): %3%") %  \
                                __FILE__ % __LINE__ % err)                     \
                                   .str());                                    \
    }                                                                          \
  } while (false)

namespace bam {
namespace dsp {

//
// tors and initializers
//

Channelizer2::Channelizer2(size_t num_chan, size_t N, size_t nbatch,
                           std::vector<SubChannel> const &subchannel_table)
    // clang-format off
    : _num_chan(num_chan),
      _N(N),
      _nbatch(nbatch),
       // take the maximum of all filter taps
      _ntaps(std::max_element(begin(subchannel_table), end(subchannel_table),
                              [](auto const &a, auto const &b) {
                                return a.taps.size() < b.taps.size();
                              })->taps.size()),
      _L(N - _ntaps + 1),
      _nos(subchannel_table.size()),
      _nload(_L * _nbatch),
      _fft_in_size((_nbatch - 1) * _L + _N),
      _fft_out_size(_nbatch * _N),
      _dec_in_size(_nbatch * _L),
      _max_out_size(_nbatch * ( _L / std::min_element(begin(subchannel_table), end(subchannel_table),
                                                      [](auto const &a, auto const &b){
                                                        return a.os < b.os;
                                                      })->os)),
      _bw_idx(_num_chan, 0)
{ // clang-format on

  // check whether all decimation factors divide _L
  //
  // n.b. this might not be necessary. revisit in the future.
  for (auto const &sc : subchannel_table) {
    if (_L % sc.os != 0) {
      throw std::runtime_error(
          (boost::format("Channelizer: (%1%: %2%): incompatible L size / "
                         "decimation factor: L = %3%, D = %4") %
           __FILE__ % __LINE__ % (int)_L % (int)sc.os)
              .str());
    }
  }

  // initialize rest of object
  _initTaps(subchannel_table);
  _initPhase();
  _initPlans();
  _initMemory();
}

Channelizer2::~Channelizer2() {
  for (auto &s : _streams) {
    cudaStreamDestroy(s);
  }

  for (auto &filter : _filters_d) {
    cudaFree(filter);
  }

  for (auto &plan : _fft_plans) {
    cufftDestroy(plan);
  }

  for (auto &plan : _ifft_plans) {
    cufftDestroy(plan);
  }

  cudaFree(_d_in);
  cudaFree(_d_fft_in);
  cudaFree(_d_fft_out);
  cudaFree(_d_dec_in);
  cudaFree(_d_out);
}

void Channelizer2::_initTaps(std::vector<SubChannel> const &subchannel_table) {
  // we first need to compute the frequency response of our dear filters. for
  // this, we use cuFFT.
  cuComplex *tmp;
  CUDA(cudaMallocManaged(&tmp, _N * sizeof(*tmp)));
  cufftHandle plan;
  CUFFT(cufftPlan1d(&plan, _N, CUFFT_C2C, 1));

  // compute freq response of all filters and save on device
  _filters_d.resize(subchannel_table.size());
  _os.resize(subchannel_table.size());
  for (size_t i = 0; i < subchannel_table.size(); ++i) {
    auto const &sc = subchannel_table[i];
    // allocate space on device
    cuComplex *filt;
    CUDA(cudaMalloc(&filt, _N * sizeof(*filt)));
    // zero pad and fill the input vector (we are doing a C2C transform, this is
    // a little awkward)
    CUDA(cudaMemset(tmp, 0x00, _N * sizeof(*tmp)));
    for (size_t j = 0; j < sc.taps.size(); ++j) {
      tmp[j] = make_cuComplex(sc.taps[j], 0.0f);
    }
    // execute FFT and save frequency response of filter on device.
    CUFFT(cufftExecC2C(plan, tmp, filt, CUFFT_FORWARD));
    // save the pointers to those filters for later. also save decimation rate.
    _filters_d[i] = filt;
    _os[i] = sc.os;
  }

  // free scratch memory and destroy fft handle.
  CUDA(cudaFree(tmp));
  CUFFT(cufftDestroy(plan));
}

void Channelizer2::_initPhase() {
  _phase.resize(_num_chan);
  _freq.resize(_num_chan);
  for (size_t i = 0; i < _num_chan; ++i) {
    setOffsetFreq(i, 0.0f);
  }
}

void Channelizer2::_initPlans() {
  // first we create _num_chan streams
  _streams.resize(_num_chan);
  std::generate(begin(_streams), end(_streams), [] {
    cudaStream_t s;
    CUDA(cudaStreamCreate(&s));
    return s;
  });

  // make the fft plan for each stream
  auto n = static_cast<int>(_N);
  auto ell = static_cast<int>(_L);
  auto nb = static_cast<int>(_nbatch);
  _fft_plans.resize(_num_chan);
  _ifft_plans.resize(_num_chan);
  std::generate(begin(_fft_plans), end(_fft_plans), [&, this] {
    cufftHandle plan;
    // FFT Size: _N, idist (here _L) gives us the overlap. we plan _nbatch FFTs.
    CUFFT(cufftPlanMany(&plan, 1, &n, &n, 1, ell, &n, 1, n, CUFFT_C2C, nb));
    return plan;
  });
  std::generate(begin(_ifft_plans), end(_ifft_plans), [&, this] {
    cufftHandle plan;
    // IFFT does not overlap.
    CUFFT(cufftPlanMany(&plan, 1, &n, &n, 1, n, &n, 1, n, CUFFT_C2C, nb));
    return plan;
  });

  // assign the plans to the corresponding stream
  for (size_t i = 0; i < _num_chan; ++i) {
    CUFFT(cufftSetStream(_fft_plans[i], _streams[i]));
    CUFFT(cufftSetStream(_ifft_plans[i], _streams[i]));
  }
}

void Channelizer2::_initMemory() {
  // the input buffer
  CUDA(cudaMalloc(&_d_in, _nload * sizeof(*_d_in)));

  // the fft input buffers
  CUDA(cudaMalloc(&_d_fft_in, _fft_in_size * _num_chan * sizeof(*_d_fft_in)));

  // the fft output buffers / overall processing buffers
  CUDA(
      cudaMalloc(&_d_fft_out, _fft_out_size * _num_chan * sizeof(*_d_fft_out)));

  // the decimator input buffers
  CUDA(cudaMalloc(&_d_dec_in, _dec_in_size * _num_chan * sizeof(*_d_dec_in)));

  // the output buffers
  CUDA(cudaMalloc(&_d_out, _max_out_size * _num_chan * sizeof(*_d_out)));
}

//
// Business logic
//

bool Channelizer2::load(bamradio::PinnedComplexRingBuffer *buf) {
  if (_nload <= buf->items_avail()) {
    CUDA(cudaMemcpy(_d_in, buf->read_ptr(), _nload * sizeof(*_d_in),
                    cudaMemcpyHostToDevice));
    buf->consume(_nload);
    return true;
  } else {
    return false;
  }
}

void Channelizer2::execute() {
  _mix();
  cudaDeviceSynchronize();
  _fft();
  cudaDeviceSynchronize();
  _filter();
  cudaDeviceSynchronize();
  _ifft();
  cudaDeviceSynchronize();
  _discard();
  cudaDeviceSynchronize();
  _decimate();
  _update_history();
  cudaDeviceSynchronize();
}

void Channelizer2::getNumOutput(std::vector<size_t> &vn) const {
  if (vn.size() != _num_chan) {
    // this should NEVER happen.
    vn.resize(_num_chan);
  }
  for (size_t i = 0; i < _num_chan; ++i) {
    vn[i] = _nbatch * _L / _os[_bw_idx[i]];
  }
}

cuComplex *Channelizer2::getOutPtr() const { return _d_out; }

size_t Channelizer2::getOutStride() const { return _max_out_size; }

std::vector<cudaStream_t> const &Channelizer2::getStreams() const {
  return _streams;
}

void Channelizer2::setBandwidth(size_t channel, size_t bw_idx) {
  // Error check
  if (channel >= _num_chan) {
    throw std::runtime_error("Invalid channel.");
  }
  if (bw_idx >= _nos) {
    throw std::runtime_error("Invalid bw_idx.");
  }
  _bw_idx[channel] = bw_idx;
}

void Channelizer2::setOffsetFreq(size_t channel, float freq_normalized) {
  // Error check
  if (channel >= _num_chan) {
    throw std::runtime_error("Invalid channel.");
  }
  if ((freq_normalized < -0.5) || (freq_normalized > 0.5)) {
    throw std::runtime_error(
        "Offset frequency has to be normalized to -0.5 to 0.5.");
  }
  // Multiply by -1 since we need to move the signal BACK to the original freq
  _phase[channel] = 0.0f;
  _freq[channel] = -1.0f * freq_normalized;
}

//
// Signal Processing
//

/// CUDA Kernels

// s = z * c where z is complex, c is real
__host__ __device__ static __inline__ cuFloatComplex cuCmulRf(cuFloatComplex z,
                                                              float c) {
  auto s = make_cuFloatComplex(cuCrealf(z) * c, cuCimagf(z) * c);
  return s;
}

__global__ void k_mix(cuComplex *in, cuComplex *out, float phase_offset,
                      float f_normalized, size_t n) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < n; i += stride) {
    auto const phase = phase_offset + 2.0 * f_normalized * i;
    auto const c = make_cuComplex(cospif(phase), sinpif(phase));
    out[i] = cuCmulf(in[i], c);
  }
}

__global__ void k_filter(cuComplex *p, cuComplex *filt, size_t N,
                         size_t nbatch) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  auto const n = nbatch * N;
  for (size_t i = offset; i < n; i += stride) {
    p[i] = cuCmulf(p[i], filt[i % N]);
  }
}

__global__ void k_discard(cuComplex *in, cuComplex *out, size_t N, size_t L,
                          size_t nbatch) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  auto const n = nbatch * N;
  auto const M = N - L + 1; // number of samples to discard from every batch
  for (size_t i = offset; i < n; i += stride) {
    if (i % N >= M - 1) {
      size_t const batch_idx = i / N;
      out[batch_idx * L + (i % N) - (M - 1)] = in[i];
    }
  }
}

__global__ void k_decimate(cuComplex *in, cuComplex *out, size_t os,
                           size_t isize, float scale) {
  auto const offset = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;
  auto const n = isize / os;
  for (size_t i = offset; i < n; i += stride) {
    out[i] = cuCmulRf(in[i * os], scale);
  }
}

/// Signal processing functions

void Channelizer2::_mix() {
  // FIXME find better place for # of tpb. also, does this make sense?
  int const ngrid = _nbatch;
  int const tpb = 512;
  // input pointer is the same for all kernel launches here
  auto ip = _d_in;
  for (size_t i = 0; i < _num_chan; ++i) {
    // find pointer to beginning of channel samples and advance past history
    auto op = _d_fft_in + i * _fft_in_size + _ntaps - 1;
    // run kernel on stream for channel i
    k_mix<<<ngrid, tpb, 0, _streams[i]>>>(ip, op, _phase[i], _freq[i], _nload);
    // update phase
    auto phase_new = _phase[i] + (2.0f * _freq[i] * _nload);
    _phase[i] = fmod(phase_new, 2.0f);
  }
}

void Channelizer2::_fft() {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = _d_fft_in + i * _fft_in_size;
    auto op = _d_fft_out + i * _fft_out_size;
    cufftExecC2C(_fft_plans[i], ip, op, CUFFT_FORWARD);
  }
}

void Channelizer2::_filter() {
  int const ngrid = _nbatch;
  int const tpb = 512;
  // filtering is in-place because _d_fft_out will be re-used every time.
  for (size_t i = 0; i < _num_chan; ++i) {
    auto p = _d_fft_out + i * _fft_out_size;
    k_filter<<<ngrid, tpb, 0, _streams[i]>>>(p, _filters_d[_bw_idx[i]], _N,
                                             _nbatch);
  }
}

void Channelizer2::_ifft() {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto p = _d_fft_out + i * _fft_out_size;
    cufftExecC2C(_ifft_plans[i], p, p, CUFFT_INVERSE);
  }
}

void Channelizer2::_discard() {
  int const ngrid = _nbatch;
  int const tpb = 512;
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = _d_fft_out + i * _fft_out_size;
    auto op = _d_dec_in + i * _dec_in_size;
    k_discard<<<ngrid, tpb, 0, _streams[i]>>>(ip, op, _N, _L, _nbatch);
  }
}

void Channelizer2::_decimate() {
  int const ngrid = _nbatch;
  int const tpb = 512;
  float const scale = 1.0f / ((float)_N);
  for (size_t i = 0; i < _num_chan; ++i) {
    auto ip = _d_dec_in + i * _dec_in_size;
    auto op = _d_out + i * _max_out_size;
    k_decimate<<<ngrid, tpb, 0, _streams[i]>>>(ip, op, _os[_bw_idx[i]],
                                               _dec_in_size, scale);
  }
}

void Channelizer2::_update_history() {
  for (size_t i = 0; i < _num_chan; ++i) {
    auto src = _d_fft_in + (i + 1) * _fft_in_size - (_ntaps - 1);
    auto dst = _d_fft_in + i * _fft_in_size;
    cudaMemcpyAsync(dst, src, (_ntaps - 1) * sizeof(*dst),
                    cudaMemcpyDeviceToDevice, _streams[i]);
  }
}

} // namespace dsp
} // namespace bam
