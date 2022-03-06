// -*- c++ -*-
//
// cuFFT wrapper

#include "bamcufft.h"

#include <boost/format.hpp>

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

namespace bamradio {
namespace fft {
//
// GPU FFT interface (cuFFT)
//
GPUFFT::GPUFFT(std::vector<size_t> const &sizes, bool forward,
               cudaStream_t stream)
    : _type(forward ? CUFFT_FORWARD : CUFFT_INVERSE) {
  // compute fft plans for all requested sizes
  for (auto const &size : sizes) {
    if (_plans.count(size) > 0)
      continue;
    cufftHandle h;
    CUFFT(cufftPlan1d(&h, size, CUFFT_C2C, 1));
    CUFFT(cufftSetStream(h, stream));
    _plans[size] = h;
  }
  assert(_plans.size() == sizes.size());
}

GPUFFT::~GPUFFT() {
  for (auto &plan : _plans) {
    cufftDestroy(plan.second);
  }
}

void GPUFFT::execute(size_t size, fcomplex *in, fcomplex *out) const {
  CUFFT(cufftExecC2C(_plans.at(size), (cufftComplex *)in, (cufftComplex *)out,
                     _type));
}

std::vector<size_t> GPUFFT::sizes() const {
  decltype(sizes()) o;
  for (auto const &p : _plans) {
    o.push_back(p.first);
  }
  std::sort(begin(o), end(o));
  return o;
}

bool GPUFFT::forward() const { return _type == CUFFT_FORWARD; }
} // namespace fft
} // namespace bamradio
