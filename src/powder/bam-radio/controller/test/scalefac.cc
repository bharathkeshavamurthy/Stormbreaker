// Scale factor sanity
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE scalefac
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "test_extra.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

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

// question: if I do cuifft(cufft(signal)), do I get a scaled version of the
// input signal?

using namespace bamradio::test;
using fcomplex = bamradio::fcomplex;

BOOST_AUTO_TEST_CASE(fftscale) {
  auto nfft = 128;
  auto signal = make_complex_gaussian(nfft);
  cufftHandle h;
  CUFFT(cufftPlan1d(&h, nfft, CUFFT_C2C, 1));

  cuComplex *buf;
  CUDA(cudaMallocManaged(&buf, nfft * sizeof(*buf)));
  CUDA(cudaMemcpy(buf, signal.data(), signal.size() * sizeof(*buf),
                  cudaMemcpyHostToDevice));

  CUFFT(cufftExecC2C(h, buf, buf, CUFFT_FORWARD));
  CUFFT(cufftExecC2C(h, buf, buf, CUFFT_INVERSE));

  std::vector<fcomplex> out_signal((fcomplex *)buf, (fcomplex *)buf + nfft);

  // print both vectors
  print_two_vec_vert(signal, out_signal);

  // print both vectors, the output one scaled
  decltype(out_signal) out_signal_scaled(out_signal);
  for (auto &s : out_signal_scaled) {
    s /= (float)nfft;
  }
  print_two_vec_vert(signal, out_signal_scaled);
}
