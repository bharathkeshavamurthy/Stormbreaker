// Channelizer2 unit tests
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE channelizer2
#define BOOST_TEST_DYN_LINK
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include "bamfftw.h"
#include "bandwidth.h"
#include "buffers.h"
#include "channelizer2.h"
#include "common.h"
#include "test_extra.h"

#include <cuda_runtime.h>
#include <volk/volk.h>

#include <boost/format.hpp>

#include <algorithm>
#include <random>
#include <vector>

// I really need to move this macro to some central place...
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

using fcomplex = bamradio::fcomplex;
using bamradio::PI;
using namespace bamradio::test;

//
// test class
//
class channelizer_test : public bam::dsp::Channelizer2 {
public:
  channelizer_test(size_t num_chan, size_t N, size_t nbatch,
                   std::vector<bam::dsp::SubChannel> const &subchannel_table)
      : bam::dsp::Channelizer2(num_chan, N, nbatch, subchannel_table) {}

  void init() {
    // TODO: clear all buffers/history/etc.
  }

  void test_mix() {
    // we need some random samples
    auto samples = make_complex_gaussian(_nload, 22);
    // we also need some random mixing frequencies to test
    auto freqs = make_real_uniform(_num_chan, 34, -0.5, 0.5);
    for (size_t i = 0; i < freqs.size(); ++i) {
      setOffsetFreq(i, freqs[i]);
    }

    // load a ringbuffer with those random samples and call _execute. The state
    // of the channelizer should now be predictable
    auto rb = bamradio::PinnedComplexRingBuffer::make(2 * _nload);
    std::copy(begin(samples), end(samples), rb->write_ptr());
    rb->produce(samples.size());
    BOOST_REQUIRE(load(rb.get()));
    _mix();
    CUDA(cudaDeviceSynchronize());
    // copy the output to host.
    std::vector<std::vector<fcomplex>> gpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      gpu_out[i] = std::vector<fcomplex>(_fft_in_size);
      CUDA(cudaMemcpy(gpu_out[i].data(), _d_fft_in + i * _fft_in_size,
                      _fft_in_size * sizeof(fcomplex), cudaMemcpyDeviceToHost));
    }

    // compute the mixed signal using libvolk's rotator
    std::vector<std::vector<fcomplex>> volk_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      auto phase_incr = std::exp(fcomplex(0.0, freqs[i] * -2.0 * PI()));
      fcomplex phase = 1.;
      volk_out[i] = std::vector<fcomplex>(samples.size());
      volk_32fc_s32fc_x2_rotator_32fc(volk_out[i].data(), samples.data(),
                                      phase_incr, &phase, samples.size());
    }

    // compare the results. remember that there need to be _numtaps - 1 zeros
    // before the signal begins.
    std::vector<fcomplex> zeros(_ntaps - 1, fcomplex(0.0, 0.0));
    for (size_t i = 0; i < _num_chan; ++i) {
      auto samp_begin = gpu_out[i].data() + _ntaps - 1;
      decltype(zeros) should_be_zero(gpu_out[i].data(), samp_begin);
      std::vector<fcomplex> mixed;
      mixed.assign(samp_begin, samp_begin + samples.size());
#if 1
      // detailed differences
      complex_vec_close_2(zeros, should_be_zero, 1e-3);
      complex_vec_close_2(volk_out[i], mixed, 1e-3);
#else
      // this tells you specifically which test failed
      BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
          zeros, should_be_zero));
      BOOST_REQUIRE(
          gr::bamofdm::assert_complex_vectors_almost_equal(volk_out[i], mixed));
#endif
    }
  }

  void test_fft() {
    using namespace bamradio::fft;
    // get some random samples and copy them in the fft input buffer.
    auto samples = make_complex_gaussian(_fft_in_size * _num_chan, 33);
    CUDA(cudaMemcpy(_d_fft_in, samples.data(),
                    sizeof(samples[0]) * samples.size(),
                    cudaMemcpyHostToDevice));

    // call _fft() and copy the result out
    _fft();
    CUDA(cudaDeviceSynchronize());
    std::vector<std::vector<fcomplex>> gpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      gpu_out[i] = std::vector<fcomplex>(_fft_out_size);
      CUDA(cudaMemcpy(gpu_out[i].data(), _d_fft_out + i * _fft_out_size,
                      _fft_out_size * sizeof(fcomplex),
                      cudaMemcpyDeviceToHost));
    }

    // compute the expected outcome with FFTW using my CPU
    auto fft = CPUFFT::make({_N}, true, 1, "channelizer2_test_fftw_wisdom");
    decltype(gpu_out) cpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      cpu_out[i] = std::vector<fcomplex>(_fft_out_size);
      auto istart = samples.data() + i * _fft_in_size;
      auto istop = istart + _nbatch * _L;
      auto o = cpu_out[i].data();
      for (auto p = istart; p < istop; p += _L) {
        fft->execute(_N, p, o);
        o += _N;
      }
    }

    // compare the two results
    for (size_t i = 0; i < _num_chan; ++i) {
#if 1
      complex_vec_close(cpu_out[i], gpu_out[i]);
#else
      BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
          cpu_out[i], gpu_out[i]));
#endif
    }
  }

  void test_filter() {
    // get some random samples and copy them in the fft output buffer
    auto samples = make_complex_gaussian(_fft_out_size * _num_chan, 33);
    CUDA(cudaMemcpy(_d_fft_out, samples.data(),
                    samples.size() * sizeof(samples[0]),
                    cudaMemcpyHostToDevice));

    // lets have each channel use a different filter.
    for (size_t i = 0; i < _num_chan; ++i) {
      setBandwidth(i, i % _nos);
    }

    // call _filter and copy the results out
    _filter();
    CUDA(cudaDeviceSynchronize());
    std::vector<std::vector<fcomplex>> gpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      gpu_out[i] = std::vector<fcomplex>(_fft_out_size);
      CUDA(cudaMemcpy(gpu_out[i].data(), _d_fft_out + i * _fft_out_size,
                      _fft_out_size * sizeof(fcomplex),
                      cudaMemcpyDeviceToHost));
    }

    // compute the expected outcome with libvolk using my CPU
    std::vector<std::vector<fcomplex>> filters(_nos);
    for (size_t i = 0; i < _num_chan; ++i) {
      // TIL that AVX on GPU memory (cudaMallocManaged(...)) does not work...
      filters[i] = std::vector<fcomplex>(_N);
      CUDA(cudaMemcpy(filters[i].data(), _filters_d[i], _N * sizeof(fcomplex),
                      cudaMemcpyDeviceToHost));
    }
    decltype(gpu_out) cpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      cpu_out[i] = std::vector<fcomplex>(_fft_out_size);
      // we advance in steps of _N and multiply the
      auto ip = samples.data() + i * _fft_out_size;
      auto op = cpu_out[i].data();
      for (size_t j = 0; j < _nbatch; ++j) {
        volk_32fc_x2_multiply_32fc(op, ip, filters[_bw_idx[i]].data(), _N);
        ip += _N;
        op += _N;
      }
    }

#if 0
    // dump these to file to inspect them in matlab
    for (size_t i = 0; i < _num_chan; ++i) {
      auto gpufn = boost::format("test_filt_gpu_%1%.32fc") % (int)i;
      auto cpufn = boost::format("test_filt_cpu_%1%.32fc") % (int)i;
      gr::bamofdm::dump_vec(gpufn.str(), gpu_out[i]);
      gr::bamofdm::dump_vec(cpufn.str(), cpu_out[i]);
    }
#endif

    // compare the two results
    for (size_t i = 0; i < _num_chan; ++i) {
#if 1
      complex_vec_close(cpu_out[i], gpu_out[i]);
#else
      BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
          cpu_out[i], gpu_out[i]));
#endif
    }

    // FIXME add test where I take the FFT of the filters and not pull them from
    // the GPU
  }

  void test_discard() {
    // get some random samples and copy them in the fft_output buffer
    auto samples = make_complex_gaussian(_fft_out_size * _num_chan, 33);
    CUDA(cudaMemcpy(_d_fft_out, samples.data(),
                    samples.size() * sizeof(samples[0]),
                    cudaMemcpyHostToDevice));

    // call discard and copy the results out
    _discard();
    CUDA(cudaDeviceSynchronize());
    std::vector<std::vector<fcomplex>> gpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      gpu_out[i] = std::vector<fcomplex>(_dec_in_size);
      CUDA(cudaMemcpy(gpu_out[i].data(), _d_dec_in + i * _dec_in_size,
                      _dec_in_size * sizeof(fcomplex), cudaMemcpyDeviceToHost));
    }

    // compute the expected outcome with memcpy using my CPU
    decltype(gpu_out) cpu_out(_num_chan);
    auto M = _N - _L + 1;
    for (size_t i = 0; i < _num_chan; ++i) {
      cpu_out[i] = std::vector<fcomplex>(_dec_in_size);
      auto ip = samples.data() + i * _fft_out_size;
      auto op = cpu_out[i].data();
      for (size_t j = 0; j < _nbatch; ++j) {
        // advance ip past samples to skip
        memcpy(op, ip + M - 1, _L * sizeof(samples[0]));
        // advance both pointers
        ip += _N;
        op += _L;
      }
    }

    // compare the two results
    for (size_t i = 0; i < _num_chan; ++i) {
#if 1
      complex_vec_close(cpu_out[i], gpu_out[i]);
#else
      BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
          cpu_out[i], gpu_out[i]));
#endif
    }
  }

  void test_decimate() {
    // get some random samples and copy them in the fft_output buffer
    auto samples = make_complex_gaussian(_dec_in_size * _num_chan, 33);
    CUDA(cudaMemcpy(_d_dec_in, samples.data(),
                    samples.size() * sizeof(samples[0]),
                    cudaMemcpyHostToDevice));

    // lets have each channel use a different oversample rate
    for (size_t i = 0; i < _num_chan; ++i) {
      setBandwidth(i, i % _nos);
    }

    // call discard and copy the results out
    _decimate();
    CUDA(cudaDeviceSynchronize());
    std::vector<std::vector<fcomplex>> gpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      gpu_out[i] = std::vector<fcomplex>(_max_out_size);
      CUDA(cudaMemcpy(gpu_out[i].data(), _d_out + i * _max_out_size,
                      _max_out_size * sizeof(fcomplex),
                      cudaMemcpyDeviceToHost));
    }

    // compute the expected outcome using my CPU
    decltype(gpu_out) cpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      cpu_out[i] = std::vector<fcomplex>(_max_out_size);
      auto ip = samples.data() + i * _dec_in_size;
      auto op = cpu_out[i].data();
      size_t k = 0;
      for (size_t j = 0; j < _dec_in_size; j += _os[_bw_idx[i]]) {
        op[k++] = ip[j];
      }
    }

    // compare the two results
    for (size_t i = 0; i < _num_chan; ++i) {
#if 1
      complex_vec_close(cpu_out[i], gpu_out[i]);
#else
      BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
          cpu_out[i], gpu_out[i]));
#endif
    }
  }

  void test_update_history() {
    // get some random samples and copy them in the fft input buffer
    auto samples = make_complex_gaussian(_fft_in_size * _num_chan, 33);
    CUDA(cudaMemcpy(_d_fft_in, samples.data(),
                    samples.size() * sizeof(samples[0]),
                    cudaMemcpyHostToDevice));

    // call discard and copy the results out
    _update_history();
    CUDA(cudaDeviceSynchronize());
    std::vector<std::vector<fcomplex>> gpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      gpu_out[i] = std::vector<fcomplex>(_ntaps - 1);
      CUDA(cudaMemcpy(gpu_out[i].data(), _d_fft_in + i * _fft_in_size,
                      (_ntaps - 1) * sizeof(fcomplex), cudaMemcpyDeviceToHost));
    }

    // gpu_out should contain the last _ntaps-1 samples of the samples buffer
    decltype(gpu_out) cpu_out(_num_chan);
    for (size_t i = 0; i < _num_chan; ++i) {
      cpu_out[i] = std::vector<fcomplex>(_ntaps - 1);
      auto dst = cpu_out[i].data();
      auto src = samples.data() + (i + 1) * _fft_in_size - (_ntaps - 1);
      memcpy(dst, src, (_ntaps - 1) * sizeof(fcomplex));
    }

    // compare the two results
    for (size_t i = 0; i < _num_chan; ++i) {
#if 1
      complex_vec_close(cpu_out[i], gpu_out[i]);
#else
      BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
          cpu_out[i], gpu_out[i]));
#endif
    }
  }
};

//
// test cases
//

BOOST_AUTO_TEST_CASE(mix) {
  using namespace bam::dsp;
  size_t num_chan = 2;
  size_t nbatch = 2;
  size_t K = 2;
  auto N = K * (SubChannel::table()[0].taps.size() - 1);

  channelizer_test t(num_chan, N, nbatch, SubChannel::table());
  t.init();
  t.test_mix();
}

BOOST_AUTO_TEST_CASE(fft) {
  using namespace bam::dsp;
  size_t num_chan = 2;
  size_t nbatch = 2;
  size_t K = 2;
  auto N = K * (SubChannel::table()[0].taps.size() - 1);

  channelizer_test t(num_chan, N, nbatch, SubChannel::table());
  t.init();
  t.test_fft();
}

BOOST_AUTO_TEST_CASE(filter) {
  using namespace bam::dsp;
  size_t num_chan = SubChannel::table().size();
  size_t nbatch = 5;
  size_t K = 3;
  auto N = K * (SubChannel::table()[0].taps.size() - 1);

  channelizer_test t(num_chan, N, nbatch, SubChannel::table());
  t.init();
  t.test_filter();
}

BOOST_AUTO_TEST_CASE(discard) {
  using namespace bam::dsp;
  size_t num_chan = 2;
  size_t nbatch = 2;
  size_t K = 2;
  auto N = K * (SubChannel::table()[0].taps.size() - 1);

  channelizer_test t(num_chan, N, nbatch, SubChannel::table());
  t.init();
  t.test_discard();
}

BOOST_AUTO_TEST_CASE(decimate) {
  using namespace bam::dsp;
  size_t num_chan = 2;
  size_t nbatch = 2;
  size_t K = 2;
  auto N = K * (SubChannel::table()[0].taps.size() - 1);

  channelizer_test t(num_chan, N, nbatch, SubChannel::table());
  t.init();
  t.test_decimate();
}

BOOST_AUTO_TEST_CASE(history) {
  using namespace bam::dsp;
  size_t num_chan = 2;
  size_t nbatch = 2;
  size_t K = 2;
  auto N = K * (SubChannel::table()[0].taps.size() - 1);

  channelizer_test t(num_chan, N, nbatch, SubChannel::table());
  t.init();
  t.test_update_history();
}
