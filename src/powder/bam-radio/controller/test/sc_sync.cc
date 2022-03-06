// GPU Schmidl & Cox sync tests
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE gpu_sc_sync
#define BOOST_TEST_DYN_LINK
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include "common.h"
#include "phy.h"
#include "sc_sync.h"

#include "legacy_phy.h"
#include "test_extra.h"

#if __has_include(<gnuradio/blocks/vector_sink_c.h>)
#include <gnuradio/blocks/vector_sink_c.h>
#include <gnuradio/blocks/vector_sink_f.h>
#include <gnuradio/blocks/vector_source_c.h>
#else
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#endif
#include <gnuradio/top_block.h>

#include <cuda_runtime.h>
#include <volk/volk.h>

#include <algorithm>
#include <random>
#include <vector>

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

using namespace bamradio;
using namespace bamradio::test;

class sc_sync_test : public bam::dsp::SCSync {
public:
  sc_sync_test(size_t L, size_t mis, std::vector<cudaStream_t> const &streams)
      : bam::dsp::SCSync(L, 1, mis, streams) {}

  void init() {
    using namespace std::complex_literals;
    // this function clears the history
    std::vector<fcomplex> chistory(_L, 0.0if);
    std::vector<fcomplex> fhistory(_L, 0.0);
    CUDA(cudaMemcpy(_d_samp_hist, chistory.data(), _L * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(_d_Pd_hist, chistory.data(), _L * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(_d_Rd2_hist, fhistory.data(), _L * sizeof(float),
                    cudaMemcpyHostToDevice));
  }

  void test_mul_conj_delay(std::vector<fcomplex> const &samples,
                           std::vector<fcomplex> const &history) {
    init();
    cuComplex *d_in;
    CUDA(cudaMalloc(&d_in, samples.size() * sizeof(fcomplex)));
    CUDA(cudaMemcpy(d_in, samples.data(), samples.size() * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));

    // copy the history on the GPU
    assert(_L == history.size());
    CUDA(cudaMemcpy(_d_samp_hist, history.data(), _L * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));

    // execute the computation
    execute({samples.size()}, d_in);
    std::vector<fcomplex> out(samples.size());
    CUDA(cudaMemcpy(out.data(), _d_mul_conj_delay,
                    samples.size() * sizeof(fcomplex), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(d_in));

    // we should be able to compute the output using volk
    std::vector<fcomplex> volk_in;
    volk_in.insert(end(volk_in), begin(history), end(history));
    volk_in.insert(end(volk_in), begin(samples), end(samples));
    decltype(volk_in) volk_out(samples.size());
    volk_32fc_x2_multiply_conjugate_32fc(volk_out.data(), volk_in.data() + _L,
                                         volk_in.data(), samples.size());

#if 1
    // does it match up? (check if results are within 1% of each other)
    complex_vec_close(out, volk_out);
#else
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(out, volk_out));
#endif
  }

  void test_Pd_1(std::vector<fcomplex> const &samples) {
    init();
    cuComplex *d_in;
    CUDA(cudaMalloc(&d_in, samples.size() * sizeof(fcomplex)));
    CUDA(cudaMemcpy(d_in, samples.data(), samples.size() * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));

    // execute the computation
    execute({samples.size()}, d_in);
    std::vector<fcomplex> Pdout(samples.size());
    std::vector<fcomplex> Mout(samples.size());
    CUDA(cudaMemcpy(Pdout.data(), _d_Pd, samples.size() * sizeof(fcomplex),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(Mout.data(), _d_mul_conj_delay,
                    samples.size() * sizeof(fcomplex), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(d_in));

    // copy/paste the sync blocks from the legacy version and compare
    auto const tb = gr::make_top_block("bamradio-gpu-sc-sync-test");
    auto const vsrc = gr::blocks::vector_source_c::make(samples);
    auto const Pdsnk = gr::blocks::vector_sink_c::make();
    auto const Msnk = gr::blocks::vector_sink_c::make();

    auto L = _L;
    auto const mul_conj_delay =
        gr::bamofdm::lambda_sync_block_11<gr_complex, gr_complex>::make(
            [L](auto b) {
              b->set_tag_propagation_policy(gr::block::TPP_DONT);
              b->set_history(L + 1);
              b->declare_sample_delay(L);
            },
            [L](auto, auto in, auto out, auto N) {
              volk_32fc_x2_multiply_conjugate_32fc(out, in + L, in, N);
              return N;
            });
    auto const Pd =
        gr::bamofdm::lambda_sync_block_11<gr_complex, gr_complex>::make(
            [L](auto b) {
              b->set_tag_propagation_policy(gr::block::TPP_DONT);
              b->set_history(L);
              b->declare_sample_delay(L - 1);
              // b->set_max_noutput_items(256);
            },
            [L](auto, auto in, auto Pd, auto N) {
              gr_complex s =
                  std::accumulate(in, in + L, gr_complex(0.0f, 0.0f));
              Pd[0] = s;
              bool reset = false;
              for (size_t i = 1; i < N; ++i) {
                bool nreset = std::norm(in[0] / in[L]) > (500.0f * 500.0f);
                if (reset && !nreset) {
                  return i;
                } else {
                  reset = nreset;
                }
                s = s - in[0] + in[L];
                ++in;
                Pd[i] = s;
              }
              return N;
            });

    tb->connect(vsrc, 0, mul_conj_delay, 0);
    tb->connect(mul_conj_delay, 0, Msnk, 0);
    tb->connect(mul_conj_delay, 0, Pd, 0);
    tb->connect(Pd, 0, Pdsnk, 0);
    tb->run();
    auto const ref_Pd = Pdsnk->data();
    auto const ref_mcd = Msnk->data();

#if 0
    // for extra verbose output
    print_two_vec_vert(ref_mcd, Mout);
    print_two_vec_vert(ref_Pd, Pdout);

    complex_vec_close(ref_mcd, Mout);
    complex_vec_close(ref_Pd, Pdout);
#else
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(ref_mcd, Mout));
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(ref_Pd, Pdout));
#endif
  }

  void test_Rd2(std::vector<fcomplex> const &samples) {
    init();

    cuComplex *d_in;
    CUDA(cudaMalloc(&d_in, samples.size() * sizeof(fcomplex)));
    CUDA(cudaMemcpy(d_in, samples.data(), samples.size() * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));

    // execute the computation
    execute({samples.size()}, d_in);
    std::vector<float> Rd2out(samples.size());
    std::vector<float> Sout(samples.size());
    CUDA(cudaMemcpy(Rd2out.data(), _d_Rd2, samples.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(Sout.data(), _d_samp_energy, samples.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaFree(d_in));

    // copy/paste the sync blocks from the legacy version and compare
    auto const tb = gr::make_top_block("bamradio-gpu-sc-sync-test");
    auto const vsrc = gr::blocks::vector_source_c::make(samples);
    auto const Rd2snk = gr::blocks::vector_sink_f::make();
    auto const Ssnk = gr::blocks::vector_sink_f::make();

    auto L = _L;
    // Compute (R(d))^2
    // R(d) = sum_{i=0}^{L-1} |r2[d+L+i]|^2

    // o[d] = |r[d+L]|^2 = r[d+L] * conj(r[d+L])
    auto const samp_energy_delay =
        gr::bamofdm::lambda_sync_block_11<gr_complex, float>::make(
            [L](auto b) {
              b->set_tag_propagation_policy(gr::block::TPP_DONT);
              b->set_history(L + 1);
              b->declare_sample_delay(L);
            },
            [L](auto, auto in, auto out, auto N) {
              volk_32fc_magnitude_squared_32f(out, in + L, N);
              return N;
            });

    // o[n] = | sum_{i=0}^{L-1} r[n+i] |^2
    // TODO: determine if max_noutput_items needs to be set for numerical
    // stability
    auto const Rd2 = gr::bamofdm::lambda_sync_block_11<float, float>::make(
        [L](auto b) {
          b->set_tag_propagation_policy(gr::block::TPP_DONT);
          b->set_history(L);
          b->declare_sample_delay(L - 1);
          // b->set_max_noutput_items(256);
        },
        [L](auto, auto in, auto Rd2, auto N) {
          float s = std::accumulate(in, in + L, 0.0f);
          Rd2[0] = s * s;
          bool reset = false;
          for (size_t i = 1; i < N; ++i) {
            bool nreset = in[0] / in[L] > 500.0f;
            if (reset && !nreset) {
              return i;
            } else {
              reset = nreset;
            }
            s = s - in[0] + in[L];
            ++in;
            Rd2[i] = s * s;
          }
          return N;
        });

    tb->connect(vsrc, 0, samp_energy_delay, 0);
    tb->connect(samp_energy_delay, 0, Ssnk, 0);
    tb->connect(samp_energy_delay, 0, Rd2, 0);
    tb->connect(Rd2, 0, Rd2snk, 0);
    tb->run();
    auto const ref_Rd2 = Rd2snk->data();
    auto const ref_sed = Ssnk->data();

#if 0
    print_two_vec_vert(ref_sed, Sout);
    print_two_vec_vert(ref_Rd2, Rd2out);

    float_vec_close(ref_sed, Sout);
    float_vec_close(ref_Rd2, Rd2out);
#else
    BOOST_REQUIRE(
        gr::bamofdm::assert_float_vectors_almost_equal(ref_sed, Sout));
    BOOST_REQUIRE(
        gr::bamofdm::assert_float_vectors_almost_equal(ref_Rd2, Rd2out));
#endif
  }

  void test_Md_1(std::vector<fcomplex> const &samples) {
    init();
    cuComplex *d_in;
    CUDA(cudaMalloc(&d_in, samples.size() * sizeof(fcomplex)));
    CUDA(cudaMemcpy(d_in, samples.data(), samples.size() * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));

    // execute the computation
    execute({samples.size()}, d_in);
    std::vector<fcomplex> Pdout(samples.size());
    std::vector<float> Mdout(samples.size());
    CUDA(cudaMemcpy(Pdout.data(), _d_Pd, samples.size() * sizeof(fcomplex),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(Mdout.data(), _d_Md, samples.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaFree(d_in));

    auto const tb = gr::make_top_block("bamradio-gpu-sc-sync-test");
    auto const ref_sc_sync = ofdm::legacy::sc_sync::make(_L, 1e6);
    auto const vsrc = gr::blocks::vector_source_c::make(samples);
    auto const Md_snk = gr::blocks::vector_sink_f::make();
    auto const Pd_snk = gr::blocks::vector_sink_c::make();
    tb->connect(vsrc, 0, ref_sc_sync, 0);
    tb->connect(ref_sc_sync, 0, Md_snk, 0);
    tb->connect(ref_sc_sync, 1, Pd_snk, 0);
    tb->run();
    auto const ref_Md = Md_snk->data();
    auto const ref_Pd = Pd_snk->data();

#if 0
    print_two_vec_vert(ref_Md, Mdout);
    print_two_vec_vert(ref_Pd, Pdout);

    float_vec_close(ref_Md, Mdout);
    complex_vec_close(ref_Pd, Pdout);
#else
    BOOST_REQUIRE(
        gr::bamofdm::assert_float_vectors_almost_equal(ref_Md, Mdout));
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(ref_Pd, Pdout));
#endif
  }

  void test_Md_2(std::vector<fcomplex> const &samples) {
    init();
    cuComplex *d_in;
    CUDA(cudaMalloc(&d_in, samples.size() * sizeof(fcomplex)));
    CUDA(cudaMemcpy(d_in, samples.data(), samples.size() / 2 * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));

    // execute the computation in two steps
    execute({samples.size() / 2}, d_in);
    std::vector<fcomplex> Pdout(samples.size());
    std::vector<float> Mdout(samples.size());
    CUDA(cudaMemcpy(Pdout.data(), _d_Pd, samples.size() / 2 * sizeof(fcomplex),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(Mdout.data(), _d_Md, samples.size() / 2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    CUDA(cudaMemcpy(d_in, samples.data() + samples.size() / 2,
                    samples.size() / 2 * sizeof(fcomplex),
                    cudaMemcpyHostToDevice));
    execute({samples.size() / 2}, d_in);
    CUDA(cudaMemcpy(Pdout.data() + samples.size() / 2, _d_Pd,
                    samples.size() / 2 * sizeof(fcomplex),
                    cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(Mdout.data() + samples.size() / 2, _d_Md,
                    samples.size() / 2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    CUDA(cudaFree(d_in));

    auto const tb = gr::make_top_block("bamradio-gpu-sc-sync-test");
    auto const ref_sc_sync = ofdm::legacy::sc_sync::make(_L, 8);
    auto const vsrc = gr::blocks::vector_source_c::make(samples);
    auto const Md_snk = gr::blocks::vector_sink_f::make();
    auto const Pd_snk = gr::blocks::vector_sink_c::make();
    tb->connect(vsrc, 0, ref_sc_sync, 0);
    tb->connect(ref_sc_sync, 0, Md_snk, 0);
    tb->connect(ref_sc_sync, 1, Pd_snk, 0);
    tb->run();
    auto const ref_Md = Md_snk->data();
    auto const ref_Pd = Pd_snk->data();

#if 0
    print_two_vec_vert(ref_Md, Mdout);
    print_two_vec_vert(ref_Pd, Pdout);

    float_vec_close(ref_Md, Mdout);
    complex_vec_close(ref_Pd, Pdout);
#else
    BOOST_REQUIRE(
        gr::bamofdm::assert_float_vectors_almost_equal(ref_Md, Mdout));
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(ref_Pd, Pdout));
#endif
  }

private:
  std::vector<cudaStream_t> _cuda_streams;
};

void compare_sc_sync(std::vector<fcomplex> const &samples, size_t L) {
  auto const mis = 1e6;
  // compute the desired output using the tried-and-true legacy sc_sync block
  // (n.b. copy the code for this into this test code once we are done writing
  // the GPU implementation)
  auto const tb = gr::make_top_block("bamradio-gpu-sc-sync-test");
  auto const ref_sc_sync =
      ofdm::legacy::sc_sync::make(L, mis); // Nc is never used in this block
  auto const vsrc = gr::blocks::vector_source_c::make(samples);
  auto const Md_snk = gr::blocks::vector_sink_f::make();
  auto const Pd_snk = gr::blocks::vector_sink_c::make();
  tb->connect(vsrc, 0, ref_sc_sync, 0);
  tb->connect(ref_sc_sync, 0, Md_snk, 0);
  tb->connect(ref_sc_sync, 1, Pd_snk, 0);
  tb->run();
  auto const ref_Md = Md_snk->data();
  auto const ref_Pd = Pd_snk->data();

  // compute the output using the GPU implementation
  BOOST_REQUIRE_GE(mis, samples.size());
  std::vector<cudaStream_t> streams(1);
  std::generate(begin(streams), end(streams), [] {
    cudaStream_t s;
    CUDA(cudaStreamCreate(&s));
    return s;
  });

  auto const my_sc_sync = bam::dsp::SCSync::make(L, 1, mis, streams);
  std::vector<size_t> const nsamp = {samples.size()};

  cuComplex *d_in;
  CUDA(cudaMalloc(&d_in, mis * sizeof(fcomplex)));
  CUDA(cudaMemcpy(d_in, samples.data(), samples.size() * sizeof(samples[0]),
                  cudaMemcpyHostToDevice));

  auto my_out = std::make_shared<bamradio::ofdm::ChannelOutputBuffer>(mis);
  std::vector<decltype(my_out)> h_out = {my_out};

  my_sc_sync->execute(nsamp, d_in);
  my_sc_sync->write_output(nsamp, d_in, h_out);
  CUDA(cudaFree(d_in));
  decltype(ref_Md) my_Md(my_out->Md->read_ptr(),
                         my_out->Md->read_ptr() + my_out->Md->items_avail());
  decltype(ref_Pd) my_Pd(my_out->Pd->read_ptr(),
                         my_out->Pd->read_ptr() + my_out->Pd->items_avail());

  gr::bamofdm::dump_vec("ref_Md.32f", ref_Md);
  gr::bamofdm::dump_vec("my_Md.32f", my_Md);
  gr::bamofdm::dump_vec("ref_Pd.32fc", ref_Pd);
  gr::bamofdm::dump_vec("my_Pd.32fc", my_Pd);

  // compare the outputs. they need to be very close
  BOOST_REQUIRE_EQUAL(ref_Md.size(), my_Md.size());
  BOOST_REQUIRE_EQUAL(ref_Pd.size(), my_Pd.size());

#if 1
  // print_two_vec_vert(ref_Md, Mdout);
  // print_two_vec_vert(ref_Pd, Pdout);

  float_vec_close(ref_Md, my_Md, 5);
  complex_vec_close(ref_Pd, my_Pd, 5);
#else
  BOOST_REQUIRE(gr::bamofdm::assert_float_vectors_almost_equal(ref_Md, my_Md));
  BOOST_REQUIRE(
      gr::bamofdm::assert_complex_vectors_almost_equal(ref_Pd, my_Pd));
#endif

  for (auto &stream : streams) {
    cudaStreamDestroy(stream);
  }
}

BOOST_AUTO_TEST_CASE(gpu_sc_sync_mul_conj) {
  std::vector<cudaStream_t> streams(1);
  std::generate(begin(streams), end(streams), [] {
    cudaStream_t s;
    CUDA(cudaStreamCreate(&s));
    return s;
  });
  size_t L = 5;
  sc_sync_test t(L, 50, streams);
  auto samples = make_complex_gaussian(20, 33);

  // test with zero history
  auto history = make_complex_gaussian(L, 22);
  t.test_mul_conj_delay(samples, history);

  // test the Pd calculation
  t.test_Pd_1(samples);

  // test the Rd2 calculation
  t.test_Rd2(samples);

  // test the Md/Pd one-shot calc
  t.test_Md_1(samples);

  // test the Md/Pd two-shot calc
  t.test_Md_2(samples);

  // clean up streams
  for (auto &stream : streams) {
    cudaStreamDestroy(stream);
  }
}

BOOST_AUTO_TEST_CASE(gpu_sc_sync_long) {
  // compare using gaussian noise
  // size_t const nsamples = 1e3;
  // size_t const L = 64;
  // auto samples = make_complex_gaussian(nsamples, 666);
  size_t L = 64;
  auto samples = make_complex_gaussian(20000, 33);
  compare_sc_sync(samples, L);
}
