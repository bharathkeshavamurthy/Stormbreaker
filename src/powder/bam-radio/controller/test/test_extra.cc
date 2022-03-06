#include "test_extra.h"
#include "common.h"

#include "legacy_phy.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <random>

namespace bamradio {
namespace test {

void deletefile(std::string const &fn) {
  boost::filesystem::path p(fn);
  if (boost::filesystem::exists(p)) {
    boost::filesystem::remove(p);
  }
}

COBSink::COBSink(size_t const L, ofdm::ChannelOutputBuffer::sptr cob)
    : gr::hier_block2("cobsink",
                      gr::io_signature::make(1, 1, sizeof(gr_complex)),
                      gr::io_signature::make(0, 0, 0)),
      _cob(cob) {
  auto synch = ofdm::legacy::sc_sync::make(L, 0);
  auto sample_sink = gr::bamofdm::lambda_sync_block_10<gr_complex>::make(
      [](auto b) {},
      [this](auto b, auto i, auto N) -> ssize_t {
        auto n = std::min((ssize_t)N, _cob->samples->space_avail());
        if (n < 0) {
          return 0;
        }
        memcpy(_cob->samples->write_ptr(), i, sizeof(gr_complex) * n);
        _cob->samples->produce(n);
        return n;
      });
  auto Md_sink = gr::bamofdm::lambda_sync_block_10<float>::make(
      [](auto b) {},
      [this](auto b, auto i, auto N) -> ssize_t {
        auto n = std::min((ssize_t)N, _cob->Md->space_avail());
        if (n < 0) {
          return 0;
        }
        memcpy(_cob->Md->write_ptr(), i, sizeof(float) * n);
        _cob->Md->produce(n);
        return n;
      });
  auto Pd_sink = gr::bamofdm::lambda_sync_block_10<gr_complex>::make(
      [](auto b) {},
      [this](auto b, auto i, auto N) -> ssize_t {
        auto n = std::min((ssize_t)N, _cob->Pd->space_avail());
        if (n < 0) {
          return 0;
        }
        memcpy(_cob->Pd->write_ptr(), i, sizeof(gr_complex) * n);
        _cob->Pd->produce(n);
        return n;
      });
  connect(self(), 0, synch, 0);
  connect(self(), 0, sample_sink, 0);
  connect(synch, 0, Md_sink, 0);
  connect(synch, 1, Pd_sink, 0);
}

std::vector<fcomplex> make_complex_gaussian(size_t n, int seed, float mean,
                                            float stddev) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);
  std::vector<fcomplex> samples(n);
  std::generate(begin(samples), end(samples), [&] {
    using namespace std::complex_literals;
    return dist(gen) + dist(gen) * 1.0if;
  });
  return samples;
}

std::vector<float> make_real_uniform(size_t n, int seed, float a, float b) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(a, b);
  std::vector<float> samples(n);
  std::generate(begin(samples), end(samples), [&] { return dist(gen); });
  return samples;
}

std::vector<uint8_t> make_random_bytes(size_t n, int seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::vector<uint8_t> o(n);
  std::generate(begin(o), end(o), [&] { return dist(gen); });
  return o;
}

void float_vec_close(std::vector<float> const &v1, std::vector<float> const &v2,
                     float percentage) {
  BOOST_REQUIRE_EQUAL(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    BOOST_REQUIRE_CLOSE(v1[i], v2[i], percentage);
  }
}

void complex_vec_close(std::vector<fcomplex> const &v1,
                       std::vector<fcomplex> const &v2, float percentage) {
  BOOST_REQUIRE_EQUAL(v1.size(), v2.size());

  std::vector<float> of(v1.size() * 2);
  auto ofp = (float *)v1.data();
  std::generate(begin(of), end(of), [&] { return *ofp++; });

  std::vector<float> vof(v1.size() * 2);
  auto vofp = (float *)v2.data();
  std::generate(begin(vof), end(vof), [&] { return *vofp++; });
  float_vec_close(of, vof, percentage);
}

// "dumber" implementations of the above
void float_vec_close_2(std::vector<float> const &v1,
                       std::vector<float> const &v2, float tol) {
  BOOST_REQUIRE_EQUAL(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    BOOST_REQUIRE_SMALL(std::abs(v1[i] - v2[i]), tol);
  }
}

void complex_vec_close_2(std::vector<fcomplex> const &v1,
                         std::vector<fcomplex> const &v2, float tol) {
  BOOST_REQUIRE_EQUAL(v1.size(), v2.size());

  std::vector<float> of(v1.size() * 2);
  auto ofp = (float *)v1.data();
  std::generate(begin(of), end(of), [&] { return *ofp++; });

  std::vector<float> vof(v1.size() * 2);
  auto vofp = (float *)v2.data();
  std::generate(begin(vof), end(vof), [&] { return *vofp++; });
  float_vec_close_2(of, vof, tol);
}

gr::tag_t make_tag(uint64_t offset, pmt::pmt_t key, pmt::pmt_t value,
                   pmt::pmt_t srcid) {
  gr::tag_t t;
  t.offset = offset;
  t.key = key;
  t.value = value;
  t.srcid = srcid;
  return t;
}

} // namespace test
} // namespace bamradio
