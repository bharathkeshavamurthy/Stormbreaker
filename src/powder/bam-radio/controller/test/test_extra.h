// common code for unit tests that is not used anywhere elsen
//
// Copyright (c) 2018 Dennis Ogbe

#ifndef n64ce3f39a64535ed8b7
#define n64ce3f39a64535ed8b7

#include "phy.h"
#include <boost/filesystem.hpp>
#include <gnuradio/hier_block2.h>

namespace bamradio {

constexpr double PI() { return 3.141592653589793238463; }

namespace test {

// delete a file if it exists
void deletefile(std::string const &fn);

// input: critically sampled, unsynchronized stream of complex valued samples
// output: none, but write metrics and samples to ChannelOutputBuffer using
// legacy sc_synch block
class COBSink : public gr::hier_block2 {
public:
  typedef boost::shared_ptr<COBSink> sptr;
  static sptr make(size_t const L, ofdm::ChannelOutputBuffer::sptr cob) {
    return gnuradio::get_initial_sptr(new COBSink(L, cob));
  }

private:
  COBSink(size_t const L, ofdm::ChannelOutputBuffer::sptr cob);
  ofdm::ChannelOutputBuffer::sptr _cob;
};

std::vector<fcomplex> make_complex_gaussian(size_t n, int seed = 33,
                                            float mean = 0.0,
                                            float stddev = 1.0);

std::vector<float> make_real_uniform(size_t n, int seed, float a = 0,
                                     float b = 1);

std::vector<uint8_t> make_random_bytes(size_t n, int seed = 33);

void float_vec_close(std::vector<float> const &v1, std::vector<float> const &v2,
                     float percentage = 1.0f);

void complex_vec_close(std::vector<fcomplex> const &v1,
                       std::vector<fcomplex> const &v2,
                       float percentage = 1.0f);

void float_vec_close_2(std::vector<float> const &v1,
                       std::vector<float> const &v2, float tol = 1.0e-5);

void complex_vec_close_2(std::vector<fcomplex> const &v1,
                         std::vector<fcomplex> const &v2, float tol = 1.0e-5);

template <class T> void print_vec_vert(std::vector<T> const &v) {
  for (auto const &elem : v) {
    std::cout << elem << "\n";
  }
  std::cout << std::endl;
}

template <class T>
void print_two_vec_vert(std::vector<T> const &v1, std::vector<T> const &v2) {
  if (v1.size() != v2.size()) {
    std::cout << "WARNING: vectors are not the same size" << std::endl;
  }
  auto n = std::min(v1.size(), v2.size());
  for (int i = 0; i < n; ++i) {
    std::cout << v1[i] << "\t" << v2[i] << "\n";
  }
  std::cout << std::endl;
}

gr::tag_t make_tag(uint64_t offset, pmt::pmt_t key, pmt::pmt_t value,
                   pmt::pmt_t srcid = pmt::PMT_F);

} // namespace test
} // namespace bamradio

#endif // n64ce3f39a64535ed8b7
