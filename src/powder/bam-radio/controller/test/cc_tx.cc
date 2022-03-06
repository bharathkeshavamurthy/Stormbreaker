// control channel transmitter unit tests
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE cc_tx
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "cc_data.h"
#include "phy.h"

#include "test_extra.h"

#include <gnuradio/basic_block.h>
#include <gnuradio/blocks/repack_bits_bb.h>
#if __has_include(<gnuradio/blocks/vector_sink_b.h>)
#include <gnuradio/blocks/vector_sink_b.h>
#include <gnuradio/blocks/vector_sink_c.h>
#include <gnuradio/blocks/vector_sink_f.h>
#include <gnuradio/blocks/vector_source_b.h>
#include <gnuradio/blocks/vector_source_c.h>
#include <gnuradio/blocks/vector_source_f.h>
#else
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#endif
#include <gnuradio/gr_complex.h>
#include <gnuradio/top_block.h>
#include <pmt/pmt.h>

#include <bamfsk/insertPreamble_bb.h>
#include <bamfsk/mfskMod_fc.h>
#include <bamfsk/rs_ccsds_encode_bb.h>
#include <bamfsk/tsb_chunks_to_symbols_bf.h>

#include <chrono>
#include <thread>

using fcomplex = bamradio::fcomplex;
using namespace bamradio::test;

// test class for CC tx
class phy_ctrl_mod_test : public bamradio::controlchannel::phy_ctrl_mod {
public:
  phy_ctrl_mod_test(float sample_rate, float bandwidth, size_t rs_k,
                    size_t npoints, float scale)
      : bamradio::controlchannel::phy_ctrl_mod(sample_rate, bandwidth, rs_k,
                                               npoints, scale) {}

  bamradio::controlchannel::phy_ctrl_mod::Resource r;

  void test_encode() {
    std::string tsb_length_tag("length");
    // random bytes and call encode
    size_t const nbytes = 700;
    auto const bytes = make_random_bytes(nbytes);
    r.bytes = bytes;
    _encode(nbytes, &r);
    // gnu radio mock
    auto tb = gr::make_top_block("test_rs");
    auto vsrc = gr::blocks::vector_source_b::make({});
    auto vsnk = gr::blocks::vector_sink_b::make();
    auto rs_enc = gr::bamfsk::rs_ccsds_encode_bb::make(_rs_k, tsb_length_tag);
    auto add_pre =
        gr::bamfsk::insertPreamble_bb::make(preamble, tsb_length_tag);
    tb->connect(vsrc, 0, rs_enc, 0);
    tb->connect(rs_enc, 0, add_pre, 0);
    tb->connect(add_pre, 0, vsnk, 0);
    auto length_tag = bamradio::test::make_tag(0, pmt::intern(tsb_length_tag),
                         pmt::from_long(nbytes));
    vsrc->set_data(bytes, {length_tag});
    tb->run();
    // compare results
    auto const ref_out = vsnk->data();
    auto const my_out = r.coded_with_preamble;
    BOOST_REQUIRE_EQUAL_COLLECTIONS(begin(my_out), end(my_out), begin(ref_out),
                                    end(ref_out));
  }

  void test_unpack() {
    std::string tsb_length_tag("length");
    // some random bytes
    size_t nbytes = 836;
    auto const bytes = make_random_bytes(nbytes);
    r.coded_with_preamble = bytes;
    _unpack(nbytes, &r);
    // gnu radio mock
    auto tb = gr::make_top_block("test_unpack");
    auto vsrc = gr::blocks::vector_source_b::make({});
    auto vsnk = gr::blocks::vector_sink_b::make();
    auto unpack = gr::blocks::repack_bits_bb::make(8, _bps, tsb_length_tag);
    tb->connect(vsrc, 0, unpack, 0);
    tb->connect(unpack, 0, vsnk, 0);
    auto length_tag = bamradio::test::make_tag(0, pmt::intern(tsb_length_tag),
                         pmt::from_long(nbytes));
    vsrc->set_data(bytes, {length_tag});
    tb->run();
    // compare results
    auto const ref_out = vsnk->data();
    auto const my_out = r.chunks;
    BOOST_REQUIRE_EQUAL_COLLECTIONS(begin(my_out), end(my_out), begin(ref_out),
                                    end(ref_out));
  }

  void test_mod() {
    std::string tsb_length_tag("length");
    // some random bytes
    size_t nbytes = 836;
    auto const bytes = make_random_bytes(nbytes);
    r.coded_with_preamble = bytes;
    auto const nchunks = _unpack(nbytes, &r);
    std::vector<fcomplex> my_out;
    auto const nsamp = _mfsk_mod(nchunks, 0.0f, &r, my_out);
    // gnu radio mock
    auto tb = gr::make_top_block("test_mod");
    auto vsrc = gr::blocks::vector_source_f::make({});
    auto vsnk = gr::blocks::vector_sink_c::make();
    auto mod = gr::bamfsk::mfskMod_fc::make(_sample_rate, _scale);
    tb->connect(vsrc, 0, mod, 0);
    tb->connect(mod, 0, vsnk, 0);
    auto length_tag = bamradio::test::make_tag(0, pmt::intern(tsb_length_tag), pmt::from_long(nsamp));
    vsrc->set_data(r.symbols, {length_tag});
    tb->run();
    auto const ref_out = vsnk->data();
    complex_vec_close(ref_out, my_out);
  }
};

BOOST_AUTO_TEST_CASE(encode) {
  float const sample_rate = 480e3;
  float const scale = 600;
  size_t const rs_k = 188;
  size_t const num_points = 8;
  phy_ctrl_mod_test t(sample_rate, sample_rate, rs_k, num_points, scale);
  t.test_encode();
}

BOOST_AUTO_TEST_CASE(unpack) {
  float const sample_rate = 480e3;
  float const scale = 600;
  size_t const rs_k = 188;
  size_t const num_points = 8;
  phy_ctrl_mod_test t(sample_rate, sample_rate, rs_k, num_points, scale);
  t.test_unpack();
}

BOOST_AUTO_TEST_CASE(mod) {
  float const sample_rate = 480e3;
  float const scale = 600;
  size_t const rs_k = 188;
  size_t const num_points = 8;
  phy_ctrl_mod_test t(sample_rate, sample_rate, rs_k, num_points, scale);
  t.test_mod();
}
