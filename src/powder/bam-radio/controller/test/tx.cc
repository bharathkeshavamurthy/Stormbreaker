// OFDM transmitter unit tests
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE ofdmtx
#define BOOST_TEST_DYN_LINK
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include "bamfftw.h"
#include "common.h"
#include "dll.h"
#include "ippacket.h"
#include "options.h"
#include "phy.h"
#include "statistics.h"

#include "test_extra.h"

// see the yaldpc/ext/FpXX headers for a reason why we are doing this
#define __4140ec62e7fc__YALDPC_FIXED_NEED_BOOST_SERIALZE__
#include "llr_format.h"
#include "ldpc/yaldpc.hpp"

#include <array>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>

#if __has_include(<gnuradio/analog/noise_source_c.h>)
#include <gnuradio/analog/noise_source_c.h>
#include <gnuradio/blocks/add_cc.h>
#else
#include <gnuradio/analog/noise_source.h>
#include <gnuradio/blocks/add_blk.h>
#endif

#include <gnuradio/blocks/rotator_cc.h>
#include <gnuradio/filter/fft_filter_ccf.h>
#include <gnuradio/filter/fft_filter_ccc.h>
#include <gnuradio/filter/firdes.h>
#include <gnuradio/top_block.h>
#include <pmt/pmt.h>

#include <uhd/stream.hpp>

#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>

#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>

using std::cerr;
using std::cout;
using std::endl;
namespace po = boost::program_options;
using bamradio::fcomplex;

// test options
double noise_floor_db;
int nframes;

void init_options() {
  using namespace bamradio;

  // Fake arguments.
  std::vector<char const *> argv = {
      "name",
      "--RF.center_freq=5.79e9",
      "--RF.rf_bandwidth=20e6",
      "--phy.args=addr=0.0.0.0",
      "--phy.fftw_wisdom=fftw_wisdom",
      "--phy.max-noutput-items=20000",
      "--phy.adaptive-mcs-threshold=0.6",
      "--phy.snr-upper-bound=24",
      "--phy_data.freq-offset=5e6",
      "--phy_data.tx-gain=20",
      "--phy_data.rx-gain=20",
      "--phy_data.initial-waveform=DFT_S_OFDM_128_1M",
      "--phy_data.sample-rate=20e6",
      "--phy_data.sync-threshold=0.95",
      "--phy_data.rx-frame-queue-size=20",
      "--phy_data.guard-band=100e3",
      "--phy_data.header-demod-nthreads=1",
      "--phy_data.payload-demod-nthreads=1",
      "--phy_control.freq-offset=-3e6",
      "--phy_control.tx-gain=20",
      "--phy_control.rx-gain=20",
      "--phy_control.sample-rate=500e3",
      "--phy_control.num_fsk_points=3",
      "--phy_control.rs_k=2",
      "--phy_control.min_soft_decs=2",
      "--phy_control.id=0",
      "--phy_control.t_slot=0",
      "--phy_control.atten=0.1",
      "--global.verbose=1",
      "--global.uid=root",
      "--net.tun-iface-prefix=tun",
      "--net.tun-ip4=10.20.30.1",
      "--net.tun-ip4-netmask=255.255.255.0",
      "--net.tun-mtu=1500",
      "--psd_sensing.fft_len=128",
      "--psd_sensing.mov_avg_len=30",
      "--psd_sensing.reset_period=10000",
      "--psd_sensing.bin_size=0.2",
      "--psd_sensing.sn_gap_bins=30",
      "--psd_sensing.empty_bin_items=2",
      "--psd_sensing.hist_avg_len=5",
      "--psd_sensing.noise_floor_db=-70",
      "--psd_sensing.holes_select_mode=0",
      "--psd_sensing.snr_threshold=15",
      "--psd_sensing.contain_count_th=3",
      "--psd_sensing.interf_th=1.2e-5"};
  // Add in true command line args.
  for (int i = 0; i < boost::unit_test::framework::master_test_suite().argc;
       ++i) {
    argv.push_back(boost::unit_test::framework::master_test_suite().argv[i]);
  }

  // Custom test options.
  po::options_description testops("test options");
  testops.add_options()(
      // Option
      "test.noise-floor-db",
      po::value<decltype(noise_floor_db)>(&noise_floor_db)
          ->required()
          ->default_value(-60.),
      "noise floor (dB)")(
      // Option
      "test.nframes",
      po::value<decltype(nframes)>(&nframes)->required()->default_value(9),
      "number of frames to test in e2e test");

  // Init options.
  options::init(argv.size(), &argv[0], &testops);
}

bamradio::ofdm::DFTSOFDMFrame::sptr make_frame(int nseg, int nbytes,
                                               int seed = 33) {
  using namespace bamradio;
  using namespace bamradio::ofdm;
  static int seqnum = 0;
  std::minstd_rand mt(seed);
  std::uniform_int_distribution<> dis(0, 255);
  std::uniform_int_distribution<> pridis(0, 15);
  std::uniform_int_distribution<> classdis(0, 3);

  std::vector<dll::Segment::sptr> segVec;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> bsVec;

  for (size_t j = 0; j < nseg; ++j) {
    auto ip = std::make_shared<net::IPPacket>(nbytes);
    auto b = buffer(ip->get_buffer());
    auto bb = boost::asio::buffer_cast<uint8_t *>(b);
    std::generate(bb + 20, bb + nbytes, [&] { return dis(mt); });
    ip->tos() = (pridis(mt) << 4) | classdis(mt);
    ip->resize(nbytes);
    // set header (UDP, src: 0, dst: 0, no fragment)
    auto const hdr = (struct ip *)bb;
    hdr->ip_hl = 5;
    hdr->ip_v = 4;
    hdr->ip_p = IPPROTO_UDP;
    hdr->ip_tos = 0x00;
    hdr->ip_off = htons(IP_DF);
    auto const l4hdr = (udphdr *)(bb + hdr->ip_hl * 4);
    l4hdr->uh_sport = 0;
    l4hdr->uh_dport = 0;
    auto const bs =
        std::make_shared<std::vector<uint8_t>>(std::move(ip->get_buffer_vec()));
    auto const ip4seg = std::make_shared<net::IP4PacketSegment>(
        0, boost::asio::buffer(*bs), std::chrono::system_clock::now());
    segVec.push_back(ip4seg);
    bsVec.push_back(bs);
  }

  auto f = std::make_shared<DFTSOFDMFrame>(
      0, segVec[0]->destNodeID(), segVec,
      MCS::stringNameToIndex(options::phy::data::header_mcs_name),
      MCS::stringNameToIndex(options::phy::data::initial_payload_mcs_name),
      SeqID::stringNameToIndex(
          options::phy::data::initial_payload_symbol_seq_name),
      seqnum++, 0);

  return f;
}

std::tuple<int, std::string> check_good_float(float *array, int n, bool nz) {
  int nwrong = 0;
  int firstwrong = -1;
  auto end = array + n;
  if (nz) {
    // enforce abs(x) > 0
    for (auto pp = array; pp < end; pp++) {
      if ((!(std::abs(*pp) > 0)) || (!std::isfinite(*pp))) {
        nwrong++;
        if (firstwrong == -1) {
          firstwrong = pp - array;
        }
      }
    }
  } else {
    // just look at finiteness
    for (auto pp = array; pp < end; pp++) {
      if (!std::isfinite(*pp)) {
        nwrong++;
        if (firstwrong == -1) {
          firstwrong = pp - array;
        }
      }
    }
  }
  if (nwrong == 0) {
    return std::tuple<int, std::string>(
        nwrong,
        (boost::format("check_good_float has passed for %1% values.") % n)
            .str());
  } else {
    return std::tuple<int, std::string>(
        nwrong, (boost::format("check_good_float has failed [%1% wrong, %2% "
                               "values]. first error: %3% at position %4%.") %
                 nwrong % n % array[firstwrong] % firstwrong)
                    .str());
  }
}

// fake USRP tx streamer to simulate phy things
class fake_tx_streamer : public uhd::tx_streamer {
public:
  typedef ringbuffer::Ringbuffer<
      gr_complex, ringbuffer::rb_detail::memfd_nocuda_circbuf<gr_complex>>
      rb_type;
  rb_type::sptr rb;
  size_t buf_sz;

  fake_tx_streamer(size_t buf_sz)
      : rb(std::make_shared<rb_type>(buf_sz)), buf_sz(buf_sz) {}
  ~fake_tx_streamer() {}

  size_t get_num_channels(void) const { return 1; }
  size_t get_max_num_samps(void) const { return buf_sz; }

  size_t send(const uhd::tx_streamer::buffs_type &buffs,
              const size_t nsamps_per_buff, const uhd::tx_metadata_t &metadata,
              const double timeout = 0.1) {
    auto n = std::min((size_t const)rb->space_avail(), nsamps_per_buff);
    memcpy(rb->write_ptr(), buffs[0], n * sizeof(*rb->write_ptr()));
    rb->produce(n);
    return n;
  }

  bool recv_async_msg(uhd::async_metadata_t &async_md, double timeout) {
    return false;
  }
};

// test harness class to unit test the individual modulator functions
class phy_tx_test : public bamradio::ofdm::phy_tx {
public:
  phy_tx_test(bamradio::ofdm::phy_tx::prepared_frame pf_)
      : phy_tx(1), r(_resources[0]), pf(pf_),
        params(pf_.frame->allParams(true, pf_.channel.interp_factor, 0)),
        symbols([this] {
          using bamradio::ofdm::OFDMSymbolParams;
          std::vector<OFDMSymbolParams *> o;
          for (auto const &symbol : params.symbols)
            for (size_t i = 0; i < symbol.first; ++i)
              o.push_back(symbol.second.get());
          return o;
        }()) {}

  // need to be able to directly write to the resources
  bamradio::ofdm::phy_tx::Resource *r;
  // keep a frame around to do some testing with
  bamradio::ofdm::phy_tx::prepared_frame pf;
  // all frame params
  bamradio::ofdm::DFTSOFDMFrameParams const params;
  // build the symbol vector like in phy.cc
  std::vector<bamradio::ofdm::OFDMSymbolParams *> const symbols;

  //
  // test methods
  //

  void test_get_window() {
    auto require_close_vec = [](auto const &a, auto const &b) {
      BOOST_REQUIRE_EQUAL(a.size(), b.size());
      for (size_t i = 0; i < a.size(); ++i) {
        BOOST_REQUIRE_CLOSE(a[i], b[i], 0.1);
      }
    };
    // generate the truth in MATLAB (sin^2 window):
    // >> format long
    // >> w = @(n) sin(pi*((0:n-1)+0.5)/(2*n)).^2
    // >> assert(sum(w(100) + fliplr(w(100))) == 100)
    std::vector<float> matlab_head{0.006155829702431, 0.054496737905816,
                                   0.146446609406726, 0.273004750130227,
                                   0.421782767479885, 0.578217232520115,
                                   0.726995249869773, 0.853553390593274,
                                   0.945503262094184, 0.993844170297569};
    std::vector<float> matlab_tail(begin(matlab_head), end(matlab_head));
    std::reverse(begin(matlab_tail), end(matlab_tail));

    require_close_vec(matlab_head, get_window(true, 10, 1));
    require_close_vec(matlab_tail, get_window(false, 10, 1));

    require_close_vec(matlab_head, get_window(true, 5, 2));
  }

  void test_encode_bits() {
    // first get decoders for the header and the payload
    auto get_decoder = [](auto mcs) {
      auto max_its = 30;
      uint64_t rate_idx, block_size;
      std::tie(rate_idx, block_size) =
          bamradio::ofdm::MCS::table[mcs].codePair();
      auto dd_code = yaldpc::ieee80211::get(rate_idx, block_size);
      auto full_code = yaldpc::expand(dd_code);
      return yaldpc::SerialCMinSumDecoder<bamradio::llr_type, uint8_t>::make(
          full_code, max_its, true, true);
    };
    auto hdr_dec = get_decoder(pf.frame->headerMcs());
    auto payl_dec = get_decoder(pf.frame->payloadMcs());

    r->init(params);
    auto nbits = encode_bits(pf.frame, params, 0);

    // decode all blocks and compare to the contents of r->raw_bits
    // layout should be [h1 h2 .. hN p1 p2 ... pM] (all blocks back-to-back in
    // r->coded_bits)
    BOOST_REQUIRE_EQUAL(nbits, params.numBits());
    static_assert(std::is_same<bamradio::llr_type, float>::value,
                  "llr_type must be float for this test");
    std::vector<float> fake_llr(nbits);
    int k = 0;
    std::generate(begin(fake_llr), end(fake_llr),
                  [&] { return r->coded_bits[k++] == 0 ? 1.0f : -1.0f; });
    decltype(r->raw_bits) decoded(r->raw_bits.size());
    auto decode = [&](auto header) -> int {
      auto nblocks = pf.frame->numBlocks(header);
      auto dec = header ? hdr_dec : payl_dec;
      auto in = fake_llr.data() +
                (header ? 0 : pf.frame->numBlocks(true) * hdr_dec->n());
      auto out = decoded.data() +
                 (header ? 0 : pf.frame->numBlocks(true) * hdr_dec->k());
      int nd = 0;
      for (int i = 0; i < nblocks; ++i) {
        dec->decode(in, out);
        in += dec->n();
        out += dec->k();
        nd += dec->n();
      }
      return nd;
    };
    int ndec = 0;
    ndec += decode(true);
    ndec += decode(false);

    // check whether the decoded bits are equivalent to the raw bits written
    BOOST_REQUIRE_EQUAL(ndec, nbits);
    BOOST_REQUIRE_EQUAL_COLLECTIONS(begin(decoded), end(decoded),
                                    begin(r->raw_bits), end(r->raw_bits));

    // now check whether I can re-construct the frame from the decoded header
    // bits
    auto nhdr = pf.frame->numBlocks(true) * hdr_dec->k();
    auto npayl = pf.frame->numBlocks(false) * payl_dec->k();
    auto hbits = boost::asio::const_buffer(decoded.data(), nhdr);
    auto pbits = boost::asio::const_buffer(decoded.data() + nhdr, npayl);
    decltype(pf.frame) f2(nullptr);
    BOOST_REQUIRE_NO_THROW(do {
      f2 = std::make_shared<decltype(pf.frame)::element_type>(
          pf.frame->headerMcs(), hbits);
    } while (0));
    int j = 0;
    while (boost::asio::buffer_size(pbits) >= payl_dec->k()) {
      auto const nread = f2->readBlock(j, pbits);
      BOOST_REQUIRE_NE(nread.second, 0);
      for (int bn = j; bn < j + nread.second; ++bn) {
        BOOST_REQUIRE(f2->blockIsValid(bn));
      }
      j += nread.second;
      pbits = pbits + nread.first * nread.second;
    }
  }

  void test_modulate_bits() {
    r->init(params);
    auto nbits = encode_bits(pf.frame, params, 0);
    auto nwritten = modulate_bits(symbols, 0);

    // we should have as many constellation symbols as there are data carriers
    // on every OFDM symbol of the frame.
    auto n_constellation_symbols =
        std::accumulate(begin(symbols), end(symbols), 0, [](auto a, auto b) {
          return a + b->numDataCarriers();
        });
    BOOST_REQUIRE_EQUAL(n_constellation_symbols, nwritten);

    // there should be no zero-valued symbol any more. compare the absolute
    // value of the real and imaginary parts and make sure they are larger than
    // zero. also check whether it is finite
    int nwrong;
    std::string msg;
    std::tie(nwrong, msg) =
        check_good_float((float *)r->symbols.data(), 2 * nwritten, true);
    BOOST_TEST_MESSAGE(__FILE__ << "(" << __LINE__ << "): " << msg);
    BOOST_REQUIRE_EQUAL(nwrong, 0);

    // demodulate every symbol and compare to the original bits
    std::vector<float> demod_llr(nbits);
    std::vector<uint8_t> demod_bits(nbits);
    auto ndemod = 0;
    auto ip = r->symbols.data();
    auto op = demod_llr.data();
    for (auto const &symbol : symbols) {
      auto constellation = symbol->constellation;
      auto const sidx = constellation->get_snr_idx(20.0);
      assert(constellation != nullptr);
      auto bps = constellation->bits_per_symbol();
      for (int i = 0; i < symbol->numDataCarriers(); ++i) {
        constellation->make_soft_decision(ip[i], op, sidx);
        op += bps;
        ndemod += bps;
      }
      ip += symbol->numDataCarriers();
    }
    BOOST_REQUIRE_EQUAL(ndemod, nbits);
    int k = 0;
    std::generate(begin(demod_bits), end(demod_bits),
                  [&] { return demod_llr[k++] > 0.0 ? 0 : 1; });
    BOOST_REQUIRE_EQUAL_COLLECTIONS(begin(demod_bits), end(demod_bits),
                                    begin(r->coded_bits),
                                    begin(r->coded_bits) + nbits);
  }

  void test_interleave() {
    r->init(params);
    auto nbits = encode_bits(pf.frame, params, 0);
    auto nsymb = modulate_bits(symbols, 0);
    auto nwritten = interleave(nsymb, 0);
    // check that the number of interleaved symbols is the same as the number of
    // deinterleaved symbols
    BOOST_REQUIRE_EQUAL(nsymb, nwritten);

    // now if we deinterleave, we should have the same symbols as before
    auto const n = params.dft_spread_length * symbols.size();
    std::vector<fcomplex> my_deinterleaved(n);
    auto const &interleaver = bamradio::ofdm::Interleaver::get(n);
    for (size_t i = 0; i < n; ++i) {
      my_deinterleaved[i] = r->inter_symbols[interleaver[i]];
    }
    BOOST_REQUIRE(gr::bamofdm::assert_complex_vectors_almost_equal(
        my_deinterleaved, r->symbols));
  }

  void test_spread_symbols() {
    auto spreadlen = params.dft_spread_length;
    r->init(params);
    encode_bits(pf.frame, params, 0);
    auto nwritten = modulate_bits(symbols, 0);
    // hack to skip interleaving step
    r->spread_symbols.resize(nwritten); // hack
    memcpy(r->inter_symbols.data(), r->symbols.data(),
           sizeof(fcomplex) * nwritten);
    auto nspread = spread_symbols(symbols, params, 0);

    // we should not lose any symbols with this
    BOOST_REQUIRE_EQUAL(nspread, nwritten);
    BOOST_REQUIRE_EQUAL(nspread % spreadlen, 0);

    // check that all symbols are finite (contrary to above, some here can be
    // zero...)
    int nwrong;
    std::string msg;
    std::tie(nwrong, msg) = check_good_float((float *)r->spread_symbols.data(),
                                             2 * nwritten, false);
    BOOST_TEST_MESSAGE(__FILE__ << "(" << __LINE__ << "): " << msg);
    BOOST_REQUIRE_EQUAL(nwrong, 0);

    // unspread by hand and see whether we are close to the input symbols
    auto ifft = bamradio::fft::CPUFFT::make(
        {spreadlen}, false, 1, bamradio::options::phy::fftw_wisdom);
    decltype(r->symbols) unspread(nspread);
    auto ip = r->spread_symbols.data();
    auto op = unspread.data();
    for (int i = 0; i < nspread / spreadlen; ++i) {
      ifft->execute(spreadlen, ip, op);
      ip += spreadlen;
      op += spreadlen;
    }
    // normalize
    for (auto &elem : unspread) {
      elem *= (float)1 / spreadlen;
    }
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(r->symbols, unspread));
  }

  void test_map_subcarriers() {
    using namespace std::complex_literals;
    r->init(params);
    // a test symbol with 8 subcarriers, two of them unoccupied, 4 data symbols,
    // and 2 pilot symbols. the data symbols count from 0 to 3 in the real part,
    // the pilots do the same in the imaginary part.
    std::vector<std::complex<float>> const ref = {
        3.0f + 0.0if, // (ind. 0, s.c. 0)
        0.0f + 2.0if, // (ind. 1, s.c. 1)
        4.0f + 0.0if, // (ind. 2, s.c. 2)
        0.0f + 0.0if, // (ind. 3, s.c. 3)
        0.0f + 0.0if, // (ind. 4, s.c. -4)
        1.0f + 0.0if, // (ind. 5, s.c. -3)
        0.0f + 1.0if, // (ind. 6, s.c. -2)
        2.0f + 0.0if  // (ind. 7, s.c. -1)
    };
    bamradio::ofdm::OFDMSymbolParams test_symbol{
        .data_carrier_mapping = pmt::init_s32vector(4, {-3, -1, 0, 2}),
        .pilot_carrier_mapping = pmt::init_s32vector(2, {-2, 1}),
        .pilot_symbols = pmt::init_c32vector(2, {0.0f + 1.0if, 0.0f + 2.0if}),
        .constellation = nullptr,
        .symbol_length = 8,
        .oversample_rate = 1,
        .cyclic_prefix_length = 4,
        .cyclic_postfix_length = 2,
        .prefix = nullptr};
    // write the data carriers
    for (int i = 0; i < 4; ++i) {
      r->spread_symbols[i] = ((float)i + 1) + 0.0if;
    }

    // map this symbol and check whether we were successful
    auto nmapped = map_subcarriers({&test_symbol}, 0);
    BOOST_REQUIRE_EQUAL(nmapped, 6);
    decltype(r->ifft_in) result(begin(r->ifft_in),
                                begin(r->ifft_in) + ref.size());
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(ref, result));

    // with an oversample factor of two, we expect a similar mapping but now
    // with 10 zeros (instead of 2) between the two halves.
    test_symbol.oversample_rate = 2;
    std::fill_n(begin(r->ifft_in), ref.size(), 0.0f + 0.0if);
    nmapped = map_subcarriers({&test_symbol}, 0);
    BOOST_REQUIRE_EQUAL(nmapped, 6);
    decltype(result) result2(begin(r->ifft_in),
                             begin(r->ifft_in) + ref.size() * 2);
    auto ref2 = [&] {
      auto sz = ref.size() * 2;
      decltype(result) o(sz, 0.0if);
      for (int i = 0; i < 3; ++i) {
        o[i] = ref[i];
        o[sz - 1 - i] = ref[ref.size() - 1 - i];
      }
      return o;
    }();
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(ref2, result2));

    // finally, just verify the output after mapping a full frame (count only)
    r->init(params);
    encode_bits(pf.frame, params, 0);
    auto const nsymb = modulate_bits(symbols, 0);
    interleave(nsymb, 0);
    spread_symbols(symbols, params, 0);
    nmapped = map_subcarriers(symbols, 0);
    auto nsymbols =
        std::accumulate(begin(symbols), end(symbols), 0, [](auto a, auto b) {
          return a + b->numDataCarriers() + b->numPilotCarriers();
        });
    BOOST_REQUIRE_EQUAL(nsymbols, nmapped);
  }

  void test_shift_upsample() {
    using namespace std::complex_literals;

    // another test symbol with oversampling and a prefix.
    uint16_t symlen = 8;
    uint16_t os = 2;
    uint16_t ncp = 3;
    uint16_t nw = 2;
    auto prefix = [] {
      std::vector<fcomplex> o;
      for (int i = 0; i < 16; ++i)
        o.push_back(100.0f + i + 0.0if);
      return o;
    }();
    bamradio::ofdm::OFDMSymbolParams sym1{
        .data_carrier_mapping = pmt::init_s32vector(4, {-3, -1, 0, 2}),
        .pilot_carrier_mapping = pmt::init_s32vector(2, {-2, 1}),
        .pilot_symbols = pmt::init_c32vector(2, {0.0f + 1.0if, 0.0f + 2.0if}),
        .constellation = nullptr,
        .symbol_length = symlen,
        .oversample_rate = os,
        .cyclic_prefix_length = ncp,
        .cyclic_postfix_length = nw,
        .prefix = pmt::init_c32vector(prefix.size(), prefix)};
    // take the same symbol, but remove the prefix
    decltype(sym1) sym2(sym1);
    sym2.prefix = nullptr;
    std::vector<decltype(&sym1)> const test_symbols{&sym1, &sym2};

    // take the result (ref2) from the test above (this is the test_symbol
    // mapped to the correct subcarriers)
    auto ref = [=] {
      std::vector<fcomplex> r = {3.0f + 0.0if, 0.0f + 2.0if, 4.0f + 0.0if,
                                 0.0f + 0.0if, 0.0f + 0.0if, 1.0f + 0.0if,
                                 0.0f + 1.0if, 2.0f + 0.0if};
      auto sz = r.size() * os;
      decltype(r) o(sz, 0.0if);
      for (int i = 0; i < 3; ++i) {
        o[i] = r[i];
        o[sz - 1 - i] = r[r.size() - 1 - i];
      }
      return o;
    }();

    // generate the reference symbol in the time domain
    auto ref_td = [&] {
      size_t sz = symlen * os;
      auto ifft = bamradio::fft::CPUFFT::make(
          {sz}, false, 1, bamradio::options::phy::fftw_wisdom);
      decltype(ref) o(sz);
      ifft->execute(sz, ref.data(), o.data());
      return o;
    }();

    // copy two of the "mapped" OFDM symbols back-to-back in the input buffer
    r->init(params);
    std::copy(begin(ref), end(ref), begin(r->ifft_in));
    std::copy(begin(ref), end(ref), begin(r->ifft_in) + ref.size());

    // after shift_upsample, we expect the following layout:
    // [ncp*os zeros] [prefix] [ncp*os zeros] [ref_td] [ncp*os zeros] [ref_td]
    decltype(ref) const zeros(ncp * os, 0.0if);
    decltype(ref) full_ref;
    auto append = [](auto &a, auto const &v) {
      a.insert(end(a), begin(v), end(v));
    };
    append(full_ref, zeros);
    append(full_ref, prefix);
    append(full_ref, zeros);
    append(full_ref, ref_td);
    append(full_ref, zeros);
    append(full_ref, ref_td);
    BOOST_REQUIRE_EQUAL(full_ref.size(), 3 * (symlen + ncp) * os);

    // run the function and check the output
    decltype(full_ref) result(full_ref.size(), 0.0if);
    auto nsamp = shift_upsample(test_symbols, result, 0);
    BOOST_REQUIRE_EQUAL(nsamp, 3 * symlen * os);
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(full_ref, result));

    // now finally check the "full" frame
    // first make sure that our memory allocations are correct
    auto total_samp = std::accumulate(
        begin(symbols), end(symbols),
        // add the last postfix to this
        symbols.back()->cyclic_postfix_length * symbols.back()->oversample_rate,
        [](auto a, auto s) {
          auto const symlen = s->symbol_length;
          auto const cplen = s->cyclic_prefix_length;
          auto const os = s->oversample_rate;
          int n = 0;
          if (s->prefix) {
            size_t prefix_vec_size = 0;
            pmt::uniform_vector_elements(s->prefix, prefix_vec_size);
            n = prefix_vec_size / sizeof(fcomplex);
            assert(n == symlen * os); // only allow same-length prefixes
            n += cplen * os;          // add the cylic prefix to this
          }
          return a + n + (symlen + cplen) * os;
        });
    BOOST_REQUIRE_EQUAL(total_samp, params.numTXSamples());

    // now shift the frame and check the number of produced samples
    r->init(params);
    encode_bits(pf.frame, params, 0);
    auto const nsymb = modulate_bits(symbols, 0);
    interleave(nsymb, 0);
    spread_symbols(symbols, params, 0);
    map_subcarriers(symbols, 0);
    decltype(full_ref) fresult(params.numTXSamples(), 0.0if);
    auto fnsamp = shift_upsample(symbols, fresult, 0);
    auto expected =
        std::accumulate(begin(symbols), end(symbols), 0, [](auto a, auto s) {
          int n = 0;
          if (s->prefix) {
            size_t prefix_vec_size = 0;
            pmt::uniform_vector_elements(s->prefix, prefix_vec_size);
            n = prefix_vec_size / sizeof(fcomplex);
          }
          return a + n + s->symbol_length * s->oversample_rate;
        });
    BOOST_REQUIRE_EQUAL(fnsamp, expected);
    int nwrong;
    std::string msg;
    std::tie(nwrong, msg) =
        check_good_float((float *)fresult.data(), 2 * expected, false);
    BOOST_TEST_MESSAGE(__FILE__ << "(" << __LINE__ << "): " << msg);
    BOOST_REQUIRE_EQUAL(nwrong, 0);
  }

  void test_window_cp() {
    using namespace std::complex_literals;

    // test symbol set-up similar to before. from this we can construct an
    // expected output and we compare to produced output.
    uint16_t symlen = 8;
    uint16_t os = 2;
    uint16_t ncp = 3;
    uint16_t nw = 2;
    auto prefix = [&] {
      std::vector<fcomplex> o;
      for (int i = 0; i < os * symlen; ++i)
        o.push_back((float)(i + 1) * 1.0if);
      return o;
    }();
    auto symb1 = [&] {
      std::vector<fcomplex> o;
      for (int i = 0; i < os * symlen; ++i)
        o.push_back((float)i + 1 + 0.0if);
      return o;
    }();
    auto symb2 = [&] {
      std::vector<fcomplex> o;
      for (int i = 0; i < os * symlen; ++i)
        o.push_back((float)10 * (i + 1) + 0.0if);
      return o;
    }();
    // alternatively (this helps for a quick visual MATLAB check)
    // prefix = std::vector<fcomplex>(prefix.size(), 1.0f + 0.0if);
    // symb1 = std::vector<fcomplex>(prefix.size(), 1.0f + 0.0if);
    // symb2 = std::vector<fcomplex>(prefix.size(), 1.0f + 0.0if);

    bamradio::ofdm::OFDMSymbolParams sym1{
        // the mappings are basically irrelevant for this test
        .data_carrier_mapping = pmt::init_s32vector(4, {-3, -1, 0, 2}),
        .pilot_carrier_mapping = pmt::init_s32vector(0, {}),
        .pilot_symbols = pmt::init_c32vector(0, {}),
        .constellation = nullptr,
        .symbol_length = symlen,
        .oversample_rate = os,
        .cyclic_prefix_length = ncp,
        .cyclic_postfix_length = nw,
        .prefix = pmt::init_c32vector(prefix.size(), prefix)};
    // take the same symbol, but remove the prefix
    decltype(sym1) sym2(sym1);
    sym2.prefix = nullptr;
    std::vector<decltype(&sym1)> const test_symbols{&sym1, &sym2};

    // we expect the following output:
    // clang-format off
  // [ncp*os windowed CP of prefix] [prefix] [nw*os windowed CPost+CP] [(ncp-nw)*os CP of sym1] [sym1] [nw*os windowed CPost+CP] [(ncp-nw)*os CP of sym2] [sym2] [nw*os CPost of sym2]
    // clang-format on

    // VERY NAIVE implementation of the windowing algorithm.
    auto head = get_window(true, nw, os);
    auto tail = get_window(false, nw, os);
    auto get_cp = [&](auto const &v) {
      auto n = v.size();
      std::vector<fcomplex> o;
      o.reserve(ncp * os);
      for (int i = 0; i < ncp * os; ++i) {
        fcomplex val;
        if (i < nw * os) {
          val = head[i] * v[n - ncp * os + i];
        } else {
          val = v[n - ncp * os + i];
        }
        o.push_back(val);
      }
      return o;
    };
    auto get_cpost = [&](auto const &v) {
      std::vector<fcomplex> o;
      o.reserve(nw * os);
      for (int i = 0; i < nw * os; ++i) {
        o.push_back(tail[i] * v[i]);
      }
      return o;
    };
    auto cp1 = get_cp(prefix);
    auto cp2 = get_cp(symb1);
    auto cp3 = get_cp(symb2);
    auto cpost1 = get_cpost(prefix);
    auto cpost2 = get_cpost(symb1);
    auto cpost3 = get_cpost(symb2);
    auto addto = [](auto dst, auto src, auto n) {
      for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
      }
    };
    auto nexpect = (3 * (symlen + ncp) + nw) * os;
    std::vector<fcomplex> ref(nexpect, 0.0if);
    auto d = ref.data();
    addto(d, cp1.data(), cp1.size());
    d += cp1.size();
    addto(d, prefix.data(), prefix.size());
    d += prefix.size();
    addto(d, cpost1.data(), cpost1.size());
    addto(d, cp2.data(), cp2.size());
    d += cp2.size();
    addto(d, symb1.data(), symb1.size());
    d += symb1.size();
    addto(d, cpost2.data(), cpost2.size());
    addto(d, cp3.data(), cp3.size());
    d += cp3.size();
    addto(d, symb2.data(), symb2.size());
    d += symb2.size();
    addto(d, cpost3.data(), cpost3.size());
    int nwrong;
    std::string msg;
    std::tie(nwrong, msg) =
        check_good_float((float *)ref.data(), 2 * nexpect, false);
    BOOST_TEST_MESSAGE(__FILE__ << "(" << __LINE__ << "): " << msg);
    BOOST_REQUIRE_EQUAL(nwrong, 0);

    // now run the function under test
    decltype(ref) result;
    decltype(ref) const zeros(ncp * os, 0.0if);
    auto append = [](auto &a, auto const &v) {
      a.insert(end(a), begin(v), end(v));
    };
    append(result, zeros);
    append(result, prefix);
    append(result, zeros);
    append(result, symb1);
    append(result, zeros);
    append(result, symb2);
    append(result, std::vector<fcomplex>(nw * os, 0.0if));

    auto nsamp = window_cp(test_symbols, result, 0);

    BOOST_REQUIRE_EQUAL(nsamp, nexpect);
    BOOST_REQUIRE(
        gr::bamofdm::assert_complex_vectors_almost_equal(result, ref));

    // as usual, check this for the "real" frame
    decltype(ref) fresult(params.numTXSamples(), 0.0if);
    r->init(params);
    encode_bits(pf.frame, params, 0);
    auto const nsymb = modulate_bits(symbols, 0);
    interleave(nsymb, 0);
    spread_symbols(symbols, params, 0);
    map_subcarriers(symbols, 0);
    shift_upsample(symbols, fresult, 0);
    auto fnsamp = window_cp(symbols, fresult, 0);
    auto fnexpected = params.numTXSamples();
    BOOST_REQUIRE_EQUAL(fnsamp, fnexpected);
    std::tie(nwrong, msg) =
        check_good_float((float *)fresult.data(), 2 * fnexpected, false);
    BOOST_TEST_MESSAGE(__FILE__ << "(" << __LINE__ << "): " << msg);
    BOOST_REQUIRE_EQUAL(nwrong, 0);

    // dump an example frame to check out in matlab
    gr::bamofdm::dump_vec("ofdm-tx-test-frame.32fc", fresult);
  }
};

// run all unit tests defined in the test class above
BOOST_AUTO_TEST_CASE(ofdm_tx_unit) {
  init_options();
  phy_tx_test t({make_frame(2, 1500, 33), bamradio::Channel(1, 0, 20)});
  t.test_get_window();
  t.test_encode_bits();
  t.test_modulate_bits();
  t.test_interleave();
  t.test_spread_symbols();
  t.test_map_subcarriers();
  t.test_shift_upsample();
  t.test_window_cp();
}

// end-to-end test (tx -> channel emulator -> rx)
BOOST_AUTO_TEST_CASE(ofdm_tx_endtoend) {
  using namespace bamradio;
  using namespace bamradio::ofdm;

  init_options();

  // fake traffic constants
  int const ppf = 2;       // packets per frame
  int const nbytes = 1500; // bytes per ip packet

  // Main thread and net IO service.
  boost::asio::io_service ios;
  boost::posix_time::seconds ti(5); // maximum runtime
  boost::asio::deadline_timer to(ios);

  // Grab some PHY params.
  auto const waveform = bam::dsp::SubChannel::table()[4];
  auto const bw = waveform.bw();

  // Create our OFDM DLL (and PHY).
  auto ofdmdll = std::make_shared<bamradio::ofdm::DataLinkLayer>(
      "ofdm0", ios, ofdm::phy_tx::make(2), 50, nbytes + 8, 0);
  ofdmdll->enableSendToSelf();

  // PHY emulation
  auto const tb = gr::make_top_block("bamradio-ofdm-tx-test");
  boost::shared_ptr<uhd::tx_streamer> dummy_streamer(new fake_tx_streamer(1e6));
  bamradio::ofdm::phy_tx::sptr tx = ofdmdll->tx();
  tx->connect(dummy_streamer);

  // feed the TX samples into a GNU radio flowgraph.
  auto inbuf = static_cast<fake_tx_streamer *>(dummy_streamer.get())->rb;
  auto src = gr::bamofdm::lambda_sync_block_01<gr_complex>::make(
      [](auto) {},
      [inbuf](auto b, auto o, auto N) -> ssize_t {
        auto const n = std::min(inbuf->items_avail(), N);
        memcpy(o, inbuf->read_ptr(), sizeof(gr_complex) * n);
        inbuf->consume(n);
        return n;
      });

  // Filter and downsample for RX.
  auto const filtertaps = gr::filter::firdes::low_pass_2(
      1.0, options::phy::data::sample_rate, bw / 2 + 100e3, 300e3, 40.0,
      gr::filter::firdes::WIN_HAMMING);
  auto const lpf = gr::filter::fft_filter_ccf::make(waveform.os, filtertaps, 1);
  auto const chan_out = gr::blocks::add_cc::make();
  auto const cfo = gr::blocks::rotator_cc::make(0.0001);
  auto const chan_in = gr::filter::fft_filter_ccc::make(
      1, {
             gr_complex(1.0, 0) /*, gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(-0.1, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.0, 0), gr_complex(0.0, 0),
              gr_complex(0.0, 0), gr_complex(0.3,0)*/
         });
  auto const noise = gr::analog::noise_source_c::make(
      gr::analog::GR_GAUSSIAN,
      pow(10,
          (noise_floor_db + 10 * log10(options::phy::data::sample_rate / bw)) /
              20.0),
      -1);

  auto cob = bamradio::ofdm::ChannelOutputBuffer::make(1e6);
  auto const sink = bamradio::test::COBSink::make(
      [] {
        using namespace bamradio::ofdm;
        auto hparams =
            DFTSOFDMFrame(
                0, 0, {},
                MCS::stringNameToIndex(options::phy::data::header_mcs_name),
                MCS::stringNameToIndex(
                    options::phy::data::initial_payload_mcs_name),
                SeqID::stringNameToIndex(
                    options::phy::data::initial_payload_symbol_seq_name),
                0, 0)
                .headerParams(false, 20);
        auto sym_len = hparams.symbols[0].second->symbol_length;
        return static_cast<size_t>(sym_len / 2);
      }(),
      cob);

  // connect the mock GNU Radio graph
  tb->connect(src, 0, chan_in, 0);
  tb->connect(chan_in, 0, cfo, 0);
  tb->connect(cfo, 0, chan_out, 0);
  tb->connect(noise, 0, chan_out, 1);
  tb->connect(chan_out, 0, lpf, 0);
  tb->connect(lpf, 0, sink, 0);

  // connect to the receiver
  ofdmdll->rx()->connect({cob}, {waveform.os});

  ofdmdll->setTxChannel({bamradio::Channel(1, 0, 20)});
  ofdmdll->start();

  // generate fake traffic
  size_t numFrames = 0;
  tx->stop();
  tx->start([&]() -> bamradio::ofdm::phy_tx::prepared_frame {
    auto f = numFrames++ >= nframes ? nullptr : make_frame(ppf, nbytes);
    return {f, bamradio::Channel(1, 0, waveform.os)};
  });

  // kill after ti seconds or on keyboard interrupt
  auto kill = [&](auto const &e, auto n) {
    tx->stop();
    ios.stop();
    ofdmdll->stop();
    tb->stop();
  };
  boost::asio::signal_set signals(ios, SIGINT);
  signals.async_wait(kill);
  to.expires_from_now(ti);
  to.async_wait([&](auto &e) { kill(e, 0); });

  // Receive the packets and check 'em.
  // n.b. due to GNU radio madness, the last payload never quite makes it to the
  // payload decoder. I don't know why, but for the purpose of this test, I will
  // make the educated executive decision to not count the last packet. this is
  // a TODO, but is it really?
  auto const n_segments_expected = (nframes - 1) * ppf;
  int n_segments_received = 0;
  NodeID rxnodeid;
  std::vector<uint8_t> packet(1024 * 8);
  std::function<void(void)> rxld = [&] {
    ofdmdll->asyncReceiveFrom(
        boost::asio::buffer(packet), &rxnodeid, [&](auto s) {
          BOOST_REQUIRE_EQUAL(rxnodeid, 0);
          BOOST_REQUIRE_EQUAL(s->length(), nbytes + 8);
          auto const ip4s = std::dynamic_pointer_cast<net::IP4PacketSegment>(s);
          // this is the sequence number, unused here
          size_t *iii = (size_t *)(boost::asio::buffer_cast<uint8_t const *>(
                                       ip4s->packetContentsBuffer()) +
                                   20);
          if (++n_segments_received < n_segments_expected) {
            // continue receiving
            rxld();
          } else {
            // stop GNU radio and the io service after receiving all frames
            kill(0, 0);
          }
        });
  };
  rxld();

  // start the emulation flow graph and io service
  tb->start();
  ios.run();

  // did we receive all of the packets?
  BOOST_REQUIRE_EQUAL(n_segments_received, n_segments_expected);
}
