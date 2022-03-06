//  Copyright © 2017 Stephen Larew

#include "ofdm.h"
#include "common.h"
#include "events.h"
#include "options.h"
#include "phy.h"

#include <numeric>

#include <gnuradio/fft/fft.h>

namespace bamradio {
namespace ofdm {

pmt::pmt_t const data_carrier_mapping_tag_key =
    pmt::intern("data_carrier_mapping");
pmt::pmt_t const pilot_carrier_mapping_tag_key =
    pmt::intern("pilot_carrier_mapping");
pmt::pmt_t const pilot_symbols_tag_key = pmt::intern("pilot_symbols");
pmt::pmt_t const constellation_tag_key = pmt::intern("constellation");
pmt::pmt_t const fft_length_tag_key = pmt::intern("fft_length");
pmt::pmt_t const symbol_length_tag_key = pmt::intern("symbol_length");
pmt::pmt_t const cyclic_prefix_length_tag_key = pmt::intern("cp_length");
pmt::pmt_t const prefix_tag_key = pmt::intern("prefix");
pmt::pmt_t const postfix_pad_tag_key = pmt::intern("postfix_pad");
pmt::pmt_t const reset_tag_key = pmt::intern("reset");
pmt::pmt_t const dft_spread_length_tag_key = pmt::intern("dft_spread_length");
pmt::pmt_t const block_size_tag_key = pmt::intern("bsize");
pmt::pmt_t const rate_idx_tag_key = pmt::intern("rate_idx");
pmt::pmt_t const snr_tag_key = pmt::intern("SNR_sync");
pmt::pmt_t const frame_ptr_tag_key = pmt::intern("frame_obj");
pmt::pmt_t const chanest_msg_port_key = pmt::intern("chanest_msg");
pmt::pmt_t const rx_time_tag_key = pmt::intern("rx_time_fdd");
pmt::pmt_t const dec_extra_item_consumption_tag_key = pmt::intern("dec_fat");

size_t OFDMFrameParams::numSymbols() const {
  return std::accumulate(symbols.begin(), symbols.end(), 0,
                         [](auto a, auto b) { return a + b.first; });
}

size_t OFDMFrameParams::numTXSamples() const {
  return std::accumulate(symbols.begin(), symbols.end(), (size_t)0,
                         [](auto a, auto b) {
                           auto const &s = *b.second;
                           return a + b.first * s.numTXSamples();
                         }) +
         symbols.back().second->oversample_rate *
             symbols.back().second->cyclic_postfix_length;
}

size_t OFDMFrameParams::numBits() const {
  return std::accumulate(symbols.begin(), symbols.end(), (size_t)0,
                         [](auto a, auto b) {
                           auto const &s = *b.second;
                           return a + b.first * s.numBits();
                         });
}

std::pair<size_t, size_t> OFDMFrameParams::offsetOfBit(size_t bitNumber) const {
  size_t i = 0, count = 0;
  for (auto const &s : symbols) {
    auto const numBits = s.first * s.second->numBits();
    if (bitNumber >= count && bitNumber < count + numBits) {
      auto const o = (bitNumber - count) / s.second->numBits();
      return {i + o, (bitNumber - count) % s.second->numBits()};
    }
    i += s.first;
    count += numBits;
  }
}

DFTSOFDMFrameParams::DFTSOFDMFrameParams(decltype(symbols) const &s)
    : OFDMFrameParams(s), dft_spread_length(0) {
  for (auto const &ss : s) {
    auto const a = pmt::length(ss.second->data_carrier_mapping);
    if (a != 0) {
      if (dft_spread_length == 0) {
        dft_spread_length = a;
      } else if (a != dft_spread_length) {
        log::doomsday("Fatal error", __FILE__, __LINE__);
      }
    }
  }
}

void normalize_carriers(std::vector<int32_t> &m, int32_t w) {
  for (auto &mm : m) {
    while (mm < 0)
      mm += w;
    while (mm >= w)
      mm -= w;
  }
  std::sort(begin(m), end(m));
}

// return vector of integers in [b,e) with stride s (HACK)
std::vector<int32_t> vrange(int32_t b, int32_t e, int32_t s) {
  std::vector<int32_t> v;
  for (auto i = b; i < e; i += s)
    v.push_back(i);
  return v;
}

std::pair<pmt::pmt_t, pmt::pmt_t>
make_evenly_spaced_carrier_map(size_t Ncarr, size_t const Nocc,
                               size_t const stride, size_t const offset) {
  auto v = vrange(-(int32_t)Nocc / 2, Nocc / 2, 1);
  auto p = vrange(-(int32_t)Nocc / 2 + offset, Nocc / 2, stride);
  normalize_carriers(v, Ncarr);
  normalize_carriers(p, Ncarr);
  std::vector<int32_t> c;
  std::set_difference(begin(v), end(v), begin(p), end(p),
                      std::back_inserter(c));
  return std::make_pair(pmt::init_s32vector(c.size(), c),
                        pmt::init_s32vector(p.size(), p));
}

// map:
// "symlen.#data-carriers.#pilot-carriers.pilot-pattern-index" ->
// (data-carrier-mapping, pilot-carrier-mapping)
//
static std::map<std::string, std::pair<pmt::pmt_t, pmt::pmt_t>> const
    data_carrier_mappings = {
#ifndef HIDEFROMYCM
        {"128.108.12.0", make_evenly_spaced_carrier_map(128, 120, 10, 0)},
        {"128.108.12.1", make_evenly_spaced_carrier_map(128, 120, 10, 1)},
        {"128.108.12.2", make_evenly_spaced_carrier_map(128, 120, 10, 2)},
        {"128.108.12.3", make_evenly_spaced_carrier_map(128, 120, 10, 3)},
        {"128.108.12.4", make_evenly_spaced_carrier_map(128, 120, 10, 4)},
        {"128.108.12.5", make_evenly_spaced_carrier_map(128, 120, 10, 5)},
        {"128.108.12.6", make_evenly_spaced_carrier_map(128, 120, 10, 6)},
        {"128.108.12.7", make_evenly_spaced_carrier_map(128, 120, 10, 7)},
        {"128.108.12.8", make_evenly_spaced_carrier_map(128, 120, 10, 8)},
        {"128.108.12.9", make_evenly_spaced_carrier_map(128, 120, 10, 9)},
        {"128.96.24.0", make_evenly_spaced_carrier_map(128, 120, 5, 0)},
        {"128.96.24.1", make_evenly_spaced_carrier_map(128, 120, 5, 1)},
        {"128.96.24.2", make_evenly_spaced_carrier_map(128, 120, 5, 2)},
        {"128.96.24.3", make_evenly_spaced_carrier_map(128, 120, 5, 3)},
        {"128.96.24.4", make_evenly_spaced_carrier_map(128, 120, 5, 4)},
#endif
};

static std::vector<OFDMSymbolParams::sptr>
makeSymbolTable(uint16_t oversample_rate) {
  std::vector<OFDMSymbolParams::sptr> st;

  double const a0_128_108 = 1.0 / sqrt((double)128 * 108);
  double const a1_128_108 = 0.45;
  double const a0_128_96 = 1.0 / sqrt((double)128 * 96);
  double const a1_128_96 = 0.45;

  auto const const_qpsk_128_108 = constellation::qpsk<llr_type>::make(
      a0_128_108, a1_128_108, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0,
      40, "qpsk_6bit_0to40_20pts.llrw");

  auto const const_qpsk_128_96 = constellation::qpsk<llr_type>::make(
      a0_128_96, a1_128_96, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0, 40,
      "qpsk_6bit_0to40_20pts.llrw");

  auto const pilot128_108_12 = pmt::init_c32vector(12, [=] {
    auto cz = gr::bamofdm::generate_cazac_seq(12, 5);
    for (auto &v : cz) {
      v *= a0_128_108 * a1_128_108 * sqrt(108.0);
    }
    return cz;
  }());

  auto const pilot128_96_24 = pmt::init_c32vector(24, [=] {
    auto cz = gr::bamofdm::generate_cazac_seq(24, 5);
    for (auto &v : cz) {
      v *= a0_128_96 * a1_128_96 * sqrt(96.0);
    }
    return cz;
  }());

  auto const const_qam16_128_108 = constellation::qam16<llr_type>::make(
      a0_128_108, a1_128_108, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0,
      40, "qam16_6bit_0to40_20pts.llrw");

  auto const const_qam32_128_108 = constellation::qam32<llr_type>::make(
      a0_128_108, a1_128_108, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0,
      40, "qam32_6bit_0to40_20pts.llrw");

  auto const const_qam64_128_108 = constellation::qam64<llr_type>::make(
      a0_128_108, a1_128_108, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0,
      40, "qam64_6bit_0to40_20pts.llrw");

  auto const const_qam128_128_108 = constellation::qam128<llr_type>::make(
      a0_128_108, a1_128_108, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0,
      40, "qam128_6bit_0to40_20pts.llrw");

  auto const const_qam256_128_108 = constellation::qam256<llr_type>::make(
      a0_128_108, a1_128_108, sqrt(1.0 / ((double)oversample_rate)), 6, 20, 0,
      40, "qam256_6bit_0to40_20pts.llrw");

  // ZC_SYNC_RX_128_12
  // ZC sync word symbol (for receiver)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = pmt::init_s32vector(0, {}),
      .pilot_carrier_mapping = pmt::init_s32vector(64, ([] {
                                                     std::vector<int32_t> a(64);
                                                     for (size_t i = 0, j = 0;
                                                          i < a.size();
                                                          ++i, j += 2) {
                                                       a[i] = j;
                                                     }
                                                     return a;
                                                   })()),
      .pilot_symbols = pmt::init_c32vector(
          64, get_preamble_pilot_symbols(64, oversample_rate)),
      .constellation = nullptr,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = nullptr}));

  // ZC_SYNC_TX_DATA_128_12_108_QPSK
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.108.12.0").first,
      .pilot_carrier_mapping = pmt::init_s32vector(0, {}),
      .pilot_symbols = pmt::init_c32vector(0, {}),
      .constellation = const_qpsk_128_108,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = pmt::init_c32vector(128 * oversample_rate, [=] {
        auto const sw = get_preamble_td_oversampled(64, oversample_rate);
        assert(sw.size() == 128 * oversample_rate);
        return sw;
      }())}));

  // ZC_SYNC_TX_DATA_128_12_96_24_0_QPSK
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.96.24.0").first,
      .pilot_carrier_mapping = data_carrier_mappings.at("128.96.24.0").second,
      .pilot_symbols = pilot128_96_24,
      .constellation = const_qpsk_128_96,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = pmt::init_c32vector(128 * oversample_rate, [=] {
        auto const sw = get_preamble_td_oversampled(64, oversample_rate);
        assert(sw.size() == 128 * oversample_rate);
        return sw;
      }())}));

  // ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.108.12.0").first,
      .pilot_carrier_mapping = data_carrier_mappings.at("128.108.12.0").second,
      .pilot_symbols = pilot128_108_12,
      .constellation = const_qpsk_128_108,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = pmt::init_c32vector(128 * oversample_rate, [=] {
        auto const sw = get_preamble_td_oversampled(64, oversample_rate);
        assert(sw.size() == 128 * oversample_rate);
        return sw;
      }())}));

  // ZC_SYNC_TX_DATA_128_12_108_QAM16
  // ZC sync word symbol (for transmitter)
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[ZC_SYNC_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam16_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_12_0_QAM16
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam16_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_QAM32
  // ZC sync word symbol (for transmitter)
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[ZC_SYNC_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam32_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_12_0_QAM32
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam32_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_QAM64
  // ZC sync word symbol (for transmitter)
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[ZC_SYNC_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam64_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_12_0_QAM64
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam64_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_QAM128
  // ZC sync word symbol (for transmitter)
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[ZC_SYNC_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam128_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_12_0_QAM128
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam128_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_QAM256
  // ZC sync word symbol (for transmitter)
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[ZC_SYNC_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam256_128_108;

  // ZC_SYNC_TX_DATA_128_12_108_12_0_QAM256
  // ZC sync word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam256_128_108;

  // ZC_CHANEST_RX_128_12,
  // ZC chanest word symbol (for receiver)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = pmt::init_s32vector(0, {}),
      .pilot_carrier_mapping =
          pmt::init_s32vector(128, ([] {
                                std::vector<int32_t> a(128);
                                for (size_t i = 0; i < a.size(); ++i) {
                                  a[i] = i;
                                }
                                normalize_carriers(a, 128);
                                return a;
                              })()),
      .pilot_symbols = pmt::init_c32vector(
          128,
          [=] {
            auto const N = 128;
            auto p = gr::bamofdm::generate_cazac_seq(N, 3);

            auto const osf = oversample_rate;

            // Take IFFT of cazac.
            gr::fft::fft_complex ifft(osf * N, false);
            memset(ifft.get_inbuf(), 0x00,
                   sizeof(gr_complex) * ifft.inbuf_length());
            for (size_t i = 0; i < N / 2; ++i) {
              ifft.get_inbuf()[i] = p[i];
            }
            for (size_t i = N / 2; i < N; ++i) {
              ifft.get_inbuf()[osf * N - N + i] = p[i];
            }
            ifft.execute();

            // Copy out IFFT result
            std::vector<gr_complex> tdp(osf * N);
            std::copy_n(ifft.get_outbuf(), osf * N, tdp.begin());

            // get max value after ifft as scalar
            auto const scalar = [&] {
              // i'm so sorry, this is terrible
              auto r = std::numeric_limits<double>::lowest();
              for (auto const &tdpp : tdp) {
                auto const a = std::abs(tdpp);
                if (a > r)
                  r = a;
              }
              return r;
            }();

            for (auto &x : p) {
              x /= scalar;
            }

            return p;
          }()),
      .constellation = nullptr,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = nullptr}));

  // ZC_CHANEST_TX_DATA_128_12_108_QPSK
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.108.12.0").first,
      .pilot_carrier_mapping = pmt::init_s32vector(0, {}),
      .pilot_symbols = pmt::init_c32vector(0, {}),
      .constellation = const_qpsk_128_108,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = pmt::init_c32vector(128 * oversample_rate, [=] {
        auto const Ns = 128;
        auto const Nc = 12;
        auto const p = gr::bamofdm::generate_cazac_seq(Ns, 3);

        auto const osf = oversample_rate;

        // Take IFFT of cazac.
        gr::fft::fft_complex ifft(osf * Ns, false);
        memset(ifft.get_inbuf(), 0x00,
               sizeof(gr_complex) * ifft.inbuf_length());
        for (size_t i = 0; i < Ns / 2; ++i) {
          ifft.get_inbuf()[i] = p[i];
        }
        for (size_t i = Ns / 2; i < Ns; ++i) {
          ifft.get_inbuf()[osf * Ns - Ns + i] = p[i];
        }
        ifft.execute();

        // Copy out IFFT result
        std::vector<gr_complex> tdp(osf * Ns);
        std::copy_n(ifft.get_outbuf(), osf * Ns, tdp.begin());

        // get max value after ifft as scalar
        auto const scalar = [&] {
          // i'm so sorry, this is terrible
          auto r = std::numeric_limits<double>::lowest();
          for (auto const &tdpp : tdp) {
            auto const a = std::abs(tdpp);
            if (a > r)
              r = a;
          }
          return r;
        }();

        for (auto &x : tdp) {
          x /= scalar;
        }
        return tdp;
      }())}));

  // DATA_128_12_108_QPSK
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.108.12.0").first,
      .pilot_carrier_mapping = pmt::init_s32vector(0, {}),
      .pilot_symbols = pmt::init_c32vector(0, {}),
      .constellation = const_qpsk_128_108,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = nullptr}));

  // ZC_CHANEST_TX_DATA_128_12_96_24_0_QPSK
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.96.24.0").first,
      .pilot_carrier_mapping = data_carrier_mappings.at("128.96.24.0").second,
      .pilot_symbols = pilot128_96_24,
      .constellation = const_qpsk_128_96,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]->prefix}));

  // ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(OFDMSymbolParams{
      .data_carrier_mapping = data_carrier_mappings.at("128.108.12.0").first,
      .pilot_carrier_mapping = data_carrier_mappings.at("128.108.12.0").second,
      .pilot_symbols = pilot128_108_12,
      .constellation = const_qpsk_128_108,
      .symbol_length = 128,
      .oversample_rate = oversample_rate,
      .cyclic_prefix_length = 12,
      .cyclic_postfix_length = 6,
      .prefix = st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]->prefix}));

  // DATA_128_12_108_12_0_QPSK AND 1 - 9
  auto dcm = data_carrier_mappings.find("128.108.12.0");
  for (auto i = 0; i < 10; ++i, ++dcm) {
    assert(dcm != end(data_carrier_mappings));
    st.emplace_back(std::make_shared<OFDMSymbolParams>(
        OFDMSymbolParams{.data_carrier_mapping = dcm->second.first,
                         .pilot_carrier_mapping = dcm->second.second,
                         .pilot_symbols = pilot128_108_12,
                         .constellation = const_qpsk_128_108,
                         .symbol_length = 128,
                         .oversample_rate = oversample_rate,
                         .cyclic_prefix_length = 12,
                         .cyclic_postfix_length = 6,
                         .prefix = nullptr}));
  }

  // DATA_128_12_96_24_0_QPSK AND 1 - 4
  dcm = data_carrier_mappings.find("128.96.24.0");
  for (auto i = 0; i < 5; ++i, ++dcm) {
    assert(dcm != end(data_carrier_mappings));
    st.emplace_back(std::make_shared<OFDMSymbolParams>(
        OFDMSymbolParams{.data_carrier_mapping = dcm->second.first,
                         .pilot_carrier_mapping = dcm->second.second,
                         .pilot_symbols = pilot128_108_12,
                         .constellation = const_qpsk_128_96,
                         .symbol_length = 128,
                         .oversample_rate = oversample_rate,
                         .cyclic_prefix_length = 12,
                         .cyclic_postfix_length = 6,
                         .prefix = nullptr}));
  }

  // ZC_CHANEST_TX_DATA_128_12_108_QAM16
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam16_128_108;

  // DATA_128_12_108_QAM16
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam16_128_108;

  // ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM16
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam16_128_108;

  // DATA_128_12_108_12_0_QAM16 AND 1 - 9
  for (auto i = 0; i < 10; ++i, ++dcm) {
    st.emplace_back(
        std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_12_0_QPSK + i]));
    st.back()->constellation = const_qam16_128_108;
  }

  // ZC_CHANEST_TX_DATA_128_12_108_QAM32
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam32_128_108;

  // DATA_128_12_108_QAM32
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam32_128_108;

  // ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM32
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam32_128_108;

  // DATA_128_12_108_12_0_QAM32 AND 1 - 9
  for (auto i = 0; i < 10; ++i, ++dcm) {
    st.emplace_back(
        std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_12_0_QPSK + i]));
    st.back()->constellation = const_qam32_128_108;
  }

  // ZC_CHANEST_TX_DATA_128_12_108_QAM64
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam64_128_108;

  // DATA_128_12_108_QAM64
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam64_128_108;

  // ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM64
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam64_128_108;

  // DATA_128_12_108_12_0_QAM64 AND 1 - 9
  for (auto i = 0; i < 10; ++i, ++dcm) {
    st.emplace_back(
        std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_12_0_QPSK + i]));
    st.back()->constellation = const_qam64_128_108;
  }

  // ZC_CHANEST_TX_DATA_128_12_108_QAM128
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam128_128_108;

  // DATA_128_12_108_QAM128
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam128_128_108;

  // ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM128
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam128_128_108;

  // DATA_128_12_108_12_0_QAM128 AND 1 - 9
  for (auto i = 0; i < 10; ++i, ++dcm) {
    st.emplace_back(
        std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_12_0_QPSK + i]));
    st.back()->constellation = const_qam128_128_108;
  }

  // ZC_CHANEST_TX_DATA_128_12_108_QAM256
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam256_128_108;

  // DATA_128_12_108_QAM256
  st.emplace_back(
      std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_QPSK]));
  st.back()->constellation = const_qam256_128_108;

  // ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM256
  // ZC chanest word symbol (for transmitter)
  st.emplace_back(std::make_shared<OFDMSymbolParams>(
      *st[ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK]));
  st.back()->constellation = const_qam256_128_108;

  // DATA_128_12_108_12_0_QAM256 AND 1 - 9
  for (auto i = 0; i < 10; ++i, ++dcm) {
    st.emplace_back(
        std::make_shared<OFDMSymbolParams>(*st[DATA_128_12_108_12_0_QPSK + i]));
    st.back()->constellation = const_qam256_128_108;
  }

  assert(st.size() == size_t(SymbolName::NUM_SYMBOLS));
  return st;
};

static std::map<uint16_t, std::vector<OFDMSymbolParams::sptr> const>
    SymbolTable;

OFDMSymbolParams::sptr lookupSymbol(uint16_t oversample_rate, SymbolName name) {
  static std::shared_timed_mutex m;
  std::shared_lock<decltype(m)> ls(m);
  auto const it = SymbolTable.find(oversample_rate);
  if (it == SymbolTable.end()) {
    ls.unlock();
    std::lock_guard<decltype(m)> le(m);
    auto const r =
        SymbolTable.emplace(oversample_rate, makeSymbolTable(oversample_rate));
    return r.first->second[name];
  } else {
    return it->second[name];
  }
}

namespace SeqID {

ID stringNameToIndex(std::string const &n) {
  if (n == "P19FULL_128_12_108_QPSK") {
    return ID::P19FULL_128_12_108_QPSK;
  } else if (n == "P10FULL_128_12_108_QAM16") {
    return ID::P10FULL_128_12_108_QAM16;
  } else if (n == "P10FULL_128_12_108_QAM32") {
    return ID::P10FULL_128_12_108_QAM32;
  } else if (n == "P10FULL_128_12_108_QAM64") {
    return ID::P10FULL_128_12_108_QAM64;
  } else if (n == "P10FULL_128_12_108_QAM128") {
    return ID::P10FULL_128_12_108_QAM128;
  } else if (n == "P10FULL_128_12_108_QAM256") {
    return ID::P10FULL_128_12_108_QAM256;
  } else if (n == "ZIG_128_12_108_12_QPSK") {
    return ID::ZIG_128_12_108_12_QPSK;
  } else if (n == "ZIG_128_12_108_12_QAM16") {
    return ID::ZIG_128_12_108_12_QAM16;
  } else if (n == "ZIG_128_12_108_12_QAM32") {
    return ID::ZIG_128_12_108_12_QAM32;
  } else if (n == "ZIG_128_12_108_12_QAM64") {
    return ID::ZIG_128_12_108_12_QAM64;
  } else if (n == "ZIG_128_12_108_12_QAM128") {
    return ID::ZIG_128_12_108_12_QAM128;
  } else if (n == "ZIG_128_12_108_12_QAM256") {
    return ID::ZIG_128_12_108_12_QAM256;
  } else if (n == "PFULL_ZIG_128_12_108_12_QPSK") {
    return ID::PFULL_ZIG_128_12_108_12_QPSK;
  } else if (n == "PFULL_ZIG_128_12_96_24_QPSK") {
    return ID::PFULL_ZIG_128_12_96_24_QPSK;
  } else {
    throw std::runtime_error("Bad SeqID name.");
  }
}

std::pair<size_t, SymbolName> begin(ID const s, bool const tx) {
  switch (s) {
  case ID::P19FULL_128_12_108_QPSK: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QPSK
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::P10FULL_128_12_108_QAM16: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM16
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::P10FULL_128_12_108_QAM32: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM32
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::P10FULL_128_12_108_QAM64: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM64
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::P10FULL_128_12_108_QAM128: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM128
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::P10FULL_128_12_108_QAM256: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM256
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::ZIG_128_12_108_12_QAM16: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM16
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::ZIG_128_12_108_12_QAM32: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM32
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::ZIG_128_12_108_12_QAM64: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM64
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::ZIG_128_12_108_12_QAM128: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM128
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::ZIG_128_12_108_12_QAM256: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM256
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::ZIG_128_12_108_12_QPSK:
  // FALLTHROUGH
  case ID::PFULL_ZIG_128_12_108_12_QPSK: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  case ID::PFULL_ZIG_128_12_96_24_QPSK: {
    return {1, tx ? SymbolName::ZC_CHANEST_TX_DATA_128_12_96_24_0_QPSK
                  : SymbolName::ZC_CHANEST_RX_128_12};
  }
  default:
    log::doomsday("Fatal error", __FILE__, __LINE__);
  }
}

std::pair<size_t, SymbolName> next(ID const s, bool const tx,
                                   size_t const prevIdx) {
  switch (s) {
  case ID::P19FULL_128_12_108_QPSK: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QPSK};
      } else {
        return {17, SymbolName::DATA_128_12_108_QPSK};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        return {18, SymbolName::DATA_128_12_108_QPSK};
      }
    }
  }
  case ID::P10FULL_128_12_108_QAM16: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM16};
      } else {
        return {8, SymbolName::DATA_128_12_108_QAM16};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        return {9, SymbolName::DATA_128_12_108_QAM16};
      }
    }
  }
  case ID::P10FULL_128_12_108_QAM32: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM32};
      } else {
        return {8, SymbolName::DATA_128_12_108_QAM32};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        return {9, SymbolName::DATA_128_12_108_QAM32};
      }
    }
  }
  case ID::P10FULL_128_12_108_QAM64: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM64};
      } else {
        return {8, SymbolName::DATA_128_12_108_QAM64};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        return {9, SymbolName::DATA_128_12_108_QAM64};
      }
    }
  }
  case ID::P10FULL_128_12_108_QAM128: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM128};
      } else {
        return {8, SymbolName::DATA_128_12_108_QAM128};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        return {9, SymbolName::DATA_128_12_108_QAM128};
      }
    }
  }
  case ID::P10FULL_128_12_108_QAM256: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_QAM256};
      } else {
        return {8, SymbolName::DATA_128_12_108_QAM256};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 2;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        return {9, SymbolName::DATA_128_12_108_QAM256};
      }
    }
  }
  case ID::ZIG_128_12_108_12_QAM256: {
    if (prevIdx == 0 && !tx) {
      return {1, SymbolName::DATA_128_12_108_12_0_QAM256};
    }
    auto const ns = SymbolName::DATA_128_12_108_12_0_QAM256 +
                    ((prevIdx + (tx ? 1 : 0)) % 10);
    return {1, (SymbolName)ns};
  }
  case ID::ZIG_128_12_108_12_QAM128: {
    if (prevIdx == 0 && !tx) {
      return {1, SymbolName::DATA_128_12_108_12_0_QAM128};
    }
    auto const ns = SymbolName::DATA_128_12_108_12_0_QAM128 +
                    ((prevIdx + (tx ? 1 : 0)) % 10);
    return {1, (SymbolName)ns};
  }
  case ID::ZIG_128_12_108_12_QAM64: {
    if (prevIdx == 0 && !tx) {
      return {1, SymbolName::DATA_128_12_108_12_0_QAM64};
    }
    auto const ns = SymbolName::DATA_128_12_108_12_0_QAM64 +
                    ((prevIdx + (tx ? 1 : 0)) % 10);
    return {1, (SymbolName)ns};
  }
  case ID::ZIG_128_12_108_12_QAM32: {
    if (prevIdx == 0 && !tx) {
      return {1, SymbolName::DATA_128_12_108_12_0_QAM32};
    }
    auto const ns = SymbolName::DATA_128_12_108_12_0_QAM32 +
                    ((prevIdx + (tx ? 1 : 0)) % 10);
    return {1, (SymbolName)ns};
  }
  case ID::ZIG_128_12_108_12_QAM16: {
    if (prevIdx == 0 && !tx) {
      return {1, SymbolName::DATA_128_12_108_12_0_QAM16};
    }
    auto const ns = SymbolName::DATA_128_12_108_12_0_QAM16 +
                    ((prevIdx + (tx ? 1 : 0)) % 10);
    return {1, (SymbolName)ns};
  }
  case ID::ZIG_128_12_108_12_QPSK: {
    if (prevIdx == 0 && !tx) {
      return {1, SymbolName::DATA_128_12_108_12_0_QPSK};
    }
    auto const ns =
        SymbolName::DATA_128_12_108_12_0_QPSK + ((prevIdx + (tx ? 1 : 0)) % 10);
    return {1, (SymbolName)ns};
  }
  case ID::PFULL_ZIG_128_12_108_12_QPSK: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 10;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK};
      } else {
        auto const ns = SymbolName::DATA_128_12_108_12_0_QPSK + thisIdx;
        return {1, (SymbolName)ns};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 11;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        auto const ns = SymbolName::DATA_128_12_108_12_0_QPSK + thisIdx - 1;
        return {1, (SymbolName)ns};
      }
    }
  }
  case ID::PFULL_ZIG_128_12_96_24_QPSK: {
    if (tx) {
      auto const thisIdx = (prevIdx + 1) % 5;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_TX_DATA_128_12_96_24_0_QPSK};
      } else {
        auto const ns = SymbolName::DATA_128_12_96_24_0_QPSK + thisIdx;
        return {1, (SymbolName)ns};
      }
    } else {
      auto const thisIdx = (prevIdx + 1) % 6;
      if (thisIdx == 0) {
        return {1, SymbolName::ZC_CHANEST_RX_128_12};
      } else {
        auto const ns = SymbolName::DATA_128_12_96_24_0_QPSK + thisIdx - 1;
        return {1, (SymbolName)ns};
      }
    }
  }
  default:
    log::doomsday("Fatal error", __FILE__, __LINE__);
  }
}

int bitsPerSymbol(ID s) {
  switch (s) {
  case ID::P19FULL_128_12_108_QPSK:
    return 2;
  case ID::P10FULL_128_12_108_QAM16:
    return 4;
  case ID::P10FULL_128_12_108_QAM32:
    return 5;
  case ID::P10FULL_128_12_108_QAM64:
    return 6;
  case ID::P10FULL_128_12_108_QAM128:
    return 7;
  case ID::P10FULL_128_12_108_QAM256:
    return 8;
  case ID::ZIG_128_12_108_12_QPSK:
    return 2;
  case ID::ZIG_128_12_108_12_QAM16:
    return 4;
  case ID::ZIG_128_12_108_12_QAM32:
    return 5;
  case ID::ZIG_128_12_108_12_QAM64:
    return 6;
  case ID::ZIG_128_12_108_12_QAM128:
    return 7;
  case ID::ZIG_128_12_108_12_QAM256:
    return 8;
  case ID::PFULL_ZIG_128_12_108_12_QPSK:
    return 2;
  case ID::PFULL_ZIG_128_12_96_24_QPSK:
    return 2;
  default:
    panic("Bad SeqID name.");
  }
}

/// Returns the number of data carriers per OFDM symbol
int symLen(ID) { return 128; }
/// Returns the number of occupied data carriers per OFDM symbol
int occupiedCarriers(ID s) {
  if (s == ID::PFULL_ZIG_128_12_96_24_QPSK) {
    return 96;
  } else {
    return 108;
  }
}
/// Returns the number of cyclic prefix symbols
int cpLen(ID) { return 12; }
float bpos(ID s) {
  auto const bpos = occupiedCarriers(s) * bitsPerSymbol(s);
  switch (s) {
  case ID::P19FULL_128_12_108_QPSK:
    return bpos * 18.0f / 19.0f;
  case ID::P10FULL_128_12_108_QAM16:
  case ID::P10FULL_128_12_108_QAM32:
  case ID::P10FULL_128_12_108_QAM64:
  case ID::P10FULL_128_12_108_QAM128:
  case ID::P10FULL_128_12_108_QAM256:
    return bpos * 9.0f / 10.0f;
  case ID::ZIG_128_12_108_12_QPSK:
  case ID::ZIG_128_12_108_12_QAM16:
  case ID::ZIG_128_12_108_12_QAM32:
  case ID::ZIG_128_12_108_12_QAM64:
  case ID::ZIG_128_12_108_12_QAM128:
  case ID::ZIG_128_12_108_12_QAM256:
    return bpos;
  case ID::PFULL_ZIG_128_12_108_12_QPSK:
    return bpos * 10.0f / 11.0f;
  case ID::PFULL_ZIG_128_12_96_24_QPSK:
    return bpos * 5.0f / 6.0f;
  default:
    panic("Bad SeqID name.");
  }
}
} // namespace SeqID
} // namespace ofdm
} // namespace bamradio
