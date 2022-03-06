// -*- c++ -*-
/* Control Channel Hier Block
 * Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu> */

#include "ctrl_ch.h"
#include "common.h"
#include "mac_ctrl.h"

#include "options.h"

#include "rs_ccsds.h"
#include <bamfsk/access_code_detector.h>
#include <bamfsk/insertPreamble_bb.h>
#include <bamfsk/mfskMod_fc.h>
#include <bamfsk/ndaSync_ff.h>
#include <bamfsk/rs_ccsds_decode_bb.h>
#include <bamfsk/rs_ccsds_encode_bb.h>
#include <bamfsk/tsb_chunks_to_symbols_bf.h>

#include <gnuradio/blocks/complex_to_mag.h>
#include <gnuradio/blocks/float_to_char.h>
#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/blocks/repack_bits_bb.h>
#include <gnuradio/blocks/tag_debug.h>
#include <gnuradio/digital/crc32_bb.h>
#include <gnuradio/expj.h>
#include <gnuradio/filter/fft_filter_ccc.h>
#include <gnuradio/filter/fft_filter_ccf.h>
#include <gnuradio/filter/firdes.h>
#include <gnuradio/gr_complex.h>
// deprecated GNU radio blocks... will be out in newer versions
#if __has_include(<gnuradio/blocks/argmax_fs.h>)
#include <gnuradio/blocks/argmax_fs.h>
#else
#include <gnuradio/blocks/argmax.h>
#endif
#if __has_include(<gnuradio/digital/chunks_to_symbols_sf.h>)
#include <gnuradio/digital/chunks_to_symbols_sf.h>
#else
#include <gnuradio/digital/chunks_to_symbols.h>
#endif

namespace bamradio {
namespace controlchannel {

using namespace gr;
using namespace gr::bamfsk;

ctrl_ch::ctrl_ch(CCData::sptr cc_data, unsigned int max_nodes,
                 unsigned int node_num, double sample_rate, double t_slot_sec,
                 double scale, unsigned int num_fsk_points, int rs_k,
                 int min_soft_decs)
    : gr::hier_block2("phy_ctrl_ch",
                      gr::io_signature::make(1, 1, sizeof(gr_complex)),
                      gr::io_signature::make(0, 0, sizeof(gr_complex))) {
  namespace rs = gr::bamfsk::rs::ccsds;

  // FSK pulse
  std::vector<float> pulse(10, 1.0);
  std::vector<float> freq_dev_vec(num_fsk_points);
  size_t k = 0;
  double start = options::phy::control::bandwidth *
                 (-1.0 / 2.0 + 2.0 / (2.0 * pulse.size() + 1));
  double stop = options::phy::control::bandwidth *
                (1.0 / 2.0 - 2.0 / (2.0 * pulse.size() + 1));
  double step = (stop - start) / (freq_dev_vec.size() - 1);
  std::generate(freq_dev_vec.begin(), freq_dev_vec.end(),
                [&k, start, step] { return start + (step * k++); });

  /*
   * ===  Rx  ===
   */

  // length name
  std::string tsb_length_tag("length");
  // preamble
  std::vector<uint8_t> preamble({73, 25, 128, 247, 8, 95, 134, 4});
  // Generate filter
  auto bits_per_sym = (int)std::log2(num_fsk_points);
  size_t len_pulse = pulse.size();
  std::vector<gr::filter::fft_filter_ccc::sptr> filter_blocks;
  std::vector<gr::blocks::complex_to_mag::sptr> c_to_mag_blocks;
  for (int i = 0; i < num_fsk_points; ++i) {
    std::vector<gr_complex> tone(len_pulse, 1.0);
    double f = start + step * i;
    for (int j = 0; j < len_pulse; ++j) {
      double t = j / sample_rate;
      gr_complex rc_factor =
          (1 - std::cos(2 * M_PI * j / len_pulse)) / 2.0; // raised cosine
      tone[j] *= rc_factor * gr_expj(2 * M_PI * f * t);
    }
    filter_blocks.push_back(filter::fft_filter_ccc::make(1, tone));
  }

  // Compute magnitude
  for (int i = 0; i < num_fsk_points; ++i)
    c_to_mag_blocks.push_back(blocks::complex_to_mag::make(1));
  // NDA sync
  auto nda_sync = ndaSync_ff::make(len_pulse, min_soft_decs);
  // find abs max output
  auto amax = blocks::argmax_fs::make(1);
  // null sink (throw away 1st output of amax)
  auto nsink = blocks::null_sink::make(sizeof(short));
  // C2S
  std::vector<float> c2s_table;
  for (int i = 0; i < num_fsk_points; ++i) {
    for (int j = 0; j < bits_per_sym; ++j) {
      uint8_t val = i;
      c2s_table.push_back(1 & (val >> j));
    }
  }
  auto c2s_rx = digital::chunks_to_symbols_sf::make(c2s_table, bits_per_sym);
  // access code detector
  auto const ncrc = cc_data->getNbytes() + 4;
  auto const nparity =
      (size_t)std::ceil(((double)ncrc) / (double)options::phy::control::rs_k);
  auto const phy_payload_len = (ncrc + nparity * rs::PARITY) * 8;
  auto access_code = access_code_detector::make(pmt::intern(tsb_length_tag),
                                                preamble, phy_payload_len, 1);
  // float to byte
  auto f2b = blocks::float_to_char::make(1, 1.0);
  // repack
  auto repack = blocks::repack_bits_bb::make(1, 8, tsb_length_tag, false);
  // RS decode
  auto rs_dec = rs_ccsds_decode_bb::make(rs_k, tsb_length_tag);
  // CRC check
  auto crc_check = digital::crc32_bb::make(true, tsb_length_tag);
  // deserialize data
  d_c_recv = gr::bamfsk::ctrl_recv::make(tsb_length_tag, cc_data);

  for (int i = 0; i < num_fsk_points; ++i) {
    connect(self(), 0, filter_blocks[i], 0);
    connect(filter_blocks[i], 0, c_to_mag_blocks[i], 0);
    connect(c_to_mag_blocks[i], 0, nda_sync, i);
    connect(nda_sync, i, amax, i);
  }
  connect(amax, 0, nsink, 0);
  connect(amax, 1, c2s_rx, 0);
  connect(c2s_rx, 0, f2b, 0);
  connect(f2b, 0, access_code, 0);
  connect(access_code, 0, repack, 0);
  connect(repack, 0, rs_dec, 0);
  connect(rs_dec, 0, crc_check, 0);
  connect(crc_check, 0, d_c_recv, 0);
}

void ctrl_ch::set_cc_data(CCData::sptr ccd) { d_c_recv->set_cc_data(ccd); }

} // namespace controlchannel
} // namespace bamradio
