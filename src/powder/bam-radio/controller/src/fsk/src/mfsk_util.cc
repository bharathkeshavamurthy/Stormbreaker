/* -*- c++ -*- */

#include "mfsk_util.h"

namespace bamradio {
namespace mfsk_util {
// number of samples for nbytes of data and given preamble length
size_t calc_nsamples(size_t nbytes, size_t preamble_len, size_t pulse_len) {
  // these are hardcoded in the RS code implementation, no need to include the
  // header.
  constexpr int rs_N = 255;
  constexpr int rs_K = 223;
  auto rs_PARITY = rs_N - rs_K;

  auto bits_per_sym = bps();

  // the input to the RS encoder consists of the data + CRC32 (4 bytes)
  auto rs_in_len = nbytes + 4;
  // see rs_ccds_encode_bb_impl.cc:127 for this
  int num_rs_blocks = std::ceil((double)rs_in_len / options::phy::control::rs_k);
  auto rs_out_len = rs_in_len + num_rs_blocks * rs_PARITY;

  // add the preamble bytes
  auto tx_len = rs_out_len + preamble_len;

  // 8 bits in a byte
  auto tx_bits = 8 * tx_len;

  // how many bits in an MFSK "Symbol"
  auto num_mfsk_symbols = tx_bits / bits_per_sym;

  // now we know the number of samples
  return num_mfsk_symbols * pulse_len;
}

// transmit time for nbytes of data and preamble length
uhd::time_spec_t calc_tx_time(size_t nbytes, size_t preamble_len,
                              size_t pulse_len) {
  return uhd::time_spec_t(
      (double)calc_nsamples(nbytes, preamble_len, pulse_len) /
      options::phy::control::sample_rate);
}

// compute the frequency table for the chunks_to symbols block
std::vector<float> get_freq_table(std::vector<float> pulse) {
  // determine the carrier frequencies (fsk constellation)
  std::vector<float> freq_dev_vec(options::phy::control::num_fsk_points);
  // "numpy.linspace(start, stop, freq_dev_vec.size()"
  size_t k = 0;
  double start = options::phy::control::sample_rate *
                 (-1.0 / 2.0 + 2.0 / (2.0 * pulse.size() + 1));
  double stop = options::phy::control::sample_rate *
                (1.0 / 2.0 - 2.0 / (2.0 * pulse.size() + 1));
  double step = (stop - start) / (freq_dev_vec.size() - 1);
  std::generate(freq_dev_vec.begin(), freq_dev_vec.end(),
                [&k, start, step] { return start + (step * k++); });

  // numpy.kron(freq_dev_vec, pulse)
  std::vector<float> freq_table;
  for (auto const &fp : freq_dev_vec) {
    std::vector<float> freq_pulse;
    for (auto const &p : pulse) {
      freq_pulse.push_back(p * fp);
    }
    freq_table.insert(freq_table.end(), freq_pulse.begin(), freq_pulse.end());
  }
  return freq_table;
}

// compute the number of bits per symbol
size_t bps() { return (int)std::log2(options::phy::control::num_fsk_points); }

} // namespace mfsk_util
} // namespace bamradio
