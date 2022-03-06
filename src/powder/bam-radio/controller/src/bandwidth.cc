#include "bandwidth.h"

#include <stdexcept>

namespace bam {
namespace dsp {

// constants
const double sample_rate = 46.08e6;
const double control_sample_rate = 480e3;
const std::string master_clock_rate = "184.32e6";

// pre-computed filter coefficients (generate with rx_filters.py)
#include "subchannel.cc"

int SubChannel::stringNameToIndex(std::string const &n) {
  if (n == "DFT_S_OFDM_128_144K") {
    return 0;
  } else if (n == "DFT_S_OFDM_128_288K") {
    return 1;
  } else if (n == "DFT_S_OFDM_128_500K") {
    return 2;
  } else if (n == "DFT_S_OFDM_128_715K") {
    return 3;
  } else if (n == "DFT_S_OFDM_128_1M") {
    return 4;
  } else if (n == "DFT_S_OFDM_128_1_25M") {
    return 5;
  } else if (n == "DFT_S_OFDM_128_2M") {
    return 6;
  } else if (n == "DFT_S_OFDM_128_2_5M") {
    return 7;
  } else if (n == "DFT_S_OFDM_128_4M") {
    return 8;
  } else if (n == "DFT_S_OFDM_128_5M") {
    return 9;
  } else if (n == "DFT_S_OFDM_128_10M") {
    return 10;
  } else {
    throw std::runtime_error("Bad waveform name.");
  }
}

} // namespace dsp
} // namespace bam
