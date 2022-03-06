/* -*- c++ -*- */

#include <algorithm>
#include <uhd/types/time_spec.hpp>
#include <vector>

#include "options.h"

namespace bamradio {
namespace mfsk_util {
// number of samples for nbytes of data and given preamble length
size_t calc_nsamples(size_t nbytes, size_t preamble_len, size_t pulse_len);

// transmit time for nbytes of data and preamble length
uhd::time_spec_t calc_tx_time(size_t nbytes, size_t preamble_len,
                              size_t pulse_len);

// compute the frequency table for the chunks_to symbols block
std::vector<float> get_freq_table(std::vector<float> pulse);

// compute the number of bits per symbol
size_t bps();
}
}
