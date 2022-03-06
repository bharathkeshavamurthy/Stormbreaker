// Sampling and channelization
//
// Copyright (c) 2018 Dennis Ogbe

#ifndef e67161190662fdacac6abb2dd51c
#define e67161190662fdacac6abb2dd51c

#include <string>
#include <vector>

namespace bam {
namespace dsp {

//
// Sampling Rate. For various reasons, this is a program-wide constant
//
extern const double sample_rate;
extern const double control_sample_rate;
extern const std::string master_clock_rate;

//
// sub-channels for this sample rate
//

struct SubChannel {
  int os;                  // interpolation/decimation rate
  std::vector<float> taps; // FIR taps

  // compute bandwidth
  double bw() const { return 1. / ((double)os) * bam::dsp::sample_rate; }

  // all possible subchannels
  static std::vector<SubChannel> const &table();

  // keeping this for legacy sake.
  static int stringNameToIndex(std::string const &n);
};

} // namespace dsp
} // namespace bam

#endif // e67161190662fdacac6abb2dd51c
