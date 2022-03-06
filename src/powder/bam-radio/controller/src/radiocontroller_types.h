// -*- c++ -*-
// Copyright (c) 2019 Dennis Ogbe

#ifndef a8521c7a666892f21fd5
#define a8521c7a666892f21fd5

namespace bamradio {
//
// A low-level representation of a Channel from the radiocontroller's
// perspective.
//
class Channel {
public:
  // interpolate by sample_rate / bandwidth; OFDM oversampling is same as
  // interp_factor
  int interp_factor;
  // frequency offset in Hz
  double offset;
  // bandwidth of channel in Hz
  double bandwidth;
  // sample rate of PHY (UHD) underlying the channel
  double sample_rate;
  // gain applied to samples before tranmission
  float sample_gain;

  double rotator_phase() const;

  Channel(double bandwidth, double offset, double sample_rate);
  bool overlaps(Channel c, double guard) const;
  double lower() const;
  double upper() const;
};

} // namespace bamradio

#endif // a8521c7a666892f21fd5
