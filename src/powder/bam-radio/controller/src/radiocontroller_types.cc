#include "radiocontroller_types.h"

static const float PI = 3.14159265358979323846264338327950f;

namespace bamradio {

Channel::Channel(double bw, double ofst, double sr)
    : interp_factor((int)sr / bw), offset(ofst), bandwidth(bw), sample_rate(sr),
      sample_gain(1.0f) {}

bool Channel::overlaps(Channel c, double guard) const {
  auto const ml = lower() - guard;
  auto const mu = upper() + guard;
  auto const cl = c.lower();
  auto const cu = c.upper();
  return !((cu < ml && cl < ml) || (cl > mu && cu > mu));
}

double Channel::lower() const { return offset - bandwidth / 2.0; }

double Channel::upper() const { return offset + bandwidth / 2.0; }

double Channel::rotator_phase() const { return 2 * PI * offset / sample_rate; }

} // namespace bamradio
