// -*- c++ -*-
//
// Copyright (c) 2017 Dennis Ogbe

#include "bam_constellation.h"
#include "events.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>
#include <gnuradio/math.h>
#include <iostream>

// debug
#include <boost/format.hpp>

namespace bamradio {
namespace constellation {

// return the constellation points for a given value
template <typename LLR>
void base<LLR>::map_to_points_and_scale(unsigned int value,
                                        gr_complex *points) const {
  assert(_scaled_constellation.size() != 0);
  for (unsigned int i = 0; i < d_dimensionality; i++)
    points[i] = _scaled_constellation[value * d_dimensionality + i];
}

template <typename LLR>
std::vector<gr_complex>
base<LLR>::map_to_points_and_scale_v(unsigned int value) const {
  std::vector<gr_complex> points_v(d_dimensionality);
  map_to_points_and_scale(value, points_v.data());
  return points_v;
}

template <typename LLR> size_t base<LLR>::get_snr_idx(float snr) const {
  if (!std::isfinite(snr)) {
    snr = 40;
  }
  if (has_luts()) {
    // find the index of the right lookup table
    if (snr < _lut.snr_min) {
      return 0;
    } else if (snr > _lut.snr_max) {
      return _lut.snr_npoints;
    } else {
      // XXX there might be a better way (using floors instead of a loop)
      for (int i = 0; i < _lut.snr_lut.size() - 1; ++i) {
        if ((snr > _lut.snr_lut[i]) && (snr <= _lut.snr_lut[i + 1])) {
          return i + 1;
        }
      }
    }
  }
}

template <typename LLR>
std::vector<LLR> base<LLR>::_calc_soft_dec(gr_complex sample,
                                           float noise_var) const {
  std::vector<double> p0(bits_per_symbol(), 0.0);
  std::vector<double> p1(bits_per_symbol(), 0.0);
  for (int i = 0; i < d_constellation.size(); ++i) {
    double dist = std::pow(std::abs(sample - d_constellation[i]), 2.0);
    double d = std::exp(-dist / noise_var);
    for (int j = 0; j < bits_per_symbol(); ++j) {
      // FIXME if we have differential constellation this will fail. see the
      // gnu radio version of this function for details.
      auto bit = (i & (1 << j)) >> j;
      if (bit == 0) {
        p0[j] += d;
      } else {
        p1[j] += d;
      }
    }
  }
  std::vector<LLR> out;
  out.resize(bits_per_symbol());
  for (int j = 0; j < p0.size(); ++j) {
    if ((p0[j] == 0.0) && (p1[j] == 0.0)) {
      /*
       * bear with me here. this case happens when both the likelihood of the
       * bit being zero and the likelihood of the bit being 1 are so small that
       * due to floating-point precision limits they are computed to be
       * zero. This happens, for example, when I want to compute the LLR at a
       * noise variance of 0.001 of a sample at, say (0.0, 0.0) in the complex
       * plane.
       *
       * I can set the LLR to zero because:
       *
       * (1) If the estimate of the SNR is correct, the probability that I get a
       * sample point at that spot tends to zero
       *
       * (2) both p0[j] and p1[j] are reaaaally small numbers. I compute their
       * ratio. Here, the ratio is -Inf/-Inf, which is not really -∞/-∞ in this
       * case, but rather a indication of the lack of bits when computing 'dist'
       * above. So. what I am basically saying is that the ratio of a really
       * small number to a really small number is 1, of which the log is zero.
       *
       * The other special cases (p0[j] == 0 or p1[j] == 0) either give log(0),
       * which is -Inf or log(Inf) which is Inf. Both values are fine and if
       * there are problems with infinities, we can always clip them.
       */
      out[bits_per_symbol() - 1 - j] = 0.0;
    } else {
      /*
       * We clip right here, saves some cycles.
       */
      auto llr = LLR(std::log(p0[j] / p1[j]));
      out[bits_per_symbol() - 1 - j] =
          llr < 0.0 ? std::max(-_llr_clip(), llr) : std::min(_llr_clip(), llr);
    }
  }
  return out;
}

// initialize the lookup tables
template <typename LLR> void base<LLR>::_init_lut() {
  using namespace boost::filesystem;
  using namespace boost::archive;
  path p(_lut_wisdom_file_name);
  if (exists(p)) {
    //
    // Load a pre-computed LLR lookup table from disk to memory
    //
    std::ifstream file(_lut_wisdom_file_name, std::ios::binary);
    binary_iarchive ar(file);
    ar >> _lut;
  } else {
    //
    // this lambda generates a LLR lookup table for a specific SNR
    //
    auto gen_lut = [this](int precision, float npwr) {
      std::vector<std::vector<LLR>> lut;
      auto const lut_scale = powf(2.0f, static_cast<float>(_lut.precision));
      float maxd = _llr_range();
      float step = (2.0f * maxd) / (lut_scale - 1);
      float y = -maxd;
      while (y < maxd + step / 2) {
        float x = -maxd;
        while (x < maxd + step / 2) {
          gr_complex pt = gr_complex(x, y);
          lut.push_back(_calc_soft_dec(pt, npwr));
          x += step;
        }
        y += step;
      }
      return lut;
    };

    //
    // populate the SNR table
    //
    _lut.snr_lut.resize(_lut.snr_npoints);
    assert(_lut.snr_max > _lut.snr_min);
    float incr = (_lut.snr_max - _lut.snr_min) / (_lut.snr_npoints - 1);
    int k = 0;
    std::generate(begin(_lut.snr_lut), end(_lut.snr_lut),
                  [&] { return _lut.snr_min + incr * k++; });

    //
    // compute a LLR table for every SNR range
    //
    _lut.llr_lut.resize(_lut.snr_npoints + 1);
    float npwr = 0.0;
    // the SNR_MIN table is first
    npwr = std::pow(10.0, (-1.0) * _lut.snr_min / 10.0);
    _lut.llr_lut[0] = gen_lut(_lut.precision, npwr);
    // SNR_MAX table
    npwr = std::pow(10.0, (-1.0) * _lut.snr_max / 10.0);
    _lut.llr_lut[_lut.snr_npoints] = gen_lut(_lut.precision, npwr);
    // compute the rest of the tables
    for (int i = 0; i < _lut.snr_lut.size() - 1; ++i) {
      // find the mean value of the SNR interval [_lut.snr_lut[i],
      // _lut.snr_lut[i+1]]
      // XXX: Is this the right thing to do when we index in units of dB?
      auto mean = (_lut.snr_lut[i] + _lut.snr_lut[i + 1]) / 2.0;
      npwr = std::pow(10.0, (-1.0) * mean / 10.0);
      // like above, calculate the lookup table and save it
      // gen_soft_dec_lut(_lut.precision, npwr);
      _lut.llr_lut[i + 1] = gen_lut(_lut.precision, npwr);
    }

    //
    // write the new LLR table to the wisdom file
    //
    std::ofstream file(_lut_wisdom_file_name, std::ios::binary);
    binary_oarchive ar(file);
    ar << _lut;
  }
}

// scale the constellation according to scale factors
template <typename LLR> void base<LLR>::_init_scaled_constellation() {
  _scaled_constellation.resize(d_constellation.size());
  for (size_t i = 0; i < d_constellation.size(); ++i) {
    _scaled_constellation[i] = d_constellation[i] * _scal;
  }
}

// lookup the soft decision value as a function of the SNR
template <typename LLR>
void base<LLR>::make_soft_decision(gr_complex sample, LLR *out,
                                   size_t snr_idx) const {
  if (!has_luts()) {
    log::doomsday("Cannot lookup without table", __FILE__, __LINE__);
  }
  // look up the soft decision value in the right lookup table
  // see gr::digital::constellation::soft_decision_maker for original
  static const float lut_scale = powf(2.0f, static_cast<float>(_lut.precision));

  static const float clip = _llr_range() - 0.01;

  // We normalize the constellation in the ctor, so we know that
  // the maximum dimenions go from -1 to +1. We can infer the x
  // and y scale directly.
  static const float scale = lut_scale / (2.0f * _llr_range());

  // scale the sample to an average energy of 1
  sample *= _inv_scal;

  // Clip to just below 1 --> at 1, we can overflow the index
  // that will put us in the next row of the 2D LUT.
  float xre = gr::branchless_clip(sample.real(), clip);
  float xim = gr::branchless_clip(sample.imag(), clip);

  // DEBUG -- take out eventually
  if (std::isnan(xre) || std::isnan(xim)) {
    log::doomsday("NaN after FFT", __FILE__, __LINE__);
  }

  // Convert the clipped x and y samples to nearest index offset
  xre = floorf((_llr_range() + xre) * scale);
  xim = floorf((_llr_range() + xim) * scale);
  int index = static_cast<int>(lut_scale * xim + xre);

  int max_index = lut_scale * lut_scale;

  // Make sure we are in bounds of the index
  while (index >= max_index) {
    index -= lut_scale;
  }
  while (index < 0) {
    index += lut_scale;
  }

  auto const &llr = _lut.llr_lut[snr_idx][index];
  std::copy(llr.begin(), llr.end(), out);
}

template <typename LLR>
void base<LLR>::calculate_soft_decision(gr_complex sample, LLR *out,
                                        float snr) const {
  sample *= _inv_scal;
  float const npwr = std::pow(10.0f, (-1.0f) * snr / 10.0f);
  auto const res = _calc_soft_dec(sample, npwr);
  // see block comment in clip_and_flip above
  for (size_t k = 0; k < res.size(); ++k)
    out[k] = res[k];
}

// gnu radio constellation object artifact
template <typename LLR> void base<LLR>::calc_arity() {
  if (d_constellation.size() % d_dimensionality != 0)
    log::doomsday(
        "Constellation vector size must be a multiple of the dimensionality.",
        __FILE__, __LINE__);
  d_arity = d_constellation.size() / d_dimensionality;
}

// BPSK
// with soft decision LUTs
template <typename LLR>
typename bpsk<LLR>::sptr
bpsk<LLR>::make(float a0, float a1, float a2, int lut_precision,
                int snr_npoints, float snr_min, float snr_max,
                std::string const &lut_wisdom_file_name) {
  return bpsk::sptr(new bpsk(a0, a1, a2, lut_precision, snr_npoints, snr_min,
                             snr_max, lut_wisdom_file_name));
}
template <typename LLR>
bpsk<LLR>::bpsk(float a0, float a1, float a2, int lut_precision,
                int snr_npoints, float snr_min, float snr_max,
                std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decicion LUTs
template <typename LLR>
typename bpsk<LLR>::sptr bpsk<LLR>::make(float a0, float a1, float a2) {
  return bpsk::sptr(new bpsk(a0, a1, a2));
}
template <typename LLR>
bpsk<LLR>::bpsk(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void bpsk<LLR>::_init_const() {
  this->d_constellation.resize(2);
  this->d_constellation[0] = gr_complex(-1, 0);
  this->d_constellation[1] = gr_complex(1, 0);
  this->_init_scaled_constellation();
  this->d_rotational_symmetry = 2;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int bpsk<LLR>::decision_maker(const gr_complex *sample) {
  return (real(*sample) > 0);
}

// QPSK
// with soft decision LUTs
template <typename LLR>
typename qpsk<LLR>::sptr
qpsk<LLR>::make(float a0, float a1, float a2, int lut_precision,
                int snr_npoints, float snr_min, float snr_max,
                std::string const &lut_wisdom_file_name) {
  return qpsk::sptr(new qpsk(a0, a1, a2, lut_precision, snr_npoints, snr_min,
                             snr_max, lut_wisdom_file_name));
}
template <typename LLR>
qpsk<LLR>::qpsk(float a0, float a1, float a2, int lut_precision,
                int snr_npoints, float snr_min, float snr_max,
                std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decicion LUTs
template <typename LLR>
typename qpsk<LLR>::sptr qpsk<LLR>::make(float a0, float a1, float a2) {
  return qpsk::sptr(new qpsk(a0, a1, a2));
}
template <typename LLR>
qpsk<LLR>::qpsk(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void qpsk<LLR>::_init_const() {
  double val = 1.0 / std::sqrt(2.0);
  // copied verbatim from constellation.cc
  this->d_constellation.resize(4);
  // Gray-coded
  this->d_constellation[0] = gr_complex(-val, -val);
  this->d_constellation[1] = gr_complex(val, -val);
  this->d_constellation[2] = gr_complex(-val, val);
  this->d_constellation[3] = gr_complex(val, val);
  this->_init_scaled_constellation();

  this->d_rotational_symmetry = 4;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int qpsk<LLR>::decision_maker(const gr_complex *sample) {
  return 2 * (imag(*sample) > 0) + (real(*sample) > 0);
}

// QAM16
// with soft decision LUTs
template <typename LLR>
typename qam16<LLR>::sptr
qam16<LLR>::make(float a0, float a1, float a2, int lut_precision,
                 int snr_npoints, float snr_min, float snr_max,
                 std::string const &lut_wisdom_file_name) {
  return qam16::sptr(new qam16(a0, a1, a2, lut_precision, snr_npoints, snr_min,
                               snr_max, lut_wisdom_file_name));
}
template <typename LLR>
qam16<LLR>::qam16(float a0, float a1, float a2, int lut_precision,
                  int snr_npoints, float snr_min, float snr_max,
                  std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decision LUTs
template <typename LLR>
typename qam16<LLR>::sptr qam16<LLR>::make(float a0, float a1, float a2) {
  return qam16::sptr(new qam16(a0, a1, a2));
}
template <typename LLR>
qam16<LLR>::qam16(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "qam16_wisdom") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void qam16<LLR>::_init_const() {
  const float level = sqrt(float(0.1));
  this->d_constellation.resize(16);
  this->d_constellation[0] = gr_complex(-3.0000 * level, 3.0000 * level);
  this->d_constellation[1] = gr_complex(-3.0000 * level, 1.0000 * level);
  this->d_constellation[2] = gr_complex(-3.0000 * level, -1.0000 * level);
  this->d_constellation[3] = gr_complex(-3.0000 * level, -3.0000 * level);
  this->d_constellation[4] = gr_complex(-1.0000 * level, 3.0000 * level);
  this->d_constellation[5] = gr_complex(-1.0000 * level, 1.0000 * level);
  this->d_constellation[6] = gr_complex(-1.0000 * level, -1.0000 * level);
  this->d_constellation[7] = gr_complex(-1.0000 * level, -3.0000 * level);
  this->d_constellation[8] = gr_complex(1.0000 * level, 3.0000 * level);
  this->d_constellation[9] = gr_complex(1.0000 * level, 1.0000 * level);
  this->d_constellation[10] = gr_complex(1.0000 * level, -1.0000 * level);
  this->d_constellation[11] = gr_complex(1.0000 * level, -3.0000 * level);
  this->d_constellation[12] = gr_complex(3.0000 * level, 3.0000 * level);
  this->d_constellation[13] = gr_complex(3.0000 * level, 1.0000 * level);
  this->d_constellation[14] = gr_complex(3.0000 * level, -1.0000 * level);
  this->d_constellation[15] = gr_complex(3.0000 * level, -3.0000 * level);
  this->_init_scaled_constellation();
  this->d_rotational_symmetry = 4;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int qam16<LLR>::decision_maker(const gr_complex *sample) {
  unsigned int ret = 0;
  const float level = sqrt(float(0.1));
  float re = sample->real();
  float im = sample->imag();

  /////////////////////////////////////////////////////////////////////////////
  // RED ALERT RED ALERT
  //
  //  DO NOT USE THE HARD SLICER, IT IS NOT IMPLEMENTED
  /////////////////////////////////////////////////////////////////////////////

  log::doomsday("Hard slicer is not implemented", __FILE__, __LINE__);

  return ret;
}

//
// higher-order qam. the proper way to do this is to use templates but it's
// november, so ¯\_(ツ)_/¯
//

// QAM32
// with soft decision LUTs
template <typename LLR>
typename qam32<LLR>::sptr
qam32<LLR>::make(float a0, float a1, float a2, int lut_precision,
                 int snr_npoints, float snr_min, float snr_max,
                 std::string const &lut_wisdom_file_name) {
  return qam32::sptr(new qam32(a0, a1, a2, lut_precision, snr_npoints, snr_min,
                               snr_max, lut_wisdom_file_name));
}
template <typename LLR>
qam32<LLR>::qam32(float a0, float a1, float a2, int lut_precision,
                  int snr_npoints, float snr_min, float snr_max,
                  std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decision LUTs
template <typename LLR>
typename qam32<LLR>::sptr qam32<LLR>::make(float a0, float a1, float a2) {
  return qam32::sptr(new qam32(a0, a1, a2));
}
template <typename LLR>
qam32<LLR>::qam32(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "qam32_wisdom") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void qam32<LLR>::_init_const() {
  // paste constellation points below
  const float level = 1.0 / sqrt(20.00 * sqrt(2.0));
  this->d_constellation.resize(32);
  this->d_constellation[0] = gr_complex(-3.0000 * level, 5.0000 * level);
  this->d_constellation[1] = gr_complex(-1.0000 * level, 5.0000 * level);
  this->d_constellation[2] = gr_complex(-1.0000 * level, -5.0000 * level);
  this->d_constellation[3] = gr_complex(-3.0000 * level, -5.0000 * level);
  this->d_constellation[4] = gr_complex(-5.0000 * level, 3.0000 * level);
  this->d_constellation[5] = gr_complex(-5.0000 * level, 1.0000 * level);
  this->d_constellation[6] = gr_complex(-5.0000 * level, -1.0000 * level);
  this->d_constellation[7] = gr_complex(-5.0000 * level, -3.0000 * level);
  this->d_constellation[8] = gr_complex(-3.0000 * level, 3.0000 * level);
  this->d_constellation[9] = gr_complex(-3.0000 * level, 1.0000 * level);
  this->d_constellation[10] = gr_complex(-3.0000 * level, -1.0000 * level);
  this->d_constellation[11] = gr_complex(-3.0000 * level, -3.0000 * level);
  this->d_constellation[12] = gr_complex(-1.0000 * level, 3.0000 * level);
  this->d_constellation[13] = gr_complex(-1.0000 * level, 1.0000 * level);
  this->d_constellation[14] = gr_complex(-1.0000 * level, -1.0000 * level);
  this->d_constellation[15] = gr_complex(-1.0000 * level, -3.0000 * level);
  this->d_constellation[16] = gr_complex(1.0000 * level, 3.0000 * level);
  this->d_constellation[17] = gr_complex(1.0000 * level, 1.0000 * level);
  this->d_constellation[18] = gr_complex(1.0000 * level, -1.0000 * level);
  this->d_constellation[19] = gr_complex(1.0000 * level, -3.0000 * level);
  this->d_constellation[20] = gr_complex(3.0000 * level, 3.0000 * level);
  this->d_constellation[21] = gr_complex(3.0000 * level, 1.0000 * level);
  this->d_constellation[22] = gr_complex(3.0000 * level, -1.0000 * level);
  this->d_constellation[23] = gr_complex(3.0000 * level, -3.0000 * level);
  this->d_constellation[24] = gr_complex(5.0000 * level, 3.0000 * level);
  this->d_constellation[25] = gr_complex(5.0000 * level, 1.0000 * level);
  this->d_constellation[26] = gr_complex(5.0000 * level, -1.0000 * level);
  this->d_constellation[27] = gr_complex(5.0000 * level, -3.0000 * level);
  this->d_constellation[28] = gr_complex(3.0000 * level, 5.0000 * level);
  this->d_constellation[29] = gr_complex(1.0000 * level, 5.0000 * level);
  this->d_constellation[30] = gr_complex(1.0000 * level, -5.0000 * level);
  this->d_constellation[31] = gr_complex(3.0000 * level, -5.0000 * level);
  this->_init_scaled_constellation();
  this->d_rotational_symmetry = 4;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int qam32<LLR>::decision_maker(const gr_complex *sample) {
  unsigned int ret = 0;
  const float level = sqrt(float(0.1));
  float re = sample->real();
  float im = sample->imag();

  /////////////////////////////////////////////////////////////////////////////
  // RED ALERT RED ALERT
  //
  //  DO NOT USE THE HARD SLICER, IT IS NOT IMPLEMENTED
  /////////////////////////////////////////////////////////////////////////////
  log::doomsday("Hard slicer is not implemented", __FILE__, __LINE__);

  return ret;
}

// QAM64
// with soft decision LUTs
template <typename LLR>
typename qam64<LLR>::sptr
qam64<LLR>::make(float a0, float a1, float a2, int lut_precision,
                 int snr_npoints, float snr_min, float snr_max,
                 std::string const &lut_wisdom_file_name) {
  return qam64::sptr(new qam64(a0, a1, a2, lut_precision, snr_npoints, snr_min,
                               snr_max, lut_wisdom_file_name));
}
template <typename LLR>
qam64<LLR>::qam64(float a0, float a1, float a2, int lut_precision,
                  int snr_npoints, float snr_min, float snr_max,
                  std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decision LUTs
template <typename LLR>
typename qam64<LLR>::sptr qam64<LLR>::make(float a0, float a1, float a2) {
  return qam64::sptr(new qam64(a0, a1, a2));
}
template <typename LLR>
qam64<LLR>::qam64(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "qam64_wisdom") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void qam64<LLR>::_init_const() {
  // paste constellation points below
  const float level = 1.0 / sqrt(42.00 * sqrt(2.0));
  this->d_constellation.resize(64);
  this->d_constellation[0] = gr_complex(-7.0000 * level, 7.0000 * level);
  this->d_constellation[1] = gr_complex(-7.0000 * level, 5.0000 * level);
  this->d_constellation[2] = gr_complex(-7.0000 * level, 3.0000 * level);
  this->d_constellation[3] = gr_complex(-7.0000 * level, 1.0000 * level);
  this->d_constellation[4] = gr_complex(-7.0000 * level, -1.0000 * level);
  this->d_constellation[5] = gr_complex(-7.0000 * level, -3.0000 * level);
  this->d_constellation[6] = gr_complex(-7.0000 * level, -5.0000 * level);
  this->d_constellation[7] = gr_complex(-7.0000 * level, -7.0000 * level);
  this->d_constellation[8] = gr_complex(-5.0000 * level, 7.0000 * level);
  this->d_constellation[9] = gr_complex(-5.0000 * level, 5.0000 * level);
  this->d_constellation[10] = gr_complex(-5.0000 * level, 3.0000 * level);
  this->d_constellation[11] = gr_complex(-5.0000 * level, 1.0000 * level);
  this->d_constellation[12] = gr_complex(-5.0000 * level, -1.0000 * level);
  this->d_constellation[13] = gr_complex(-5.0000 * level, -3.0000 * level);
  this->d_constellation[14] = gr_complex(-5.0000 * level, -5.0000 * level);
  this->d_constellation[15] = gr_complex(-5.0000 * level, -7.0000 * level);
  this->d_constellation[16] = gr_complex(-3.0000 * level, 7.0000 * level);
  this->d_constellation[17] = gr_complex(-3.0000 * level, 5.0000 * level);
  this->d_constellation[18] = gr_complex(-3.0000 * level, 3.0000 * level);
  this->d_constellation[19] = gr_complex(-3.0000 * level, 1.0000 * level);
  this->d_constellation[20] = gr_complex(-3.0000 * level, -1.0000 * level);
  this->d_constellation[21] = gr_complex(-3.0000 * level, -3.0000 * level);
  this->d_constellation[22] = gr_complex(-3.0000 * level, -5.0000 * level);
  this->d_constellation[23] = gr_complex(-3.0000 * level, -7.0000 * level);
  this->d_constellation[24] = gr_complex(-1.0000 * level, 7.0000 * level);
  this->d_constellation[25] = gr_complex(-1.0000 * level, 5.0000 * level);
  this->d_constellation[26] = gr_complex(-1.0000 * level, 3.0000 * level);
  this->d_constellation[27] = gr_complex(-1.0000 * level, 1.0000 * level);
  this->d_constellation[28] = gr_complex(-1.0000 * level, -1.0000 * level);
  this->d_constellation[29] = gr_complex(-1.0000 * level, -3.0000 * level);
  this->d_constellation[30] = gr_complex(-1.0000 * level, -5.0000 * level);
  this->d_constellation[31] = gr_complex(-1.0000 * level, -7.0000 * level);
  this->d_constellation[32] = gr_complex(1.0000 * level, 7.0000 * level);
  this->d_constellation[33] = gr_complex(1.0000 * level, 5.0000 * level);
  this->d_constellation[34] = gr_complex(1.0000 * level, 3.0000 * level);
  this->d_constellation[35] = gr_complex(1.0000 * level, 1.0000 * level);
  this->d_constellation[36] = gr_complex(1.0000 * level, -1.0000 * level);
  this->d_constellation[37] = gr_complex(1.0000 * level, -3.0000 * level);
  this->d_constellation[38] = gr_complex(1.0000 * level, -5.0000 * level);
  this->d_constellation[39] = gr_complex(1.0000 * level, -7.0000 * level);
  this->d_constellation[40] = gr_complex(3.0000 * level, 7.0000 * level);
  this->d_constellation[41] = gr_complex(3.0000 * level, 5.0000 * level);
  this->d_constellation[42] = gr_complex(3.0000 * level, 3.0000 * level);
  this->d_constellation[43] = gr_complex(3.0000 * level, 1.0000 * level);
  this->d_constellation[44] = gr_complex(3.0000 * level, -1.0000 * level);
  this->d_constellation[45] = gr_complex(3.0000 * level, -3.0000 * level);
  this->d_constellation[46] = gr_complex(3.0000 * level, -5.0000 * level);
  this->d_constellation[47] = gr_complex(3.0000 * level, -7.0000 * level);
  this->d_constellation[48] = gr_complex(5.0000 * level, 7.0000 * level);
  this->d_constellation[49] = gr_complex(5.0000 * level, 5.0000 * level);
  this->d_constellation[50] = gr_complex(5.0000 * level, 3.0000 * level);
  this->d_constellation[51] = gr_complex(5.0000 * level, 1.0000 * level);
  this->d_constellation[52] = gr_complex(5.0000 * level, -1.0000 * level);
  this->d_constellation[53] = gr_complex(5.0000 * level, -3.0000 * level);
  this->d_constellation[54] = gr_complex(5.0000 * level, -5.0000 * level);
  this->d_constellation[55] = gr_complex(5.0000 * level, -7.0000 * level);
  this->d_constellation[56] = gr_complex(7.0000 * level, 7.0000 * level);
  this->d_constellation[57] = gr_complex(7.0000 * level, 5.0000 * level);
  this->d_constellation[58] = gr_complex(7.0000 * level, 3.0000 * level);
  this->d_constellation[59] = gr_complex(7.0000 * level, 1.0000 * level);
  this->d_constellation[60] = gr_complex(7.0000 * level, -1.0000 * level);
  this->d_constellation[61] = gr_complex(7.0000 * level, -3.0000 * level);
  this->d_constellation[62] = gr_complex(7.0000 * level, -5.0000 * level);
  this->d_constellation[63] = gr_complex(7.0000 * level, -7.0000 * level);
  this->_init_scaled_constellation();
  this->d_rotational_symmetry = 4;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int qam64<LLR>::decision_maker(const gr_complex *sample) {
  unsigned int ret = 0;
  const float level = sqrt(float(0.1));
  float re = sample->real();
  float im = sample->imag();

  /////////////////////////////////////////////////////////////////////////////
  // RED ALERT RED ALERT
  //
  //  DO NOT USE THE HARD SLICER, IT IS NOT IMPLEMENTED
  /////////////////////////////////////////////////////////////////////////////
  log::doomsday("Hard slicer is not implemented", __FILE__, __LINE__);

  return ret;
}

// QAM128
// with soft decision LUTs
template <typename LLR>
typename qam128<LLR>::sptr
qam128<LLR>::make(float a0, float a1, float a2, int lut_precision,
                  int snr_npoints, float snr_min, float snr_max,
                  std::string const &lut_wisdom_file_name) {
  return qam128::sptr(new qam128(a0, a1, a2, lut_precision, snr_npoints,
                                 snr_min, snr_max, lut_wisdom_file_name));
}
template <typename LLR>
qam128<LLR>::qam128(float a0, float a1, float a2, int lut_precision,
                    int snr_npoints, float snr_min, float snr_max,
                    std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decision LUTs
template <typename LLR>
typename qam128<LLR>::sptr qam128<LLR>::make(float a0, float a1, float a2) {
  return qam128::sptr(new qam128(a0, a1, a2));
}
template <typename LLR>
qam128<LLR>::qam128(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "qam128_wisdom") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void qam128<LLR>::_init_const() {
  // paste constellation points below
  const float level = 1.0 / sqrt(82.00 * 1.7);
  this->d_constellation.resize(128);
  this->d_constellation[0] = gr_complex(-7.0000 * level, 9.0000 * level);
  this->d_constellation[1] = gr_complex(-7.0000 * level, 11.0000 * level);
  this->d_constellation[2] = gr_complex(-1.0000 * level, 11.0000 * level);
  this->d_constellation[3] = gr_complex(-1.0000 * level, 9.0000 * level);
  this->d_constellation[4] = gr_complex(-1.0000 * level, -9.0000 * level);
  this->d_constellation[5] = gr_complex(-1.0000 * level, -11.0000 * level);
  this->d_constellation[6] = gr_complex(-7.0000 * level, -11.0000 * level);
  this->d_constellation[7] = gr_complex(-7.0000 * level, -9.0000 * level);
  this->d_constellation[8] = gr_complex(-5.0000 * level, 9.0000 * level);
  this->d_constellation[9] = gr_complex(-5.0000 * level, 11.0000 * level);
  this->d_constellation[10] = gr_complex(-3.0000 * level, 11.0000 * level);
  this->d_constellation[11] = gr_complex(-3.0000 * level, 9.0000 * level);
  this->d_constellation[12] = gr_complex(-3.0000 * level, -9.0000 * level);
  this->d_constellation[13] = gr_complex(-3.0000 * level, -11.0000 * level);
  this->d_constellation[14] = gr_complex(-5.0000 * level, -11.0000 * level);
  this->d_constellation[15] = gr_complex(-5.0000 * level, -9.0000 * level);
  this->d_constellation[16] = gr_complex(-11.0000 * level, 7.0000 * level);
  this->d_constellation[17] = gr_complex(-11.0000 * level, 5.0000 * level);
  this->d_constellation[18] = gr_complex(-11.0000 * level, 3.0000 * level);
  this->d_constellation[19] = gr_complex(-11.0000 * level, 1.0000 * level);
  this->d_constellation[20] = gr_complex(-11.0000 * level, -1.0000 * level);
  this->d_constellation[21] = gr_complex(-11.0000 * level, -3.0000 * level);
  this->d_constellation[22] = gr_complex(-11.0000 * level, -5.0000 * level);
  this->d_constellation[23] = gr_complex(-11.0000 * level, -7.0000 * level);
  this->d_constellation[24] = gr_complex(-9.0000 * level, 7.0000 * level);
  this->d_constellation[25] = gr_complex(-9.0000 * level, 5.0000 * level);
  this->d_constellation[26] = gr_complex(-9.0000 * level, 3.0000 * level);
  this->d_constellation[27] = gr_complex(-9.0000 * level, 1.0000 * level);
  this->d_constellation[28] = gr_complex(-9.0000 * level, -1.0000 * level);
  this->d_constellation[29] = gr_complex(-9.0000 * level, -3.0000 * level);
  this->d_constellation[30] = gr_complex(-9.0000 * level, -5.0000 * level);
  this->d_constellation[31] = gr_complex(-9.0000 * level, -7.0000 * level);
  this->d_constellation[32] = gr_complex(-7.0000 * level, 7.0000 * level);
  this->d_constellation[33] = gr_complex(-7.0000 * level, 5.0000 * level);
  this->d_constellation[34] = gr_complex(-7.0000 * level, 3.0000 * level);
  this->d_constellation[35] = gr_complex(-7.0000 * level, 1.0000 * level);
  this->d_constellation[36] = gr_complex(-7.0000 * level, -1.0000 * level);
  this->d_constellation[37] = gr_complex(-7.0000 * level, -3.0000 * level);
  this->d_constellation[38] = gr_complex(-7.0000 * level, -5.0000 * level);
  this->d_constellation[39] = gr_complex(-7.0000 * level, -7.0000 * level);
  this->d_constellation[40] = gr_complex(-5.0000 * level, 7.0000 * level);
  this->d_constellation[41] = gr_complex(-5.0000 * level, 5.0000 * level);
  this->d_constellation[42] = gr_complex(-5.0000 * level, 3.0000 * level);
  this->d_constellation[43] = gr_complex(-5.0000 * level, 1.0000 * level);
  this->d_constellation[44] = gr_complex(-5.0000 * level, -1.0000 * level);
  this->d_constellation[45] = gr_complex(-5.0000 * level, -3.0000 * level);
  this->d_constellation[46] = gr_complex(-5.0000 * level, -5.0000 * level);
  this->d_constellation[47] = gr_complex(-5.0000 * level, -7.0000 * level);
  this->d_constellation[48] = gr_complex(-3.0000 * level, 7.0000 * level);
  this->d_constellation[49] = gr_complex(-3.0000 * level, 5.0000 * level);
  this->d_constellation[50] = gr_complex(-3.0000 * level, 3.0000 * level);
  this->d_constellation[51] = gr_complex(-3.0000 * level, 1.0000 * level);
  this->d_constellation[52] = gr_complex(-3.0000 * level, -1.0000 * level);
  this->d_constellation[53] = gr_complex(-3.0000 * level, -3.0000 * level);
  this->d_constellation[54] = gr_complex(-3.0000 * level, -5.0000 * level);
  this->d_constellation[55] = gr_complex(-3.0000 * level, -7.0000 * level);
  this->d_constellation[56] = gr_complex(-1.0000 * level, 7.0000 * level);
  this->d_constellation[57] = gr_complex(-1.0000 * level, 5.0000 * level);
  this->d_constellation[58] = gr_complex(-1.0000 * level, 3.0000 * level);
  this->d_constellation[59] = gr_complex(-1.0000 * level, 1.0000 * level);
  this->d_constellation[60] = gr_complex(-1.0000 * level, -1.0000 * level);
  this->d_constellation[61] = gr_complex(-1.0000 * level, -3.0000 * level);
  this->d_constellation[62] = gr_complex(-1.0000 * level, -5.0000 * level);
  this->d_constellation[63] = gr_complex(-1.0000 * level, -7.0000 * level);
  this->d_constellation[64] = gr_complex(1.0000 * level, 7.0000 * level);
  this->d_constellation[65] = gr_complex(1.0000 * level, 5.0000 * level);
  this->d_constellation[66] = gr_complex(1.0000 * level, 3.0000 * level);
  this->d_constellation[67] = gr_complex(1.0000 * level, 1.0000 * level);
  this->d_constellation[68] = gr_complex(1.0000 * level, -1.0000 * level);
  this->d_constellation[69] = gr_complex(1.0000 * level, -3.0000 * level);
  this->d_constellation[70] = gr_complex(1.0000 * level, -5.0000 * level);
  this->d_constellation[71] = gr_complex(1.0000 * level, -7.0000 * level);
  this->d_constellation[72] = gr_complex(3.0000 * level, 7.0000 * level);
  this->d_constellation[73] = gr_complex(3.0000 * level, 5.0000 * level);
  this->d_constellation[74] = gr_complex(3.0000 * level, 3.0000 * level);
  this->d_constellation[75] = gr_complex(3.0000 * level, 1.0000 * level);
  this->d_constellation[76] = gr_complex(3.0000 * level, -1.0000 * level);
  this->d_constellation[77] = gr_complex(3.0000 * level, -3.0000 * level);
  this->d_constellation[78] = gr_complex(3.0000 * level, -5.0000 * level);
  this->d_constellation[79] = gr_complex(3.0000 * level, -7.0000 * level);
  this->d_constellation[80] = gr_complex(5.0000 * level, 7.0000 * level);
  this->d_constellation[81] = gr_complex(5.0000 * level, 5.0000 * level);
  this->d_constellation[82] = gr_complex(5.0000 * level, 3.0000 * level);
  this->d_constellation[83] = gr_complex(5.0000 * level, 1.0000 * level);
  this->d_constellation[84] = gr_complex(5.0000 * level, -1.0000 * level);
  this->d_constellation[85] = gr_complex(5.0000 * level, -3.0000 * level);
  this->d_constellation[86] = gr_complex(5.0000 * level, -5.0000 * level);
  this->d_constellation[87] = gr_complex(5.0000 * level, -7.0000 * level);
  this->d_constellation[88] = gr_complex(7.0000 * level, 7.0000 * level);
  this->d_constellation[89] = gr_complex(7.0000 * level, 5.0000 * level);
  this->d_constellation[90] = gr_complex(7.0000 * level, 3.0000 * level);
  this->d_constellation[91] = gr_complex(7.0000 * level, 1.0000 * level);
  this->d_constellation[92] = gr_complex(7.0000 * level, -1.0000 * level);
  this->d_constellation[93] = gr_complex(7.0000 * level, -3.0000 * level);
  this->d_constellation[94] = gr_complex(7.0000 * level, -5.0000 * level);
  this->d_constellation[95] = gr_complex(7.0000 * level, -7.0000 * level);
  this->d_constellation[96] = gr_complex(9.0000 * level, 7.0000 * level);
  this->d_constellation[97] = gr_complex(9.0000 * level, 5.0000 * level);
  this->d_constellation[98] = gr_complex(9.0000 * level, 3.0000 * level);
  this->d_constellation[99] = gr_complex(9.0000 * level, 1.0000 * level);
  this->d_constellation[100] = gr_complex(9.0000 * level, -1.0000 * level);
  this->d_constellation[101] = gr_complex(9.0000 * level, -3.0000 * level);
  this->d_constellation[102] = gr_complex(9.0000 * level, -5.0000 * level);
  this->d_constellation[103] = gr_complex(9.0000 * level, -7.0000 * level);
  this->d_constellation[104] = gr_complex(11.0000 * level, 7.0000 * level);
  this->d_constellation[105] = gr_complex(11.0000 * level, 5.0000 * level);
  this->d_constellation[106] = gr_complex(11.0000 * level, 3.0000 * level);
  this->d_constellation[107] = gr_complex(11.0000 * level, 1.0000 * level);
  this->d_constellation[108] = gr_complex(11.0000 * level, -1.0000 * level);
  this->d_constellation[109] = gr_complex(11.0000 * level, -3.0000 * level);
  this->d_constellation[110] = gr_complex(11.0000 * level, -5.0000 * level);
  this->d_constellation[111] = gr_complex(11.0000 * level, -7.0000 * level);
  this->d_constellation[112] = gr_complex(5.0000 * level, 9.0000 * level);
  this->d_constellation[113] = gr_complex(5.0000 * level, 11.0000 * level);
  this->d_constellation[114] = gr_complex(3.0000 * level, 11.0000 * level);
  this->d_constellation[115] = gr_complex(3.0000 * level, 9.0000 * level);
  this->d_constellation[116] = gr_complex(3.0000 * level, -9.0000 * level);
  this->d_constellation[117] = gr_complex(3.0000 * level, -11.0000 * level);
  this->d_constellation[118] = gr_complex(5.0000 * level, -11.0000 * level);
  this->d_constellation[119] = gr_complex(5.0000 * level, -9.0000 * level);
  this->d_constellation[120] = gr_complex(7.0000 * level, 9.0000 * level);
  this->d_constellation[121] = gr_complex(7.0000 * level, 11.0000 * level);
  this->d_constellation[122] = gr_complex(1.0000 * level, 11.0000 * level);
  this->d_constellation[123] = gr_complex(1.0000 * level, 9.0000 * level);
  this->d_constellation[124] = gr_complex(1.0000 * level, -9.0000 * level);
  this->d_constellation[125] = gr_complex(1.0000 * level, -11.0000 * level);
  this->d_constellation[126] = gr_complex(7.0000 * level, -11.0000 * level);
  this->d_constellation[127] = gr_complex(7.0000 * level, -9.0000 * level);
  this->_init_scaled_constellation();
  this->d_rotational_symmetry = 4;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int qam128<LLR>::decision_maker(const gr_complex *sample) {
  unsigned int ret = 0;
  const float level = sqrt(float(0.1));
  float re = sample->real();
  float im = sample->imag();

  /////////////////////////////////////////////////////////////////////////////
  // RED ALERT RED ALERT
  //
  //  DO NOT USE THE HARD SLICER, IT IS NOT IMPLEMENTED
  /////////////////////////////////////////////////////////////////////////////
  log::doomsday("Hard slicer is not implemented", __FILE__, __LINE__);

  return ret;
}

// QAM256
// with soft decision LUTs
template <typename LLR>
typename qam256<LLR>::sptr
qam256<LLR>::make(float a0, float a1, float a2, int lut_precision,
                  int snr_npoints, float snr_min, float snr_max,
                  std::string const &lut_wisdom_file_name) {
  return qam256::sptr(new qam256(a0, a1, a2, lut_precision, snr_npoints,
                                 snr_min, snr_max, lut_wisdom_file_name));
}
template <typename LLR>
qam256<LLR>::qam256(float a0, float a1, float a2, int lut_precision,
                    int snr_npoints, float snr_min, float snr_max,
                    std::string const &lut_wisdom_file_name)
    : base<LLR>(a0, a1, a2, lut_precision, snr_npoints, snr_min, snr_max,
                lut_wisdom_file_name) {
  this->_init_const();
  // initialize the LUT
  this->_init_lut();
  this->_has_luts = true;
}

// without soft decision LUTs
template <typename LLR>
typename qam256<LLR>::sptr qam256<LLR>::make(float a0, float a1, float a2) {
  return qam256::sptr(new qam256(a0, a1, a2));
}
template <typename LLR>
qam256<LLR>::qam256(float a0, float a1, float a2)
    : base<LLR>(a0, a1, a2, 0, 0, 0.0, 0.0, "qam256_wisdom") {
  this->_init_const();
  this->_has_luts = false;
}

// constellation points
template <typename LLR> void qam256<LLR>::_init_const() {
  // paste constellation points below
  const float level = 1.0 / sqrt(170.00 * sqrt(2.0));
  this->d_constellation.resize(256);
  this->d_constellation[0] = gr_complex(-15.0000 * level, 15.0000 * level);
  this->d_constellation[1] = gr_complex(-15.0000 * level, 13.0000 * level);
  this->d_constellation[2] = gr_complex(-15.0000 * level, 11.0000 * level);
  this->d_constellation[3] = gr_complex(-15.0000 * level, 9.0000 * level);
  this->d_constellation[4] = gr_complex(-15.0000 * level, 7.0000 * level);
  this->d_constellation[5] = gr_complex(-15.0000 * level, 5.0000 * level);
  this->d_constellation[6] = gr_complex(-15.0000 * level, 3.0000 * level);
  this->d_constellation[7] = gr_complex(-15.0000 * level, 1.0000 * level);
  this->d_constellation[8] = gr_complex(-15.0000 * level, -1.0000 * level);
  this->d_constellation[9] = gr_complex(-15.0000 * level, -3.0000 * level);
  this->d_constellation[10] = gr_complex(-15.0000 * level, -5.0000 * level);
  this->d_constellation[11] = gr_complex(-15.0000 * level, -7.0000 * level);
  this->d_constellation[12] = gr_complex(-15.0000 * level, -9.0000 * level);
  this->d_constellation[13] = gr_complex(-15.0000 * level, -11.0000 * level);
  this->d_constellation[14] = gr_complex(-15.0000 * level, -13.0000 * level);
  this->d_constellation[15] = gr_complex(-15.0000 * level, -15.0000 * level);
  this->d_constellation[16] = gr_complex(-13.0000 * level, 15.0000 * level);
  this->d_constellation[17] = gr_complex(-13.0000 * level, 13.0000 * level);
  this->d_constellation[18] = gr_complex(-13.0000 * level, 11.0000 * level);
  this->d_constellation[19] = gr_complex(-13.0000 * level, 9.0000 * level);
  this->d_constellation[20] = gr_complex(-13.0000 * level, 7.0000 * level);
  this->d_constellation[21] = gr_complex(-13.0000 * level, 5.0000 * level);
  this->d_constellation[22] = gr_complex(-13.0000 * level, 3.0000 * level);
  this->d_constellation[23] = gr_complex(-13.0000 * level, 1.0000 * level);
  this->d_constellation[24] = gr_complex(-13.0000 * level, -1.0000 * level);
  this->d_constellation[25] = gr_complex(-13.0000 * level, -3.0000 * level);
  this->d_constellation[26] = gr_complex(-13.0000 * level, -5.0000 * level);
  this->d_constellation[27] = gr_complex(-13.0000 * level, -7.0000 * level);
  this->d_constellation[28] = gr_complex(-13.0000 * level, -9.0000 * level);
  this->d_constellation[29] = gr_complex(-13.0000 * level, -11.0000 * level);
  this->d_constellation[30] = gr_complex(-13.0000 * level, -13.0000 * level);
  this->d_constellation[31] = gr_complex(-13.0000 * level, -15.0000 * level);
  this->d_constellation[32] = gr_complex(-11.0000 * level, 15.0000 * level);
  this->d_constellation[33] = gr_complex(-11.0000 * level, 13.0000 * level);
  this->d_constellation[34] = gr_complex(-11.0000 * level, 11.0000 * level);
  this->d_constellation[35] = gr_complex(-11.0000 * level, 9.0000 * level);
  this->d_constellation[36] = gr_complex(-11.0000 * level, 7.0000 * level);
  this->d_constellation[37] = gr_complex(-11.0000 * level, 5.0000 * level);
  this->d_constellation[38] = gr_complex(-11.0000 * level, 3.0000 * level);
  this->d_constellation[39] = gr_complex(-11.0000 * level, 1.0000 * level);
  this->d_constellation[40] = gr_complex(-11.0000 * level, -1.0000 * level);
  this->d_constellation[41] = gr_complex(-11.0000 * level, -3.0000 * level);
  this->d_constellation[42] = gr_complex(-11.0000 * level, -5.0000 * level);
  this->d_constellation[43] = gr_complex(-11.0000 * level, -7.0000 * level);
  this->d_constellation[44] = gr_complex(-11.0000 * level, -9.0000 * level);
  this->d_constellation[45] = gr_complex(-11.0000 * level, -11.0000 * level);
  this->d_constellation[46] = gr_complex(-11.0000 * level, -13.0000 * level);
  this->d_constellation[47] = gr_complex(-11.0000 * level, -15.0000 * level);
  this->d_constellation[48] = gr_complex(-9.0000 * level, 15.0000 * level);
  this->d_constellation[49] = gr_complex(-9.0000 * level, 13.0000 * level);
  this->d_constellation[50] = gr_complex(-9.0000 * level, 11.0000 * level);
  this->d_constellation[51] = gr_complex(-9.0000 * level, 9.0000 * level);
  this->d_constellation[52] = gr_complex(-9.0000 * level, 7.0000 * level);
  this->d_constellation[53] = gr_complex(-9.0000 * level, 5.0000 * level);
  this->d_constellation[54] = gr_complex(-9.0000 * level, 3.0000 * level);
  this->d_constellation[55] = gr_complex(-9.0000 * level, 1.0000 * level);
  this->d_constellation[56] = gr_complex(-9.0000 * level, -1.0000 * level);
  this->d_constellation[57] = gr_complex(-9.0000 * level, -3.0000 * level);
  this->d_constellation[58] = gr_complex(-9.0000 * level, -5.0000 * level);
  this->d_constellation[59] = gr_complex(-9.0000 * level, -7.0000 * level);
  this->d_constellation[60] = gr_complex(-9.0000 * level, -9.0000 * level);
  this->d_constellation[61] = gr_complex(-9.0000 * level, -11.0000 * level);
  this->d_constellation[62] = gr_complex(-9.0000 * level, -13.0000 * level);
  this->d_constellation[63] = gr_complex(-9.0000 * level, -15.0000 * level);
  this->d_constellation[64] = gr_complex(-7.0000 * level, 15.0000 * level);
  this->d_constellation[65] = gr_complex(-7.0000 * level, 13.0000 * level);
  this->d_constellation[66] = gr_complex(-7.0000 * level, 11.0000 * level);
  this->d_constellation[67] = gr_complex(-7.0000 * level, 9.0000 * level);
  this->d_constellation[68] = gr_complex(-7.0000 * level, 7.0000 * level);
  this->d_constellation[69] = gr_complex(-7.0000 * level, 5.0000 * level);
  this->d_constellation[70] = gr_complex(-7.0000 * level, 3.0000 * level);
  this->d_constellation[71] = gr_complex(-7.0000 * level, 1.0000 * level);
  this->d_constellation[72] = gr_complex(-7.0000 * level, -1.0000 * level);
  this->d_constellation[73] = gr_complex(-7.0000 * level, -3.0000 * level);
  this->d_constellation[74] = gr_complex(-7.0000 * level, -5.0000 * level);
  this->d_constellation[75] = gr_complex(-7.0000 * level, -7.0000 * level);
  this->d_constellation[76] = gr_complex(-7.0000 * level, -9.0000 * level);
  this->d_constellation[77] = gr_complex(-7.0000 * level, -11.0000 * level);
  this->d_constellation[78] = gr_complex(-7.0000 * level, -13.0000 * level);
  this->d_constellation[79] = gr_complex(-7.0000 * level, -15.0000 * level);
  this->d_constellation[80] = gr_complex(-5.0000 * level, 15.0000 * level);
  this->d_constellation[81] = gr_complex(-5.0000 * level, 13.0000 * level);
  this->d_constellation[82] = gr_complex(-5.0000 * level, 11.0000 * level);
  this->d_constellation[83] = gr_complex(-5.0000 * level, 9.0000 * level);
  this->d_constellation[84] = gr_complex(-5.0000 * level, 7.0000 * level);
  this->d_constellation[85] = gr_complex(-5.0000 * level, 5.0000 * level);
  this->d_constellation[86] = gr_complex(-5.0000 * level, 3.0000 * level);
  this->d_constellation[87] = gr_complex(-5.0000 * level, 1.0000 * level);
  this->d_constellation[88] = gr_complex(-5.0000 * level, -1.0000 * level);
  this->d_constellation[89] = gr_complex(-5.0000 * level, -3.0000 * level);
  this->d_constellation[90] = gr_complex(-5.0000 * level, -5.0000 * level);
  this->d_constellation[91] = gr_complex(-5.0000 * level, -7.0000 * level);
  this->d_constellation[92] = gr_complex(-5.0000 * level, -9.0000 * level);
  this->d_constellation[93] = gr_complex(-5.0000 * level, -11.0000 * level);
  this->d_constellation[94] = gr_complex(-5.0000 * level, -13.0000 * level);
  this->d_constellation[95] = gr_complex(-5.0000 * level, -15.0000 * level);
  this->d_constellation[96] = gr_complex(-3.0000 * level, 15.0000 * level);
  this->d_constellation[97] = gr_complex(-3.0000 * level, 13.0000 * level);
  this->d_constellation[98] = gr_complex(-3.0000 * level, 11.0000 * level);
  this->d_constellation[99] = gr_complex(-3.0000 * level, 9.0000 * level);
  this->d_constellation[100] = gr_complex(-3.0000 * level, 7.0000 * level);
  this->d_constellation[101] = gr_complex(-3.0000 * level, 5.0000 * level);
  this->d_constellation[102] = gr_complex(-3.0000 * level, 3.0000 * level);
  this->d_constellation[103] = gr_complex(-3.0000 * level, 1.0000 * level);
  this->d_constellation[104] = gr_complex(-3.0000 * level, -1.0000 * level);
  this->d_constellation[105] = gr_complex(-3.0000 * level, -3.0000 * level);
  this->d_constellation[106] = gr_complex(-3.0000 * level, -5.0000 * level);
  this->d_constellation[107] = gr_complex(-3.0000 * level, -7.0000 * level);
  this->d_constellation[108] = gr_complex(-3.0000 * level, -9.0000 * level);
  this->d_constellation[109] = gr_complex(-3.0000 * level, -11.0000 * level);
  this->d_constellation[110] = gr_complex(-3.0000 * level, -13.0000 * level);
  this->d_constellation[111] = gr_complex(-3.0000 * level, -15.0000 * level);
  this->d_constellation[112] = gr_complex(-1.0000 * level, 15.0000 * level);
  this->d_constellation[113] = gr_complex(-1.0000 * level, 13.0000 * level);
  this->d_constellation[114] = gr_complex(-1.0000 * level, 11.0000 * level);
  this->d_constellation[115] = gr_complex(-1.0000 * level, 9.0000 * level);
  this->d_constellation[116] = gr_complex(-1.0000 * level, 7.0000 * level);
  this->d_constellation[117] = gr_complex(-1.0000 * level, 5.0000 * level);
  this->d_constellation[118] = gr_complex(-1.0000 * level, 3.0000 * level);
  this->d_constellation[119] = gr_complex(-1.0000 * level, 1.0000 * level);
  this->d_constellation[120] = gr_complex(-1.0000 * level, -1.0000 * level);
  this->d_constellation[121] = gr_complex(-1.0000 * level, -3.0000 * level);
  this->d_constellation[122] = gr_complex(-1.0000 * level, -5.0000 * level);
  this->d_constellation[123] = gr_complex(-1.0000 * level, -7.0000 * level);
  this->d_constellation[124] = gr_complex(-1.0000 * level, -9.0000 * level);
  this->d_constellation[125] = gr_complex(-1.0000 * level, -11.0000 * level);
  this->d_constellation[126] = gr_complex(-1.0000 * level, -13.0000 * level);
  this->d_constellation[127] = gr_complex(-1.0000 * level, -15.0000 * level);
  this->d_constellation[128] = gr_complex(1.0000 * level, 15.0000 * level);
  this->d_constellation[129] = gr_complex(1.0000 * level, 13.0000 * level);
  this->d_constellation[130] = gr_complex(1.0000 * level, 11.0000 * level);
  this->d_constellation[131] = gr_complex(1.0000 * level, 9.0000 * level);
  this->d_constellation[132] = gr_complex(1.0000 * level, 7.0000 * level);
  this->d_constellation[133] = gr_complex(1.0000 * level, 5.0000 * level);
  this->d_constellation[134] = gr_complex(1.0000 * level, 3.0000 * level);
  this->d_constellation[135] = gr_complex(1.0000 * level, 1.0000 * level);
  this->d_constellation[136] = gr_complex(1.0000 * level, -1.0000 * level);
  this->d_constellation[137] = gr_complex(1.0000 * level, -3.0000 * level);
  this->d_constellation[138] = gr_complex(1.0000 * level, -5.0000 * level);
  this->d_constellation[139] = gr_complex(1.0000 * level, -7.0000 * level);
  this->d_constellation[140] = gr_complex(1.0000 * level, -9.0000 * level);
  this->d_constellation[141] = gr_complex(1.0000 * level, -11.0000 * level);
  this->d_constellation[142] = gr_complex(1.0000 * level, -13.0000 * level);
  this->d_constellation[143] = gr_complex(1.0000 * level, -15.0000 * level);
  this->d_constellation[144] = gr_complex(3.0000 * level, 15.0000 * level);
  this->d_constellation[145] = gr_complex(3.0000 * level, 13.0000 * level);
  this->d_constellation[146] = gr_complex(3.0000 * level, 11.0000 * level);
  this->d_constellation[147] = gr_complex(3.0000 * level, 9.0000 * level);
  this->d_constellation[148] = gr_complex(3.0000 * level, 7.0000 * level);
  this->d_constellation[149] = gr_complex(3.0000 * level, 5.0000 * level);
  this->d_constellation[150] = gr_complex(3.0000 * level, 3.0000 * level);
  this->d_constellation[151] = gr_complex(3.0000 * level, 1.0000 * level);
  this->d_constellation[152] = gr_complex(3.0000 * level, -1.0000 * level);
  this->d_constellation[153] = gr_complex(3.0000 * level, -3.0000 * level);
  this->d_constellation[154] = gr_complex(3.0000 * level, -5.0000 * level);
  this->d_constellation[155] = gr_complex(3.0000 * level, -7.0000 * level);
  this->d_constellation[156] = gr_complex(3.0000 * level, -9.0000 * level);
  this->d_constellation[157] = gr_complex(3.0000 * level, -11.0000 * level);
  this->d_constellation[158] = gr_complex(3.0000 * level, -13.0000 * level);
  this->d_constellation[159] = gr_complex(3.0000 * level, -15.0000 * level);
  this->d_constellation[160] = gr_complex(5.0000 * level, 15.0000 * level);
  this->d_constellation[161] = gr_complex(5.0000 * level, 13.0000 * level);
  this->d_constellation[162] = gr_complex(5.0000 * level, 11.0000 * level);
  this->d_constellation[163] = gr_complex(5.0000 * level, 9.0000 * level);
  this->d_constellation[164] = gr_complex(5.0000 * level, 7.0000 * level);
  this->d_constellation[165] = gr_complex(5.0000 * level, 5.0000 * level);
  this->d_constellation[166] = gr_complex(5.0000 * level, 3.0000 * level);
  this->d_constellation[167] = gr_complex(5.0000 * level, 1.0000 * level);
  this->d_constellation[168] = gr_complex(5.0000 * level, -1.0000 * level);
  this->d_constellation[169] = gr_complex(5.0000 * level, -3.0000 * level);
  this->d_constellation[170] = gr_complex(5.0000 * level, -5.0000 * level);
  this->d_constellation[171] = gr_complex(5.0000 * level, -7.0000 * level);
  this->d_constellation[172] = gr_complex(5.0000 * level, -9.0000 * level);
  this->d_constellation[173] = gr_complex(5.0000 * level, -11.0000 * level);
  this->d_constellation[174] = gr_complex(5.0000 * level, -13.0000 * level);
  this->d_constellation[175] = gr_complex(5.0000 * level, -15.0000 * level);
  this->d_constellation[176] = gr_complex(7.0000 * level, 15.0000 * level);
  this->d_constellation[177] = gr_complex(7.0000 * level, 13.0000 * level);
  this->d_constellation[178] = gr_complex(7.0000 * level, 11.0000 * level);
  this->d_constellation[179] = gr_complex(7.0000 * level, 9.0000 * level);
  this->d_constellation[180] = gr_complex(7.0000 * level, 7.0000 * level);
  this->d_constellation[181] = gr_complex(7.0000 * level, 5.0000 * level);
  this->d_constellation[182] = gr_complex(7.0000 * level, 3.0000 * level);
  this->d_constellation[183] = gr_complex(7.0000 * level, 1.0000 * level);
  this->d_constellation[184] = gr_complex(7.0000 * level, -1.0000 * level);
  this->d_constellation[185] = gr_complex(7.0000 * level, -3.0000 * level);
  this->d_constellation[186] = gr_complex(7.0000 * level, -5.0000 * level);
  this->d_constellation[187] = gr_complex(7.0000 * level, -7.0000 * level);
  this->d_constellation[188] = gr_complex(7.0000 * level, -9.0000 * level);
  this->d_constellation[189] = gr_complex(7.0000 * level, -11.0000 * level);
  this->d_constellation[190] = gr_complex(7.0000 * level, -13.0000 * level);
  this->d_constellation[191] = gr_complex(7.0000 * level, -15.0000 * level);
  this->d_constellation[192] = gr_complex(9.0000 * level, 15.0000 * level);
  this->d_constellation[193] = gr_complex(9.0000 * level, 13.0000 * level);
  this->d_constellation[194] = gr_complex(9.0000 * level, 11.0000 * level);
  this->d_constellation[195] = gr_complex(9.0000 * level, 9.0000 * level);
  this->d_constellation[196] = gr_complex(9.0000 * level, 7.0000 * level);
  this->d_constellation[197] = gr_complex(9.0000 * level, 5.0000 * level);
  this->d_constellation[198] = gr_complex(9.0000 * level, 3.0000 * level);
  this->d_constellation[199] = gr_complex(9.0000 * level, 1.0000 * level);
  this->d_constellation[200] = gr_complex(9.0000 * level, -1.0000 * level);
  this->d_constellation[201] = gr_complex(9.0000 * level, -3.0000 * level);
  this->d_constellation[202] = gr_complex(9.0000 * level, -5.0000 * level);
  this->d_constellation[203] = gr_complex(9.0000 * level, -7.0000 * level);
  this->d_constellation[204] = gr_complex(9.0000 * level, -9.0000 * level);
  this->d_constellation[205] = gr_complex(9.0000 * level, -11.0000 * level);
  this->d_constellation[206] = gr_complex(9.0000 * level, -13.0000 * level);
  this->d_constellation[207] = gr_complex(9.0000 * level, -15.0000 * level);
  this->d_constellation[208] = gr_complex(11.0000 * level, 15.0000 * level);
  this->d_constellation[209] = gr_complex(11.0000 * level, 13.0000 * level);
  this->d_constellation[210] = gr_complex(11.0000 * level, 11.0000 * level);
  this->d_constellation[211] = gr_complex(11.0000 * level, 9.0000 * level);
  this->d_constellation[212] = gr_complex(11.0000 * level, 7.0000 * level);
  this->d_constellation[213] = gr_complex(11.0000 * level, 5.0000 * level);
  this->d_constellation[214] = gr_complex(11.0000 * level, 3.0000 * level);
  this->d_constellation[215] = gr_complex(11.0000 * level, 1.0000 * level);
  this->d_constellation[216] = gr_complex(11.0000 * level, -1.0000 * level);
  this->d_constellation[217] = gr_complex(11.0000 * level, -3.0000 * level);
  this->d_constellation[218] = gr_complex(11.0000 * level, -5.0000 * level);
  this->d_constellation[219] = gr_complex(11.0000 * level, -7.0000 * level);
  this->d_constellation[220] = gr_complex(11.0000 * level, -9.0000 * level);
  this->d_constellation[221] = gr_complex(11.0000 * level, -11.0000 * level);
  this->d_constellation[222] = gr_complex(11.0000 * level, -13.0000 * level);
  this->d_constellation[223] = gr_complex(11.0000 * level, -15.0000 * level);
  this->d_constellation[224] = gr_complex(13.0000 * level, 15.0000 * level);
  this->d_constellation[225] = gr_complex(13.0000 * level, 13.0000 * level);
  this->d_constellation[226] = gr_complex(13.0000 * level, 11.0000 * level);
  this->d_constellation[227] = gr_complex(13.0000 * level, 9.0000 * level);
  this->d_constellation[228] = gr_complex(13.0000 * level, 7.0000 * level);
  this->d_constellation[229] = gr_complex(13.0000 * level, 5.0000 * level);
  this->d_constellation[230] = gr_complex(13.0000 * level, 3.0000 * level);
  this->d_constellation[231] = gr_complex(13.0000 * level, 1.0000 * level);
  this->d_constellation[232] = gr_complex(13.0000 * level, -1.0000 * level);
  this->d_constellation[233] = gr_complex(13.0000 * level, -3.0000 * level);
  this->d_constellation[234] = gr_complex(13.0000 * level, -5.0000 * level);
  this->d_constellation[235] = gr_complex(13.0000 * level, -7.0000 * level);
  this->d_constellation[236] = gr_complex(13.0000 * level, -9.0000 * level);
  this->d_constellation[237] = gr_complex(13.0000 * level, -11.0000 * level);
  this->d_constellation[238] = gr_complex(13.0000 * level, -13.0000 * level);
  this->d_constellation[239] = gr_complex(13.0000 * level, -15.0000 * level);
  this->d_constellation[240] = gr_complex(15.0000 * level, 15.0000 * level);
  this->d_constellation[241] = gr_complex(15.0000 * level, 13.0000 * level);
  this->d_constellation[242] = gr_complex(15.0000 * level, 11.0000 * level);
  this->d_constellation[243] = gr_complex(15.0000 * level, 9.0000 * level);
  this->d_constellation[244] = gr_complex(15.0000 * level, 7.0000 * level);
  this->d_constellation[245] = gr_complex(15.0000 * level, 5.0000 * level);
  this->d_constellation[246] = gr_complex(15.0000 * level, 3.0000 * level);
  this->d_constellation[247] = gr_complex(15.0000 * level, 1.0000 * level);
  this->d_constellation[248] = gr_complex(15.0000 * level, -1.0000 * level);
  this->d_constellation[249] = gr_complex(15.0000 * level, -3.0000 * level);
  this->d_constellation[250] = gr_complex(15.0000 * level, -5.0000 * level);
  this->d_constellation[251] = gr_complex(15.0000 * level, -7.0000 * level);
  this->d_constellation[252] = gr_complex(15.0000 * level, -9.0000 * level);
  this->d_constellation[253] = gr_complex(15.0000 * level, -11.0000 * level);
  this->d_constellation[254] = gr_complex(15.0000 * level, -13.0000 * level);
  this->d_constellation[255] = gr_complex(15.0000 * level, -15.0000 * level);
  this->_init_scaled_constellation();
  this->d_rotational_symmetry = 4;
  this->d_dimensionality = 1;
  this->calc_arity();
}

// hard decision maker
template <typename LLR>
unsigned int qam256<LLR>::decision_maker(const gr_complex *sample) {
  unsigned int ret = 0;
  const float level = sqrt(float(0.1));
  float re = sample->real();
  float im = sample->imag();

  /////////////////////////////////////////////////////////////////////////////
  // RED ALERT RED ALERT
  //
  //  DO NOT USE THE HARD SLICER, IT IS NOT IMPLEMENTED
  /////////////////////////////////////////////////////////////////////////////
  log::doomsday("Hard slicer is not implemented", __FILE__, __LINE__);

  return ret;
}

// explicit template instantiations. we support single-precision floating point,
// YALDPC's Q16.16 32-bit fixed-point, and YALDPC's Q48.16 64-bit floating point
// format.
//
// Edit Nov. 2017: We don't.
template class base<float>;
// template class base<yaldpc::fixed>;
// template class base<yaldpc::fixed64>;
template class bpsk<float>;
// template class bpsk<yaldpc::fixed>;
// template class bpsk<yaldpc::fixed64>;
template class qpsk<float>;
// template class qpsk<yaldpc::fixed>;
// template class qpsk<yaldpc::fixed64>;
template class qam16<float>;
// template class qam16<yaldpc::fixed>;
// template class qam16<yaldpc::fixed64>;
template class qam32<float>;
// template class qam32<yaldpc::fixed>;
// template class qam32<yaldpc::fixed64>;
template class qam64<float>;
// template class qam64<yaldpc::fixed>;
// template class qam64<yaldpc::fixed64>;
template class qam128<float>;
// template class qam128<yaldpc::fixed>;
// template class qam128<yaldpc::fixed64>;
template class qam256<float>;
// template class qam256<yaldpc::fixed>;
// template class qam256<yaldpc::fixed64>;
} // namespace constellation
} // namespace bamradio
