// -*- c++ -*-
//
// Copyright (c) 2017 Dennis Ogbe
//
// BAM-Wireless constellations. Currently supports soft decoding for BPSK, QPSK,
// {16,32,64,128,256}-QAM

#ifndef CONTROLLER_BAM_CONSTELLATION_H
#define CONTROLLER_BAM_CONSTELLATION_H

#include <boost/any.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <gnuradio/gr_complex.h>
#include <pmt/pmt.h>
#include <string>

// see the yaldpc/ext/FpXX headers for a reason why we are doing this
#define __4140ec62e7fc__YALDPC_FIXED_NEED_BOOST_SERIALZE__
#include "ldpc/yaldpc.hpp" // for fixed number type

namespace bamradio {
namespace constellation {

template <typename LLR> class LLRLut {
public:
  std::vector<float> snr_lut;
  std::vector<std::vector<std::vector<LLR>>> llr_lut;

  int precision;
  int snr_npoints;
  float snr_min;
  float snr_max;

  LLRLut(int precision, int npoints, float min, float max)
      : precision(precision), snr_npoints(npoints), snr_min(min), snr_max(max) {
  }
  LLRLut() {}

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &precision;
    ar &snr_npoints;
    ar &snr_min;
    ar &snr_max;
    ar &snr_lut;
    ar &llr_lut;
  }
};

// base:
// the functionality that all of the constellation objects share
template <typename LLR> class base {
public:
  typedef boost::shared_ptr<base> sptr;

  // return the constellation points for a given value
  void map_to_points_and_scale(unsigned int value, gr_complex *points) const;
  std::vector<gr_complex> map_to_points_and_scale_v(unsigned int value) const;

  // lookup the soft decision value
  void make_soft_decision(gr_complex sample, LLR *out, size_t snr_idx) const;
  void calculate_soft_decision(gr_complex sample, LLR *out, float snr) const;

  // get index for lookup table
  size_t get_snr_idx(float snr) const;

  // does this object have the pre-computed soft decision LUT values?
  bool has_luts() const { return _has_luts; }

  virtual unsigned int bits_per_symbol() const = 0;

  // a0 -- the symmetric component. Multiply with it at the transmitter as well
  // as the receiver. (Use to compensate for IFFT and FFT scaling)
  //
  // a1 -- the reciprocal component. Multiply with it at the transmitter, divide
  // by it at the receiver (Use for PAPR backoff)
  //
  // a2 -- extra scale factor for scaled constellation. Useful for oversampled
  // OFDM symbols at the transmitter.
  base(float a0, float a1, float a2, int lut_precision, int snr_npoints,
       float snr_min, float snr_max, std::string const &lut_wisdom_file_name)
      : _lut(lut_precision, snr_npoints, snr_min, snr_max),
        _lut_wisdom_file_name(lut_wisdom_file_name), _scal(a0 * a1),
        _inv_scal(a0 * a2 / a1), _has_luts(false) {}
  ~base() {}

  LLRLut<LLR> get_lut() const { return _lut; }
  float get_scale() const { return _scal; }
  float get_inv_scale() const { return _inv_scal; }

  // gnu radio const object legacy
  std::vector<gr_complex> points() const { return d_constellation; }
  unsigned int rotational_symmetry() const { return d_rotational_symmetry; }
  unsigned int dimensionality() const { return d_dimensionality; }
  unsigned int arity() const { return d_arity; }

protected:
  // we save the hardcoded LLR clipping factor as part of the base class.
  float _llr_clip() const { return 4000.0; }
  // the range for the LLR table lookup function is hardcoded as well.
  float _llr_range() const { return 1.8; }

  // initialize the lookup tables
  void _init_lut();

  // initialize the constellation
  virtual void _init_const() = 0;

  LLRLut<LLR> _lut;
  std::string _lut_wisdom_file_name;

  float const _scal;
  float const _inv_scal;
  // TODO: consolidate naming convention (remove d_*)
  std::vector<gr_complex> d_constellation;
  unsigned int d_rotational_symmetry;
  unsigned int d_dimensionality;
  unsigned int d_arity;
  // constellations must initialize this in _init_const()
  std::vector<gr_complex> _scaled_constellation;
  void _init_scaled_constellation();
  std::vector<LLR> _calc_soft_dec(gr_complex sample, float noise_var) const;
  void calc_arity();

  bool _has_luts;
};
// some typedefs for less typing
typedef base<float>::sptr float_sptr;
typedef base<yaldpc::fixed>::sptr fixed_sptr;
typedef base<yaldpc::fixed64>::sptr fixed64_sptr;

// return a pmt object containing a sptr to a constellation object.
// XXX for some reason, this does not work as a member function of the base
// class.
template <typename LLR> pmt::pmt_t to_pmt(typename base<LLR>::sptr c) {
  return pmt::make_any(boost::any(c));
}
// convert a pmt holding a constellation back to a constellation
template <typename LLR> typename base<LLR>::sptr from_pmt(pmt::pmt_t p) {
  return boost::any_cast<typename base<LLR>::sptr>(pmt::any_ref(p));
}

// BPSK
template <typename LLR> class bpsk : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<bpsk> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~bpsk() {}

  unsigned int bits_per_symbol() const { return 1; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  bpsk(float a0, float a1, float a2, int lut_precision, int snr_npoints,
       float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  bpsk(float a0, float a1, float a2);

  void _init_const();
};

// QPSK
template <typename LLR> class qpsk : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<qpsk> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~qpsk() {}

  unsigned int bits_per_symbol() const { return 2; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  qpsk(float a0, float a1, float a2, int lut_precision, int snr_npoints,
       float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  qpsk(float a0, float a1, float a2);

  void _init_const();
};

// QAM16
template <typename LLR>
class qam16 : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<qam16> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~qam16() {}

  unsigned int bits_per_symbol() const { return 4; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  qam16(float a0, float a1, float a2, int lut_precision, int snr_npoints,
        float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  qam16(float a0, float a1, float a2);

  void _init_const();
};

// QAM32
template <typename LLR>
class qam32 : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<qam32> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~qam32() {}

  unsigned int bits_per_symbol() const { return 5; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  qam32(float a0, float a1, float a2, int lut_precision, int snr_npoints,
        float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  qam32(float a0, float a1, float a2);

  void _init_const();
};

// QAM64
template <typename LLR>
class qam64 : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<qam64> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~qam64() {}

  unsigned int bits_per_symbol() const { return 6; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  qam64(float a0, float a1, float a2, int lut_precision, int snr_npoints,
        float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  qam64(float a0, float a1, float a2);

  void _init_const();
};

// QAM128
template <typename LLR>
class qam128 : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<qam128> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~qam128() {}

  unsigned int bits_per_symbol() const { return 7; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  qam128(float a0, float a1, float a2, int lut_precision, int snr_npoints,
         float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  qam128(float a0, float a1, float a2);

  void _init_const();
};

// QAM256
template <typename LLR>
class qam256 : public bamradio::constellation::base<LLR> {
public:
  typedef boost::shared_ptr<qam256> sptr;

  static sptr make(float a0, float a1, float a2, int lut_precision,
                   int snr_npoints, float snr_min, float snr_max,
                   std::string const &lut_wisdom_file_name);
  static sptr make(float a0, float a1, float a2);

  ~qam256() {}

  unsigned int bits_per_symbol() const { return 8; }

  // have a hard decision
  unsigned int decision_maker(const gr_complex *sample);

protected:
  qam256(float a0, float a1, float a2, int lut_precision, int snr_npoints,
         float snr_min, float snr_max, std::string const &lut_wisdom_file_name);
  qam256(float a0, float a1, float a2);

  void _init_const();
};

} // namespace constellation
} // namespace bamradio

#endif // CONTROLLER_BAM_CONSTELLATION_H
