/*
 * serial_c_decoder.cpp
 * Implementation of the serial-c decoder base class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "detail.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <utility>
#include <vector>

namespace yaldpc {
namespace detail {

template <typename LLR, typename Out>
SerialCDecoder<LLR, Out>::SerialCDecoder(GF2Matrix H, unsigned int m,
                                         unsigned int n, uint64_t max_its,
                                         bool output_value,
                                         bool do_parity_check)
    : _max_its(max_its), _output_value(output_value),
      _do_parity_check(do_parity_check), H(H), _m(m), _n(n), _k(n - m),
      _B(m, std::vector<size_t>()), _R(m, std::vector<LLR>()), _P(n, 0.0),
      _Q(n, 0.0), _z(n) {
  // search the PCM to get the connections between bit and check nodes
  for (size_t j = 0; j < m; ++j) {
    auto row = H.slice(j, j + 1, 0, n);
    for (size_t i = 0; i < n; ++i) {
      if (__ONE__ == row.data[i]) {
        _B[j].push_back(i);
        _R[j].push_back(detail::Zero<LLR>());
      }
    }
  }
  // find the highest degree of a check node to resize the signbit storage
  int sz = 0;
  for (auto const &b : _B)
    if (b.size() >= sz)
      sz = b.size();
  _si2.resize(sz + 1);
}

template <typename LLR, typename Out>
void SerialCDecoder<LLR, Out>::_serial_c_decode(LLR const *in_llr, Out *out) {
  // initialization
  _P.assign(in_llr, in_llr + _n);
  _Q.assign(in_llr, in_llr + _n);
  for (size_t j = 0; j < _m; ++j)
    std::fill(begin(_R[j]), end(_R[j]), detail::Zero<LLR>());
  // iteration
  size_t II = 0;
  while (true) {
    // messages
    _message_passing();
    // test
    if (II == _max_its) {
      // zero copy write to output
      _write_results_zc(out);
      break;
    } else if (_do_parity_check) {
      // compute the hard decisions and do the syndrome check
      assert(_z.size() == _n);
      size_t i = 0;
      std::generate(begin(_z), end(_z),
                    [&i, this]() { return detail::HardDecision(_P[i++]); });
      auto syndrome = H.mult_vec(_z);
      // declare valid codeword if syndrome is all zeros
      if (end(syndrome) ==
          std::find_if(begin(syndrome), end(syndrome),
                       [](auto s) { return s != detail::__ZERO__; })) {
        _write_results(out);
        break;
      }
    } // else swap & continue
    std::swap(_P, _Q);
    ++II;
  }
}

// write results to the output buffers
template <> void SerialCDecoder<double, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <> void SerialCDecoder<double, double>::_write_results(double *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}
template <> void SerialCDecoder<float, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <> void SerialCDecoder<float, float>::_write_results(float *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}
template <> void SerialCDecoder<fixed, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <> void SerialCDecoder<fixed, fixed>::_write_results(fixed *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}
template <>
void SerialCDecoder<fixed64, uint8_t>::_write_results(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _z[i];
}
template <>
void SerialCDecoder<fixed64, fixed64>::_write_results(fixed64 *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}

// write results to output buffers without copying
template <>
void SerialCDecoder<double, uint8_t>::_write_results_zc(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = detail::HardDecision(_P[i]);
}
template <>
void SerialCDecoder<double, double>::_write_results_zc(double *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}
template <>
void SerialCDecoder<float, uint8_t>::_write_results_zc(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = detail::HardDecision(_P[i]);
}
template <> void SerialCDecoder<float, float>::_write_results_zc(float *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}
template <>
void SerialCDecoder<fixed, uint8_t>::_write_results_zc(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = detail::HardDecision(_P[i]);
}
template <> void SerialCDecoder<fixed, fixed>::_write_results_zc(fixed *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}
template <>
void SerialCDecoder<fixed64, uint8_t>::_write_results_zc(uint8_t *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = detail::HardDecision(_P[i]);
}
template <>
void SerialCDecoder<fixed64, fixed64>::_write_results_zc(fixed64 *out) {
  size_t end = _output_value ? _k : _n;
  for (size_t i = 0; i < end; ++i)
    out[i] = _P[i];
}

// instantiate templates
template class SerialCDecoder<double, uint8_t>;
template class SerialCDecoder<double, double>;
template class SerialCDecoder<float, uint8_t>;
template class SerialCDecoder<float, float>;
template class SerialCDecoder<fixed, uint8_t>;
template class SerialCDecoder<fixed, fixed>;
template class SerialCDecoder<fixed64, uint8_t>;
template class SerialCDecoder<fixed64, fixed64>;
} /* namespace detail */
} /* namespace yaldpc */
