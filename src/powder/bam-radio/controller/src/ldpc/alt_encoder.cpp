/*
 * alt_encoder.cpp
 * Implementation of the ALT Encoder class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "yaldpc.hpp"

namespace yaldpc {
using detail::CSRGF2Matrix;
using detail::GF2Matrix;

// constructor
ALTEncoder::ALTEncoder(LDPCCode const &code, std::size_t g)
    : Encoder(), H(code.H), g(g), _m(code.m), _n(code.n), _k(code.k) {
  // compute phi_inv if it exists
  compute_phi_inv();

  // initialize encoding matrices
  this->A = CSRGF2Matrix(H.slice(0, _m - g, 0, _n - _m));
  this->B = CSRGF2Matrix(H.slice(0, _m - g, _n - _m, _n - _m + g));
  this->T = CSRGF2Matrix(H.slice(0, _m - g, _n - _m + g, _n));
  this->C = CSRGF2Matrix(H.slice(_m - g, _m, 0, _n - _m));
  this->E = CSRGF2Matrix(H.slice(_m - g, _m, _n - _m + g, _n));
}

ALTEncoder::sptr ALTEncoder::make(LDPCCode const &code, std::size_t g) {
  return std::make_shared<ALTEncoder>(code, g);
}

void ALTEncoder::compute_phi_inv() {
  // copy the PCM for this endeavour
  GF2Matrix H1(H);

  // clear the E submatrix by adding rows from T
  // for each row in E
  for (std::size_t i = _m - g; i < _m; ++i) {
    // start from the right. if there is a '1' in the current column,
    // add the corresponding row from E
    for (std::size_t j = _n - 1; j >= _n - _m + g; --j) {
      // first, get the slice for the current row
      std::valarray<uint8_t> row = H1.data[std::slice(i * _n, _n, 1)];
      if (detail::__ONE__ == row[j]) {
        auto row_to_add = j - _n + _m - g;
        H1.row_add(row_to_add, i);
      }
    }
  }

  // extract the resulting g x g submatrix phi and save its inverse
  auto phi = H1.slice(_m - g, _m, _n - _m, _n - _m + g);
  // note. if the inverse does not exist (TODO raise exception in the
  // inversion code), we could try to permute the columns of H1 until phi
  // becomes invertible. This might be worth looking into for some LDPC
  // codes, I will not worry about it for now as I think that the 802.11
  // LDPC codes are specifically designed for this not to happen.
  this->phi_inv = CSRGF2Matrix(phi.inv());
}

// encode information bits in linear time. the codeword has the
// following structure: c = (s, p1, p2), where s is the systematic
// part (the information bits)
void ALTEncoder::encode(uint8_t const *in, uint8_t *out) {
  // (0) copy the input data
  std::valarray<uint8_t> in_data(in, _k);
  // (1) obtain p1
  auto t1 = A.mult_vec(in_data);
  auto t2 = T.lt_solve(t1);
  auto t3 = E.mult_vec(t2);
  auto t4 = C.mult_vec(in_data);
  t3 ^= t4;
  auto p1 = phi_inv.mult_vec(t3);
  // (2) obtain p2
  auto s1 = B.mult_vec(p1);
  t1 ^= s1;
  auto p2 = T.lt_solve(t1);
  // (3) construct the codeword
  for (std::size_t i = 0; i < _k; ++i)
    out[i] = in[i];
  size_t i = _k;
  for (auto p : p1)
    out[i++] = p;
  for (auto p : p2)
    out[i++] = p;
}

std::vector<uint8_t> ALTEncoder::encode(std::vector<uint8_t> const &in) {
  assert(in.size() == _k);
  std::vector<uint8_t> o(_n);
  encode(in.data(), o.data());
  return o;
}

// code rate
std::size_t ALTEncoder::n() { return _n; }
std::size_t ALTEncoder::m() { return _m; }
std::size_t ALTEncoder::k() { return _k; }

} /*namespace yaldpc */
