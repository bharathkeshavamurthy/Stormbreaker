/*
 * dd_encoder.cpp
 * Implementation of the dual-diagonal encoder class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "yaldpc.hpp"

#include <cstdio>

namespace yaldpc {

DDEncoder::DDEncoder(IEEE802Code const &code)
    : Encoder(), Hb(&code.Hb[0], code.mb * code.nb), Z(code.Z), mb(code.mb),
      nb(code.nb), kb(code.nb - code.mb), _m(code.Z * code.mb),
      _n(code.Z * code.nb), _k(code.Z * kb) {
  // find 'x'
  x = 0;
  std::valarray<int64_t> x_col(Hb[std::slice(kb, mb, nb)]);
  for (size_t i = 0; i < mb; ++i) {
    if (0 == x_col[i]) {
      x = i;
      break;
    }
  }

  // I should probably also verify the dual diagonal structure but oh well...
  if (x == 0)
    throw NoDDStructureException();
}

DDEncoder::sptr DDEncoder::make(IEEE802Code const &code) {
  return std::make_shared<DDEncoder>(code);
}

// encode a block of data
void DDEncoder::encode(uint8_t const *in, uint8_t *out) {
  // (1) compute the systematic components using circular shifts and
  // accumulators
  // save the lambda components in a vector of valarrays
  std::vector<std::valarray<uint8_t>> lambda(
      mb, std::valarray<uint8_t>(detail::__ZERO__, Z));
  // iterate through rows and cols of the systematic part of Hb
  for (size_t j = 0; j < kb; ++j) { // columns
    // grab a segment of length Z from the input
    size_t segment_start = j * Z;
    std::valarray<uint8_t> s(in + segment_start, Z);
    for (size_t i = 0; i < mb; ++i) { // rows
      // grab the value of the current shift.
      auto p = Hb[i * nb + j];
      switch (p) {
      case -1:
        // p == -1 means that we are multiplying by a zero
        // matrix. nothing left to do.
        continue;
      case 0:
        // zero shift, no function call
        lambda[i] ^= s;
        break;
      default:
        // cyclically shift the current segment p positions
        // to the left and add to the current column of lambda
        lambda[i] ^= s.cshift(p);
      }
    }
  }

  // (2) compute the arbitrary parity bits using forward and backward
  // substitution
  // assume that the first parity block is all zeros
  std::vector<std::valarray<uint8_t>> p(
      mb, std::valarray<uint8_t>(detail::__ZERO__, Z));
  // forward substitution
  p[1] = lambda[0];
  for (size_t i = 2; i <= x; ++i)
    p[i] = lambda[i - 1] ^ p[i - 1];
  // backward substitution
  p[mb - 1] = lambda[mb - 1];
  for (size_t i = mb - 2; i > x; --i)
    p[i] = lambda[i] ^ p[i + 1];

  // (3) compute the correction factor
  p[0] = (p[x] ^ p[x + 1]) ^ lambda[x];
  std::valarray<uint8_t> f(p[0].cshift(Hb[kb]));

  // (4) correct by the correction factor and output codeword
  for (size_t i = 0; i < _k; ++i)
    out[i] = in[i];
  // the first segment is already corrected
  for (size_t u = _k, i = 0; i < Z; ++i)
    out[u++] = p[0][i];
  // correct the other segments
  for (size_t i = 1; i < mb; ++i) {
    size_t segment_start = _k + i * Z;
    for (size_t j = segment_start, l = 0; l < Z; ++l) {
      out[j++] = p[i][l] ^ f[l];
    }
  }
}

std::vector<uint8_t> DDEncoder::encode(std::vector<uint8_t> const &in) {
  assert(in.size() == _k);
  std::vector<uint8_t> o(_n);
  encode(in.data(), o.data());
  return o;
}

// print the base matrix for debugging...
void DDEncoder::print_Hb() {
  std::printf("yaldpc::DDEncoder with %zu x %zu base matrix:\n\n", mb, nb);
  for (size_t i = 0; i < mb; ++i) {
    for (auto j = 0; j < nb; ++j) {
      std::printf("%ld ", Hb[i * nb + j]);
    }
    std::printf("\n");
  }
  std::printf("\n");
}

// code rate
std::size_t DDEncoder::n() { return _n; }
std::size_t DDEncoder::m() { return _m; }
std::size_t DDEncoder::k() { return _k; }
} /* namespace yaldpc */
