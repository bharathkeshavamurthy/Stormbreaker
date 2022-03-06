/*
 * systematic_encoder.cpp
 * Implementation of the systematic encoder class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "yaldpc.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>

namespace yaldpc {
using detail::GF2Matrix;
using detail::CSRGF2Matrix;

SystematicEncoder::sptr make(LDPCCode const &code) {
  return std::make_shared<SystematicEncoder>(code);
}

SystematicEncoder::SystematicEncoder(LDPCCode const &code)
    : Encoder(), H(code.H), col_perm(code.n, 0), _m(code.m), _n(code.n),
      _k(code.k) {
  // initialize the column permutations
  for (auto i = 0; i < col_perm.size(); ++i)
    col_perm[i] = i;

  // compute the generator matrix
  preprocess_H();
}

// attempt to bring the parity check matrix into systematic form
void SystematicEncoder::preprocess_H() {
  // get a copy of the PCM
  GF2Matrix H1(H);
  // (1) row-echelon form
  size_t j = _k;
  while (j < _n) {
    // get the slice into the current column
    std::valarray<uint8_t> col(H1.data[std::slice(j, _m, _n)]);
    // for column j, the pivot needs to go in row i2
    size_t i2 = j - _k;
    bool have_one = false;
    for (auto i = i2; i < _m; ++i) {
      have_one = col[i] == detail::__ONE__ ? true : false;
      if (have_one) {
        // found a pivot. if we found a pivot, we swap rows
        // ii and i2 and add row ii to all of the rows below
        // it
        H1.row_swap(i, i2);
        break;
      }
    }
    // if we found a pivot, everything is good
    if (have_one) {
      // add row i2 to all rows below it which have a '1' in column j
      // need a new slice into the column
      std::valarray<uint8_t> col(H1.data[std::slice(j, _m, _n)]);
      for (auto i = i2 + 1; i < _m; ++i) {
        have_one = col[i] == detail::__ONE__ ? true : false;
        if (have_one) {
          H1.row_add(i2, i);
        }
      }
      // go to the next column
      ++j;
    } else {
      // if there is no pivot, theres is a problem.
      // search for a column on the left half of the PCM with a
      // '1' on or below the main diagonal and swap with the
      // current one. Keep track of the column permutations.
      for (auto u = 0; u < _k; ++u) {
        std::valarray<uint8_t> col(H1.data[std::slice(u, _m, _n)]);
        for (auto v = i2; v < _m; ++v) {
          have_one = col[v] == detail::__ONE__ ? true : false;
          if (have_one) {
            H1.col_swap(j, u);
            // need to update the column permutations
            std::swap(col_perm[j], col_perm[u]);
            break;
          }
        }
        if (have_one)
          break;
      }
      if (!have_one) {
        // if we hit this point without a '1' in a suitable spot,
        // we cannot do much more...
        throw SystematicConversionException();
        return;
      }
    }
  }

  // (2) reduced row echelon form
  for (j = _k + 1; j < _n; ++j) {
    // look upwards and cancel out any '1's above the main
    // diagonal. Not needed for the first column, of course.
    std::valarray<uint8_t> col(H1.data[std::slice(j, _m, _n)]);
    for (int i = j - _k - 1; i >= 0; --i) {
      bool have_one = col[i] == detail::__ONE__ ? true : false;
      if (have_one) {
        // add the row with the '1' on the main diagonal to
        // the row in which we found a '1' to cancel it out.
        H1.row_add(j - _k, i);
      }
    }
  }

  // (3) extract the generator matrix
  std::valarray<uint8_t> Ad(_k * _m);
  for (j = 0; j < _k; ++j) {
    Ad[std::slice(j, _m, _k)] = H1.data[std::slice(j, _m, _n)];
  }
  this->A = CSRGF2Matrix(Ad, _m, _k);
}

// encode a block of data
void SystematicEncoder::encode(uint8_t const *in, uint8_t *out) {
  // pack the incoming data in a valarray
  std::valarray<uint8_t> in_data(in, _k);
  // get the parity bits
  auto out_parity = A.mult_vec(in_data);
  // apply reverse permutation as we fill the output buffer
  for (auto i = 0; i < _k; ++i) {
    auto dst_index = col_perm[i];
    out[dst_index] = in[i];
  }
  for (size_t u = 0, i = _k; i < _n; ++i) {
    auto dst_index = col_perm[i];
    out[dst_index] = out_parity[u++];
  }
}

std::vector<uint8_t> SystematicEncoder::encode(std::vector<uint8_t> const &in) {
  assert(in.size() == _k);
  std::vector<uint8_t> o(_n);
  encode(in.data(), o.data());
  return o;
}

// code rate
std::size_t SystematicEncoder::n() { return _n; }
std::size_t SystematicEncoder::m() { return _m; }
std::size_t SystematicEncoder::k() { return _k; }

} /* namespace yaldpc */
