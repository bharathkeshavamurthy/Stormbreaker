/*
 * csr_gf2matrix.cpp
 * Implementation of the CSRGF2Matrix class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "detail.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <vector>

namespace yaldpc {
namespace detail {
// construct from a dense GF2 Matrix
CSRGF2Matrix::CSRGF2Matrix(GF2Matrix M) : IA(), JA(), m(M.m), n(M.n) {
  // first element of IA is always zero
  IA.push_back(__ZERO__);
  // iterate through the GF2matrix and find the ones
  for (size_t i = 0; i < M.m; ++i) {
    size_t nnz_row = 0;
    // get the slice of this row
    std::valarray<uint8_t> row(M.data[std::slice(i * M.n, M.n, 1)]);
    for (size_t j = 0; j < M.n; ++j) {
      if (row[j] != __ZERO__) {
        JA.push_back(j);
        ++nnz_row;
      }
    }
    // update the IA array
    IA.push_back(nnz_row + IA.back());
  }
}

// construct from a valarray holding a row-major matrix (TODO this is
// dirty, find a way to fix and consolidate with the algo above...)
CSRGF2Matrix::CSRGF2Matrix(std::valarray<uint8_t> va, size_t m, size_t n)
    : IA(), JA(), m(m), n(n) {
  // first element of IA is always zero
  IA.push_back(__ZERO__);
  // iterate through the GF2matrix and find the ones
  for (size_t i = 0; i < m; ++i) {
    size_t nnz_row = 0;
    // get the slice of this row
    std::valarray<uint8_t> row(va[std::slice(i * n, n, 1)]);
    for (size_t j = 0; j < n; ++j) {
      if (row[j] != __ZERO__) {
        JA.push_back(j);
        ++nnz_row;
      }
    }
    // update the IA array
    IA.push_back(nnz_row + IA.back());
  }
}

// construct from a flat row-major array
CSRGF2Matrix::CSRGF2Matrix(uint8_t *flat, size_t m, size_t n)
    : IA(), JA(), m(m), n(n) {
  // first element of IA is always zero
  IA.push_back(__ZERO__);
  // iterate through the data and find the ones
  for (size_t i = 0; i < m; ++i) {
    size_t nnz_row = 0;
    for (size_t j = 0; j < n; ++j) {
      if (flat[i * n + j] != __ZERO__) {
        JA.push_back(j);
        ++nnz_row;
      }
    }
    // update the IA array
    IA.push_back(nnz_row + IA.back());
  }
}

// construct from a flat row-major array
CSRGF2Matrix::CSRGF2Matrix(std::vector<uint8_t> v, size_t m, size_t n)
    : IA(), JA(), m(m), n(n) {
  // first element of IA is always zero
  IA.push_back(__ZERO__);
  // iterate through the data and find the ones
  for (size_t i = 0; i < m; ++i) {
    size_t nnz_row = 0;
    for (size_t j = 0; j < n; ++j) {
      if (v[i * n + j] != __ZERO__) {
        JA.push_back(j);
        ++nnz_row;
      }
    }
    // update the IA array
    IA.push_back(nnz_row + IA.back());
  }
}

GF2Matrix CSRGF2Matrix::to_gf2matrix() {
  std::valarray<uint8_t> data((uint8_t)0, m * n);
  for (auto i = 0; i < m; ++i) {
    // the column indices of the '1's in each row can be read using the
    // array IA. 'c' below will hold the column indices.
    std::set<uint8_t> c;
    for (auto l = IA[i]; l < IA[i + 1]; ++l) {
      c.insert(JA[l]);
    }
    // now iterate through the row. print a '1' if we are at a column
    // that contains one. (i.e. a column index that is in 'cols')
    for (size_t j = 0; j < n; ++j) {
      if (c.find(j) != c.end()) {
        data[i * n + j] = 1;
      }
    }
  }
  return GF2Matrix(data, m, n);
}

// solve the lower triangular system u = T * v in GF(2). (compute v =
// T^(-1) * u)
std::valarray<uint8_t> CSRGF2Matrix::lt_solve(std::valarray<uint8_t> u) {
  if (u.size() != n)
    throw DimensionMismatchException();

  std::valarray<uint8_t> v(__ZERO__, m);
  v[0] = u[0];
  for (auto i = 1; i < m; ++i) {
    v[i] = u[i];
    for (auto j = IA[i]; j < IA[i + 1] - 1; ++j) {
      v[i] ^= v[JA[j]];
    }
  }
  return v;
}

// multiply with a vector in GF(2)
std::valarray<uint8_t> CSRGF2Matrix::mult_vec(std::valarray<uint8_t> v) {
  if (v.size() != n)
    throw DimensionMismatchException();

  std::valarray<uint8_t> u(__ZERO__, m);
  for (auto i = 0; i < m; ++i) {
    for (auto j = IA[i]; j < IA[i + 1]; ++j) {
      u[i] ^= v[JA[j]];
    }
  }
  return u;
}

// derrty, but oh well
std::vector<uint8_t> CSRGF2Matrix::mult_vec(std::vector<uint8_t> const &v) {
  if (v.size() != n)
    throw DimensionMismatchException();

  std::vector<uint8_t> u(m, __ZERO__);
  for (auto i = 0; i < m; ++i) {
    for (auto j = IA[i]; j < IA[i + 1]; ++j) {
      u[i] ^= v[JA[j]];
    }
  }
  return u;
}

// print the matrix to the screen. print a '.' instead of a '0'
void CSRGF2Matrix::print() {
  std::printf("yaldpc::detail::CSRGF2Matrix of size %zu by %zu\n\n", m, n);
  for (auto i = 0; i < m; ++i) {
    // the column indices of the '1's in each row can be read using the
    // array IA. 'c' below will hold the column indices.
    std::set<uint8_t> c;
    for (auto l = IA[i]; l < IA[i + 1]; ++l) {
      c.insert(JA[l]);
    }
    // now iterate through the row. print a '1' if we are at a column
    // that contains one. (i.e. a column index that is in 'cols')
    for (size_t j = 0; j < n; ++j) {
      if (c.find(j) != c.end()) {
        std::printf("1 ");
      } else {
        std::printf(". ");
      }
    }
    // newline at the end of each row
    std::printf("\n");
  }
}
} /* namespace detail */
} /* namespace yaldpc */
