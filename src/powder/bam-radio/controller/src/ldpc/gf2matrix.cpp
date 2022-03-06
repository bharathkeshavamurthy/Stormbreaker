/*
 * gf2matrix.cpp
 * Implementation of the GF2Matrix class
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 *
 * See the corresponding header file for documentation
 */

#include "detail.hpp"

#include <cstdio>

namespace yaldpc {
namespace detail {
// swap col1 with col2
void GF2Matrix::col_swap(size_t col1, size_t col2) {
  // early exit
  if (col1 == col2)
    return;
  if (col1 > n - 1 || col2 > n - 1)
    throw IndexOutOfBoundsException();
  // define the slices into the data
  auto s1 = std::slice(col1, m, n);
  auto s2 = std::slice(col2, m, n);
  // copy the second column
  std::valarray<uint8_t> tmp(data[s2]);
  // set the second column
  data[s2] = data[s1];
  // set the first column
  data[s1] = tmp;
}

// swap row1 with row2
void GF2Matrix::row_swap(size_t row1, size_t row2) {
  // early exit
  if (row1 == row2)
    return;
  if (row1 > m - 1 || row2 > m - 1)
    throw IndexOutOfBoundsException();
  // get the slices
  auto s1 = std::slice(row1 * n, n, 1);
  auto s2 = std::slice(row2 * n, n, 1);
  // copy the second row
  std::valarray<uint8_t> tmp(data[s2]);
  // set the second row
  data[s2] = data[s1];
  // set the first row
  data[s1] = tmp;
}

// add row1 to row2 over GF(2)
void GF2Matrix::row_add(size_t row1, size_t row2) {
  if (row1 > m - 1 || row2 > m - 1)
    throw IndexOutOfBoundsException();
  auto s1 = std::slice(row1 * n, n, 1);
  auto s2 = std::slice(row2 * n, n, 1);
  data[s2] ^= data[s1];
}

// concatenate horizontally with another GF2 Matrix
GF2Matrix GF2Matrix::hcat(GF2Matrix A) {
  // not sure how to handle this error. for now, just return a copy of
  // itself
  if (m != A.m)
    throw HCatDimensionMismatchException();
  // the resulting matrix will have new dimensionts
  size_t m_new = m;
  size_t n_new = n + A.n;
  // we make a new data array
  std::valarray<uint8_t> newdata(m_new * n_new);
  // copy the data over. in hindsight, I should have just done this with
  // slices
  for (auto i = 0; i < m_new; ++i) {
    for (auto j = 0; j < n; ++j) {
      newdata[i * n_new + j] = data[i * n + j];
    }
    for (auto j = 0; j < A.n; ++j) {
      newdata[i * n_new + n + j] = A.data[i * A.n + j];
    }
  }
  // return a new matrix
  return GF2Matrix(newdata, m_new, n_new);
}

// compute the inverse using gaussian elimination
GF2Matrix GF2Matrix::inv() {
  // (1) form augmented matrix [A | I]
  auto AI = this->hcat(GF2Matrix(m));

  // (2) row echelon form
  for (auto j = 0; j < n; ++j) { // columns
    // get the slice into AI for this column
    std::valarray<uint8_t> col(AI.data[std::slice(j, m, 2 * m)]);
    // for column j, we need to put the pivot in row j
    bool have_one = false;
    for (auto i = j; i < m; ++i) { // rows
      // look down the rows to find a pivot. If you find one, swap rows i
      // and j and add row i (now at row j really) to all of the rows
      // below.
      have_one = col[i] == __ONE__ ? true : false;
      if (have_one) {
        AI.row_swap(i, j);
        break;
      }
    }
    if (have_one) {
      // add row j to all of the rows below it which have a '1' in column j
      // need a new copy of the column
      std::valarray<uint8_t> col(AI.data[std::slice(j, m, 2 * m)]);
      for (auto i = j + 1; i < m; ++i) {
        have_one = col[i] == __ONE__ ? true : false;
        if (have_one) {
          AI.row_add(j, i);
        }
      }
    } else {
      // the matrix is not invertible
      throw NotInvertibleException();
    }
  }

  // (3) reduced row echelon form
  // for each column, look upwards and cancel out any '1's above the main
  // diagonal. For obvious reasons, we can skip the first column
  for (auto j = 1; j < n; ++j) { // columns
    // get the slice for column j
    std::valarray<uint8_t> col(AI.data[std::slice(j, m, 2 * m)]);
    for (int i = j - 1; i >= 0; --i) { // rows from the main diagonal up
      bool have_one = col[i] == __ONE__ ? true : false;
      if (have_one) {
        AI.row_add(j, i);
      }
    }
  }

  // (4) extract the inverse
  std::valarray<uint8_t> Ainvd(m * m);
  size_t k = 0;
  for (auto j = m; j < 2 * m; ++j) {
    Ainvd[std::slice(k++, m, m)] = AI.data[std::slice(j, m, 2 * m)];
  }

  return GF2Matrix(Ainvd, m, m);
}

// print the matrix. print a '.' instead of a zero
void GF2Matrix::print() {
  std::printf("yaldpc::detail::GF2Matrix of size %zu x %zu\n\n", m, n);
  for (auto i = 0; i < m; ++i) {
    for (auto j = 0; j < n; ++j) {
      auto val = data[i * n + j];
      if (0 == val) {
        std::printf(". ");
      } else {
        std::printf("%d ", val);
      }
    }
    std::printf("\n");
  }
  std::printf("\n");
}

// slice into a GF2 matrix, similar to MATLAB's syntax. in MATLAB, we'd do:
// Hsub = H[row_start:row_end, col_start:col_end]
// n.b. row_end and col_end are non-inclusive here
GF2Matrix GF2Matrix::slice(size_t row_start, size_t row_end, size_t col_start,
                           size_t col_end) const {
  // construct the data array
  size_t new_m = row_end - row_start;
  size_t new_n = col_end - col_start;

  std::valarray<uint8_t> out_data(new_m * new_n);

  // copy the data over. works only for row-major matrices.
  for (size_t v = 0, i = row_start; i < row_end; ++i) {
    for (size_t u = 0, j = col_start; j < col_end; ++j) {
      out_data[v * new_n + u++] = data[i * n + j];
    }
    v++;
  }

  // return a new GF2Matrix
  return GF2Matrix(out_data, new_m, new_n);
}
} /* namespace detail */
} /* namespace yaldpc */
