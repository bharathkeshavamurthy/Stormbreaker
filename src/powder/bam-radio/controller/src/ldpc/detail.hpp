/*
 * detail.hpp
 * YALDPC implementation details
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 */

#ifndef r1383db6433d17876c9eeeb6a81be346a52dcf3dffa837573f
#define r1383db6433d17876c9eeeb6a81be346a52dcf3dffa837573f

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <valarray>
#include <vector>

/*
 * https://github.com/mbedded-ninja/MFixedPoint
 */
#include "Fp32f.hpp"
#include "Fp64f.hpp"

namespace yaldpc {
// 'fixed' -> Q16.16
// 'fixed64' -> Q48.16
typedef Fp::Fp32f<16> fixed;
typedef Fp::Fp64f<16> fixed64;

namespace detail {

/*
 * Constants & helper templates
 */

// infinity
template <typename LLR> inline LLR Inf() {
  return std::numeric_limits<LLR>::infinity();
}
constexpr float __FIXED_INF__ = 32000.0;
template <> inline fixed Inf() {
  fixed f(__FIXED_INF__);
  return f;
}
template <> inline fixed64 Inf() {
  fixed64 f(__FIXED_INF__);
  return f;
}
// bits
constexpr uint8_t __ZERO__ = 0;
constexpr uint8_t __ONE__ = 1;
// zero
const fixed __FIXED_ZERO__(0.0);
const fixed64 __FIXED64_ZERO__(0.0);
template <typename LLR> inline LLR Zero() { return 0.0; }
template <> inline fixed Zero() { return __FIXED_ZERO__; }
template <> inline fixed64 Zero() { return __FIXED64_ZERO__; }
// absolute value
const fixed __FIXED_MINUSONE__(-1.0);
const fixed64 __FIXED64_MINUSONE__(-1.0);
template <typename LLR> inline LLR Abs(LLR const val) { return std::abs(val); }
template <> inline fixed Abs(fixed const val) {
  return val < const_cast<fixed &>(__FIXED_ZERO__)
             ? const_cast<fixed &>(__FIXED_MINUSONE__) * val
             : val;
}
template <> inline fixed64 Abs(fixed64 const val) {
  return val < const_cast<fixed64 &>(__FIXED64_ZERO__)
             ? const_cast<fixed64 &>(__FIXED64_MINUSONE__) * val
             : val;
}
// signbit (true -> negative, false -> positive)
template <typename LLR> inline bool Sign(LLR const val) {
  return std::signbit(val);
};
template <> inline bool Sign(fixed const val) {
  return val < const_cast<fixed &>(__FIXED_ZERO__) ? true : false;
}
template <> inline bool Sign(fixed64 const val) {
  return val < const_cast<fixed64 &>(__FIXED64_ZERO__) ? true : false;
}
// hard decision
template <typename LLR> inline uint8_t HardDecision(LLR const val) {
  return val <= 0.0 ? __ONE__ : __ZERO__;
}
template <> inline uint8_t HardDecision(fixed const val) {
  return val <= const_cast<fixed &>(__FIXED_ZERO__) ? __ONE__ : __ZERO__;
}
template <> inline uint8_t HardDecision(fixed64 const val) {
  return val <= const_cast<fixed64 &>(__FIXED64_ZERO__) ? __ONE__ : __ZERO__;
}

/*
 * This class provides some convenience functions to manipulate matrices over
 * GF(2). The implemented methods are all that is needed for LDPC encoding and
 * are
 *
 * - Row or column swap
 * - Row addition
 * - Horizontal concatenation of two matrices
 * - Matrix inversion (using Gaussian Elimination)
 * - Arbitrary row and column slicing
 *
 * Most of the operations are implemented using the std::valarray class and the
 * XOR operators.
 */
class GF2Matrix {
public:
  // create a new matrix from a flat data array
  GF2Matrix(uint8_t *in_data, std::size_t m, std::size_t n)
      : data(in_data, m * n), m(m), n(n) {}
  // create new matrix from a valarray
  GF2Matrix(std::valarray<uint8_t> in_array, std::size_t m, std::size_t n)
      : data(in_array), m(m), n(n) {}
  // can also construct an identity matrix
  explicit GF2Matrix(std::size_t eye_size)
      : data((uint8_t)0, eye_size * eye_size), m(eye_size), n(eye_size) {
    data[std::slice(0, eye_size, eye_size + 1)] = 1;
  }

  // matrix manipulation functions
  // swap col1 with col2
  void col_swap(std::size_t col1, std::size_t col2);
  // swap row1 with row2
  void row_swap(std::size_t row1, std::size_t row2);
  // add row1 to row2 over GF(2)
  void row_add(std::size_t row1, std::size_t row2);

  // concatenate horizontally with another GF2 matrix
  GF2Matrix hcat(GF2Matrix A);
  // compute the inverse using gaussian elimination
  GF2Matrix inv();
  // return a submatrix
  GF2Matrix slice(std::size_t row_start, std::size_t row_end,
                  std::size_t col_start, std::size_t col_end) const;

  // print the matrix
  void print();

  // the data is stored in a valarray, which allows for nice slicing
  std::valarray<uint8_t> data;
  std::size_t m, n; // dimensions
};

class IndexOutOfBoundsException : public std::exception {
  virtual const char *what() const throw() {
    return "Row or column index out of bounds.";
  }
};

class NotInvertibleException : public std::exception {
  virtual const char *what() const throw() {
    return "The matrix is not invertible.";
  }
};

class HCatDimensionMismatchException : public std::exception {
  virtual const char *what() const throw() {
    return "Dimension mismatch during concatenation.";
  }
};

/*
 * Compressed Sparse Row storage of a sparse GF(2) matrix. Also known
 * as 'Yale Format'. This class does not implement all of the necessary
 * operations, just the ones that I need for LDPC coding, which are
 * - matrix/vector multiplication in GF(2)
 * - construction from a multidimensional GF(2) array.
 * - solving a lower triangular system u = Tv, where T is sparse
 *
 * Some references:
 * [1] https://en.wikipedia.org/wiki/Sparse_matrix
 * [2] http://www.cslab.ntua.gr/~kkourt/papers/cf08-spmv-kkourt-pr.pdf
 */

// FIXME: It seems like the to_gf2matrix() method is buggy. See the tests
// regarding IEEE 80211 LDPC codes for this. Luckily, by using the DDEncder and
// Sum-Product decoder, we are not using this matrix impl. Investigate when
// there is time.

class CSRGF2Matrix {
public:
  // construct from a GF2Matrix
  explicit CSRGF2Matrix(GF2Matrix M);
  // construct from a flat row-major array
  CSRGF2Matrix(uint8_t *flat, std::size_t m, std::size_t n);
  // construct from a valarray holding a row-major matrix
  CSRGF2Matrix(std::valarray<uint8_t> va, std::size_t m, std::size_t n);
  // construct from a flat row-major array
  CSRGF2Matrix(std::vector<uint8_t> v, std::size_t m, std::size_t n);
  // construct an empty object
  CSRGF2Matrix() : IA(), JA(), m(0), n(0){};

  GF2Matrix to_gf2matrix(); // FIXME make sure this works correctly

  // solve the lower triangular system u = T * v in GF(2). (compute v =
  // T^(-1) * u)
  std::valarray<uint8_t> lt_solve(std::valarray<uint8_t> u);

  // multiply with a vector in GF(2)
  std::valarray<uint8_t> mult_vec(std::valarray<uint8_t> v);
  std::vector<uint8_t> mult_vec(std::vector<uint8_t> const &v);

  // print the contents of the matrix to the screen
  void print();

  // data is stored in two vectors
  std::vector<std::size_t> IA;
  std::vector<std::size_t> JA;
  std::size_t m, n; // dimensions
};

class DimensionMismatchException : public std::exception {
  virtual const char *what() const throw() {
    return "Dimensions for matrix-vector multiplication do not match up.";
  }
};

/*
 * LDPC decoders with a flooding schedule
 */

template <typename LLR, typename Out> class FloodingDecoder {
public:
  FloodingDecoder(GF2Matrix H, unsigned int m, unsigned int n, uint64_t max_its,
                  bool output_value, bool do_parity_check);

  // settings
  uint64_t _max_its;
  bool _output_value;
  bool _do_parity_check;

  // hold the PCM
  CSRGF2Matrix H;
  size_t _m;
  size_t _n;
  size_t _k;

  // connections
  // A holds indices of the check nodes that bit node i is connected to
  // B holds indices of the bit nodes that check node j is connected to
  std::vector<std::vector<size_t>> A, B;

  // messages
  // messages from check nodes to bit nodes
  std::vector<LLR> E;
  // messages from bit nodes to check nodes
  std::vector<LLR> M;

  // storage for sign bits
  std::vector<bool> _si2;

  // results
  std::vector<LLR> _L;
  std::vector<uint8_t> _z;

  // decode a block of data
  // compute the check messages (this is where most algorithms differ)
  virtual void check_messages() = 0;
  // compute the bit messages
  void bit_messages(LLR const *in_llr);
  // decode the data
  void flood_decode(LLR const *in_llr, Out *out);

  // write the results to the output buffers
  void _write_results(Out *out);
};

/*
 * LDPC decoders with the serial-c schedule (iterate through check nodes)
 */
template <typename LLR, typename Out> class SerialCDecoder {
protected:
  SerialCDecoder(GF2Matrix H, unsigned int m, unsigned int n, uint64_t max_its,
                 bool output_value, bool do_parity_check);

  // settings
  uint64_t _max_its;
  bool _output_value;
  bool _do_parity_check;

  // hold the PCM
  CSRGF2Matrix H;
  size_t _m;
  size_t _n;
  size_t _k;

  // B holds the indices of the bit nodes that check node j is connected to
  std::vector<std::vector<size_t>> _B;

  // R holds the messages from check nodes to the bit nodes
  // P holds the a-posteriori log-likelihood
  // Q is a temporary buffer for the bit messages
  std::vector<std::vector<LLR>> _R;
  std::vector<LLR> _P, _Q;

  // storage for sign bits
  std::vector<bool> _si2;

  // results
  std::vector<uint8_t> _z;

  // decode
  void _serial_c_decode(LLR const *in_llr, Out *out);
  virtual void _message_passing() = 0;
  void _write_results(Out *out);
  void _write_results_zc(Out *out);
};
} /* namespace detail */
} /* namespace yaldpc */

#endif /* r1383db6433d17876c9eeeb6a81be346a52dcf3dffa837573f */
