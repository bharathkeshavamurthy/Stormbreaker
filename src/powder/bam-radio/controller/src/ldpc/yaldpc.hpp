/*
 * yaldpc.hpp
 * LDPC Encoder and Decoder main header
 *
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 */

#ifndef e10adbec294ab8cbf37b47ea8d03ec64aae51c54b5ecc83556
#define e10adbec294ab8cbf37b47ea8d03ec64aae51c54b5ecc83556

#include <cassert>
#include <cstdint>
#include <exception>
#include <memory>
#include <valarray>
#include <vector>

#include "detail.hpp"

namespace yaldpc {

/*
 * ----------------------------------------------------------------------------
 * Base Classes
 * ----------------------------------------------------------------------------
 */

/*
 * All encoders inherit this base encoder interface.
 */
class Encoder {
public:
  typedef std::shared_ptr<Encoder> sptr;
  virtual ~Encoder() {}

  /*
   * Encode a data vector of length 'n'. the data format is unpacked bits (1 bit
   * per byte)
   */
  virtual void encode(uint8_t const *in, uint8_t *out) = 0;
  virtual std::vector<uint8_t> encode(std::vector<uint8_t> const &in) = 0;

  /*
   * Info about the code rate
   */
  virtual std::size_t m() = 0;
  virtual std::size_t n() = 0;
  virtual std::size_t k() = 0;
};

/*
 * All decoders inherit this base decoder interface.
 */
template <typename LLR, typename Out> class Decoder {
public:
  typedef std::shared_ptr<Decoder> sptr;
  virtual ~Decoder() {}
  /*
   * Decode a block of n LLRs and save the desired output. Output data is either
   * a posteriori LLRs or unpacked bits.
   */
  virtual void decode(LLR const *in, Out *out) = 0;
  virtual std::vector<Out> decode(std::vector<LLR> const &in) = 0;

  /*
   * Info about the code rate
   */
  virtual std::size_t m() = 0;
  virtual std::size_t n() = 0;
  virtual std::size_t k() = 0;

  /*
   * Decoder parameters
   */
  virtual uint64_t max_its() = 0;
  virtual bool output_value() = 0;
  virtual bool do_parity_check() = 0;
};

/*
 * ----------------------------------------------------------------------------
 * Code & PCM Data structures
 * ----------------------------------------------------------------------------
 */

/*
 * We store our LDPC Codes using this simple data structure. The constructors
 * take a flat vector or array of zeros and ones
 */
struct LDPCCode {
  detail::GF2Matrix H;
  unsigned int m;
  unsigned int n;
  unsigned int k;

  LDPCCode(std::vector<uint8_t> H_gf2, unsigned int m, unsigned int n)
      : H(&H_gf2[0], m, n), m(m), n(n), k(n - m) {}
  LDPCCode(uint8_t *H_gf2, unsigned int m, unsigned int n)
      : H(H_gf2, m, n), m(m), n(n), k(n - m) {}
};

/*
 * The IEEE 802.11 and 802.16 standards provide a good selection of codes for
 * wireless applications.
 */
struct IEEE802Code {
  std::vector<int64_t> Hb;
  unsigned int Z;
  unsigned int mb;
  unsigned int nb;

  IEEE802Code(std::vector<int64_t> Hb, unsigned int Z, unsigned int mb,
              unsigned int nb)
      : Hb(Hb), Z(Z), mb(mb), nb(nb) {}
};

namespace ieee80211 {
/*
 * IEEE 802.11-2012 [1] LDPC Parity check matrices.
 *
 * Available Rates: [index] rate
 *  [0] 1/2
 *  [1] 2/3
 *  [2] 3/4
 *  [3] 5/6
 *
 * Available block sizes:
 *  648
 *  1296
 *  1944
 *
 * [1] IEEE 802.11™-2012: IEEE Standard for Information
 *     technology--Telecommunications and information exchange between systems
 *     Local and metropolitan area networks--Specific requirements Part 11:
 *     Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY)
 *     Specifications

 */
IEEE802Code get(unsigned int rate_idx, unsigned int blocksize);

const std::vector<unsigned int> valid_rates({0, 1, 2, 3});
const std::vector<unsigned int> valid_sizes({648, 1296, 1944});
} /* namespace ieee80211 */

namespace ieee80216 {
/*
 * IEEE 802.16 [2] LDPC Parity check matrices.
 *
 * [2] IEEE 802.16™-2012: IEEE Standard for Air Interface for Broadband
 *     Wireless Access Systems
 *
 * FIXME: Write down
 */
} /* namespace ieee80216 */

/*
 * Expand a QC IEEE 802.11/16 code to a full PCM.
 *
 */
LDPCCode expand(IEEE802Code code);

/*
 * ----------------------------------------------------------------------------
 * Encoder Implementations
 * ----------------------------------------------------------------------------
 *
 * If your code is a dual-diagonal QC-LDPC code, use the DDEncoder. Otherwise,
 * the ALT Encoder might work if you read the paper and figure out the 'gap'
 * parameter. As a last resort, the systematic encoder should work for most code
 * matrices.
 */

/*
 * LDPC Encoding using using a generator matrix
 *
 * This encoder attempts to find the generator matrix for the PCM H using row
 * and column operations, i.e., it attempts to find
 *
 *   H = [A | I], where I is the n-k identity matrix
 *
 * and encodes a codeword as
 *
 *   c = [c1 c2], where
 *
 *   c1 = u (the message bits) and
 *   c2 = u * A^T, with the matrix multiplication in GF(2).
 */
class SystematicEncoder : public Encoder {
public:
  typedef std::shared_ptr<SystematicEncoder> sptr;
  SystematicEncoder(LDPCCode const &code);
  static sptr make(LDPCCode const &code);

  // encode the data
  void encode(uint8_t const *in, uint8_t *out);
  std::vector<uint8_t> encode(std::vector<uint8_t> const &in);

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

private:
  // pre-process the parity check matrix and return the partial generator
  // matrix
  void preprocess_H();

  // we save the generator matrix as a CSR GF2 matrix to skip a few zero
  // multiplications. save the parity check matrix as reguluar GF2 matrix
  detail::GF2Matrix H;
  detail::CSRGF2Matrix A;
  std::vector<std::size_t> col_perm; // column permutations

  std::size_t _m, _n, _k;
};

// in case something goes wrong...
class SystematicConversionException : public std::exception {
  virtual const char *what() const throw() {
    return "Could not bring PCM into systematic form using elementary row "
           "operations.";
  }
};

/*
 * ALT LDPC Encoding
 *
 * LDPC Encoding with approx O(n) complexity using the Approximate Lower
 * Triangular (ALT) technique from [3]. This encoder makes use of many
 * sparsity properties of the parity check matrix in ALT form.
 *
 * [3] Richardson, Thomas J., and Rüdiger L. Urbanke. "Efficient encoding of
 *     low-density parity-check codes." IEEE Transactions on Information
 *     Theory 47.2 (2001): 638-656.
 */
class ALTEncoder : public Encoder {
public:
  typedef std::shared_ptr<ALTEncoder> sptr;
  ALTEncoder(LDPCCode const &code, std::size_t g);
  static sptr make(LDPCCode const &code, std::size_t g);

  // encode the data
  void encode(uint8_t const *in, uint8_t *out);
  std::vector<uint8_t> encode(std::vector<uint8_t> const &in);

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

private:
  // compute the phi_inv dense encoding matrix
  void compute_phi_inv();

  // parity check matrix
  //
  //      |         |  |       |
  //  H = |    A    |B |   T   |
  //      |         |  |       |
  //      |---------+--+-------|
  //      |    C    |D |   E   |
  //
  detail::GF2Matrix H;
  // sparse encoding matrices
  detail::CSRGF2Matrix A, B, T, C, E;
  // possibly dense encoding matrices
  detail::CSRGF2Matrix phi_inv;
  // the 'gap' between approximate and full lower triangular
  std::size_t g;

  std::size_t _m, _n, _k;
};

/*
 * Efficient encoding of dual-diagonal structured LDPC codes
 *
 * Direct LDPC Encoding for dual-diagonal parity-check matrices using the bit
 * prediction/correction method from [4]. Note that the IEEE 802.11 and 802.16
 * codes from [1] and [2] are particular suitable for this structure.
 *
 * [4] Lin, Chia-Yu, Chih-Chun Wei, and Mong-Kai Ku. "Efficient encoding for
 *     dual-diagonal structured LDPC codes based on parity bit prediction and
 *     correction." Circuits and Systems, 2008. APCCAS 2008. IEEE Asia Pacific
 *     Conference on. IEEE, 2008.
 */
class DDEncoder : public Encoder {
public:
  typedef std::shared_ptr<DDEncoder> sptr;
  static sptr make(IEEE802Code const &code);
  DDEncoder(IEEE802Code const &code);

  // encode the data
  void encode(uint8_t const *in, uint8_t *out);
  std::vector<uint8_t> encode(std::vector<uint8_t> const &in);

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

private:
  // mainly for debugging purposes
  void print_Hb();

  // store the base matrix as valarray
  std::valarray<int64_t> Hb;
  // base matrix properties
  std::size_t Z, x, mb, nb, kb;
  std::size_t _m, _n, _k;
};

class NoDDStructureException : public std::exception {
  virtual const char *what() const throw() {
    return "The parity check matrix is not structured properly. Use a "
           "different encoder.";
  }
};

/*
 * ----------------------------------------------------------------------------
 * Decoder Implementations
 * ----------------------------------------------------------------------------
 *
 * The best choice here is the Min-Sum decoder.
 */

/*
 * Sum-product decoding of LDPC codes
 *
 * This decoder uses the sum-product algorithm to compute the check
 * messages. Inspired by the listings of [5].
 *
 * [5] Sarah J. Johnson, Introducing Low-Density Parity-Check Codes, ACoRN
 *     Spring School, version 1.1,
 *     http://sigpromu.org/sarah/SJohnsonLDPCintro.pdf
 */
template <typename LLR, typename Out>
class SumProductDecoder : public Decoder<LLR, Out>,
                          public detail::FloodingDecoder<LLR, Out> {
public:
  typedef std::shared_ptr<SumProductDecoder<LLR, Out>> sptr;
  SumProductDecoder(LDPCCode const &code, uint64_t max_its, bool output_value,
                    bool do_parity_check);
  static sptr make(LDPCCode const &code, uint64_t max_its, bool output_value,
                   bool do_parity_check);

  // decode
  void decode(LLR const *in, Out *out);
  std::vector<Out> decode(std::vector<LLR> const &in);

  // decoder parameters
  uint64_t max_its();
  bool output_value();
  bool do_parity_check();

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

  // compute the check messages for the current pass
  void check_messages();
};

/*
 * Sum-Product LDPC decoding using the Min* function
 *
 * This decoder is equivalent to a sum-product decoder, but uses a different
 * function, coined Min* [6] to compute the LLR additions. See [6, Sec. IV]
 * for a discussion on the advantages of this function.
 *
 * [6] Divsalar, Dariush, et al. "Capacity-approaching protograph codes." IEEE
 *     Journal on Selected Areas in Communications 27.6 (2009): 876-888.
 */
template <typename LLR, typename Out>
class MinStarDecoder : public Decoder<LLR, Out>,
                       public detail::FloodingDecoder<LLR, Out> {
public:
  typedef std::shared_ptr<MinStarDecoder<LLR, Out>> sptr;
  MinStarDecoder(LDPCCode const &code, uint64_t max_its, bool output_value,
                 bool do_parity_check);
  static sptr make(LDPCCode const &code, uint64_t max_its, bool output_value,
                   bool do_parity_check);

  // decode
  void decode(LLR const *in, Out *out);
  std::vector<Out> decode(std::vector<LLR> const &in);

  // decoder parameters
  uint64_t max_its();
  bool output_value();
  bool do_parity_check();

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

  // compute the check messages for the current pass
  void check_messages();
};

/*
 * Scaled Min-Sum decoding
 *
 * This decoder uses the scaled min-sum approximation, which was developed in
 * [7]. The parameter 'alpha' scales the check-to-bit messages.
 *
 * [7] Chen, Jinghu, and Marc PC Fossorier. "Near optimum universal belief
 *     propagation based decoding of low-density parity check codes." IEEE
 *     Transactions on communications 50.3 (2002): 406-414.
 */
template <typename LLR, typename Out>
class MinSumDecoder : public Decoder<LLR, Out>,
                      public detail::FloodingDecoder<LLR, Out> {
public:
  typedef std::shared_ptr<MinSumDecoder<LLR, Out>> sptr;
  MinSumDecoder(LDPCCode const &code, uint64_t max_its, bool output_value,
                bool do_parity_check, LLR alpha);
  static sptr make(LDPCCode const &code, uint64_t max_its, bool output_value,
                   bool do_parity_check, LLR alpha = 0.75);

  // decode
  void decode(LLR const *in, Out *out);
  std::vector<Out> decode(std::vector<LLR> const &in);

  // decoder parameters
  uint64_t max_its();
  bool output_value();
  bool do_parity_check();

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

  // compute the check messages for the current pass
  void check_messages();

private:
  LLR _alpha, _minusalpha;
};

/*
 * Scaled min-sum decoding with serial-c scheduling
 *
 * Another min-sum decoder with the scaling trick from [7]. This decoder,
 * however, computes the bit messages on-the-fly, a technique also known as
 * 'serial-c' or 'turbo' message passing scheduling [8].
 *
 * [8] Sharon, Eran, Simon Litsyn, and Jacob Goldberger. "Efficient serial
 *     message-passing schedules for LDPC decoding." IEEE Transactions on
 *     Information Theory 53.11 (2007): 4076-4091.
 */
template <typename LLR, typename Out>
class SerialCMinSumDecoder : public Decoder<LLR, Out>,
                             public detail::SerialCDecoder<LLR, Out> {
public:
  typedef std::shared_ptr<SerialCMinSumDecoder<LLR, Out>> sptr;
  SerialCMinSumDecoder(LDPCCode const &code, uint64_t max_its,
                       bool output_value, bool do_parity_check, float alpha);
  static sptr make(LDPCCode const &code, uint64_t max_its, bool output_value,
                   bool do_parity_check, float alpha = 0.75);

  // decode
  void decode(LLR const *in, Out *out);
  std::vector<Out> decode(std::vector<LLR> const &in);

  // decoder parameters
  uint64_t max_its();
  bool output_value();
  bool do_parity_check();

  // code rate
  std::size_t m();
  std::size_t n();
  std::size_t k();

private:
  LLR _alpha, _minusalpha;
  void _message_passing();
};
} /* namespace yaldpc */

#endif /* e10adbec294ab8cbf37b47ea8d03ec64aae51c54b5ecc83556 */
