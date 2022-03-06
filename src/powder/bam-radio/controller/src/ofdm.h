//  Copyright © 2017 Stephen Larew

#ifndef a465c3a3cedc693f44
#define a465c3a3cedc693f44

#include "bam_constellation.h"
#include "llr_format.h"

#include <cstdint>
#include <gnuradio/tags.h>
#include <memory>
#include <pmt/pmt.h>
#include <utility>
#include <vector>

namespace bamradio {
namespace ofdm {

/// Parameters of an OFDM symbol
struct OFDMSymbolParams {
  typedef std::shared_ptr<OFDMSymbolParams> sptr;

  /// Vector (PMT s32 uvec) of indices mapping data symbols to frequency domain
  /// sub-carriers (required, non-null)
  pmt::pmt_t data_carrier_mapping;
  /// Vector (PMT s32 uvec) of indices mapping pilot symbols to frequency domain
  /// sub-carriers (required, non-null)
  pmt::pmt_t pilot_carrier_mapping;
  /// Vector (PMT c32 uvec) of pilot symbols (required, non-null)
  pmt::pmt_t pilot_symbols;
  /// Symbol constellation (optional, nullable)
  bamradio::constellation::sptr constellation;
  /// Length of OFDM symbol
  uint16_t symbol_length;
  /// Multiple of symbol_length to give length of final OFDM symbol after
  /// oversampling (size of IFFT @ TX)
  uint16_t oversample_rate;
  /// Length of cyclic prefix
  uint16_t cyclic_prefix_length;
  /// Length of cyclic postfix
  uint16_t cyclic_postfix_length;
  /// Vector (PMT c32 uvec) of symbols to prefix the cyclically-prefixed OFDM
  /// symbol (optional, nullable)
  pmt::pmt_t prefix;
  /// Number of zeros to append.
  uint64_t postfixPad;

  size_t numTXSamples() const {
    assert(oversample_rate > 0);
    return (symbol_length + cyclic_prefix_length) * oversample_rate +
           // <2018-06-12 Tue> hoo boy we are in some trouble here. some
           // explanation is necessary at this point.
           //
           // pre-tx-rewrite we regarded the prefix as a simple memcopy
           // operation. whatever is in the prefix, copy before the OFDM
           // symbol. Now (post-tx-rewrite), we want to pulse shape/ window our
           // OFDM symbols and have to thus add a cylic postfix. Furthermore, we
           // need to ADD a part of the cyclic prefix to the previous' symbols
           // cyclic postfix (see phy_tx::window_cp) for details.
           //
           // to enable this, we make the following breaking change: we assume
           // that a prefix has to be the same length as the OFDM symbols it's
           // attached to and we further assume that the prefix comes WITHOUT
           // the cyclic prefix and the cyclic postfix attached (but is already
           // oversampled), i.e., we let the cyclic prefixer compute the cyclic
           // prefix of the prefix vector.
           //
           // this is why we now need to add the length of the cyclic prefix to
           // the length of the prefix in this calculation.
           (prefix == nullptr ? 0
                              : pmt::length(prefix) +
                                    cyclic_prefix_length * oversample_rate) +
           // <2018-06-12 Tue> n.b. postfixPad has been ignored for a
           // while. even less necessary with windowing/pulse shaping.
           // FIXME remove postfixPad.
           postfixPad;
  }
  size_t numDataCarriers() const { return pmt::length(data_carrier_mapping); }
  size_t numBits() const {
    return constellation == nullptr
               ? 0
               : numDataCarriers() * constellation->bits_per_symbol();
  }
  size_t numPilotCarriers() const { return pmt::length(pilot_carrier_mapping); }
};

enum SymbolName : size_t {
  /// Zadoff-Chu preamble symbol for RX (sl=128,cp=12)
  ZC_SYNC_RX_128_12 = 0,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QPSK)
  ZC_SYNC_TX_DATA_128_12_108_QPSK,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=96,pc=0,QPSK)
  ZC_SYNC_TX_DATA_128_12_96_24_0_QPSK,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM16)
  ZC_SYNC_TX_DATA_128_12_108_QAM16,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  ZC_SYNC_TX_DATA_128_12_108_12_0_QAM16,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM32)
  ZC_SYNC_TX_DATA_128_12_108_QAM32,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  ZC_SYNC_TX_DATA_128_12_108_12_0_QAM32,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM64)
  ZC_SYNC_TX_DATA_128_12_108_QAM64,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  ZC_SYNC_TX_DATA_128_12_108_12_0_QAM64,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM128)
  ZC_SYNC_TX_DATA_128_12_108_QAM128,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  ZC_SYNC_TX_DATA_128_12_108_12_0_QAM128,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM256)
  ZC_SYNC_TX_DATA_128_12_108_QAM256,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  ZC_SYNC_TX_DATA_128_12_108_12_0_QAM256,
  /// Zadoff-Chu RX chanest symbol (sl=128,cp=12)
  ZC_CHANEST_RX_128_12,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QPSK)
  ZC_CHANEST_TX_DATA_128_12_108_QPSK,
  /// Data symbol for TX (sl=128,cp=12,dc=108,pc=0,QPSK)
  DATA_128_12_108_QPSK,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=96,pc=24,QPSK)
  ZC_CHANEST_TX_DATA_128_12_96_24_0_QPSK,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  ZC_CHANEST_TX_DATA_128_12_108_12_0_QPSK,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_0_QPSK,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_1_QPSK,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_2_QPSK,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_3_QPSK,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_4_QPSK,
  /// Data symbol 5 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_5_QPSK,
  /// Data symbol 6 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_6_QPSK,
  /// Data symbol 7 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_7_QPSK,
  /// Data symbol 8 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_8_QPSK,
  /// Data symbol 9 for TX (sl=128,cp=12,dc=108,pc=12,QPSK)
  DATA_128_12_108_12_9_QPSK,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=96,pc=24,QPSK)
  DATA_128_12_96_24_0_QPSK,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=96,pc=24,QPSK)
  DATA_128_12_96_24_1_QPSK,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=96,pc=24,QPSK)
  DATA_128_12_96_24_2_QPSK,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=96,pc=24,QPSK)
  DATA_128_12_96_24_3_QPSK,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=96,pc=24,QPSK)
  DATA_128_12_96_24_4_QPSK,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM16)
  ZC_CHANEST_TX_DATA_128_12_108_QAM16,
  /// Data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM16)
  DATA_128_12_108_QAM16,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM16,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_0_QAM16,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_1_QAM16,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_2_QAM16,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_3_QAM16,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_4_QAM16,
  /// Data symbol 5 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_5_QAM16,
  /// Data symbol 6 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_6_QAM16,
  /// Data symbol 7 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_7_QAM16,
  /// Data symbol 8 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_8_QAM16,
  /// Data symbol 9 for TX (sl=128,cp=12,dc=108,pc=12,QAM16)
  DATA_128_12_108_12_9_QAM16,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM32)
  ZC_CHANEST_TX_DATA_128_12_108_QAM32,
  /// Data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM32)
  DATA_128_12_108_QAM32,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM32,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_0_QAM32,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_1_QAM32,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_2_QAM32,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_3_QAM32,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_4_QAM32,
  /// Data symbol 5 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_5_QAM32,
  /// Data symbol 6 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_6_QAM32,
  /// Data symbol 7 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_7_QAM32,
  /// Data symbol 8 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_8_QAM32,
  /// Data symbol 9 for TX (sl=128,cp=12,dc=108,pc=12,QAM32)
  DATA_128_12_108_12_9_QAM32,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM64)
  ZC_CHANEST_TX_DATA_128_12_108_QAM64,
  /// Data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM64)
  DATA_128_12_108_QAM64,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM64,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_0_QAM64,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_1_QAM64,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_2_QAM64,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_3_QAM64,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_4_QAM64,
  /// Data symbol 5 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_5_QAM64,
  /// Data symbol 6 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_6_QAM64,
  /// Data symbol 7 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_7_QAM64,
  /// Data symbol 8 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_8_QAM64,
  /// Data symbol 9 for TX (sl=128,cp=12,dc=108,pc=12,QAM64)
  DATA_128_12_108_12_9_QAM64,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM128)
  ZC_CHANEST_TX_DATA_128_12_108_QAM128,
  /// Data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM128)
  DATA_128_12_108_QAM128,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM128,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_0_QAM128,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_1_QAM128,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_2_QAM128,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_3_QAM128,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_4_QAM128,
  /// Data symbol 5 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_5_QAM128,
  /// Data symbol 6 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_6_QAM128,
  /// Data symbol 7 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_7_QAM128,
  /// Data symbol 8 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_8_QAM128,
  /// Data symbol 9 for TX (sl=128,cp=12,dc=108,pc=12,QAM128)
  DATA_128_12_108_12_9_QAM128,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM256)
  ZC_CHANEST_TX_DATA_128_12_108_QAM256,
  /// Data symbol for TX (sl=128,cp=12,dc=108,pc=0,QAM256)
  DATA_128_12_108_QAM256,
  /// Zadoff-Chu prefixed data symbol for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  ZC_CHANEST_TX_DATA_128_12_108_12_0_QAM256,
  /// Data symbol 0 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_0_QAM256,
  /// Data symbol 1 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_1_QAM256,
  /// Data symbol 2 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_2_QAM256,
  /// Data symbol 3 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_3_QAM256,
  /// Data symbol 4 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_4_QAM256,
  /// Data symbol 5 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_5_QAM256,
  /// Data symbol 6 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_6_QAM256,
  /// Data symbol 7 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_7_QAM256,
  /// Data symbol 8 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_8_QAM256,
  /// Data symbol 9 for TX (sl=128,cp=12,dc=108,pc=12,QAM256)
  DATA_128_12_108_12_9_QAM256,
  NUM_SYMBOLS
};

OFDMSymbolParams::sptr lookupSymbol(uint16_t oversample_rate, SymbolName name);

namespace SeqID {
/// A pre-defined sequence of OFDM symbols.
enum class ID {
  /// Periodic full chanest OFDM symbol (every 19 OFDM symbols)
  P19FULL_128_12_108_QPSK = 0,
  /// Periodic full chanest OFDM symbol (every 10 OFDM symbols)
  P10FULL_128_12_108_QAM16,
  P10FULL_128_12_108_QAM32,
  P10FULL_128_12_108_QAM64,
  P10FULL_128_12_108_QAM128,
  P10FULL_128_12_108_QAM256,
  /// First OFDM symbol is full chanest, then zig (but don't zag) evenly spaced
  /// pilots
  ZIG_128_12_108_12_QPSK,
  ZIG_128_12_108_12_QAM16,
  ZIG_128_12_108_12_QAM32,
  ZIG_128_12_108_12_QAM64,
  ZIG_128_12_108_12_QAM128,
  ZIG_128_12_108_12_QAM256,
  /// Periodic full chanest OFDM symbol with zig evenly spaced pilots between
  PFULL_ZIG_128_12_108_12_QPSK,
  /// Periodic full chanest OFDM symbol with zig evenly spaced pilots between
  PFULL_ZIG_128_12_96_24_QPSK,
  NUM_SEQIDS
};

ID stringNameToIndex(std::string const &n);
/// Returns the first (count,SymbolName) pair for sequence s.
std::pair<size_t, SymbolName> begin(ID s, bool tx);
/// Returns the next (count,SymbolName) pair for sequence s after prevIdx.
std::pair<size_t, SymbolName> next(ID s, bool tx, size_t prevIdx);
/// Returns the number of data carriers per OFDM symbol
int symLen(ID s);
/// Returns the number of occupied data carriers per OFDM symbol
int occupiedCarriers(ID s);
/// Returns the number of cyclic prefix symbols
int cpLen(ID s);
/// Returns the average number of bits per OFDM symbol.
float bpos(ID s);
} // namespace SeqID

extern pmt::pmt_t const data_carrier_mapping_tag_key;
extern pmt::pmt_t const pilot_carrier_mapping_tag_key;
extern pmt::pmt_t const pilot_symbols_tag_key;
extern pmt::pmt_t const constellation_tag_key;
extern pmt::pmt_t const fft_length_tag_key;
extern pmt::pmt_t const symbol_length_tag_key;
extern pmt::pmt_t const cyclic_prefix_length_tag_key;
extern pmt::pmt_t const prefix_tag_key;
extern pmt::pmt_t const postfix_pad_tag_key;
extern pmt::pmt_t const reset_tag_key;
extern pmt::pmt_t const dft_spread_length_tag_key;
extern pmt::pmt_t const block_size_tag_key;
extern pmt::pmt_t const rate_idx_tag_key;
extern pmt::pmt_t const snr_tag_key;
extern pmt::pmt_t const frame_ptr_tag_key;
extern pmt::pmt_t const chanest_msg_port_key;
extern pmt::pmt_t const rx_time_tag_key;
extern pmt::pmt_t const dec_extra_item_consumption_tag_key;

/// Parameters of an OFDM frame.
struct OFDMFrameParams {
  typedef std::shared_ptr<OFDMFrameParams> sptr;

  /// Vector of (count,OFDMSymbol) pairs
  std::vector<std::pair<uint64_t, OFDMSymbolParams::sptr>> symbols;

  explicit OFDMFrameParams(decltype(symbols) const &s) : symbols(s) {}

  /// Return the total number of OFDM symbols (excluding symbol prefixes)
  size_t numSymbols() const;

  /// Return the total number of samples (including prefixes)
  size_t numTXSamples() const;

  // Return the total number of (typically coded) data bits.
  size_t numBits() const;

  /// Return the index of the symbol containing the given bit.
  std::pair<size_t, size_t> offsetOfBit(size_t bitNumber) const;
};

/// Parameters of a DFT spread OFDM frame.
struct DFTSOFDMFrameParams : public OFDMFrameParams {
  typedef std::shared_ptr<DFTSOFDMFrameParams> sptr;

  /// Length of DFT for symbol spreading
  uint32_t dft_spread_length;

  explicit DFTSOFDMFrameParams(decltype(symbols) const &s);
};
} // namespace ofdm
} // namespace bamradio

#endif
