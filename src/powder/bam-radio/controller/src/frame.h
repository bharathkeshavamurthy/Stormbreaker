/* -*-c++-*-
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 * Copyright © 2017 Stephen Larew
 */

#ifndef c6a7fc9ce25e525160
#define c6a7fc9ce25e525160

#include "dll_types.h"
#include "flowtracker.h"
#include "mcs.h"
#include "ofdm.h"
#include "segment.h"

#include <boost/asio.hpp>
#include <memory>
#include <stdexcept>
#include <uhd/types/time_spec.hpp>
#include <vector>

namespace bamradio {

/// Abstract PHY frame interface
class Frame {
public:
  /// NodeID of source
  virtual NodeID sourceNodeID() const;
  /// NodeID of destination
  virtual NodeID destinationNodeID() const;
  /// True if frame has a header
  virtual bool hasHeader() const = 0;
  /// Number of coded bits per block
  virtual size_t blockLength(bool header) const = 0;
  /// Number of header blocks
  virtual size_t numBlocks(bool header) const = 0;
  /// Write bits of n-th block of frame
  virtual void writeBlock(bool header, size_t blockNumber,
                          boost::asio::mutable_buffer b) const = 0;
  /// Read decoded bits into payload
  virtual std::pair<size_t, size_t> readBlock(size_t firstBlockNumber,
                                              boost::asio::const_buffer bits)
      __attribute__((warn_unused_result)) = 0;
  /// Bytes of the payload
  virtual boost::asio::const_buffer payload() const = 0;

protected:
  Frame(NodeID sourceNodeID, NodeID destinationNodeID);
  NodeID _sourceNodeID, _destinationNodeID;
};

namespace ofdm {

// TODO: factor out non-DFT-spread OFDM frame base class (?)

/// DFT-Spread OFDM frame
class DFTSOFDMFrame : public Frame {
public:
  typedef std::shared_ptr<DFTSOFDMFrame> sptr;

  // Frame interface

  bool hasHeader() const;
  size_t blockLength(bool header) const;
  size_t numBlocks(bool header) const;
  void writeBlock(bool header, size_t blockNumber,
                  boost::asio::mutable_buffer b) const;
  std::pair<size_t, size_t> readBlock(size_t firstBlockNumber,
                                      boost::asio::const_buffer bits)
      __attribute__((warn_unused_result));
  boost::asio::const_buffer payload() const;
  std::shared_ptr<std::vector<uint8_t>> movePayload();

  /** Construct a frame for transmission.
   *
   * \param sourceNodeID ID of source node
   * \param destinationNodeID ID of destinationID
   * \param payload Buffer of payload bytes (not bits) (copied)
   * \param headerMcs MCS index for header
   * \param payloadMcs MCS index for payload
   * \param payloadSymbolIndex Index into OFDMSymbolParams::table for payload
   * symbols
   * \param seqNum Sequence number of this frame
   * \param frameID unique frame identifier
   */
  DFTSOFDMFrame(NodeID sourceNodeID, NodeID destinationNodeID,
                std::vector<dll::Segment::sptr> segments, MCS::Name headerMcs,
                MCS::Name payloadMcs, SeqID::ID payloadSymSeqID,
                uint16_t seqNum, int64_t frameID);

  class BadCRC : public std::exception {
  public:
    uint32_t const rxMagic;
    uint32_t const rxMagicErrorBits;
    uint32_t const numMagicBitErrors;
    BadCRC(uint32_t rxMagic_, uint32_t rxMagicErrorBits_,
           uint32_t numMagicBitErrors_)
        : rxMagic(rxMagic_), rxMagicErrorBits(rxMagicErrorBits_),
          numMagicBitErrors(numMagicBitErrors_) {}
    char const *what() const noexcept { return "CRC check failed"; }
  };

#if 0
  class ProbablyNotAFrame : public BadCRC {
  public:
    ProbablyNotAFrame(uint32_t rxMagic_, uint32_t rxMagicErrorBits_,
                      uint32_t numMagicBitErrors_)
        : BadCRC(rxMagic_, rxMagicErrorBits_, numMagicBitErrors_) {}
  };

  class CorruptFrame : public BadCRC {
  public:
    CorruptFrame(uint32_t rxMagic_, uint32_t rxMagicErrorBits_,
                 uint32_t numMagicBitErrors_)
        : BadCRC(rxMagic_, rxMagicErrorBits_, numMagicBitErrors_) {}
  };
#endif

  class InvalidMCSIndex : public std::runtime_error {
  public:
    InvalidMCSIndex() : std::runtime_error("received invalid MCS index") {}
  };

  class InvalidPayloadSymbolSeqID : public std::exception {
  public:
    char const *what() const noexcept {
      return "received invalid payload symbol sequence index";
    }
  };

  class TooManySegments : public std::exception {
  public:
    char const *what() const noexcept {
      return "segments exceed max frame payload length";
    }
  };

  /** Construct a frame from a header bits
   *
   * \param headerMcs MCS index for header
   * \param headerBits buffer of uncoded header bits
   *
   * \throws BadCRC on failed validation
   */
  DFTSOFDMFrame(MCS::Name headerMCS, boost::asio::const_buffer headerBits);

  /// Return parameters describing the header of this frame.
  DFTSOFDMFrameParams headerParams(bool tx, uint16_t oversample_rate) const;
  /// Return parameters describing the payload of this frame.
  DFTSOFDMFrameParams payloadParams(bool tx, uint16_t oversample_rate,
                                    uint64_t postfixPad) const;
  /// Return parameters describing the full frame.
  DFTSOFDMFrameParams allParams(bool tx, uint16_t oversample_rate,
                                uint64_t postfixPad) const;

  ///
  size_t payloadNumInfoBits() const;

  MCS::Name payloadMcs() const { return _payloadMcs; }
  MCS::Name headerMcs() const { return _headerMcs; }
  SeqID::ID payloadSymSeqID() const { return _payloadSymSeqID; }

  /// Returns vector of complete (either non-checksummed or validated
  /// checksummed) segments.
  std::vector<dll::Segment::sptr> const &segments() const;

  uint16_t seqNum() const { return _seqNum; }
  int64_t frameID() const { return _frameID; }

  void printHeader() const;

  bool blockIsValid(int n) const;

  // ARQ
  void setARQFeedback(std::vector<dll::ARQBurstInfo> arq_feedback) {
    _arq_feedback = arq_feedback;
  }
  std::vector<dll::ARQBurstInfo> ARQFeedback() { return _arq_feedback; }

private:
  MCS::Name const _headerMcs;
  MCS::Name _payloadMcs;
  SeqID::ID _payloadSymSeqID;
  uint16_t _seqNum;
  int64_t _frameID;

  // TODO: Don't expose BlockHeader here, similar to Header.
  struct BlockHeader {
    BlockHeader()
        : checksum(0), nextSegmentBitOffset(0),
          nextSegmentType((uint16_t)dll::SegmentType::None),
          nextSegmentDestNodeID(UnspecifiedNodeID) {}

    BlockHeader(boost::asio::const_buffer const b, size_t const fbo,
                size_t const lbo, uint16_t nsbo, dll::SegmentType nstype,
                NodeID nsDestNodeID)
        : checksum(0), nextSegmentBitOffset(nsbo),
          nextSegmentType((uint16_t)nstype),
          nextSegmentDestNodeID(nsDestNodeID) {
      checksum = computeCRC(b, fbo, lbo);
    }

    uint32_t computeCRC(boost::asio::const_buffer const b, size_t const fbo,
                        size_t const lbo) const;

    uint32_t checksum;
    uint16_t nextSegmentBitOffset : 12;
    uint16_t nextSegmentType : 4;
    static_assert(ssize_t(dll::SegmentType::NUM_TYPES) < (1 << 4), "");
    uint8_t nextSegmentDestNodeID;
    static_assert(sizeof(NodeID) == 1, "");
  } __attribute__((packed));

  static_assert(sizeof(BlockHeader) == 7, "");

  struct BlockInfo {
    BlockInfo(size_t const K, size_t const bn);
    uint64_t byteOffset, firstByteSkipBits, nextByteOffsetUp, lastByteSkipBits;
  };

  struct BI {
    CodeRate R;
    size_t N;
    size_t count;
    size_t K() const { return N * R; }
  };

  BI const _bh;
  BI _bp;

  // ARQ feedback
  std::vector<dll::ARQBurstInfo> _arq_feedback;

  /// Payload bytes (packed)
  std::vector<uint8_t> _payload;
  std::vector<dll::Segment::sptr> _cachedSegments;

  std::map<int, std::pair<BlockHeader, bool>> _bhmap;

  std::vector<uint8_t> mutable _payloadBlockBits;

  void updateCachedSegments();

public:
  static constexpr size_t BlockHeaderSize = sizeof(BlockHeader);
};

// scrambling
struct Scramble {
  static std::vector<uint8_t> const bytes;
  static std::vector<uint8_t> const bits;
};

} // namespace ofdm
} // namespace bamradio

#endif
