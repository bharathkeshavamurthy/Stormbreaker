/* -*-c++-*-
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 * Copyright © 2017-2018 Stephen Larew
 */

#include "frame.h"
#include "arq.h"
#include "events.h"
#include <boost/crc.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <pmt/pmt.h>
#include <random>
#include <tuple>
#include <vector>

namespace bamradio {

using namespace boost::asio;

static_assert(SeqNumBitlength + BlockNumBitlength == 16, "");

/// Negative acknowledgement of sequence numbers from a source Node.
struct NAck {
  /// Source NodeID
  NodeID srcID;
  /// Sequence # from srcID
  uint16_t seqNum : SeqNumBitlength;
  /// Block # in frame seqNum
  uint16_t blockNum : BlockNumBitlength;
} __attribute__((packed));

struct Retransmission {
  /// Sequence # from srcID
  uint16_t seqNum : SeqNumBitlength;
  /// Block # in frame seqNum
  uint16_t blockNum : BlockNumBitlength;
} __attribute__((packed));

Frame::Frame(NodeID sourceNodeID, NodeID destinationNodeID)
    : _sourceNodeID(sourceNodeID), _destinationNodeID(destinationNodeID) {}
NodeID Frame::sourceNodeID() const { return _sourceNodeID; }
NodeID Frame::destinationNodeID() const { return _destinationNodeID; }

namespace ofdm {

// Header must fit into an R=1/2 N=648 codeword, which has 324 bits / 40.5
// bytes / 10.125 32bit words.
// CRC at end of header is 32 bits.
// => header info <= 324-32=292 bits / 36.5 bytes / 9.125 32bit words
struct Header {
  /// NodeID of source
  uint64_t sourceNodeID : 8;
  static_assert(sizeof(NodeID) == 1, "");

  /// NodeID of destination
  uint64_t destinationNodeID : 8;

  /// index into MCS::table of payload
  uint64_t mcsIndex : 6;
  static_assert(MCS::NUM_MCS <= (1 << 6), "");

  /// index into SeqID
  uint64_t payloadSymSeqID : 4;
  static_assert(ssize_t(SeqID::ID::NUM_SEQIDS) < (1 << 4), "");

  /// Sequence number of this frame on this source NodeID.
  uint64_t seqNum : SeqNumBitlength;

  /// Number of blocks/codewords in payload.
  uint64_t numPayloadBlocks : BlockNumBitlength;

  uint64_t _pad0 : 22;

  // END UINT64

  /// unique frame identifier
  int64_t frameID;

  /// ARQ feedback
  struct {
    uint16_t flow_uid;
    uint8_t burst_num;
    uint16_t seq_num;
  } __attribute__((packed)) arqFeedback[MaxNumARQFeedback];

  uint32_t headerCRC;

  uint32_t computeCRC() const {
    // TODO: determine if recreate crc_32_type everytime is expensive
    boost::crc_32_type crc_comp;
    crc_comp.process_bytes(this, sizeof(*this));
    return crc_comp();
  }

} __attribute__((packed));

static_assert(sizeof(Header) == 40, "Header must be 40 bytes");

/// Magic number to identify start of header
// On little endian (lsb first), we should see this bit sequence:
// [0 0 1 0] [1 0 1 0] [0 1 1 0] [1 1 1 0]
// [0 0 0 1] [1 0 0 1] [0 1 0 1] [1 1 0 1]
// uint32_t const HeaderMagic = 0xBA987654;

inline uint64_t roundNumberUp(uint64_t dividend, uint64_t divisor) {
  return (dividend + (divisor - 1)) / divisor;
}

// lsb pack sizeof(src_buf) bits from src_buf at dstBitOffset into dst_buf
inline void pack(mutable_buffer const dst_buf, size_t const dstBitOffset,
                 const_buffer const src_buf) {
  assert(buffer_size(dst_buf) * 8 > dstBitOffset);
  assert(buffer_size(dst_buf) * 8 - dstBitOffset >= buffer_size(src_buf));

  auto const src_begin = buffer_cast<uint8_t const *>(src_buf);
  auto const src_end = src_begin + buffer_size(src_buf);
  auto d = buffer_cast<uint8_t *>(dst_buf + (dstBitOffset / 8));

  auto i = dstBitOffset % 8;
  for (auto s = src_begin; s != src_end; ++s) {
    assert(*s <= 1);
    *d |= *s << i;
    if (++i == 8) {
      ++d;
      i = 0;
    }
  }
}

// fill dst_buf with min(sizeof(dst_buf), 8*sizeof(src_buf)-srcBitOffset) bits
// extracted lsb first from srcBitOffset into src_buf
inline void unpack(mutable_buffer const dst_buf, const_buffer const src_buf,
                   size_t const srcBitOffset) {
  assert(8 * buffer_size(src_buf) > srcBitOffset);

  size_t const bitsToUnpack =
      std::min(8 * buffer_size(src_buf) - srcBitOffset, buffer_size(dst_buf));

  auto d = buffer_cast<uint8_t *>(dst_buf);
  auto d_end = d + bitsToUnpack;

  auto src_it = buffer_cast<uint8_t const *>(src_buf + (srcBitOffset / 8));
  auto bo = srcBitOffset % 8;
  auto s = (*src_it++) >> bo;

  while (d != d_end) {
    *d++ = s & 1;
    s >>= 1;
    if (++bo == 8) {
      bo = 0;
      s = *(src_it++);
    }
  }
}

uint32_t DFTSOFDMFrame::BlockHeader::computeCRC(const_buffer const b,
                                                size_t const fbo,
                                                size_t const lbo) const {
  assert(fbo < 8);
  assert(lbo < 8);
  boost::crc_32_type crc_comp;
  auto const begin = buffer_cast<uint8_t const *>(b);
  auto const end = begin + buffer_size(b);
  assert(checksum == 0);
  crc_comp.process_bytes(this, sizeof(*this));
  crc_comp.process_block(begin + (fbo > 0 ? 1 : 0), end - (lbo > 0 ? 1 : 0));
  if (fbo > 0) {
    crc_comp.process_byte(*begin & (0xff - ((1 << fbo) - 1)));
  }
  if (lbo > 0) {
    crc_comp.process_byte(*(end - 1) & ((1 << (8 - lbo)) - 1));
  }
  return crc_comp();
}

/// Compute (byteOffset, firstByteSkipBits, nextByteOffsetUp,
// lastByteSkipBits) tuple for block number bn and block lenght K bits.
DFTSOFDMFrame::BlockInfo::BlockInfo(size_t const K, size_t const bn) {
  auto const payloadBits = K - 8 * sizeof(DFTSOFDMFrame::BlockHeader);
  auto const bitOffset = bn * payloadBits;
  auto const nextBitOffset = (bn + 1) * payloadBits;
  byteOffset = bitOffset / 8;
  nextByteOffsetUp = roundNumberUp(nextBitOffset, 8);
  firstByteSkipBits = bitOffset - byteOffset * 8;
  lastByteSkipBits = nextByteOffsetUp * 8 - nextBitOffset;
}

DFTSOFDMFrame::DFTSOFDMFrame(NodeID srcID, NodeID dstID,
                             std::vector<dll::Segment::sptr> segments,
                             MCS::Name headerMcs, MCS::Name payloadMcs,
                             SeqID::ID payloadSymSeqID, uint16_t seqNum,
                             int64_t frameID)
    : Frame(srcID, dstID), _headerMcs(headerMcs), _payloadMcs(payloadMcs),
      _payloadSymSeqID(payloadSymSeqID),
      _seqNum(seqNum & ((1 << SeqNumBitlength) - 1)), _frameID(frameID),
      _bh({MCS::table[headerMcs].codeRate, MCS::table[headerMcs].blockLength,
           1}),
      _bp({MCS::table[payloadMcs].codeRate, MCS::table[payloadMcs].blockLength,
           roundNumberUp(std::accumulate(segments.begin(), segments.end(), 0,
                                         [](auto const a, auto const &s) {
                                           return a + s->length();
                                         }) *
                             8,
                         MCS::table[payloadMcs].blockLength *
                                 MCS::table[payloadMcs].codeRate -
                             8 * sizeof(BlockHeader))}),
      _payload([this, &segments] {
        std::vector<uint8_t> buf(
            roundNumberUp(_bp.count * (_bp.K() - 8 * sizeof(BlockHeader)), 8));
        auto bufit = begin(buf);
        for (auto const &segment : segments) {
          auto const ba = segment->rawContentsBuffer();
          for (auto const &b : ba) {
            assert(buf.end() - bufit >= (ssize_t)buffer_size(b));
            std::copy_n(buffer_cast<uint8_t const *>(b), buffer_size(b), bufit);
            bufit += buffer_size(b);
          }
        }
        return buf;
      }()),
      _cachedSegments([this, &segments] {
        // copy segments with new _payload backing store
        std::vector<dll::Segment::sptr> segmentsCopy;
        const_buffer pb = buffer(_payload);
        for (auto const &segment : segments) {
          switch (segment->type()) {
          case dll::SegmentType::IPv4: {
            auto const s = std::make_shared<net::IP4PacketSegment>(
                segment->destNodeID(), pb);
            segmentsCopy.push_back(s);
            pb = pb + s->length();
            break;
          }
          case dll::SegmentType::ARQIPv4: {
            auto const s = std::make_shared<net::ARQIP4PacketSegment>(
                segment->destNodeID(), pb);
            segmentsCopy.push_back(s);
            pb = pb + s->length();
            break;
          }
          case dll::SegmentType::PSD: {
            auto const s = std::make_shared<psdsensing::PSDSegment>(
                segment->destNodeID(), pb);
            segmentsCopy.push_back(s);
            pb = pb + s->length();
            break;
          }
          case dll::SegmentType::Control: {
            auto const s =
                std::make_shared<controlchannel::ControlChannelSegment>(pb);
            segmentsCopy.push_back(s);
            pb = pb + s->length();
            break;
          }
          default:
            panic("unexpected segment type");
          }
        }
        return segmentsCopy;
      }()),
      _bhmap([this] {
        std::map<int, std::pair<BlockHeader, bool>> bhmap;
        // sit - current segment iterator
        auto sit = _cachedSegments.begin();
        // segmentEnd - # bytes for all segments including current
        auto segmentEnd = 0; //(*sit)->length();

        if (sit == _cachedSegments.end()) {
          return bhmap;
        }

        auto nextSegmentBitOffset = 0;
        auto nextSegmentType = (*sit)->type();
        auto nextSegmentDestNodeID = (*sit)->destNodeID();

        for (size_t bn = 0; bn < _bp.count && sit != _cachedSegments.end();
             ++bn) {

          auto const bi = BlockInfo(_bp.K(), bn);

          if (segmentEnd * 8 == bi.byteOffset * 8 + bi.firstByteSkipBits) {
            nextSegmentBitOffset = 0;
            segmentEnd += (*sit)->length();
            nextSegmentType = (*sit)->type();
            nextSegmentDestNodeID = (*sit)->destNodeID();
          } else if (segmentEnd * 8 <
                     bi.nextByteOffsetUp * 8 - bi.lastByteSkipBits) {
            nextSegmentBitOffset =
                (segmentEnd * 8) - (bi.byteOffset * 8 + bi.firstByteSkipBits);
            ++sit;
            if (sit != _cachedSegments.end()) {
              segmentEnd += (*sit)->length();
              nextSegmentType = (*sit)->type();
              nextSegmentDestNodeID = (*sit)->destNodeID();
            } else {
              nextSegmentType = dll::SegmentType::None;
              nextSegmentDestNodeID = UnspecifiedNodeID;
            }
          } else if (segmentEnd * 8 ==
                     bi.nextByteOffsetUp * 8 - bi.lastByteSkipBits) {
            ++sit;
            nextSegmentBitOffset =
                _bp.K(); // TODO: needs to be K - sizeof(header) ?
            nextSegmentType = dll::SegmentType::None;
            nextSegmentDestNodeID = UnspecifiedNodeID;
          } else {
            // there is no segment starting in this block so create fake one
            // past end of block to indicate so
            nextSegmentBitOffset = _bp.K();
            nextSegmentType = dll::SegmentType::None;
            nextSegmentDestNodeID = UnspecifiedNodeID;
          }
          auto const bb = buffer(buffer(_payload) + bi.byteOffset,
                                 bi.nextByteOffsetUp - bi.byteOffset);
          BlockHeader const bh(bb, bi.firstByteSkipBits, bi.lastByteSkipBits,
                               nextSegmentBitOffset, nextSegmentType,
                               nextSegmentDestNodeID);
          bhmap[bn] = std::make_pair(bh, true);
        }
        return bhmap;
      }()) {
  if (_seqNum >= (1 << SeqNumBitlength)) {
    throw std::logic_error("frame sequence number too high");
  }
  if (_bp.count >= (1 << BlockNumBitlength)) {
    throw TooManySegments();
  }
  if (_payloadMcs >= MCS::NUM_MCS) {
    throw std::runtime_error("invalid payload MCS");
  }
  if (_payloadSymSeqID >= SeqID::ID::NUM_SEQIDS) {
    throw std::runtime_error("invalid payload symseqid");
  }
  for (auto const &segment : _cachedSegments) {
    if (segment->length() < roundNumberUp(1944 * 5, 6 * 8)) {
      throw std::logic_error("Segment is shorter than min.");
    }
  }
}

DFTSOFDMFrame::DFTSOFDMFrame(MCS::Name hmcs, const_buffer headerBits)
    : Frame(0, 0), _headerMcs(hmcs),
      _bh({MCS::table[hmcs].codeRate, MCS::table[hmcs].blockLength, 1}) {
  assert(buffer_size(headerBits) >= (_bh.N * _bh.R));

  std::vector<uint8_t> headerBuffer(sizeof(Header));
  mutable_buffer headerBytes = buffer(headerBuffer);

  // truncate to headerBytes bytes
  pack(headerBytes, 0, buffer(headerBits, 8 * headerBuffer.size()));

  auto const header = (Header const *)headerBuffer.data();

  auto headerCopy = *header;
  headerCopy.headerCRC = 0;

  if (headerCopy.computeCRC() != header->headerCRC) {
#if 0
    uint32_t a = header->magic ^ HeaderMagic;
    auto bb = 0;
    while (a != 0) {
      bb += a & 1;
      a >>= 1;
    }
    if (bb > 10) {
      throw ProbablyNotAFrame(header->magic, header->magic ^ HeaderMagic, bb);
    } else {
      throw CorruptFrame(header->magic, header->magic ^ HeaderMagic, bb);
    }
#endif
    throw BadCRC(0, 0, 0);
  }

  // Extract primitive values

  _sourceNodeID = header->sourceNodeID;
  _destinationNodeID = header->destinationNodeID;
  _payloadMcs = (MCS::Name)header->mcsIndex;
  _payloadSymSeqID = (SeqID::ID)header->payloadSymSeqID;
  _seqNum = header->seqNum;
  _frameID = header->frameID;

  // Extract ARQ Feedback

  _arq_feedback.clear();
  for (size_t i = 0; i < MaxNumARQFeedback; ++i) {
    if (header->arqFeedback[i].flow_uid > 0) {
      _arq_feedback.push_back(
          dll::ARQBurstInfo{.flow_uid = header->arqFeedback[i].flow_uid,
                            .burst_num = header->arqFeedback[i].burst_num,
                            .seq_num = header->arqFeedback[i].seq_num});
    }
  }

  // Validate primitive values

  if (_payloadMcs >= MCS::table.size()) {
    throw InvalidMCSIndex();
  }

  if (_payloadSymSeqID >= SeqID::ID::NUM_SEQIDS) {
    throw InvalidPayloadSymbolSeqID();
  }

  auto const pmcs = MCS::table[_payloadMcs];
  _bp = {pmcs.codeRate, pmcs.blockLength, header->numPayloadBlocks};

  _payload.resize(
      roundNumberUp(_bp.count * (_bp.K() - 8 * sizeof(BlockHeader)), 8));
}

bool DFTSOFDMFrame::hasHeader() const { return true; }

size_t DFTSOFDMFrame::blockLength(bool header) const {
  return header ? _bh.N : _bp.N;
}

size_t DFTSOFDMFrame::numBlocks(bool header) const {
  return header ? _bh.count : _bp.count;
}

std::vector<uint8_t> const Scramble::bytes([] {
  std::vector<uint8_t> sb;
  sb.resize(2 << 14);
  std::mt19937_64 rng(33);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::generate(begin(sb), end(sb), [&] { return dist(rng); });
  return sb;
}());

std::vector<uint8_t> const Scramble::bits([] {
  std::vector<uint8_t> sb;
  sb.resize(Scramble::bytes.size() * 8);
  unpack(buffer(sb), buffer(Scramble::bytes), 0);
  return sb;
}());

const_buffer scramble_bytes(size_t offset) {
  return buffer(Scramble::bytes) + offset;
}
const_buffer scramble_bits(size_t bit_offset) {
  return buffer(Scramble::bits) + bit_offset;
}

void do_scramble(mutable_buffer w, const_buffer scramble_seq) {
  auto b = buffer_cast<uint8_t *>(w);
  auto const e = b + buffer_size(w);
  auto sb = buffer_cast<uint8_t const *>(scramble_seq);
  while (b < e) {
    *b++ ^= *sb++;
  }
}

void DFTSOFDMFrame::writeBlock(bool header, size_t bn, mutable_buffer b) const {
  auto const &bi = header ? _bh : _bp;

  assert(buffer_size(b) >= bi.K());
  assert(bn < bi.count);

  if (header) {
    // Create header, compute crc, unpack, encode

    assert(_seqNum < (1 << SeqNumBitlength));
    Header h = {
        .sourceNodeID = sourceNodeID(),
        .destinationNodeID = destinationNodeID(),
        .mcsIndex = (uint8_t)_payloadMcs,
        .payloadSymSeqID = (uint32_t)_payloadSymSeqID,
        .seqNum = _seqNum,
        .numPayloadBlocks = _bp.count,
        ._pad0 = 0x9a2f5, // "scramble bits"
        .frameID = _frameID,
    };

    for (size_t i = 0; i < MaxNumARQFeedback; ++i) {
      bool has_entry = i < _arq_feedback.size();
      h.arqFeedback[i].flow_uid = has_entry ? _arq_feedback[i].flow_uid : 0;
      h.arqFeedback[i].burst_num = has_entry ? _arq_feedback[i].burst_num : 0;
      h.arqFeedback[i].seq_num = has_entry ? _arq_feedback[i].seq_num : 0;
    }

    h.headerCRC = h.computeCRC();

    unpack(b, buffer(&h, sizeof(h)), 0);
  } else {
    // FIXME: only zero pad where necessary
    std::fill(buffer_cast<uint8_t *>(b), buffer_cast<uint8_t *>(b) + _bp.K(),
              0);
    auto pbb = b;

    auto const bi = BlockInfo(_bp.K(), bn);

    assert(bi.nextByteOffsetUp <= _payload.size());

    auto const bb = buffer(buffer(_payload) + bi.byteOffset,
                           bi.nextByteOffsetUp - bi.byteOffset);

    assert(_bhmap.at(bn).second);
    auto const &bh = _bhmap.at(bn).first; // 👀

    // Unpack the block header into bit buffer
    unpack(pbb, buffer(&bh, sizeof(bh)), 0);
    pbb = pbb + sizeof(bh) * 8;

    // Unpack the payload block bytes into bit buffer
    assert(buffer_size(pbb) == _bp.K() - sizeof(bh) * 8);
    unpack(pbb, bb, bi.firstByteSkipBits);

    // Scramble and encode whole block
    do_scramble(b, scramble_bits(0));
  }
}

std::pair<size_t, size_t>
DFTSOFDMFrame::readBlock(size_t const firstBlockNumber,
                         boost::asio::const_buffer const bgiven) {
  // A block has one codeword worth of bits minus the block header
  auto const blockBits = _bp.K();
  auto const blockPayloadBits = blockBits - 8 * sizeof(BlockHeader);

  // Need at least one blocklength to work
  if (buffer_size(bgiven) < blockBits) {
    return {blockBits, 0};
  }

  size_t i = 0;

  while (buffer_size(bgiven) >= (i + 1) * blockBits) {
    auto const b = buffer(bgiven + i * blockBits, blockBits);

    // block number:
    auto const bn = firstBlockNumber + i;

    // recompute bit offset after removing block header bits
    auto const bitOffset = bn * blockPayloadBits;

    auto const numPayloadBits = _payload.size() * 8;

    // Payload size may only partially fill last blocklength so cut it short if
    // needed:
    assert(blockPayloadBits <= numPayloadBits - bitOffset);
    auto const numPayloadBitsToPack =
        std::min(numPayloadBits - bitOffset, blockPayloadBits);

    if (numPayloadBitsToPack <= 0) {
      log::doomsday("Fatal error", __FILE__, __LINE__);
    }

    // Unscramble bits
    _payloadBlockBits.resize(blockBits);
    mutable_buffer pbb = buffer(_payloadBlockBits);
    buffer_copy(pbb, b);
    do_scramble(pbb, scramble_bits(0));

    BlockHeader bh;
    bh.nextSegmentDestNodeID = 0;
    pack(buffer(&bh, sizeof(bh)), 0, buffer(pbb, sizeof(bh) * 8));

    pack(buffer(_payload), bitOffset,
         buffer(pbb + 8 * sizeof(BlockHeader), numPayloadBitsToPack));

    auto const bi = BlockInfo(_bp.K(), bn);
    auto const bb = buffer(buffer(_payload) + bi.byteOffset,
                           bi.nextByteOffsetUp - bi.byteOffset);

    assert(bitOffset == bi.byteOffset * 8 + bi.firstByteSkipBits);
    assert(bitOffset + numPayloadBitsToPack ==
           bi.nextByteOffsetUp * 8 - bi.lastByteSkipBits);

    auto bhCopy = bh;
    bhCopy.checksum = 0;
    auto const checksumGood =
        bhCopy.computeCRC(bb, bi.firstByteSkipBits, bi.lastByteSkipBits) ==
        bh.checksum;

    _bhmap[bn] = std::make_pair(bh, checksumGood);

    ++i;
  }

  updateCachedSegments();

  return {blockBits, i};
}

const_buffer DFTSOFDMFrame::payload() const { return buffer(_payload); }

std::shared_ptr<std::vector<uint8_t>> DFTSOFDMFrame::movePayload() {
  auto const pd = _payload.data();
  auto const r = std::make_shared<std::vector<uint8_t>>(std::move(_payload));
  assert(_payload.size() == 0);
  assert(r->data() == pd);
  return r;
}

DFTSOFDMFrameParams
DFTSOFDMFrame::headerParams(bool tx, uint16_t oversample_rate) const {
  switch (_headerMcs) {
  case MCS::QPSK_R12_N648: {
    DFTSOFDMFrameParams hprx(
        {{1, lookupSymbol(oversample_rate, SymbolName::ZC_SYNC_RX_128_12)},
         {3, lookupSymbol(oversample_rate, SymbolName::DATA_128_12_108_QPSK)}});
    /*{1, SymbolTable[SymbolName::DATA_128_12_108_12_0_QPSK]},
    {1, SymbolTable[SymbolName::DATA_128_12_108_12_4_QPSK]},
    {1, SymbolTable[SymbolName::DATA_128_12_108_12_7_QPSK]}});*/
    DFTSOFDMFrameParams hptx(
        {{1, lookupSymbol(oversample_rate,
                          SymbolName::ZC_SYNC_TX_DATA_128_12_108_QPSK)},
         {2, lookupSymbol(oversample_rate, SymbolName::DATA_128_12_108_QPSK)}});
    /*{{1, SymbolTable[SymbolName::ZC_SYNC_TX_DATA_128_12_108_12_0_QPSK]},
     {1, SymbolTable[SymbolName::DATA_128_12_108_12_4_QPSK]},
     {1, SymbolTable[SymbolName::DATA_128_12_108_12_7_QPSK]}});*/
    return tx ? hptx : hprx;
  }
  default:
    log::doomsday("Fatal error", __FILE__, __LINE__);
  }
}

DFTSOFDMFrameParams DFTSOFDMFrame::payloadParams(bool tx,
                                                 uint16_t oversample_rate,
                                                 uint64_t postfixPad) const {
  decltype(DFTSOFDMFrameParams::symbols) s;

  auto const numCodedBits = _bp.N * _bp.count;

  if (numCodedBits == 0) {
    return DFTSOFDMFrameParams(s);
  }

  size_t codedBitsAdded = 0;
  auto sn = SeqID::begin(_payloadSymSeqID, tx);
  auto snsym = lookupSymbol(oversample_rate, sn.second);
  auto snbits = snsym->numBits();
  auto sni = 0;

  while (codedBitsAdded < numCodedBits &&
         codedBitsAdded + sn.first * snbits <= numCodedBits) {
    s.emplace_back(sn.first, snsym);
    codedBitsAdded += sn.first * snbits;
    sn = SeqID::next(_payloadSymSeqID, tx, sni++);
    snsym = lookupSymbol(oversample_rate, sn.second);
    snbits = snsym->numBits();
  }

  // Try to bail early
  if (codedBitsAdded == numCodedBits) {
    // add postfixPad to last symbol
    auto &lastSymbols = s.back();
    if (lastSymbols.second->postfixPad != postfixPad) {
      auto const lastSymbol =
          std::make_shared<OFDMSymbolParams>(*lastSymbols.second);
      lastSymbol->postfixPad = postfixPad;
      if (lastSymbols.first > 1) {
        --lastSymbols.first;
        s.emplace_back(1, lastSymbol);
      } else {
        s.back().second = lastSymbol;
      }
    }
    return DFTSOFDMFrameParams(s);
  } else if (snbits == 0) {
    throw std::logic_error(
        "Bits remain but next symbol has not allocated bits.");
  }

  // try to to insert snic out of sn.first symbols

  auto const snic = roundNumberUp(numCodedBits - codedBitsAdded, snbits);
  assert(snic <= sn.first);
  if (snic > 0) {
    assert(snbits > 0);
    s.emplace_back(snic, snsym);
    codedBitsAdded += snic * snbits;
    // std::cout << boost::format("added %1%/%2% whole symbols\n") % snic %
    // sn.first;
  }

  // Now compute the final symbol as a partially used whole symbol from the
  // sequence
  //
  // Err, no, scratch that, not adding partial special symbols anymore.

  auto const codedPadBits = (ssize_t)numCodedBits - (ssize_t)codedBitsAdded;

  assert(codedPadBits <= 0);

#if 0
  assert(codedPadBits % snsym->constellation->bits_per_symbol() == 0);
  auto const extraCarriers =
      codedPadBits / snsym->constellation->bits_per_symbol();
  assert(extraCarriers < snsym->numDataCarriers());

  // Copy the standard symbol and truncate its data carrier mapping
  if (extraCarriers > 0) {
    auto const extraSymbol = std::make_shared<OFDMSymbolParams>(*snsym);

    extraSymbol->data_carrier_mapping = pmt::init_s32vector(
        extraCarriers,
        pmt::s32vector_elements(extraSymbol->data_carrier_mapping));

    extraSymbol->postfixPad = postfixPad;

    // std::cout << boost::format("added partial symbol with %1% carriers\n") %
    // extraCarriers;

    s.emplace_back(1, extraSymbol);
  } else {
#endif
  // FIXME: DRY
  // add postfixPad to last symbol
  auto &lastSymbols = s.back();
  if (lastSymbols.second->postfixPad != postfixPad) {
    auto const lastSymbol =
        std::make_shared<OFDMSymbolParams>(*lastSymbols.second);
    lastSymbol->postfixPad = postfixPad;
    if (lastSymbols.first > 1) {
      --lastSymbols.first;
      s.emplace_back(1, lastSymbol);
    } else {
      s.back().second = lastSymbol;
    }
  }

  return DFTSOFDMFrameParams(s);
}

DFTSOFDMFrameParams DFTSOFDMFrame::allParams(bool tx, uint16_t oversample_rate,
                                             uint64_t postfixPad) const {
  auto const a = headerParams(tx, oversample_rate);
  auto const b = payloadParams(tx, oversample_rate, postfixPad);
  auto s = a.symbols;
  s.reserve(s.size() + b.symbols.size());
  for (auto const &p : b.symbols) {
    s.emplace_back(p);
  }
  return DFTSOFDMFrameParams(s);
}

size_t DFTSOFDMFrame::payloadNumInfoBits() const {
  return _bp.count * _bp.N * _bp.R;
}

std::vector<dll::Segment::sptr> const &DFTSOFDMFrame::segments() const {
  return _cachedSegments;
}

bool DFTSOFDMFrame::blockIsValid(int n) const { return _bhmap.at(n).second; }

void DFTSOFDMFrame::updateCachedSegments() {
  // FIXME: mutex

  _cachedSegments.resize(0);

  if (_bhmap.size() == 0) {
    // TODO: alternative to resize?
    return;
  }

  const_buffer const pb = buffer(_payload);
  assert(_payload.size() ==
         roundNumberUp(_bp.count * (_bp.K() - 8 * sizeof(BlockHeader)), 8));

  auto bhitrb = _bhmap.begin();
  auto const bhite = _bhmap.end();

  while (bhitrb != bhite) {
    // if starting on new block, must pass checksum to trust the next* params
    if (!bhitrb->second.second) {
      ++bhitrb;
      continue;
    }

    auto segmentType = (dll::SegmentType)bhitrb->second.first.nextSegmentType;
    auto segmentDestNodeID = (NodeID)bhitrb->second.first.nextSegmentDestNodeID;
    auto const segmentBitOffset = bhitrb->second.first.nextSegmentBitOffset;

    assert(segmentType < dll::SegmentType::NUM_TYPES);
    assert(segmentBitOffset <= _bp.K());

    if (segmentType == dll::SegmentType::None) {
      ++bhitrb;
      continue;
    }

    // Find range of contiguous blocks [bhitrb,bhitre] in [bhitrb,bhite)
    auto const fr = [bhite](auto bhitrb) {
      auto num = bhitrb->first;
      ++bhitrb;
      return std::find_if_not(bhitrb, bhite, [&](auto const &it) {
        // next bn == it's bn && it's block is checksummedGOOD
        return (++num == it.first) && it.second.second;
      });
    };

    auto const bhitre = --fr(bhitrb);

    auto const bib = BlockInfo(_bp.K(), bhitrb->first);
    auto const bie = BlockInfo(_bp.K(), bhitre->first);

    assert((bib.firstByteSkipBits + segmentBitOffset) % 8 == 0);
    auto const bSegmentByteOffset =
        (bib.firstByteSkipBits + segmentBitOffset) / 8;

    auto const segmentsNumBytes = bie.nextByteOffsetUp -
                                  (bie.lastByteSkipBits > 0 ? 1 : 0) -
                                  (bib.byteOffset + bSegmentByteOffset);

    const_buffer segmentsBuffer =
        buffer(pb + bib.byteOffset + bSegmentByteOffset, segmentsNumBytes);

    auto bytesAdded = 0;

    try {
      while (buffer_size(segmentsBuffer) > 0) {

        auto const newSegment = [&]() -> std::shared_ptr<dll::Segment> {
          switch (segmentType) {
          case dll::SegmentType::None:
            return nullptr;
          case dll::SegmentType::IPv4:
            return std::make_shared<net::IP4PacketSegment>(segmentDestNodeID,
                                                           segmentsBuffer);
          case dll::SegmentType::ARQIPv4:
            return std::make_shared<net::ARQIP4PacketSegment>(segmentDestNodeID,
                                                              segmentsBuffer);
          case dll::SegmentType::PSD:
            return std::make_shared<psdsensing::PSDSegment>(segmentDestNodeID,
                                                            segmentsBuffer);
          case dll::SegmentType::Control:
            return std::make_shared<controlchannel::ControlChannelSegment>(
                segmentsBuffer);
          default:
            panic("Unhandled segment type.");
          }
        }();

        // TODO: support non-checksummed segments (here?)

        if (newSegment) {
          _cachedSegments.push_back(newSegment);

          bytesAdded += newSegment->length();
          segmentsBuffer = segmentsBuffer + newSegment->length();
        } else {
          // advance to next block
          break;
        }

        if (buffer_size(segmentsBuffer) == 0) {
          // Don't try to look a next block because we aleady determined the
          // byte range for the current range of backing blocks; segmentsBuffer
          // is now empty so break out.
          break;
        }

        // Compute backing block number of next segment
        int const ibn = (bib.byteOffset + bSegmentByteOffset + bytesAdded) * 8 /
                        (_bp.K() - 8 * sizeof(BlockHeader));
        assert(ibn > bhitrb->first);
        assert(ibn <= bhitre->first);

        auto const &ibh = _bhmap[ibn];
        assert(ibh.second);

        segmentDestNodeID = ibh.first.nextSegmentDestNodeID;
        segmentType = (dll::SegmentType)ibh.first.nextSegmentType;
        assert(segmentType < dll::SegmentType::NUM_TYPES);
      }
    } catch (dll::BufferTooShort) {
      // OK
    } catch (net::InvalidIPv4SegmentLength) {
      // OK
    } catch (std::exception &e) {
      log::doomsday((boost::format("Fatal error: %1%") % e.what()).str(),
                    __FILE__, __LINE__);
    } catch (...) {
      log::doomsday("Fatal error", __FILE__, __LINE__);
    }

    bhitrb = bhitre;
    ++bhitrb;
  }
}
} // namespace ofdm
} // namespace bamradio
