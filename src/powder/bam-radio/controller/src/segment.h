// -*- c++ -*-
//  Copyright © 2017-2018 Stephen Larew

#ifndef dcsse2ee0254120
#define dcsse2ee0254120

#include "dll_types.h"
#include "json.hpp"
#include <boost/asio.hpp>
#include <chrono>
#include <cstdint>
#include <memory>

namespace bamradio {

namespace dll {

/// Enumeration of segment types
enum class SegmentType : uint8_t {
  None = 0,
  IPv4,
  ARQIPv4,
  Control,
  PSD,
  NUM_TYPES
};

/// A linked list element containing a sequence of bytes of known type & length.
class Segment {
public:
  typedef std::shared_ptr<Segment> sptr;

  /// Length in bytes of segment.
  virtual size_t length() const = 0;

  /// Type of this segment.
  virtual SegmentType type() const = 0;

  /// True if data should be checksummed before use.
  virtual bool checksummed() const = 0;

  /// True if data wants reliability.
  ///
  /// NB: Reliability only refers to notifying transmitter of failures.
  virtual bool reliable() const = 0;

  /// Source time of segment.
  virtual std::chrono::system_clock::time_point sourceTime() const = 0;

  /// NodeID for which this segment is destined.
  virtual NodeID destNodeID() const;

  virtual void setDestNodeID(NodeID n);

  /// Flow UID (individual mandate identifier in SC2).
  virtual FlowUID flowUID() const;

  /// Returns buffer with contents of this segment.
  virtual std::vector<boost::asio::const_buffer> rawContentsBuffer() const;

  /// Returns buffer for segments following this segment.
  // virtual boost::asio::const_buffer nextSegmentsBuffer() const;

  /// Returns buffer for all segments starting with this one.
  // virtual boost::asio::const_buffer rawSegmentsBuffer() const;

  virtual FlowID flowID() const = 0;

  virtual size_t packetLength() const = 0;

  virtual ~Segment(){};

protected:
  Segment(SegmentType t, NodeID destNodeID, boost::asio::const_buffer b);

  NodeID _destNodeID;
  boost::asio::const_buffer _segments;
};

class BufferTooShort : public std::exception {
public:
  char const *what() const noexcept { return "Segment buffer too short."; }
};

class NullSegment : public Segment {
public:
  NullSegment()
      : Segment(SegmentType::None, UnspecifiedNodeID,
                boost::asio::const_buffer()) {}
  size_t length() const { return 0; }
  SegmentType type() const { return SegmentType::None; }
  bool checksummed() const { return false; }
  bool reliable() const { return false; }
  FlowID flowID() const { return FlowID{}; }
};

void to_json(nlohmann::json &j, Segment const &p);
void to_json(nlohmann::json &j, Segment::sptr const &p);

} // namespace dll

namespace net {

class InvalidIPv4SegmentLength : public std::exception {
public:
  virtual char const *what() const noexcept {
    return "Invalid IPv4 segment length.";
  }
};

class InvalidUDPLength : public InvalidIPv4SegmentLength {
public:
  char const *what() const noexcept { return "Invalid UDP length."; }
};

class InvalidTCPLength : public InvalidIPv4SegmentLength {
public:
  char const *what() const noexcept { return "Invalid TCP length."; }
};

class IP4PacketSegment : public dll::Segment {
protected:
  typedef std::chrono::nanoseconds source_time_duration;
  source_time_duration::rep _sourceTime;
  std::vector<uint8_t> _zeroSuffix;

  // contructors when subclassing
  IP4PacketSegment(NodeID destNodeID, boost::asio::const_buffer b,
                   std::chrono::system_clock::time_point sourceTime,
                   dll::SegmentType t);
  IP4PacketSegment(NodeID destNodeID, boost::asio::const_buffer b,
                   dll::SegmentType t);

public:
  typedef std::shared_ptr<IP4PacketSegment> sptr;

  IP4PacketSegment(NodeID destNodeID, boost::asio::const_buffer b,
                   std::chrono::system_clock::time_point sourceTime);
  IP4PacketSegment(NodeID destNodeID, boost::asio::const_buffer b);
  virtual ~IP4PacketSegment(){};

  // implementing abstract header interface
  virtual size_t length() const;
  virtual dll::SegmentType type() const { return dll::SegmentType::IPv4; }
  virtual bool checksummed() const { return true; }
  virtual bool reliable() const { return true; }
  virtual std::chrono::system_clock::time_point sourceTime() const;
  virtual std::vector<boost::asio::const_buffer> rawContentsBuffer() const;
  virtual FlowUID flowUID() const;
  virtual FlowID flowID() const;
  virtual size_t packetLength() const;

  // overriden by subclasses
  virtual boost::asio::const_buffer packetContentsBuffer() const;
  virtual void const *ipHeaderAddress() const;

  uint8_t protocol() const;
  void setProtocol(uint8_t proto);
  std::chrono::system_clock::duration
  currentDelay(std::chrono::system_clock::time_point now =
                   std::chrono::system_clock::now()) const;
  uint8_t priority() const;
  boost::asio::ip::address_v4 srcAddr() const;
  boost::asio::ip::address_v4 dstAddr() const;
  uint16_t srcPort() const;
  void setSrcPort(uint16_t port);
  uint16_t dstPort() const;
  void setDstPort(uint16_t port);
};

void to_json(nlohmann::json &j, IP4PacketSegment const &p);

class ARQIP4PacketSegment : public net::IP4PacketSegment {
public:
  typedef std::shared_ptr<ARQIP4PacketSegment> sptr;

  // extra metadata that is transmitter over-the-air for ARQ purposes
  struct ARQData {
    uint32_t arqExtra; // bytes_remaining if FilePT, min_seqnum if StreamPT
    uint16_t seqNum;   // sequence number
    uint8_t burstNum;  // if bursty traffic, we track bursts
    constexpr static auto UnknownFileSize =
        std::numeric_limits<decltype(arqExtra)>::max();
    constexpr static auto MaxFileSize = UnknownFileSize - 1;
    inline bool fileSizeUnknown() const { return arqExtra == UnknownFileSize; }
  } __attribute__((packed));

  // tors
  ARQIP4PacketSegment(NodeID destNodeID, boost::asio::const_buffer b,
                      std::chrono::system_clock::time_point sourceTime);
  ARQIP4PacketSegment(NodeID destNodeID, boost::asio::const_buffer b);
  ARQIP4PacketSegment(IP4PacketSegment const &iseg);
  virtual ~ARQIP4PacketSegment(){};

  // implementing abstract interface
  virtual size_t length() const;
  virtual dll::SegmentType type() const { return dll::SegmentType::ARQIPv4; }
  virtual std::chrono::system_clock::time_point sourceTime() const;
  virtual std::vector<boost::asio::const_buffer> rawContentsBuffer() const;
  virtual size_t packetLength() const;

  // overriding non-ARQ IP4PacketSegment
  virtual void const *ipHeaderAddress() const;
  virtual boost::asio::const_buffer packetContentsBuffer() const;

  // metadata handling
  bool arqDataSet() const { return _arqDataSet; }
  ARQData arqData() const;
  void setArqData(ARQData ad);

private:
  ARQData _arqData;
  bool _arqDataSet;

  size_t _sizeofMetadata() const {
    return sizeof(_sourceTime) + sizeof(_arqData);
  }
  bool _metaDataEmbedded() const { return _sourceTime == 0; }
  void const *_sourceTimeAddress() const;
  void const *_arqDataAddress() const;
  void _resizeToMin();
};

bool operator<(ARQIP4PacketSegment::ARQData const &lhs,
               ARQIP4PacketSegment::ARQData const &rhs);

void to_json(nlohmann::json &j, ARQIP4PacketSegment::ARQData const &ad);
void to_json(nlohmann::json &j, ARQIP4PacketSegment const &p);

} // namespace net

namespace controlchannel {
class ControlChannelSegment : public dll::Segment {
public:
  typedef std::shared_ptr<ControlChannelSegment> sptr;

  ControlChannelSegment(boost::asio::const_buffer b,
                        std::chrono::system_clock::time_point sourceTime);
  ControlChannelSegment(boost::asio::const_buffer b);
  boost::asio::const_buffer packetContentsBuffer() const;
  size_t length() const;
  size_t packetLength() const;
  dll::SegmentType type() const { return dll::SegmentType::Control; }
  bool checksummed() const { return true; }
  bool reliable() const { return false; }
  std::chrono::system_clock::time_point sourceTime() const;
  std::vector<boost::asio::const_buffer> rawContentsBuffer() const;
  FlowID flowID() const;

private:
  typedef std::chrono::nanoseconds source_time_duration;
  source_time_duration::rep _sourceTime;
  std::vector<uint8_t> _zeroSuffix;
};
} // namespace controlchannel

namespace psdsensing {
class PSDSegment : public dll::Segment {
public:
  typedef std::shared_ptr<PSDSegment> sptr;

  PSDSegment(NodeID destNodeID, boost::asio::const_buffer b,
             std::chrono::system_clock::time_point sourceTime);
  PSDSegment(NodeID destNodeID, boost::asio::const_buffer b);
  boost::asio::const_buffer packetContentsBuffer() const;
  size_t length() const;
  size_t packetLength() const;
  dll::SegmentType type() const { return dll::SegmentType::PSD; }
  bool checksummed() const { return true; }
  bool reliable() const { return false; }
  std::chrono::system_clock::time_point sourceTime() const;
  std::vector<boost::asio::const_buffer> rawContentsBuffer() const;
  FlowID flowID() const;
  NodeID finalDestNodeID() const;

private:
  typedef std::chrono::nanoseconds source_time_duration;
  source_time_duration::rep _sourceTime;
  std::vector<uint8_t> _zeroSuffix;
};
} // namespace psdsensing

} // namespace bamradio

#endif
