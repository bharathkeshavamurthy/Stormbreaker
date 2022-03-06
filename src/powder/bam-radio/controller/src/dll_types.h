// -*- c++ -*-
//  Copyright © 2017-2018 Stephen Larew

#ifndef aa3f1bd01f8878f626
#define aa3f1bd01f8878f626

#include "json.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace bamradio {
namespace dll {
class Segment;
}

/// Local network address type
typedef uint8_t NodeID;

extern NodeID const AllNodesID;
extern NodeID const ExtNodeID;
extern NodeID const UnspecifiedNodeID;

int constexpr SeqNumBitlength = 10;
int constexpr BlockNumBitlength = 6;
int constexpr MaxNumBlocks = 1 << BlockNumBitlength;
int constexpr MaxNumARQFeedback = 4;

/// Flow UID (dst port)
typedef uint16_t FlowUID;

/// Flow identifier
struct FlowID {
  uint32_t srcIP, dstIP; // host byte order
  uint32_t proto;
  uint16_t srcPort, dstPort;

  inline FlowUID flowUID() const { return dstPort; }
  NodeID dstIPNodeID() const;

  enum Protocol : decltype(proto) {
    TCP = 6,
    UDP = 17,
    IN_MAX = 255,
    PSD,
    Control
  };
  inline bool mandated() const {
    return proto == (uint32_t)Protocol::UDP && dstPort >= 5000;
  }
  std::string description() const;
  operator std::string() const { return description(); }
  std::string srcIPString() const;
  std::string dstIPString() const;
};
void to_json(nlohmann::json &j, FlowID const &ei);

bool operator<(FlowID const &lhs, FlowID const &rhs);
bool operator==(FlowID const &lhs, FlowID const &rhs);
bool operator!=(FlowID const &lhs, FlowID const &rhs);

struct QueuedSegment {
  std::shared_ptr<dll::Segment> segment;
  std::shared_ptr<std::vector<uint8_t>> backingStore;
  bool valid() const { return segment && backingStore; }
};
namespace dll {
struct ARQBurstInfo {
  FlowUID flow_uid;
  uint8_t burst_num;
  uint16_t seq_num;
  uint32_t extra; 
};
} // namespace dll
} // namespace bamradio

#endif
