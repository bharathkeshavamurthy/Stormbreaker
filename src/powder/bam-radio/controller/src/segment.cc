// -*- c++ -*-
//  Copyright © 2017-2018 Stephen Larew

#include "segment.h"
#include "events.h"

#include <chrono>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>

namespace bamradio {

namespace asio = boost::asio;

inline uint64_t roundNumberUp(uint64_t dividend, uint64_t divisor) {
  return (dividend + (divisor - 1)) / divisor;
}

namespace dll {

Segment::Segment(SegmentType t, NodeID destNodeID, boost::asio::const_buffer b)
    : _destNodeID(destNodeID), _segments(b) {
  assert(t >= SegmentType::None && t < SegmentType::NUM_TYPES);
}

NodeID Segment::destNodeID() const { return _destNodeID; }
void Segment::setDestNodeID(NodeID n) { _destNodeID = n; }

FlowUID Segment::flowUID() const { return 0; }

std::vector<asio::const_buffer> Segment::rawContentsBuffer() const {
  if (length() > asio::buffer_size(_segments)) {
    throw BufferTooShort();
  }
  return {asio::buffer(_segments, length())};
}

void to_json(nlohmann::json &j, Segment const &p) {
  j = nlohmann::json{{"dstNodeID", p.destNodeID()},
                     {"length", p.length()},
                     {"type", (int)p.type()}};
}

void to_json(nlohmann::json &j, Segment::sptr const &p) {
  switch (p->type()) {
  case SegmentType::IPv4: {
    to_json(j, *std::dynamic_pointer_cast<net::IP4PacketSegment>(p));
    break;
  }
  case SegmentType::ARQIPv4: {
    to_json(j, *std::dynamic_pointer_cast<net::ARQIP4PacketSegment>(p));
    break;
  }
  case SegmentType::Control: {
    to_json(
        j,
        *std::dynamic_pointer_cast<controlchannel::ControlChannelSegment>(p));
    break;
  }
  case SegmentType::PSD: {
    to_json(j, *std::dynamic_pointer_cast<psdsensing::PSDSegment>(p));
    break;
  }
  default: { throw; }
  }
}

} // namespace dll

namespace net {

enum : size_t { IP4_HEADER_MIN_LENGTH = 20 };

IP4PacketSegment::IP4PacketSegment(
    NodeID destNodeID, boost::asio::const_buffer b,
    std::chrono::system_clock::time_point sourceTime)
    : dll::Segment(dll::SegmentType::IPv4, destNodeID, b),
      _sourceTime(source_time_duration(sourceTime.time_since_epoch()).count()) {
  if (_sourceTime == 0) {
    throw std::invalid_argument("sourceTime cannot be 0");
  }
  if (asio::buffer_size(_segments) < net::IP4_HEADER_MIN_LENGTH) {
    throw dll::BufferTooShort();
  }
  if (asio::buffer_size(_segments) < packetLength()) {
    throw dll::BufferTooShort();
  }
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}
IP4PacketSegment::IP4PacketSegment(NodeID destNodeID,
                                   boost::asio::const_buffer b)
    : dll::Segment(dll::SegmentType::IPv4, destNodeID, b), _sourceTime(0) {
  if (asio::buffer_size(_segments) <
      (net::IP4_HEADER_MIN_LENGTH + sizeof(_sourceTime))) {
    throw dll::BufferTooShort();
  }
  if (asio::buffer_size(_segments) < length()) {
    throw dll::BufferTooShort();
  }
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}

size_t IP4PacketSegment::length() const {
  return sizeof(_sourceTime) + packetLength() + _zeroSuffix.size();
}

void const *IP4PacketSegment::ipHeaderAddress() const {
  return (void const *)(asio::buffer_cast<uint8_t const *>(_segments) +
                        (_sourceTime == 0 ? sizeof(_sourceTime) : 0));
}

std::chrono::system_clock::time_point IP4PacketSegment::sourceTime() const {
  decltype(_sourceTime) t;
  memcpy(&t, asio::buffer_cast<uint8_t const *>(_segments), sizeof(t));
  return std::chrono::system_clock::time_point(
      source_time_duration(_sourceTime == 0 ? t : _sourceTime));
}

std::chrono::system_clock::duration IP4PacketSegment::currentDelay(
    std::chrono::system_clock::time_point now) const {
  return now - sourceTime();
}

size_t IP4PacketSegment::packetLength() const {
  auto const hdr = (struct iphdr const *)ipHeaderAddress();
  auto const l = ntohs(hdr->tot_len);
  if (l < net::IP4_HEADER_MIN_LENGTH ||
      l > (asio::buffer_size(_segments) -
           (_sourceTime == 0 ? sizeof(_sourceTime) : 0))) {
    throw InvalidIPv4SegmentLength();
  }
  return l;
}

uint8_t IP4PacketSegment::priority() const {
  auto const hdr = (struct iphdr const *)ipHeaderAddress();
  return (hdr->tos & 0xf0) >> 4;
}

uint8_t IP4PacketSegment::protocol() const {
  auto const hdr = (struct iphdr const *)ipHeaderAddress();
  return hdr->protocol;
}

void IP4PacketSegment::setProtocol(uint8_t proto) {
  auto const hdr = (struct iphdr *)ipHeaderAddress();
  hdr->protocol = proto;
}

asio::ip::address_v4 IP4PacketSegment::srcAddr() const {
  auto const hdr = (struct iphdr const *)ipHeaderAddress();
  return asio::ip::address_v4(ntohl(hdr->saddr));
}

asio::ip::address_v4 IP4PacketSegment::dstAddr() const {
  auto const hdr = (struct iphdr const *)ipHeaderAddress();
  return asio::ip::address_v4(ntohl(hdr->daddr));
}

uint16_t IP4PacketSegment::srcPort() const {
  auto const hdr = (struct ip const *)ipHeaderAddress();
  auto const ip4hdrlength = hdr->ip_hl * 4;
  assert(ip4hdrlength >= 20);
  assert(ip4hdrlength <= 60);
  auto const proto = hdr->ip_p;
  bool const fragment =
      (hdr->ip_off & htons(IP_MF)) ||          // More fragments set
      ((hdr->ip_off & htons(IP_OFFMASK)) > 0); // positive frag offset
  assert(!fragment);
  if (proto == IPPROTO_UDP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(udphdr)) {
      throw InvalidUDPLength();
    }
    auto const l4hdr = (udphdr const *)((uint8_t const *)hdr + ip4hdrlength);
    return ntohs(l4hdr->uh_sport);
  } else if (proto == IPPROTO_TCP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(tcphdr)) {
      throw InvalidTCPLength();
    }
    auto const l4hdr = (tcphdr const *)((uint8_t const *)hdr + ip4hdrlength);
    return ntohs(l4hdr->th_sport);
  }
  return 0;
}

void IP4PacketSegment::setSrcPort(uint16_t port) {
  auto const hdr = (struct ip *)ipHeaderAddress();
  auto const ip4hdrlength = hdr->ip_hl * 4;
  assert(ip4hdrlength >= 20);
  assert(ip4hdrlength <= 60);
  auto const proto = hdr->ip_p;
  bool const fragment =
      (hdr->ip_off & htons(IP_MF)) ||          // More fragments set
      ((hdr->ip_off & htons(IP_OFFMASK)) > 0); // positive frag offset
  assert(!fragment);
  if (proto == IPPROTO_UDP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(udphdr)) {
      throw InvalidUDPLength();
    }
    auto const l4hdr = (udphdr *)((uint8_t *)hdr + ip4hdrlength);
    l4hdr->uh_sport = htons(port);
  } else if (proto == IPPROTO_TCP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(tcphdr)) {
      throw InvalidTCPLength();
    }
    auto const l4hdr = (tcphdr *)((uint8_t *)hdr + ip4hdrlength);
    l4hdr->th_sport = htons(port);
  } else {
    throw std::runtime_error("IPv4 packet is neither UDP nor TCP");
  }
}

uint16_t IP4PacketSegment::dstPort() const {
  auto const hdr = (struct ip const *)ipHeaderAddress();
  auto const ip4hdrlength = hdr->ip_hl * 4;
  assert(ip4hdrlength >= 20);
  assert(ip4hdrlength <= 60);
  auto const proto = hdr->ip_p;
  bool const fragment =
      (hdr->ip_off & htons(IP_MF)) ||          // More fragments set
      ((hdr->ip_off & htons(IP_OFFMASK)) > 0); // positive frag offset
  assert(!fragment);
  if (proto == IPPROTO_UDP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(udphdr)) {
      throw InvalidUDPLength();
    }
    auto const l4hdr = (udphdr const *)((uint8_t const *)hdr + ip4hdrlength);
    return ntohs(l4hdr->uh_dport);
  } else if (proto == IPPROTO_TCP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(tcphdr)) {
      throw InvalidTCPLength();
    }
    auto const l4hdr = (tcphdr const *)((uint8_t const *)hdr + ip4hdrlength);
    return ntohs(l4hdr->th_dport);
  }
  return 0;
}

void IP4PacketSegment::setDstPort(uint16_t port) {
  auto const hdr = (struct ip *)ipHeaderAddress();
  auto const ip4hdrlength = hdr->ip_hl * 4;
  assert(ip4hdrlength >= 20);
  assert(ip4hdrlength <= 60);
  auto const proto = hdr->ip_p;
  bool const fragment =
      (hdr->ip_off & htons(IP_MF)) ||          // More fragments set
      ((hdr->ip_off & htons(IP_OFFMASK)) > 0); // positive frag offset
  assert(!fragment);
  if (proto == IPPROTO_UDP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(udphdr)) {
      throw InvalidUDPLength();
    }
    auto const l4hdr = (udphdr *)((uint8_t *)hdr + ip4hdrlength);
    l4hdr->uh_dport = htons(port);
  } else if (proto == IPPROTO_TCP && !fragment) {
    if (packetLength() < ip4hdrlength + sizeof(tcphdr)) {
      throw InvalidTCPLength();
    }
    auto const l4hdr = (tcphdr *)((uint8_t *)hdr + ip4hdrlength);
    l4hdr->th_dport = htons(port);
  } else {
    throw std::runtime_error("IPv4 packet is neither UDP nor TCP");
  }
}

FlowUID IP4PacketSegment::flowUID() const { return this->dstPort(); }

asio::const_buffer IP4PacketSegment::packetContentsBuffer() const {
  if (_sourceTime == 0) {
    return asio::buffer(_segments + sizeof(_sourceTime), packetLength());
  } else {
    return asio::buffer(_segments, packetLength());
  }
}

std::vector<asio::const_buffer> IP4PacketSegment::rawContentsBuffer() const {
  if (_sourceTime == 0) {
    return {asio::buffer(_segments, sizeof(_sourceTime) + packetLength()),
            asio::buffer(_zeroSuffix)};
  } else {
    return {asio::buffer(&_sourceTime, sizeof(_sourceTime)),
            asio::buffer(_segments, packetLength()), asio::buffer(_zeroSuffix)};
  }
}

FlowID IP4PacketSegment::flowID() const {
  return FlowID{srcAddr().to_uint(), dstAddr().to_uint(), protocol(), srcPort(),
                dstPort()};
}

void to_json(nlohmann::json &j, IP4PacketSegment const &p) {
  to_json(j, *((dll::Segment const *)&p));
  j["sourceTime"] = p.sourceTime().time_since_epoch().count();
  j["packetLength"] = p.packetLength();
  j["priority"] = p.priority();
  j["srcAddr"] = p.srcAddr().to_string();
  j["dstAddr"] = p.dstAddr().to_string();
  j["srcPort"] = p.srcPort();
  j["dstPort"] = p.dstPort();
  j["protocol"] = p.protocol();
  j["currentDelay"] = p.currentDelay().count();
}

// *special* constructors for the IP4PacketSegment class when called from a
// subclass
IP4PacketSegment::IP4PacketSegment(
    NodeID destNodeID, boost::asio::const_buffer b,
    std::chrono::system_clock::time_point sourceTime, dll::SegmentType t)
    : dll::Segment(t, destNodeID, b),
      _sourceTime(source_time_duration(sourceTime.time_since_epoch()).count()) {
}

IP4PacketSegment::IP4PacketSegment(NodeID destNodeID,
                                   boost::asio::const_buffer b,
                                   dll::SegmentType t)
    : dll::Segment(t, destNodeID, b), _sourceTime(0) {}

// ARQIP4Packet tors. tx, rx, or from IP4Segment

ARQIP4PacketSegment::ARQIP4PacketSegment(
    NodeID destNodeID, boost::asio::const_buffer b,
    std::chrono::system_clock::time_point sourceTime)
    : IP4PacketSegment(destNodeID, b, sourceTime, dll::SegmentType::ARQIPv4),
      _arqDataSet(false) {
  if (asio::buffer_size(_segments) < net::IP4_HEADER_MIN_LENGTH) {
    throw dll::BufferTooShort();
  }
  if (asio::buffer_size(_segments) < packetLength()) {
    throw dll::BufferTooShort();
  }
  _resizeToMin();
}

ARQIP4PacketSegment::ARQIP4PacketSegment(NodeID destNodeID,
                                         boost::asio::const_buffer b)
    : IP4PacketSegment(destNodeID, b, dll::SegmentType::ARQIPv4),
      _arqDataSet(true) {
  if (asio::buffer_size(_segments) <
      (net::IP4_HEADER_MIN_LENGTH + _sizeofMetadata())) {
    throw dll::BufferTooShort();
  }
  if (asio::buffer_size(_segments) < length()) {
    throw dll::BufferTooShort();
  }
  _resizeToMin();
}

ARQIP4PacketSegment::ARQIP4PacketSegment(IP4PacketSegment const &iseg)
    : IP4PacketSegment(iseg.destNodeID(),
                       *std::next(iseg.rawContentsBuffer().begin()),
                       iseg.sourceTime(), dll::SegmentType::ARQIPv4),
      _arqDataSet(false) {
  if (iseg.rawContentsBuffer().size() != 3) {
    unimplemented();
  }
  _resizeToMin();
}

void ARQIP4PacketSegment::_resizeToMin() {
  // FIXME I don't even know
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}

size_t ARQIP4PacketSegment::packetLength() const {
  auto const hdr = (struct iphdr const *)ipHeaderAddress();
  auto const l = ntohs(hdr->tot_len);
  return l;
}

size_t ARQIP4PacketSegment::length() const {
  return _sizeofMetadata() + packetLength() + _zeroSuffix.size();
}

ARQIP4PacketSegment::ARQData ARQIP4PacketSegment::arqData() const {
  return *((ARQData *)_arqDataAddress());
}

void ARQIP4PacketSegment::setArqData(
    bamradio::net::ARQIP4PacketSegment::ARQData ad) {
  *((ARQData *)_arqDataAddress()) = ad;
  _arqDataSet = true;
}

std::chrono::system_clock::time_point ARQIP4PacketSegment::sourceTime() const {
  return std::chrono::system_clock::time_point(
      source_time_duration(*((decltype(_sourceTime) *)_sourceTimeAddress())));
}

void const *ARQIP4PacketSegment::_sourceTimeAddress() const {
  if (_metaDataEmbedded()) {
    return asio::buffer_cast<uint8_t const *>(_segments);
  } else {
    return (void const *)&_sourceTime;
  }
}

void const *ARQIP4PacketSegment::_arqDataAddress() const {
  if (_metaDataEmbedded()) {
    return asio::buffer_cast<uint8_t const *>(_segments) + sizeof(_sourceTime);
  } else {
    return (void *const) & _arqData;
  }
}

void const *ARQIP4PacketSegment::ipHeaderAddress() const {
  if (_metaDataEmbedded()) {
    return asio::buffer_cast<uint8_t const *>(_segments) + _sizeofMetadata();
  } else {
    return asio::buffer_cast<uint8_t const *>(_segments);
  }
}

boost::asio::const_buffer ARQIP4PacketSegment::packetContentsBuffer() const {
  if (_metaDataEmbedded()) {
    return asio::buffer(_segments + _sizeofMetadata(), packetLength());
  } else {
    return asio::buffer(_segments, packetLength());
  }
}

std::vector<boost::asio::const_buffer>
ARQIP4PacketSegment::rawContentsBuffer() const {
  if (_metaDataEmbedded()) {
    return {asio::buffer(_segments, length()), asio::buffer(_zeroSuffix)};
  } else {
    return {asio::buffer(&_sourceTime, sizeof(_sourceTime)),
            asio::buffer(&_arqData, sizeof(_arqData)),
            asio::buffer(_segments, packetLength()), asio::buffer(_zeroSuffix)};
  }
}

bool operator<(ARQIP4PacketSegment::ARQData const &lhs,
               ARQIP4PacketSegment::ARQData const &rhs) {
  return std::forward_as_tuple(lhs.burstNum, lhs.seqNum, lhs.arqExtra) <
         std::forward_as_tuple(rhs.burstNum, rhs.seqNum, rhs.arqExtra);
}

void to_json(nlohmann::json &j, ARQIP4PacketSegment::ARQData const &ad) {
  j["arqExtra"] = ad.arqExtra;
  j["seqNum"] = ad.seqNum;
  j["burstNum"] = ad.burstNum;
}

void to_json(nlohmann::json &j, ARQIP4PacketSegment const &p) {
  to_json(j, *((net::IP4PacketSegment const *)&p));
  nlohmann::json jarq = p.arqData();
  j["arqData"] = jarq;
}

} // namespace net

namespace controlchannel {
ControlChannelSegment::ControlChannelSegment(
    boost::asio::const_buffer b,
    std::chrono::system_clock::time_point sourceTime)
    : dll::Segment(dll::SegmentType::Control, AllNodesID, b),
      _sourceTime(source_time_duration(sourceTime.time_since_epoch()).count()) {
  if (_sourceTime == 0) {
    throw std::invalid_argument("sourceTime cannot be 0");
  }
  if (asio::buffer_size(_segments) < packetLength()) {
    throw dll::BufferTooShort();
  }
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}

ControlChannelSegment::ControlChannelSegment(boost::asio::const_buffer b)
    : dll::Segment(dll::SegmentType::Control, AllNodesID, b), _sourceTime(0) {
  if (asio::buffer_size(_segments) < length()) {
    throw dll::BufferTooShort();
  }
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}

asio::const_buffer ControlChannelSegment::packetContentsBuffer() const {
  if (_sourceTime == 0) {
    return asio::buffer(_segments + sizeof(_sourceTime), packetLength());
  } else {
    return asio::buffer(_segments, packetLength());
  }
}

std::vector<asio::const_buffer>
ControlChannelSegment::rawContentsBuffer() const {
  if (_sourceTime == 0) {
    return {asio::buffer(_segments, sizeof(_sourceTime) + packetLength()),
            asio::buffer(_zeroSuffix)};
  } else {
    return {asio::buffer(&_sourceTime, sizeof(_sourceTime)),
            asio::buffer(_segments, packetLength()), asio::buffer(_zeroSuffix)};
  }
}

FlowID ControlChannelSegment::flowID() const {
  return FlowID{0, destNodeID(), FlowID::Protocol::Control, 0, 0};
}

size_t ControlChannelSegment::length() const {
  return sizeof(_sourceTime) + packetLength() + _zeroSuffix.size();
}

size_t ControlChannelSegment::packetLength() const {
  auto ptr = (asio::buffer_cast<uint8_t const *>(_segments) +
              (_sourceTime == 0 ? sizeof(_sourceTime) : 0));
  uint16_t payload_size;
  std::memcpy(&payload_size, ptr, sizeof(payload_size));
  return sizeof(payload_size) + payload_size;
}

std::chrono::system_clock::time_point
ControlChannelSegment::sourceTime() const {
  decltype(_sourceTime) t;
  memcpy(&t, asio::buffer_cast<uint8_t const *>(_segments), sizeof(t));
  return std::chrono::system_clock::time_point(
      source_time_duration(_sourceTime == 0 ? t : _sourceTime));
}
} // namespace controlchannel

namespace psdsensing {
PSDSegment::PSDSegment(NodeID destNodeID, boost::asio::const_buffer b,
                       std::chrono::system_clock::time_point sourceTime)
    : dll::Segment(dll::SegmentType::PSD, destNodeID, b),
      _sourceTime(source_time_duration(sourceTime.time_since_epoch()).count()) {
  if (_sourceTime == 0) {
    throw std::invalid_argument("sourceTime cannot be 0");
  }
  if (asio::buffer_size(_segments) < packetLength()) {
    throw dll::BufferTooShort();
  }
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}

PSDSegment::PSDSegment(NodeID destNodeID, boost::asio::const_buffer b)
    : dll::Segment(dll::SegmentType::PSD, destNodeID, b), _sourceTime(0) {
  if (asio::buffer_size(_segments) < length()) {
    throw dll::BufferTooShort();
  }
  auto const l = length();
  auto const minsegsizehack = roundNumberUp(1944 * 5, 6 * 8);
  if (l < minsegsizehack) {
    _zeroSuffix.resize(minsegsizehack - l);
  }
}

asio::const_buffer PSDSegment::packetContentsBuffer() const {
  if (_sourceTime == 0) {
    return asio::buffer(_segments + sizeof(_sourceTime), packetLength());
  } else {
    return asio::buffer(_segments, packetLength());
  }
}

std::vector<asio::const_buffer> PSDSegment::rawContentsBuffer() const {
  if (_sourceTime == 0) {
    return {asio::buffer(_segments, sizeof(_sourceTime) + packetLength()),
            asio::buffer(_zeroSuffix)};
  } else {
    return {asio::buffer(&_sourceTime, sizeof(_sourceTime)),
            asio::buffer(_segments, packetLength()), asio::buffer(_zeroSuffix)};
  }
}

FlowID PSDSegment::flowID() const {
  return FlowID{0, destNodeID(), FlowID::Protocol::PSD, 0, 0};
}

size_t PSDSegment::length() const {
  return sizeof(_sourceTime) + packetLength() + _zeroSuffix.size();
}

size_t PSDSegment::packetLength() const {
  auto ptr = (asio::buffer_cast<uint8_t const *>(_segments) +
              (_sourceTime == 0 ? sizeof(_sourceTime) : 0));
  uint16_t payload_size;
  std::memcpy(&payload_size, ptr, sizeof(payload_size));
  return sizeof(payload_size) + payload_size;
}

std::chrono::system_clock::time_point PSDSegment::sourceTime() const {
  decltype(_sourceTime) t;
  memcpy(&t, asio::buffer_cast<uint8_t const *>(_segments), sizeof(t));
  return std::chrono::system_clock::time_point(
      source_time_duration(_sourceTime == 0 ? t : _sourceTime));
}

NodeID PSDSegment::finalDestNodeID() const {
  auto ptr = (asio::buffer_cast<uint8_t const *>(_segments) +
              (_sourceTime == 0 ? sizeof(_sourceTime) : 0));
  NodeID node_id;
  std::memcpy(&node_id, ptr + sizeof(uint16_t), sizeof(NodeID));
  return node_id;
}

} // namespace psdsensing
} // namespace bamradio
