/* -*-c++-*-
 * Copyright © 2017-2018 Stephen Larew
 */

#include "dll_types.h"
#include <boost/asio.hpp>
#include <boost/format.hpp>

namespace bamradio {

NodeID const AllNodesID = 255;
NodeID const ExtNodeID = 254;
NodeID const UnspecifiedNodeID = 253;

void to_json(nlohmann::json &j, FlowID const &ei) {
  j["srcIP"] = boost::asio::ip::address_v4(ei.srcIP).to_string();
  j["dstIP"] = boost::asio::ip::address_v4(ei.dstIP).to_string();
  j["protocol"] = ei.proto;
  j["srcPort"] = ei.srcPort;
  j["dstPort"] = ei.dstPort;
}

bool operator<(FlowID const &lhs, FlowID const &rhs) {
  // Most likely to differ: dstPort (mandated flows)
  return std::forward_as_tuple(lhs.dstPort, lhs.proto, lhs.srcIP, lhs.dstIP,
                               lhs.srcPort) <
         std::forward_as_tuple(rhs.dstPort, rhs.proto, rhs.srcIP, rhs.dstIP,
                               rhs.srcPort);
}

bool operator==(FlowID const &lhs, FlowID const &rhs) {
  return lhs.srcIP == rhs.srcIP && lhs.dstIP == rhs.dstIP &&
         lhs.proto == rhs.proto && lhs.srcPort == rhs.srcPort &&
         lhs.dstPort == rhs.dstPort;
}

bool operator!=(FlowID const &lhs, FlowID const &rhs) {
  return lhs.srcIP != rhs.srcIP || lhs.dstIP != rhs.dstIP ||
         lhs.proto != rhs.proto || lhs.srcPort != rhs.srcPort ||
         lhs.dstPort != rhs.dstPort;
}

std::string FlowID::description() const {
  return (boost::format("%s:%u-[%u]->%s:%u") %
          boost::asio::ip::address_v4(srcIP).to_string() % srcPort % proto %
          boost::asio::ip::address_v4(dstIP).to_string() % dstPort)
      .str();
}

NodeID FlowID::dstIPNodeID() const {
  if (proto > Protocol::IN_MAX) {
    return dstIP;
  }
  auto const srnmask = inet_addr("0.0.255.0");
  auto const dst = htonl(dstIP);
  NodeID const dst_srnid = (ntohl(dst & srnmask) >> 8) - 100;
  return dst_srnid;
}

std::string FlowID::srcIPString() const {
  return boost::asio::ip::address_v4(srcIP).to_string();
}
std::string FlowID::dstIPString() const {
  return boost::asio::ip::address_v4(dstIP).to_string();
}
} // namespace bamradio
