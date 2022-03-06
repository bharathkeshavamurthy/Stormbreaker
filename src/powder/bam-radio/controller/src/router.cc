// -*- c++ -*-
//  Copyright (c) 2017 Tomohiro Arakawa

#include "router.h"
#include "events.h"
#include "options.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <vector>

using namespace boost::asio;

namespace bamradio {
namespace net {

AbstractRouter::AbstractRouter(NodeID my_srnid) : _my_srnid(my_srnid) {}

void AbstractRouter::setTunInterface(tun::Interface::sptr iface) {
  _tun_dev = iface;
}

void AbstractRouter::setOFDMInterface(tun::Interface::sptr iface) {
  _ofdm_dev = iface;
}

//
// Multi-hop basic router
//

BasicRouter::BasicRouter(uint8_t my_srnid) : AbstractRouter(my_srnid) {}

void BasicRouter::updateRoutingTable(NetworkMap &netmap) {
  // clear routing table
  d_routing_table.clear();
  // get network graph from NetworkMap
  auto g = netmap.getGraph();
  auto v_me = netmap.getNode(_my_srnid);
  if (v_me != g.null_vertex()) {
    std::vector<NetworkMap::Vertex> parents(boost::num_vertices(g));
    boost::dijkstra_shortest_paths(g, v_me,
                                   boost::predecessor_map(&parents[0]));
#ifndef NDEBUG
    std::cout << "== Routes ==" << std::endl;
#endif

    typedef boost::graph_traits<NetworkMap::Graph>::vertex_iterator vertex_iter;
    std::pair<vertex_iter, vertex_iter> vp;
    for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
      // do not add route entry if the destination is myself
      if (*vp.first == v_me)
        continue;

      // check if the destination node is actually reachable
      if (parents[*vp.first] == *vp.first)
        continue;

      // get route to the destination
      std::deque<uint8_t> route;
      for (NetworkMap::Vertex v = *vp.first; v != v_me; v = parents[v])
        route.push_front(g[v]);
      // add routing entry
      uint8_t dst_srnid = g[*vp.first];
      d_routing_table[dst_srnid] = route.front();

#ifndef NDEBUG
      std::cout << "Src = " << (int)_my_srnid;
      for (auto const v : route) {
        std::cout << " -> " << (int)v;
      }
      std::cout << " = Dst" << std::endl;
#endif
    }
  }
  // output routing table
  NotificationCenter::shared.post(RoutingTableUpdateEvent,
                                  RoutingTableUpdateEventInfo{d_routing_table});
}

std::pair<AbstractInterface::sptr, uint8_t>
BasicRouter::doRouting(dll::Segment::sptr segment, NodeID src_srnid) {

  auto const unknown_ip = boost::asio::ip::address_v4::any();

  // Segment type
  auto const seg_type = segment->type();

  // try to get IP4PacketSegment::sptr
  auto ipseg = std::dynamic_pointer_cast<IP4PacketSegment>(segment);

  // Get the final destination node ID
  // FIXME use Segment::flowId().dstIPNodeID()
  NodeID dst_srnid;
  switch (seg_type) {
  case dll::SegmentType::IPv4: // FALLTHROUGH
  case dll::SegmentType::ARQIPv4: {
    if (ipseg->flowUID() >= 5000) {
      // All scored traffic have FlowUID >= 5000 according to the official
      // FAQ Q25.
      auto ip = htonl(ipseg->dstAddr().to_ulong());
      dst_srnid = ((ip & 0x00ff0000) >> 16) - 100;
    } else {
      dst_srnid = UnspecifiedNodeID;
    }
    break;
  }
  case dll::SegmentType::PSD: {
    dst_srnid = std::dynamic_pointer_cast<psdsensing::PSDSegment>(segment)
                    ->finalDestNodeID();
    break;
  }
  case dll::SegmentType::Control: {
    log::text("CC segment reached the router. Something's not right...");
    //[[fallthrough]];
  }
  default: {
    dst_srnid = UnspecifiedNodeID;
    break;
  }
  }

  // drop packet if destination is not one of our SRNs
  if (dst_srnid < 1 || dst_srnid > 128) {
    if (ipseg) {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{
              RouterAction::DROP_UNKNOWN_PACKET, src_srnid, UnspecifiedNodeID,
              seg_type, ipseg->srcAddr(), ipseg->dstAddr(), ipseg->srcPort(),
              ipseg->dstPort(), ipseg->protocol(), segment->packetLength(),
              segment->sourceTime()});
    } else {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{RouterAction::DROP_UNKNOWN_PACKET, src_srnid,
                                 UnspecifiedNodeID, seg_type, unknown_ip,
                                 unknown_ip, 0, 0, 0, segment->packetLength(),
                                 segment->sourceTime()});
    }
    return std::pair<AbstractInterface::sptr, uint8_t>(nullptr,
                                                       UnspecifiedNodeID);
  } else if (_my_srnid == dst_srnid) {
    if (ipseg) {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{
              RouterAction::WRITE_TO_TUN, src_srnid, ExtNodeID, seg_type,
              ipseg->srcAddr(), ipseg->dstAddr(), ipseg->srcPort(),
              ipseg->dstPort(), ipseg->protocol(), segment->packetLength(),
              segment->sourceTime()});
    } else {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{RouterAction::WRITE_TO_TUN, src_srnid,
                                 ExtNodeID, seg_type, unknown_ip, unknown_ip, 0,
                                 0, 0, segment->packetLength(),
                                 segment->sourceTime()});
    }
    return std::pair<AbstractInterface::sptr, uint8_t>(_tun_dev, ExtNodeID);
  } else {
    // find next hop from routing table
    auto itr = d_routing_table.find(dst_srnid);
    // send packet to the destination directly if no route exists
    if (itr == d_routing_table.end()) {
      if (ipseg) {
        NotificationCenter::shared.post(
            RouteDecisionEvent,
            RouteDecisionEventInfo{RouterAction::FORWARD, src_srnid, dst_srnid,
                                   seg_type, ipseg->srcAddr(), ipseg->dstAddr(),
                                   ipseg->srcPort(), ipseg->dstPort(),
                                   ipseg->protocol(), segment->packetLength(),
                                   segment->sourceTime()});
      } else {
        NotificationCenter::shared.post(
            RouteDecisionEvent,
            RouteDecisionEventInfo{RouterAction::FORWARD, src_srnid, dst_srnid,
                                   seg_type, unknown_ip, unknown_ip, 0, 0, 0,
                                   segment->packetLength(),
                                   segment->sourceTime()});
      }
      return std::pair<AbstractInterface::sptr, uint8_t>(_ofdm_dev, dst_srnid);
    } else {
      // forward packet to next hop
      if (ipseg) {
        NotificationCenter::shared.post(
            RouteDecisionEvent,
            RouteDecisionEventInfo{
                RouterAction::FORWARD, src_srnid, itr->second, seg_type,
                ipseg->srcAddr(), ipseg->dstAddr(), ipseg->srcPort(),
                ipseg->dstPort(), ipseg->protocol(), segment->packetLength(),
                segment->sourceTime()});
      } else {
        NotificationCenter::shared.post(
            RouteDecisionEvent,
            RouteDecisionEventInfo{RouterAction::FORWARD, src_srnid,
                                   itr->second, seg_type, unknown_ip,
                                   unknown_ip, 0, 0, 0, segment->packetLength(),
                                   segment->sourceTime()});
      }
      return std::pair<AbstractInterface::sptr, uint8_t>(_ofdm_dev,
                                                         itr->second);
    }
  }
}

//
// NoRouter (no multi-hop)
//

NoRouter::NoRouter(uint8_t my_node_id) : AbstractRouter(my_node_id) {}

std::pair<AbstractInterface::sptr, uint8_t>
NoRouter::doRouting(dll::Segment::sptr segment, NodeID src_srnid) {

  auto const unknown_ip = boost::asio::ip::address_v4::any();

  // Segment type
  auto const seg_type = segment->type();

  // try to get IP4PacketSegment::sptr
  auto ipseg = std::dynamic_pointer_cast<IP4PacketSegment>(segment);

  // Get the final destination node ID
  // FIXME use Segment::flowId().dstIPNodeID()
  NodeID dst_srnid;
  switch (seg_type) {
  case dll::SegmentType::IPv4: // FALLTHROUGH
  case dll::SegmentType::ARQIPv4: {
    if (ipseg->flowUID() >= 5000) {
      // All scored traffic have FlowUID >= 5000 according to the official
      // FAQ Q25.
      auto ip = htonl(ipseg->dstAddr().to_ulong());
      dst_srnid = ((ip & 0x00ff0000) >> 16) - 100;
    } else {
      dst_srnid = UnspecifiedNodeID;
    }
    break;
  }
  case dll::SegmentType::PSD: {
    dst_srnid = std::dynamic_pointer_cast<psdsensing::PSDSegment>(segment)
                    ->finalDestNodeID();
    break;
  }
  case dll::SegmentType::Control: {
    log::text("CC segment reached the router. Something's not right...");
    //[[fallthrough]];
  }
  default: {
    dst_srnid = UnspecifiedNodeID;
    break;
  }
  }

  // drop packet if destination is not one of our SRNs
  if (dst_srnid < 1 || dst_srnid > 128) {
    if (ipseg) {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{
              RouterAction::DROP_UNKNOWN_PACKET, src_srnid, UnspecifiedNodeID,
              seg_type, ipseg->srcAddr(), ipseg->dstAddr(), ipseg->srcPort(),
              ipseg->dstPort(), ipseg->protocol(), segment->packetLength(),
              segment->sourceTime()});
    } else {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{RouterAction::DROP_UNKNOWN_PACKET, src_srnid,
                                 UnspecifiedNodeID, seg_type, unknown_ip,
                                 unknown_ip, 0, 0, 0, segment->packetLength(),
                                 segment->sourceTime()});
    }
    return std::pair<AbstractInterface::sptr, uint8_t>(nullptr,
                                                       UnspecifiedNodeID);
  } else if (_my_srnid == dst_srnid) {
    if (ipseg) {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{
              RouterAction::WRITE_TO_TUN, src_srnid, ExtNodeID, seg_type,
              ipseg->srcAddr(), ipseg->dstAddr(), ipseg->srcPort(),
              ipseg->dstPort(), ipseg->protocol(), segment->packetLength(),
              segment->sourceTime()});
    } else {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{RouterAction::WRITE_TO_TUN, src_srnid,
                                 ExtNodeID, seg_type, unknown_ip, unknown_ip, 0,
                                 0, 0, segment->packetLength(),
                                 segment->sourceTime()});
    }
    return std::pair<AbstractInterface::sptr, uint8_t>(_tun_dev, ExtNodeID);
  } else {
    // send packet to the destination directly if no route exists
    if (ipseg) {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{RouterAction::FORWARD, src_srnid, dst_srnid,
                                 seg_type, ipseg->srcAddr(), ipseg->dstAddr(),
                                 ipseg->srcPort(), ipseg->dstPort(),
                                 ipseg->protocol(), segment->packetLength(),
                                 segment->sourceTime()});
    } else {
      NotificationCenter::shared.post(
          RouteDecisionEvent,
          RouteDecisionEventInfo{RouterAction::FORWARD, src_srnid, dst_srnid,
                                 seg_type, unknown_ip, unknown_ip, 0, 0, 0,
                                 segment->packetLength(),
                                 segment->sourceTime()});
    }
    return std::pair<AbstractInterface::sptr, uint8_t>(_ofdm_dev, dst_srnid);
  }
}

} // namespace net
} // namespace bamradio
