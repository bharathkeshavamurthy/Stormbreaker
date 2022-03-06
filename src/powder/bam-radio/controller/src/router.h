// -*- c++ -*-
//  Copyright (c) 2017 Tomohiro Arakawa

#ifndef CONTROLLER_SRC_ROUTER_H_
#define CONTROLLER_SRC_ROUTER_H_

#include "dll_types.h"
#include "interface.h"
#include "json.hpp"
#include "networkmap.h"
#include "notify.h"
#include "segment.h"

#include <chrono>
#include <map>
#include <tuple>

namespace bamradio {
namespace net {

class AbstractRouter {
public:
  typedef std::shared_ptr<AbstractRouter> sptr;
  virtual std::pair<AbstractInterface::sptr, NodeID> doRouting(
      dll::Segment::sptr segment, NodeID src_srnid) = 0;
  void setTunInterface(AbstractInterface::sptr iface);
  void setOFDMInterface(AbstractInterface::sptr iface);

protected:
  AbstractRouter(NodeID my_srnid);
  NodeID const _my_srnid;
  AbstractInterface::sptr _tun_dev;
  AbstractInterface::sptr _ofdm_dev;
};

// Multi-hop router
class BasicRouter : public AbstractRouter {
public:
  BasicRouter(NodeID my_node_id);
  ~BasicRouter(){};
  void updateRoutingTable(NetworkMap &netmap);
  std::pair<AbstractInterface::sptr, NodeID> doRouting(
      dll::Segment::sptr segment, NodeID src_srnid);

private:
  std::map<NodeID, NodeID> d_routing_table; // <dstID, nextHopID>
};

/// Router for dubugging purpose. Does not "route" but always assume there is a
/// direct link to the destination.
class NoRouter : public AbstractRouter {
public:
  NoRouter(NodeID my_node_id);
  ~NoRouter(){};
  std::pair<AbstractInterface::sptr, NodeID> doRouting(
      dll::Segment::sptr segment, NodeID src_srnid);
};
} // namespace net
} // namespace bamradio

#endif /* CONTROLLER_SRC_ROUTER_H_ */
