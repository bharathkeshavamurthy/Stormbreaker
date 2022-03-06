// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#ifndef bc23392d7b152d9fe6
#define bc23392d7b152d9fe6

#include "dll.h"
#include "interface.h"
#include "notify.h"
#include "router.h"

#include <vector>

#include <boost/asio.hpp>
#include <boost/signals2.hpp>

namespace bamradio {
namespace net {

class Controller {
private:
  bool _running;
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  std::thread _work_thread;
  NotificationCenter::SubToken _subToken;
  std::vector<AbstractInterface::sptr> _infs;

  AbstractRouter::sptr _router;

  void receiveOnInterface(AbstractInterface::sptr inf);

  void handlePacket(net::IP4PacketSegment::sptr ip4seg,
                    std::shared_ptr<std::vector<uint8_t>> backingStore,
                    NodeID srcNodeID);

public:
  Controller();
  ~Controller();

  /// Return the io_service for network controller operations.
  boost::asio::io_service &io_service();

  /// Add an interface to the network.
  void addInterface(AbstractInterface::sptr i);

  /// Start network controller operations.
  void start();
};

std::thread *sendMockTraffic(size_t packet_size, NodeID myNodeID,
                             AbstractDataLinkLayer::sptr dll);

} // namespace net
} // namespace bamradio

#endif
