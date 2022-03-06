// -*- c++ -*-
//  Copyright © 2017 Stephen Larew
//  Copyright (c) 2017 Tomohiro Arakawa

#include "net.h"
#include "cc_data.h"
#include "events.h"
#include "ippacket.h"
#include "networkmap.h"
#include "options.h"
#include "tun.h"

#include <random>

#include <boost/format.hpp>

namespace bamradio {
namespace net {

using namespace boost::asio;

// TODO: thread safety for basically this whole file

Controller::Controller()
    : _running(false), _ios_work(new io_service::work(_ios)),
      // FIXME get ID dynamically as posted by cc_data or from somewhere else
      _work_thread([this] {
        bamradio::set_thread_name("network_controller_work");
        _ios.run();
        log::text("net::Controller thread exiting", __FILE__, __LINE__);
      }) {
  if (options::phy::data::multihop) {
    // multihop router
    _router = std::make_shared<BasicRouter>(options::phy::control::id);
    // route update
    _subToken = NotificationCenter::shared
                    .subscribe<controlchannel::NetworkMapEventInfo>(
                        controlchannel::NetworkMapEvent, _ios, [&](auto ei) {
                          std::static_pointer_cast<BasicRouter>(_router)
                              ->updateRoutingTable(ei.netmap);
                        });
  } else {
    // no multihop
    _router = std::make_shared<NoRouter>(options::phy::control::id);
  }
}

Controller::~Controller() {
  delete _ios_work;
  _work_thread.join();
}

io_service &Controller::io_service() { return _ios; }

void Controller::addInterface(AbstractInterface::sptr i) {
  _infs.push_back(i);

  // FIXME: should be more flexible
  if (std::dynamic_pointer_cast<bamradio::tun::Interface>(i))
    _router->setTunInterface(i);
  else if (std::dynamic_pointer_cast<BasicInterfaceAdapter>(i))
    _router->setOFDMInterface(i);
  else
    log::text("unknown network interface", __FILE__, __LINE__);

  i->observeUp([this, i](auto up) {
    if (up && _running) {
      this->receiveOnInterface(i);
    }
  });

  if (i->isUp() && _running) {
    receiveOnInterface(i);
  }
}

void Controller::start() {
  if (_running) {
    throw std::runtime_error("Controller already running.");
  }
  // FIXME: hack remove this and make better
  system("ip route add 192.168.0.0/16 dev tun0");
  system("ip link set txqueuelen 50 dev tun0");
  _running = true;

  for (auto i : _infs) {
    if (i->isUp()) {
      receiveOnInterface(i);
    }
  }
}

void Controller::receiveOnInterface(AbstractInterface::sptr i) {
  if (!_running)
    return;

  size_t const extra_alloc_bytes = sizeof(uint64_t);
  auto const bs = std::make_shared<std::vector<uint8_t>>(extra_alloc_bytes +
                                                         i->dll()->mtu() * 2);

  i->dll()->asyncReceiveFrom(buffer(buffer(*bs) + extra_alloc_bytes),
                             (NodeID *)&bs->front(), [this, bs, i](auto s) {
                               if (s->packetLength() > 0) {
                                 this->handlePacket(s, bs,
                                                    *(NodeID *)&bs->front());
                               }
                               this->receiveOnInterface(i);
                             });
}

void Controller::handlePacket(
    net::IP4PacketSegment::sptr ip4seg,
    std::shared_ptr<std::vector<uint8_t>> backingStore, NodeID srcNodeID) {
  AbstractInterface::sptr outIface;
  uint8_t nextSrn;
  std::tie(outIface, nextSrn) = _router->doRouting(ip4seg, srcNodeID);

  if (!outIface) {
    return;
  }

  ip4seg->setDestNodeID(nextSrn);

  if (outIface->isUp())
    outIface->dll()->send(ip4seg, backingStore);
  else
    log::text((boost::format("interface %1% not up") % outIface->name()).str(),
              __FILE__, __LINE__);
}

std::thread *sendMockTraffic(size_t packet_size, NodeID myNodeID,
                             AbstractDataLinkLayer::sptr dll) {
  return new std::thread([=] {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dis(0, 255);
    std::uniform_int_distribution<> disDst(0, 1);
    while (true) {
      NodeID dst = myNodeID;
      while (dst == myNodeID) {
        dst = disDst(mt);
      }
      auto const colNet = inet_addr("192.168.0.0");
      auto const dstAddr = colNet | ((((uint32_t)dst) + 100) << 16);

      net::IPPacket packet(packet_size);
      packet.setDstAddr(boost::asio::ip::address_v4(htonl(dstAddr)));
      auto b = boost::asio::buffer(packet.get_buffer_vec());
      auto bb = boost::asio::buffer_cast<uint8_t *>(b);
      std::generate(bb + 20, bb + packet_size, [&] { return dis(mt); });
      // uint8_t k = 0;
      // std::generate(bb, bb + 1500, [&k] { return k++; });
      auto const bs = std::make_shared<std::vector<uint8_t>>(
          std::move(packet.get_buffer_vec()));
      auto const ip4seg =
          std::make_shared<net::IP4PacketSegment>(dst, buffer(*bs));
      dll->send(ip4seg, bs);
    }
  });
}

} // namespace net
} // namespace bamradio
