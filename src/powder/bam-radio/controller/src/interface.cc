// -*- c++ -*-
//  Copyright © 2017 Stephen Larew
//  Copyright (c) 2017 Tomohiro Arakawa

#include "interface.h"
#include "tun.h"
#include "events.h"

#include <boost/format.hpp>

namespace bamradio {
namespace net {

using namespace boost::asio;
// TODO: thread safety for basically this whole file

AbstractDataLinkLayer::sptr AbstractInterface::dll() const { return _dll; }

AbstractInterface::AbstractInterface(AbstractDataLinkLayer::sptr dll)
    : _dll(dll) {
  _dll->observeRunning([this](bool const running) {
    if (running) {
      log::text((boost::format("%1% RUNNING & UP (auto)") % _dll->name()).str(), __FILE__, __LINE__);
      this->setUp();
    } else {
      log::text((boost::format("%1% DOWN (auto)") % _dll->name()).str(), __FILE__, __LINE__);
      this->setDown();
    }
  });
}

std::string const &AbstractInterface::name() const {
  return dll()->name();
}

boost::signals2::connection
AbstractInterface::observeUp(UpSignal::slot_type const &slot) {
  return _upSignal.connect(slot);
}

BasicInterfaceAdapter::BasicInterfaceAdapter(AbstractDataLinkLayer::sptr dll)
    : AbstractInterface(dll), _up(false) {}

// Admin up/down directly tied to the dll start/stop in this basic adapter
void BasicInterfaceAdapter::setUp() {
  if (_up) {
    return;
  }
  if (!dll()->running()) {
    throw std::runtime_error("DLL is not running.");
  }
  _up = true;
  _upSignal(_up);
}

void BasicInterfaceAdapter::setDown() {
  if (!_up) {
    return;
  }
  _up = false;
  _upSignal(_up);
}

bool BasicInterfaceAdapter::isUp() const { return _up; }

void BasicInterfaceAdapter::addAddress(boost::asio::ip::address addr, int net) {
  // TODO:
}

std::vector<boost::asio::ip::address> BasicInterfaceAdapter::addresses() {
  // TODO:
  return {};
}
}

namespace tun {

Interface::Interface(DataLinkLayer::sptr dll) : AbstractInterface(dll) {}

void Interface::setUp() { dll()->device()->setUp(); }

void Interface::setDown() {
  throw std::runtime_error("Unsupported: Cannot set the TUN device to down.");
}

bool Interface::isUp() const { return dll()->device()->isUp(); }

void Interface::addAddress(boost::asio::ip::address addr, int net) {
  if (addr.is_v4()) {
    assert(net <= 32);
  } else {
    assert(net <= 128);
  }

  if (!addr.is_v4()) {
    throw std::runtime_error("Cannot add non-IPv4 address to TUN device.");
  }

  uint32_t netmask = 0xffffffff << (32 - net);

  try {
    auto addr = dll()->device()->address(false);
    throw std::runtime_error(
        "Adding multiple addresses to a TUN device is unsupported.");
  } catch (std::system_error e) {
    log::text(e.what(), __FILE__, __LINE__);
    dll()->device()->setAddress(addr.to_v4());
    dll()->device()->setNetmask(boost::asio::ip::address_v4(netmask));
  }

  throw std::runtime_error("addAddress untested");
}

std::vector<boost::asio::ip::address> Interface::addresses() {
  return {dll()->device()->address()};
}

DataLinkLayer::sptr Interface::dll() const {
  return std::dynamic_pointer_cast<bamradio::tun::DataLinkLayer>(
      net::AbstractInterface::dll());
}
}

}
