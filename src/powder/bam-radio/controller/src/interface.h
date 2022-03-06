// -*- c++ -*-
//  Copyright © 2017 Stephen Larew
//  Copyright (c) 2017 Tomohiro Arakawa

#ifndef CONTROLLER_SRC_INTERFACE_H_
#define CONTROLLER_SRC_INTERFACE_H_

#include "dll.h"
#include <vector>
#include <boost/asio.hpp>
#include <boost/signals2.hpp>

namespace bamradio {
namespace net {

/// Network interface
class AbstractInterface {
private:
  AbstractDataLinkLayer::sptr _dll;
  typedef boost::signals2::signal<void(bool)> UpSignal;

public:
  typedef std::shared_ptr<AbstractInterface> sptr;

  explicit AbstractInterface(AbstractDataLinkLayer::sptr dll);

  virtual std::string const &name() const;

  virtual void setUp() = 0;
  virtual void setDown() = 0;
  virtual bool isUp() const = 0;

  boost::signals2::connection observeUp(UpSignal::slot_type const &slot);

  virtual void addAddress(boost::asio::ip::address addr, int net) = 0;
  virtual std::vector<boost::asio::ip::address> addresses() = 0;

  AbstractDataLinkLayer::sptr dll() const;

protected:
  UpSignal _upSignal;
};

/// Create an Interface from any AbstractDataLinkLayer
class BasicInterfaceAdapter : public AbstractInterface {
public:
  BasicInterfaceAdapter(AbstractDataLinkLayer::sptr dll);

  void setUp();
  void setDown();
  bool isUp() const;

  void addAddress(boost::asio::ip::address addr, int net);
  std::vector<boost::asio::ip::address> addresses();

private:
  bool _up;
};

}

namespace tun {

class Interface : public net::AbstractInterface {
public:
  Interface(DataLinkLayer::sptr dll);

  void setUp();
  void setDown();
  bool isUp() const;

  void addAddress(boost::asio::ip::address addr, int net);
  std::vector<boost::asio::ip::address> addresses();

  DataLinkLayer::sptr dll() const;
};
} // namespace tun

}



#endif /* CONTROLLER_SRC_INTERFACE_H_ */
