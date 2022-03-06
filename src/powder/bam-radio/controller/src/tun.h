// -*-c++-*-
//  Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
//  Copyright © 2017 Stephen Larew

#ifndef a945c223baeaf68b8c
#define a945c223baeaf68b8c

#include <memory>
#include <string>

#include <boost/asio.hpp>

namespace bamradio {
namespace tun {

class Device {
public:
  typedef std::shared_ptr<Device> sptr;

  Device(boost::asio::io_service &os, std::string const &nameFormat);
  ~Device();

  std::string const &name() const;

  void setMtu(size_t mtu);
  size_t mtu(bool use_cache = false) const;

  void setAddress(boost::asio::ip::address_v4 const &address);
  boost::asio::ip::address_v4 const &address(bool use_cache = false) const;

  void setNetmask(boost::asio::ip::address_v4 const &netmask);
  boost::asio::ip::address_v4 const &netmask(bool use_cache = false) const;

  void setUp();
  bool isUp(bool use_cache = false) const;

  boost::asio::posix::stream_descriptor &descriptor();

private:
  // socket for ioctls
  int kernel_socket;

  // file descriptor
  boost::asio::posix::stream_descriptor _tun_fd;

  // interface name
  std::string _name;

  mutable bool _up_cached;
  mutable size_t _mtu_cached;
  mutable boost::asio::ip::address_v4 _addr_cached;
  mutable boost::asio::ip::address_v4 _netmask_cached;

  void init_ifreq(void *pifr) const;
};
}
}

#endif
