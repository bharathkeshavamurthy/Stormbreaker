/* -*-c++-*-
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 * Copyright © 2017 Stephen Larew
 */

#ifndef e63b19fe40cbe90ef0
#define e63b19fe40cbe90ef0

#include <memory>
#include <vector>

#include <boost/asio.hpp>

namespace bamradio {
namespace net {

class IPPacket {
private:
  std::vector<uint8_t> _buffer;

public:
  typedef std::shared_ptr<IPPacket> sptr;

  explicit IPPacket(size_t mtu);
  explicit IPPacket(decltype(_buffer) const &packet_buffer);
  explicit IPPacket(boost::asio::const_buffer packet_buffer);
  explicit IPPacket(IPPacket &&i);
  explicit IPPacket(IPPacket const &i);

  boost::asio::ip::address_v4 srcAddr() const;
  boost::asio::ip::address_v4 dstAddr() const;

  void setSrcAddr(boost::asio::ip::address_v4 a);
  void setDstAddr(boost::asio::ip::address_v4 a);

  size_t size() const;
  void resize(size_t l);

  uint8_t &tos();

  boost::asio::mutable_buffer get_buffer();

  std::vector<uint8_t> &get_buffer_vec() { return _buffer; }

  IPPacket &operator=(IPPacket &&i);
};
}
}

#endif
