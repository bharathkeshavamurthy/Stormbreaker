/* -*-c++-*-
 * Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
 * Copyright © 2017 Stephen Larew
 */

#include "ippacket.h"

#include <iostream>

#include <arpa/inet.h>
#include <linux/ip.h>
#include <netinet/in.h>

namespace bamradio {
namespace net {

using namespace boost::asio;

IPPacket::IPPacket(size_t mtu) : _buffer(mtu) {
  assert(mtu > 20);
  auto const hdr = (struct iphdr *)&_buffer.front();
  hdr->version = IPVERSION;
  hdr->ihl = 5;
  hdr->tot_len = htons(mtu);
  hdr->ttl = MAXTTL;
  hdr->protocol = 253;
}

IPPacket::IPPacket(decltype(_buffer) const &packet_buffer)
    : _buffer(packet_buffer) {}

IPPacket::IPPacket(const_buffer packet_buffer)
    : _buffer(buffer_cast<uint8_t const *>(packet_buffer),
              buffer_cast<uint8_t const *>(packet_buffer) +
                  buffer_size(packet_buffer)) {}

IPPacket::IPPacket(IPPacket &&i) : _buffer(std::move(i._buffer)) {}

IPPacket::IPPacket(IPPacket const &i) : _buffer(i._buffer) {}

ip::address_v4 IPPacket::srcAddr() const {
  auto const hdr = (struct iphdr *)&_buffer.front();
  return ip::address_v4(ntohl(hdr->saddr));
}

ip::address_v4 IPPacket::dstAddr() const {
  auto const hdr = (struct iphdr *)&_buffer.front();
  return ip::address_v4(ntohl(hdr->daddr));
}

void IPPacket::setSrcAddr(boost::asio::ip::address_v4 a) {
  ((struct iphdr *)&_buffer.front())->saddr = htonl(a.to_ulong());
}

void IPPacket::setDstAddr(boost::asio::ip::address_v4 a) {
  ((struct iphdr *)&_buffer.front())->daddr = htonl(a.to_ulong());
}

size_t IPPacket::size() const {
  auto const hdr = (struct iphdr *)&_buffer.front();
  return ntohs(hdr->tot_len);
}

void IPPacket::resize(size_t l) {
  assert(l > 20);
  assert(l < 65535);
  _buffer.resize(l);
  auto const hdr = (struct iphdr *)&_buffer.front();
  hdr->tot_len = htons(l);
}

uint8_t &IPPacket::tos() {
  auto const hdr = (struct iphdr *)&_buffer.front();
  return hdr->tos;
}

mutable_buffer IPPacket::get_buffer() { return buffer(_buffer, size()); }

IPPacket &IPPacket::operator=(IPPacket &&i) {
  _buffer = std::move(i._buffer);
  return *this;
}
}
}
