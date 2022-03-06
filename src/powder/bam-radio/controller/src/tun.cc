// -*-c++-*-
//  Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
//  Copyright © 2017 Stephen Larew

#include "tun.h"
#include "options.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <iostream>
#include <net/if.h>
#include <linux/if_tun.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <system_error>

using std::cerr;
using std::endl;

namespace bamradio {
namespace tun {

using namespace boost::asio;

Device::Device(io_service &io_service, std::string const &name_fmt)
    : kernel_socket(-1), _tun_fd(io_service), _up_cached(false),
      _mtu_cached(0) {
  if (name_fmt.size() > IFNAMSIZ) {
    throw std::runtime_error("tun name fmt too long");
  }

  // reference:
  // https://www.kernel.org/doc/Documentation/networking/tuntap.txt

  int fd = open("/dev/net/tun", O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    throw std::system_error(errno, std::system_category(),
                            "open(\"/dev/net/tun\") failed");
  }

  _tun_fd.assign(fd);

  // initialize the interface with name and type

  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));

  strncpy(ifr.ifr_name, name_fmt.c_str(), name_fmt.size());

  // allocate a TUN device and get the new name
  /*
   * Flags: IFF_TAP   - TAP device (layer 2, ethernet frame)
   *        IFF_TUN   - TUN device (layer 3, IP packet)
   *        IFF_NO_PI - Do not provide packet information
   *        IFF_MULTI_QUEUE - Create a queue of multiqueue device
   */
  ifr.ifr_flags = IFF_TUN | IFF_NO_PI;

  if (ioctl(fd, TUNSETIFF, &ifr) != 0) {
    close(fd);
    throw std::system_error(errno, std::system_category(),
                            "ioctl(TUNSETIFF) failed");
  }

  // Save new tunnel name
  _name = ifr.ifr_name;

  // get a socket for the ioctls
  kernel_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (kernel_socket < 0) {
    close(fd);
    throw std::system_error(errno, std::system_category(),
                            "socket(AF_INET, SOCK_STREAM, 0) failed");
  }

  // cache UP and MTU values
  mtu();
}

Device::~Device() {
  if (_tun_fd.is_open()) {
    _tun_fd.close();
  }
  if (kernel_socket > 0) {
    close(kernel_socket);
  }
}

std::string const &Device::name() const { return _name; }

// initialize the ifreq struct at pifr to zeros and set the name field
void Device::init_ifreq(void *pifr) const {
  memset(pifr, 0x00, sizeof(struct ifreq));
  strncpy(((struct ifreq *)pifr)->ifr_name, _name.c_str(), _name.size());
  if (ioctl(kernel_socket, SIOCGIFFLAGS, pifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCGIFFLAGS) failed");
  }
  // update the cached UP flag
  _up_cached = (((struct ifreq *)pifr)->ifr_flags & IFF_UP) == IFF_UP;
}

void Device::setMtu(size_t mtu) {
  struct ifreq ifr;
  init_ifreq(&ifr);
  ifr.ifr_mtu = mtu;
  if (ioctl(kernel_socket, SIOCSIFMTU, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCSIFMTU) failed");
  } else {
    _mtu_cached = mtu;
  }
}

size_t Device::mtu(bool use_cache) const {
  if (use_cache) {
    return _mtu_cached;
  }

  struct ifreq ifr;
  init_ifreq(&ifr);
  if (ioctl(kernel_socket, SIOCGIFMTU, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCGIFMTU) failed");
  }
  _mtu_cached = ifr.ifr_mtu;
  return _mtu_cached;
}

void Device::setAddress(ip::address_v4 const &ip) {
  struct ifreq ifr;
  init_ifreq(&ifr);
  struct sockaddr_in *addr = (struct sockaddr_in *)&ifr.ifr_addr;
  addr->sin_family = AF_INET;
  addr->sin_addr.s_addr = *(in_addr_t *)ip.to_bytes().data();
  // set the IP address
  if (ioctl(kernel_socket, SIOCSIFADDR, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCSIFADDR) failed");
  } else {
    _addr_cached = ip;
  }
}

ip::address_v4 const &Device::address(bool use_cache) const {
  if (use_cache) {
    return _addr_cached;
  }

  struct ifreq ifr;
  init_ifreq(&ifr);
  struct sockaddr_in *addr = (struct sockaddr_in *)&ifr.ifr_addr;
  addr->sin_family = AF_INET;
  if (ioctl(kernel_socket, SIOCGIFADDR, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCGIFADDR) failed");
  }
  ip::address_v4::bytes_type ip_bytes;
  std::copy(ip_bytes.begin(), ip_bytes.end(), &addr->sin_addr.s_addr);
  _addr_cached = ip::address_v4(ip_bytes);
  return _addr_cached;
}

void Device::setNetmask(ip::address_v4 const &ip) {
  struct ifreq ifr;
  init_ifreq(&ifr);
  struct sockaddr_in *addr = (struct sockaddr_in *)&ifr.ifr_netmask;
  addr->sin_family = AF_INET;
  addr->sin_addr.s_addr = *(in_addr_t *)ip.to_bytes().data();
  // set the IP address
  if (ioctl(kernel_socket, SIOCSIFNETMASK, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCSIFNETMASK) failed");
  } else {
    _netmask_cached = ip;
  }
}

ip::address_v4 const &Device::netmask(bool use_cache) const {
  if (use_cache) {
    return _netmask_cached;
  }

  struct ifreq ifr;
  init_ifreq(&ifr);
  struct sockaddr_in *addr = (struct sockaddr_in *)&ifr.ifr_netmask;
  addr->sin_family = AF_INET;
  if (ioctl(kernel_socket, SIOCGIFNETMASK, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCGIFNETMASK) failed");
  }
  ip::address_v4::bytes_type ip_bytes;
  std::copy(ip_bytes.begin(), ip_bytes.end(), &addr->sin_addr.s_addr);
  _netmask_cached = ip::address_v4(ip_bytes);
  return _netmask_cached;
}

void Device::setUp() {
  struct ifreq ifr;
  init_ifreq(&ifr);
  // set the flags
  ifr.ifr_flags |=
      //IFF_UP | IFF_RUNNING;
      //IFF_UP | IFF_NOARP | IFF_RUNNING;
      //IFF_UP | IFF_POINTOPOINT | IFF_MULTICAST | IFF_NOARP | IFF_RUNNING;
      IFF_UP | IFF_NOARP | IFF_RUNNING;
  if (ioctl(kernel_socket, SIOCSIFFLAGS, &ifr) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "ioctl(SIOCSIFFLAGS) failed");
  } else {
    _up_cached = true;
  }
}

bool Device::isUp(bool) const {
  struct ifreq ifr;
  init_ifreq(&ifr);
  // init_ifreq updates the cache
  return _up_cached;
}

posix::stream_descriptor &Device::descriptor() {
  return _tun_fd;
}

#if 0
mutable_buffer Device::read(mutable_buffer b) {
  boost::system::error_code e;
  size_t size;
  do {
    size = _tun_fd.read_some(buffer(b), e);
    if (e) {
      cerr << "Device::read_some encountered error: (" << e.value() << ") "
           << e.message() << endl;
    }
  } while (e);
  return buffer(b, size);
}

void Device::write(const_buffer b) {
  size_t size;
  boost::system::error_code e;
  do {
    size = _tun_fd.write_some(buffer(b), e);
    if (e) {
      cerr << "Device::write encountered error: (" << e.value() << ") "
           << e.message() << endl;
    }
  } while (e);
  if (size != buffer_size(b)) {
    cerr << "Device::write only wrote " << size << "/"
         << buffer_size(b) << endl;
  }
}

void Device::async_read(IPPacket::sptr p,
                        std::function<void(IPPacket::sptr)> cb) {
  _tun_fd.async_read_some(buffer(p->get_buffer()),
                          [p, cb](auto const e, auto const size) {
                            if (e) {
                              cerr << "Device::async_read encountered error: ("
                                   << e.value() << ") " << e.message() << endl;
                              // FIXME: report error to callback
                              return;
                            }
                            p->resize(size);
                            cb(p);
                          });
}

void Device::async_write(IPPacket::sptr p, std::function<void()> cb) {
  _tun_fd.async_write_some(
      buffer(p->get_buffer()), [p, cb](auto const e, auto const size) {
        if (e) {
          cerr << "Device::async_write encountered error: (" << e.value()
               << ") " << e.message() << endl;
          return;
        }
        if (size != buffer_size(p->get_buffer())) {
          cerr << "Device::async_write only wrote " << size << "/"
               << buffer_size(p->get_buffer()) << endl;
        }
        cb();
      });
}
#endif
}
}
