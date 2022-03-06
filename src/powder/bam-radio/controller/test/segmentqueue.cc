//#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#ifndef BOOST_ASIO_DISABLE_EPOLL
#error "BOOST_ASIO_DISABLE_EPOLL not defined"
#endif
#include "../src/segmentqueue.h"
#include "../src/flowtracker.h"
#include "../src/ippacket.h"
#include "../src/tun.h"
#include <array>
#include <boost/asio/ip/network_v4.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <linux/ip.h>
#include <map>
#include <random>
#include <thread>

using namespace bamradio;
using boost::format;
namespace asio = boost::asio;
using namespace std::chrono_literals;
namespace chrono = std::chrono;

chrono::milliseconds const flowRateWindow = 1000ms;

int const mtu = 1500;
asio::ip::network_v4 const network =
    asio::ip::make_network_v4("172.16.240.0/24");
asio::ip::address_v4 addrFromNodeID(NodeID nid) {
  return asio::ip::address_v4(network.network().to_ulong() |
                              ((uint32_t)nid << 8) | 1);
}
asio::ip::address_v4 flowDestAddr(NodeID nid, NodeID fromNid) {
  return asio::ip::address_v4(network.network().to_ulong() |
                              ((uint32_t)nid << 8) | (128 + fromNid));
}
asio::ip::network_v4 flowDestNet(NodeID nid) {
  return asio::ip::network_v4(flowDestAddr(nid, 0), 25).canonical();
}
/*NodeID nodeIDFromAddr(asio::ip::address_v4 addr) {
  return addr.to_ulong() & ~network.netmask().to_ulong();
}*/

struct FlowDesc {
  uint16_t flowUID;
  NodeID tx, rx;
  chrono::microseconds startDelay;
  size_t bps;
  size_t numBytes;
  size_t bytesPerPacket;
};

auto transferDur = 2000ms;

std::array<NodeID, 6> nodeIds{1, 2, 3, 4, 5, 6};
std::array<FlowDesc, 6> flows{
    FlowDesc{5200, 1, 2, 0us, 8 * 4000 * 1000, 10000, 352},
    FlowDesc{5201, 2, 1, 0us, 8 * 3000 * 1000, 10000, 400},
    FlowDesc{5202, 2, 1, 0us, 8 * 3000 * 1000, 100000, mtu - 28},
    FlowDesc{5203, 2, 3, 0us, 8 * 10000 * 1000, 100000, mtu - 28},
    FlowDesc{5204, 3, 2, 10ms, 8 * 10000 * 1000, 100000, 1200},
    FlowDesc{5205, 4, 5, 0us, 8 * 10000 * 1000, 100000, mtu - 28}};

class Node {
public:
  typedef std::shared_ptr<Node> sptr;
  static std::map<std::string, sptr> routes;
  static std::map<NodeID, sptr> nodes;

private:
  DRRFQSegQ _sq;
  asio::io_context &_ioctx;
  tun::Device _tunDevice;
  asio::ip::address_v4 _addr;
  asio::ip::network_v4 _flowNetwork;
  asio::steady_timer _tTx, _tRx;
  struct otaFrame {
    otaFrame(NodeID dest, std::vector<dll::ARQBurstInfo> abiv_,
             std::vector<QueuedSegment> const &s)
        : destNodeID(dest), abiv(abiv_), segments(s) {}
    NodeID destNodeID;
    std::vector<dll::ARQBurstInfo> abiv;
    std::vector<QueuedSegment> segments;
    size_t bits() const {
      return std::accumulate(
          begin(segments), end(segments), 128,
          [](auto a, auto b) { return a + b.segment->length() * 8; });
    }
  };
  std::deque<otaFrame> _otaQueue;
  bool _willSend = false;
  bool _willReceive = false;
  float _drop_rate = 0.3f;
  dll::FlowTracker _ft;
  bool _triggerARQFeedback = false;

public:
  asio::ip::address_v4 address() const { return _addr; }
  asio::ip::network_v4 flowNetwork() const { return _flowNetwork; }
  std::string tunDevName() const { return _tunDevice.name(); }

  float bps = 1e6;
  NodeID const nodeID;
  DRRFQSegQ &sq() { return _sq; }

  Node(asio::io_context &ioctx, NodeID nid)
      : _sq(5,
            [this](FlowID fid) -> DRRFQSegQ::FlowQueue::sptr {
              return std::make_shared<DRRFQSegQ::FIFOBurstQueue>(
                  fid, _ioctx, [](auto const &f) { f(); },
                  [this] {
                    if (!_willSend) {
                      this->send();
                    }
                  },
                  transferDur, 40ms, 1s, 1000);
              return std::make_shared<DRRFQSegQ::LIFODelayDropQueue>(fid,
                                                                     200ms);
            },
            [](auto, auto, auto, auto) {
              return DRRFQSegQ::FlowSchedule{100ms, 50ms};
            }),
        _ioctx(ioctx), _tunDevice(ioctx, "tun%d"), _addr(addrFromNodeID(nid)),
        _flowNetwork(flowDestNet(nid)), _tTx(ioctx), _tRx(ioctx), nodeID(nid) {
    _tunDevice.setMtu(mtu);
    _tunDevice.setAddress(_addr);
    _tunDevice.setNetmask(network.netmask());
    _tunDevice.setUp();
    std::map<FlowUID, IndividualMandate> imm;
    for (auto fd : flows) {
      imm[fd.flowUID] =
          IndividualMandate{1, IndividualMandate::FilePT{-1, transferDur}};
    }
    _ft.addIndividualMandates(imm);
  }

  void start() { asyncReceiveFromTun(); }

  void asyncReceiveFromTun() {
    // std::cerr << "tun read scheduled\n";
    auto readbuf = std::make_shared<std::vector<uint8_t>>(mtu);
    _tunDevice.descriptor().async_read_some(
        asio::buffer(*readbuf),
        [this, readbuf](auto error, auto bytes_transferred) {
          if (error) {
            std::cerr << _tunDevice.name() << " tun read error "
                      << error.value() << " msg " << error.message()
                      << std::endl;
            this->asyncReceiveFromTun();
            return;
          }
          // std::cerr << format("tun read complete %lu\n") % bytes_transferred;
          std::vector<uint8_t> &bufr = *readbuf;
          bufr.resize(bytes_transferred);
          struct iphdr *hdr = (struct iphdr *)&bufr[0];
          if (hdr->version != 4) {
            std::cerr << _tunDevice.name() << " non IPv4" << std::endl;
            this->asyncReceiveFromTun();
            return;
          }
          // std::cerr << _tunDevice.name() << " tun ipv4" << std::endl;
          this->handleTunPacket(std::make_shared<net::IPPacket>(bufr));
          this->asyncReceiveFromTun();
        });
  }

  void handleTunPacket(net::IPPacket::sptr ipp) {
    /*std::cerr << "route lookup " << ipp->dstAddr().to_string() << " net "
              << asio::ip::network_v4(ipp->dstAddr(), 25).to_string()
              << std::endl;*/
    auto const dstNodeID =
        Node::routes
            [asio::ip::network_v4(ipp->dstAddr(), 25).canonical().to_string()]
                ->nodeID;
    auto const bs =
        std::make_shared<std::vector<uint8_t>>(ipp->get_buffer_vec());
    dll::Segment::sptr const segment = std::make_shared<net::IP4PacketSegment>(
        dstNodeID, asio::buffer(*bs), chrono::system_clock::now());
    QueuedSegment qs{segment, bs};
    _sq.push(qs);
    if (!_willSend) {
      // std::cerr << format("Node %1% send now\n") % (int)nodeID;
      send();
    } else {
      // std::cerr << format("Node %1% send deferred\n") % (int)nodeID;
    }
  }

  void send(std::chrono::steady_clock::time_point now =
                std::chrono::steady_clock::now()) {
    auto segments = _sq.pop(100ms, false, false);
    /*std::cerr << format("Node %1% popped %2% segments to send\n") %
                     (int)nodeID % segments.size();*/

    if (segments.empty() && !_triggerARQFeedback) {
      _willSend = false;
      // std::cerr << format("Node %1% send stop\n") % (int)nodeID;
      return;
    }

    NodeID dest =
        segments.empty() ? AllNodesID : segments.front().segment->destNodeID();

    for (auto const &qs : segments) {
      auto arqseg =
          std::dynamic_pointer_cast<net::ARQIP4PacketSegment>(qs.segment);
      if (arqseg && arqseg->arqDataSet()) {
        auto ad = arqseg->arqData();
        auto const bn = ad.burstNum;
        auto const sn = ad.seqNum;
        std::cerr << format(
                         "Node %u will send flowuid %u burst %u seqnum %u\n") %
                         (int)nodeID % arqseg->flowUID() % (int)bn % sn;
      }
    }
    auto seqnumsacked = _ft.getLastSeqNums();
    for (auto aa : seqnumsacked) {
      std::cerr << format("Node %u will ack flowuid %u burst %u seqnum %u\n") %
                       (int)nodeID % aa.flow_uid % (int)aa.burst_num %
                       aa.seq_num;
    }
    _otaQueue.emplace_back(dest, seqnumsacked, segments);
    _triggerARQFeedback = false;
    otaFrame &frame = _otaQueue.back();
    auto const frameAirtime =
        chrono::nanoseconds((uint64_t)(frame.bits() * 1000000000 / bps));
    std::cerr << chrono::duration<float>(frameAirtime).count() << std::endl;

    if (!_willReceive) {
      // std::cerr << format("Node %1% receive scheduled by send\n") %
      // (int)nodeID;
      scheduleReceive( // 10ms + // tx delay
          now + frameAirtime);
    } else {
      // std::cerr << format("Node %1% receive scheduled already\n") %
      // (int)nodeID;
    }

    // Set the next time at which to poll the segq and send.
    _willSend = true;
    // std::cerr << format("Node %1% send scheduled\n") % (int)nodeID;
    auto expireat = now + frameAirtime;
    _tTx.expires_at(expireat);
    _tTx.async_wait([this, expireat](auto error) {
      if (error) {
        std::cerr << "sched send timer error " << error.message() << std::endl;
        abort();
      }
      this->send(expireat);
    });
  }

  void scheduleReceive(std::chrono::steady_clock::time_point when) {
    _willReceive = true;
    _tRx.expires_at(when);
    _tRx.async_wait([this, when](auto error) {
      if (error == asio::error::operation_aborted) {
        std::cerr << "asioabort link rx timer error " << error << std::endl;
        abort();
      }
      if (error) {
        std::cerr << "link rx timer error " << error << std::endl;
        abort();
      }
      std::cerr << "willrx timerr "
                << chrono::duration_cast<chrono::duration<float>>(
                       chrono::steady_clock::now() - when)
                       .count()
                << std::endl;
      otaFrame &frame = this->_otaQueue.front();
      auto abiv = frame.abiv;
      std::vector<QueuedSegment> segments = std::move(frame.segments);
      auto coinflip = ((float)std::rand()) / RAND_MAX;
      // auto now = std::chrono::steady_clock::now():
      auto now = when;
      if (coinflip > _drop_rate && !segments.empty()) {
        auto const dstNodeID = segments.front().segment->destNodeID();
        Node::nodes[dstNodeID]->handleReceivedSegments(segments);
      } else {
        if (!segments.empty()) {
          std::cerr << "dropping "
                    << segments.front().segment->flowID().description()
                    << std::endl;
        }
      }
      for (auto &np : Node::nodes) {
        auto const &queues = np.second->_sq.flowQueues();
        // iterate over all feedback messages
        for (auto const &f : abiv) {
          // iterate over all flow queues
          for (auto &q : queues) {
            if (q.first.flowUID() == f.flow_uid) {
              std::cerr << "NodeID " << (int)np.second->nodeID << " rxd arqf "
                        << f.flow_uid << " " << (int)f.burst_num << " "
                        << f.seq_num << std::endl;
              auto &flow_queue = q.second.q;
              assert(flow_queue != nullptr);
              flow_queue->handleARQInfo(f, now,
                                        std::chrono::system_clock::now());
            }
          }
        }
        if (!np.second->_willSend) {
          np.second->send();
        }
      }
      this->_otaQueue.pop_front();
      this->_willReceive = false;
      if (this->_otaQueue.size() > 0) {
        /*std::cerr << format("Node %1% receive scheduled by receive\n") %
                         (int)nodeID;*/
        otaFrame &nextFrame = this->_otaQueue.front();
        this->scheduleReceive(when + chrono::nanoseconds((uint64_t)(
                                         nextFrame.bits() * 1000000000 / bps)));
      } else {
        // std::cerr << format("Node %1% receive stop\n") % (int)nodeID;
      }
    });
  }

  void handleReceivedSegments(std::vector<QueuedSegment> &segments) {
    /*std::cerr << format("Node %1% received %2% segments\n") % (int)nodeID %
                     segments.size();*/
    for (auto const &qs : segments) {
      switch (qs.segment->type()) {
      case dll::SegmentType::ARQIPv4: {
        auto const ipseg =
            std::dynamic_pointer_cast<net::ARQIP4PacketSegment>(qs.segment);
        auto ad = ipseg->arqData();
        std::cerr << "received " << ipseg->flowUID() << " seqnum " << ad.seqNum
                  << std::endl;
        _ft.markReceived(dll::ARQBurstInfo{ipseg->flowUID(), ad.burstNum,
                                           ad.seqNum, ad.arqExtra},
                         ipseg->sourceTime(), chrono::system_clock::now(),
                         ipseg->length());
        _triggerARQFeedback = true;
        if (asio::ip::network_v4(ipseg->dstAddr(), 25).canonical() ==
            _flowNetwork) {

          _tunDevice.descriptor().async_write_some(
              ipseg->packetContentsBuffer(),
              [this, qs, ipseg](auto error, auto bytes_transferred) {
                if (error) {
                  std::cerr << _tunDevice.name() << " tun write error "
                            << error.value() << " msg " << error.message()
                            << std::endl;
                  abort();
                }
                /*std::cerr << format("Node %d wrote to %s: %s:%u->%s:%u") %
                                 (int)nodeID % _tunDevice.name() %
                                 ipseg->srcAddr().to_string() %
                                 ipseg->srcPort() %
                                 ipseg->dstAddr().to_string() % ipseg->dstPort()
                          << std::endl;*/
                assert(ipseg->packetLength() == bytes_transferred);
              });
        } else {
          abort();
          auto ipp =
              std::make_shared<net::IPPacket>(ipseg->packetContentsBuffer());
          handleTunPacket(ipp);
        }
      } break;
      default:
        std::cerr << "non ip segment" << std::endl;
        abort();
      }
    }
  }
};

std::map<std::string, Node::sptr> Node::routes;
std::map<NodeID, Node::sptr> Node::nodes;

void printQueueStats(asio::steady_timer &statTimer) {
  for (auto &np : Node::nodes) {
    Node::sptr n = np.second;
    auto &sfqv = n->sq().flowQueues();
    for (auto const &sfqp : sfqv) {
      auto const &fid = sfqp.first;
      auto const fs = sfqp.second.q->stats();
      std::cerr << format("\tflow %s en %lu de %lu dr %lu") %
                       fid.description() % fs.bytesEnqueued % fs.bytesDequeued %
                       fs.bytesDropped
                << std::endl;
    }
  }
  statTimer.expires_after(1000ms);
  statTimer.async_wait([&statTimer](auto) { printQueueStats(statTimer); });
}

struct FlowSim {
public:
  FlowDesc const flowDesc;

  struct Stats {
    int64_t bytes = 0;
    int64_t windowedBits = 0;
    chrono::nanoseconds windowedDelay = 0s;
    float bps(chrono::nanoseconds window) const {
      return windowedBits * 1000000000.0f / window.count();
    }
    chrono::nanoseconds delay() const {
      return hist.size() > 0
                 ? chrono::nanoseconds(windowedDelay.count() / hist.size())
                 : -1ns;
    }
    std::deque<std::tuple<chrono::steady_clock::time_point, int64_t,
                          chrono::nanoseconds>>
        hist;
  } stats;

  void updateRates(chrono::steady_clock::time_point now) {
    if (stats.hist.empty()) {
      return;
    }
    auto st = stats.hist.front();
    while (std::get<0>(st) + flowRateWindow < now) {
      stats.windowedBits -= std::get<1>(st) * 8;
      stats.windowedDelay -= std::get<2>(st);
      stats.hist.pop_front();
      if (stats.hist.empty()) {
        break;
      }
      st = stats.hist.front();
    }
  }

private:
  chrono::steady_clock::time_point timeOrigin;
  asio::steady_timer txTimer;
  asio::ip::udp::socket rxs, txs;
  std::map<uint32_t, std::chrono::steady_clock::time_point> txpackets;
  uint32_t sendSeq = 0;
  int const perpacketaddbytes = 20 + 8;

  void trackRx(uint32_t seqNum, int64_t bytes,
               chrono::steady_clock::time_point arrivalTime =
                   chrono::steady_clock::now()) {
    if (txpackets.count(seqNum) == 0) {
      return;
    }
    bytes += perpacketaddbytes;
    auto const delay = arrivalTime - txpackets.at(seqNum);
    txpackets.erase(seqNum);
    stats.hist.emplace_back(arrivalTime, bytes, delay);
    stats.windowedBits += bytes * 8;
    stats.windowedDelay += delay;
    stats.bytes += bytes;
  }

  void receive() {
    /*std::cerr << format("Flow %u will receive on %s:%u") % flowDesc.flowUID %
                     rxs.local_endpoint().address().to_string() %
                     rxs.local_endpoint().port()
              << std::endl;*/
    auto const buf = std::make_shared<std::vector<uint8_t>>(mtu);
    auto const sender =
        std::make_shared<asio::ip::udp::socket::endpoint_type>();
    rxs.async_receive_from(
        asio::buffer(*buf), *sender,
        [this, buf, sender](auto error, auto bytes_transferred) {
          // std::cerr << "trackrx2" << std::endl;
          // std::cerr.flush();
          if (error) {
            std::cerr << format("Flow %u receive failed. %s") %
                             flowDesc.flowUID % error.message()
                      << std::endl;
            abort();
          }
          /*if (*sender != txs.local_endpoint()) {
            std::cerr << format("Flow %u unexpected sender") % flowDesc.flowUID
                      << std::endl;
            abort();
          }*/
          if (bytes_transferred != flowDesc.bytesPerPacket) {
            std::cerr << format("Flow %u receive %lu expected %lu") %
                             flowDesc.flowUID % bytes_transferred %
                             flowDesc.bytesPerPacket
                      << std::endl;
            abort();
          }
          stats.bytes += bytes_transferred;
          if (bytes_transferred > sizeof(sendSeq)) {
            auto const recvSeq = (decltype(sendSeq) *)&buf->front();
            this->trackRx(*recvSeq, bytes_transferred);
          }
          this->receive();
        });
  }

  void doSend() {
    assert(flowDesc.bytesPerPacket % sizeof(sendSeq) == 0);
    /*std::cerr << format("Flow %u will send from %s:%u") % flowDesc.flowUID %
                     txs.local_endpoint().address().to_string() %
                     txs.local_endpoint().port()
              << std::endl;*/
    txpackets[sendSeq] = chrono::steady_clock::now();
    auto const buf = std::make_shared<std::vector<decltype(sendSeq)>>(
        flowDesc.bytesPerPacket / sizeof(sendSeq), sendSeq++);
    txs.async_send_to(
        asio::buffer(*buf),
        asio::ip::udp::endpoint(flowDestAddr(flowDesc.rx, flowDesc.tx),
                                flowDesc.flowUID),
        [this, buf](auto error, auto bytes_transferred) {
          if (error) {
            std::cerr << format("Flow %u send error %s") % flowDesc.flowUID %
                             error.message()
                      << std::endl;
            abort();
          }
          assert(bytes_transferred == flowDesc.bytesPerPacket);
          if (flowDesc.bytesPerPacket * sendSeq < flowDesc.numBytes) {
            txTimer.expires_at(timeOrigin + flowDesc.startDelay +
                               chrono::nanoseconds((uint64_t)(
                                   sendSeq * flowDesc.bytesPerPacket * 8 *
                                   1000000000 / flowDesc.bps)));
            txTimer.async_wait([this](auto error) {
              if (error) {
                abort();
              }
              this->doSend();
            });
          }
        });
  }

public:
  FlowSim(asio::io_context &ioctx, FlowDesc fd)
      : flowDesc(fd), txTimer(ioctx),
        rxs(ioctx, asio::ip::udp::endpoint(addrFromNodeID(fd.rx), fd.flowUID)),
        // rxs(ioctx, asio::ip::udp::endpoint(asio::ip::udp::v4(), fd.flowUID)),
        txs(ioctx, asio::ip::udp::endpoint(addrFromNodeID(fd.tx),
                                           10000 + fd.flowUID)) {}

  void start(chrono::steady_clock::time_point origin) {
    receive();
    timeOrigin = origin;
    txTimer.expires_at(origin + flowDesc.startDelay);
    txTimer.async_wait([this](auto error) {
      if (error) {
        abort();
      }
      this->doSend();
    });
  }
};

void printNetStats(std::vector<FlowSim> &sims, asio::steady_timer &statTimer) {
  auto const now = chrono::steady_clock::now();
  for (auto &fs : sims) {
    fs.updateRates(now);
    std::cerr << format("\tflowUID %u delay %ld bytes %ld bps %f") %
                     fs.flowDesc.flowUID %
                     (fs.stats.delay().count() / 1000000) % fs.stats.bytes %
                     fs.stats.bps(flowRateWindow)
              << std::endl;
  }
  statTimer.expires_after(1000ms);
  statTimer.async_wait(
      [&sims, &statTimer](auto) { printNetStats(sims, statTimer); });
}

int main() {
  std::minstd_rand mt(1);
  std::uniform_int_distribution<> dis(0, 255);

  asio::io_context ioctx;

  std::vector<NotificationCenter::SubToken> tokens;
  tokens.push_back(NotificationCenter::shared.subscribe<dll::NewFlowEventInfo>(
      dll::NewFlowEvent, ioctx, [](auto ei) {
        std::cerr << format("NewFlowEvent ") << std::setw(4)
                  << nlohmann::json(ei) << std::endl;
      }));
  /*tokens.push_back(
      NotificationCenter::shared.subscribe<dll::CoDelStateEventInfo>(
          dll::CoDelStateEvent, ioctx, [](auto ei) {
            std::cerr << format("CoDelStateEvent ") << std::setw(4)
                      << nlohmann::json(ei) << std::endl;
          }));
  tokens.push_back(
      NotificationCenter::shared.subscribe<dll::CoDelDelayEventInfo>(
          dll::CoDelDelayEvent, ioctx, [](auto ei) {
            std::cerr << format("CoDelDelayEvent ") << std::setw(4)
                      << nlohmann::json(ei) << std::endl;
          }));*/
  tokens.push_back(
      NotificationCenter::shared.subscribe<dll::FlowQueuePushEventInfo>(
          dll::FlowQueuePushEvent, ioctx, [](auto ei) {
            std::cerr << format("FlowQueuePush ") << std::setw(4)
                      << nlohmann::json(ei) << std::endl;
          }));
  tokens.push_back(
      NotificationCenter::shared.subscribe<dll::FlowQueuePopEventInfo>(
          dll::FlowQueuePopEvent, ioctx, [](auto ei) {
            std::cerr << format("FlowQueuePop ") << std::setw(4)
                      << nlohmann::json(ei) << std::endl;
          }));
  tokens.push_back(NotificationCenter::shared.subscribe<log::DoomsdayEventInfo>(
      log::DoomsdayEvent, ioctx, [](log::DoomsdayEventInfo const &ei) {
        std::cerr << format("Doomsday ") << std::setw(4) << nlohmann::json(ei)
                  << std::endl;
        ei.judgement_day();
      }));
  tokens.push_back(NotificationCenter::shared.subscribe<log::TextLogEventInfo>(
      log::TextLogEvent, ioctx, [](log::TextLogEventInfo const &ei) {
        std::cerr << format("TextLog ") << ei.msg << std::endl;
      }));
  tokens.push_back(
      NotificationCenter::shared.subscribe<dll::FlowTrackerIMEventInfo>(
          dll::FlowTrackerIMEvent, ioctx, [](auto ei) {
            std::cerr << format("FlowTrackerIMEvent ") << std::setw(4)
                      << nlohmann::json(ei) << std::endl;
          }));
  tokens.push_back(
      NotificationCenter::shared
          .subscribe<dll::FlowTrackerStateUpdateEventInfo>(
              dll::FlowTrackerStateUpdateEvent, ioctx, [](auto ei) {
                std::cerr << format("FlowTrackerStateUpdateEvent ")
                          << std::setw(4) << nlohmann::json(ei) << std::endl;
              }));
  tokens.push_back(
      NotificationCenter::shared.subscribe<dll::FlowQueueResendEventInfo>(
          dll::FlowQueueResendEvent, ioctx, [](auto ei) {
            std::cerr << format("FlowQueueResendEvent ") << std::setw(4)
                      << nlohmann::json(ei) << std::endl;
          }));

  for (NodeID id : nodeIds) {
    auto node = std::make_shared<Node>(ioctx, id);
    Node::nodes[id] = node;
    std::cerr << "route " << node->flowNetwork().to_string() << std::endl;
    Node::routes[node->flowNetwork().to_string()] = node;
  }
  for (auto &np : Node::nodes) {
    auto node = np.second;
    for (NodeID srcId : nodeIds) {
      system((format("ip route add %s dev %s") %
              flowDestAddr(node->nodeID, srcId).to_string() %
              Node::nodes[srcId]->tunDevName())
                 .str()
                 .c_str());
      system((format("iptables -t nat -A POSTROUTING -s %s -d %s -j SNAT "
                     "--to-source %s") %
              node->address().to_string() %
              flowDestAddr(srcId, node->nodeID).to_string() %
              flowDestAddr(node->nodeID, srcId).to_string())
                 .str()
                 .c_str());
    }
    /*system(
        (format("iptables -t nat -A POSTROUTING -s %s -j SNAT --to-source %s") %
         node->address().to_string() %
         flowDestAddr(node->nodeID, 0).to_string())
            .str()
            .c_str());*/
    /*system((format("iptables -t nat -A PREROUTING -d %s -j DNAT "
                   "--to-destination %s") %
            node->flowNetwork().to_string() % node->address().to_string())*/
    system((format("iptables -t nat -A PREROUTING -i %s -j DNAT "
                   "--to-destination %s") %
            node->tunDevName() % node->address().to_string())
               .str()
               .c_str());
  }

  // Construct a signal set registered for process termination.
  boost::asio::signal_set signals(ioctx, SIGINT, SIGTERM);

  // Start an asynchronous wait for one of the signals to occur.
  signals.async_wait([](auto error, auto signum) {
    std::cerr << "SIG=" << signum << " error=" << error.value()
              << " msg: " << error.message() << std::endl;
    exit(0);
  });

  for (auto const &np : Node::nodes) {
    np.second->start();
  }

  asio::steady_timer statTimer(ioctx);
  // printQueueStats(statTimer);

  auto testnetthread = std::thread([] {
    asio::io_context ioctx;

    std::vector<FlowSim> sims;
    for (auto fd : flows) {
      sims.emplace_back(ioctx, fd);
    }
    auto const testNetOrigin = chrono::steady_clock::now() + 1100ms;
    for (auto &fs : sims) {
      fs.start(testNetOrigin);
    }

    asio::steady_timer statTimer(ioctx);
    // printNetStats(sims, statTimer);

    ioctx.run();
    std::cerr << "RIP test net" << std::endl;
  });

  ioctx.run();
  testnetthread.join();
  std::cerr << "main returning" << std::endl;
  return 1;
}
