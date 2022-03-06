// -*- c++ -*-
//  Copyright © 2017-2018 Stephen Larew

#include "dll.h"
#include "cc_data.h"
#include "common.h"
#include "events.h"
#include "options.h"
#include "statistics.h"
#include "discrete_channels.h"

#include <atomic>
#include <iostream>
#include <random>

#include <linux/ip.h>

#include <boost/format.hpp>
#include <boost/optional.hpp>

#include <pmt/pmt.h>

namespace bamradio {

using namespace std::chrono_literals;
using namespace std::string_literals;
using namespace boost::asio;
using boost::format;

typedef std::chrono::duration<float, chrono::seconds::period> fsec;

AbstractDataLinkLayer::AbstractDataLinkLayer(std::string const &name,
                                             size_t mtu, NodeID n)
    : _name(name), _mtu(mtu), _localNodeID(n) {}
std::string const &AbstractDataLinkLayer::name() const { return _name; }
size_t AbstractDataLinkLayer::mtu() const { return _mtu; }
void AbstractDataLinkLayer::setMtu(std::size_t mtu) { _mtu = mtu; }
NodeID AbstractDataLinkLayer::nodeID() const { return _localNodeID; }
void AbstractDataLinkLayer::setNodeID(NodeID id) { _localNodeID = id; }

boost::signals2::connection
AbstractDataLinkLayer::observeRunning(RunningSignalSlotType const &slot) {
  return _runningSignal.connect(slot);
}

void AbstractDataLinkLayer::checkSend(dll::Segment::sptr, NodeID) {
  if (!running()) {
    throw std::runtime_error("DLL not running.");
  }
}

LoopbackDLL::LoopbackDLL(std::string const &name, boost::asio::io_service &ios,
                         size_t queueSize)
    : AbstractDataLinkLayer(name, 0x0fff, ExtNodeID), basic_io_object(ios),
      _pkts(queueSize), _running(true) {}

void LoopbackDLL::start() {
  if (_running)
    return;
  _running = true;
  _runningSignal(_running);
}
void LoopbackDLL::stop() {
  if (!_running)
    return;
  _running = false;
  _runningSignal(_running);
}
bool LoopbackDLL::running() { return _running; }

void LoopbackDLL::send(dll::Segment::sptr seg,
                       std::shared_ptr<std::vector<uint8_t>> backingStore) {
  net::IP4PacketSegment::sptr s;
  if (!(s = std::dynamic_pointer_cast<net::IP4PacketSegment>(seg))) {
    throw std::runtime_error("Non-IP segment.");
  }
  checkSend(s, s->destNodeID());
  if (s->packetLength() > mtu()) {
    throw std::runtime_error("Packet length exceeds MTU.");
  }
  _pkts.push({s, backingStore});
}

void LoopbackDLL::asyncReceiveFrom(
    boost::asio::mutable_buffers_1 b, NodeID *node,
    std::function<void(net::IP4PacketSegment::sptr)> h) {
  if (!running()) {
    throw std::runtime_error("DLL not running.");
  }

  get_service().work_ios.post([this, b, node, h] {
    auto const p = _pkts.pop();

    if (buffer_size(b) < p.first->length()) {
      throw std::runtime_error("Buffer too small in asyncReceiveFrom");
    }

    mutable_buffer bc = buffer(b);
    for (auto const &rcb : p.first->rawContentsBuffer()) {
      bc = bc + buffer_copy(bc, rcb);
    }
    if (node) {
      *node = nodeID();
    }
    auto const s = std::make_shared<net::IP4PacketSegment>(
        p.first->destNodeID(), buffer(b, p.first->length()));
    get_io_service().post([=] { h(s); });
  });
}

namespace tun {

DataLinkLayer::DataLinkLayer(Device::sptr tunDev, boost::asio::io_service &ios)
    : AbstractDataLinkLayer(tunDev->name(), tunDev->mtu(false), ExtNodeID),
      basic_io_object(ios), _tunDev(tunDev) {}

size_t DataLinkLayer::mtu() const { return _tunDev->mtu(); }
void DataLinkLayer::setMtu(size_t mtu) { return _tunDev->setMtu(mtu); }

void DataLinkLayer::send(dll::Segment::sptr seg,
                         std::shared_ptr<std::vector<uint8_t>> backingStore) {
  checkSend(seg, seg->destNodeID());
  net::IP4PacketSegment::sptr ip4seg;
  if (!(ip4seg = std::dynamic_pointer_cast<net::IP4PacketSegment>(seg))) {
    throw std::runtime_error("Non-IP segment.");
  }
  if (ip4seg->destNodeID() != ExtNodeID) {
    throw std::runtime_error("Must send to ExtNodeID on TUN interface.");
  }
  if (ip4seg->packetLength() > mtu()) {
    throw std::runtime_error("Packet length exceeds MTU.");
  }
  // TODO: do basic check that packet is an IPv4 packet
  _tunDev->descriptor().async_write_some(
      buffer(ip4seg->packetContentsBuffer()),
      [ip4seg, backingStore](auto error, auto bytes_transferred) {
        if (error) {
          log::text(
              (boost::format("tun write error %1%") % error.value()).str());
        }
        if (bytes_transferred < ip4seg->packetLength()) {
          log::text((boost::format("warning: short write to TUN (%1%/%2%)\n") %
                     bytes_transferred % ip4seg->packetLength())
                        .str(),
                    __FILE__, __LINE__);
        }
      });
}

void DataLinkLayer::asyncReceiveFrom(
    boost::asio::mutable_buffers_1 b, NodeID *node,
    std::function<void(net::IP4PacketSegment::sptr)> h) {
  if (!running()) {
    throw std::runtime_error("DLL not running.");
  }

  if (node) {
    *node = ExtNodeID;
  }

  _tunDev->descriptor().async_read_some(b, [this, b, node,
                                            h](auto error,
                                               auto bytes_transferred) {
    auto const hdr = boost::asio::buffer_cast<struct iphdr const *>(
        buffer(b, bytes_transferred));
    if (error || bytes_transferred == 0 || hdr->version != IPVERSION) {
      if (error) {
        log::text(
            (boost::format("async tun read error %1%") % error.value()).str());
      }
      // Try again?
      this->asyncReceiveFrom(b, node, h);
      return;
    }

    auto segment = std::make_shared<net::IP4PacketSegment>(
        UnspecifiedNodeID, buffer(b, bytes_transferred),
        std::chrono::system_clock::now());
    h(segment);
  });
}

void DataLinkLayer::start() {
  if (running())
    return;

  // I think start would more accurately toggle the IFF_RUNNING flag, not the UP
  // flag, but for now just change UP cause it's all we have.
  _tunDev->setUp();
  _runningSignal(running());
}

void DataLinkLayer::stop() {
  log::text("tun::DLL:stop() not implemented", __FILE__, __LINE__);
}

bool DataLinkLayer::running() { return _tunDev->isUp(); }

Device::sptr const &DataLinkLayer::device() const { return _tunDev; }

} // namespace tun

namespace ofdm {

DataLinkLayer::DataLinkLayer(std::string const &name,
                             boost::asio::io_service &ios, phy_tx::sptr phy_tx,
                             size_t rxFrameQueueSize, size_t mtu, NodeID n)
    : AbstractDataLinkLayer(name, mtu, n), basic_io_object(ios),
      // Deficit Round Robin Flow Queue
      _txsq(std::make_shared<DRRFQSegQ>(
          -1, // disable auto purge
          [this](FlowID const &flowId) -> DRRFQSegQ::FlowQueue::sptr {
            switch (flowId.proto) {
            case FlowID::Protocol::UDP: {
              if (!flowId.mandated()) {
                // FIXME non-mandated UDP flow... what to do? see
                // updateSchedule()
                return std::make_shared<DRRFQSegQ::LIFODelayDropQueue>(flowId,
                                                                       500ms);
              }
              auto const imp = _im.find(flowId.flowUID());
              if (imp == _im.end()) {
                if (_imprev.count(flowId.flowUID()) == 0) {
                  // No individual mandate found.
                  // UFO 🛸  (Unidentified Flow Object)
                  // Queue 'em up and pray.
                  //
                  // Actually, just hold the segments until we get the IM.
                  return std::make_shared<DRRFQSegQ::HoldQueue>(flowId);
                } else {
                  // Flow is for an old expired mandated. Drop it.
                  return std::make_shared<DRRFQSegQ::DropAllQueue>(flowId);
                }
              }
              // Now we have the im for the flow.
              auto &im = imp->second;
              return im.visit<DRRFQSegQ::FlowQueue::sptr>(
                  [flowId](auto const &streampt) {
                    // Last-in-first-out so we only send the fresh packets.
                    // Don't bother sending old packets.
                    return std::make_shared<DRRFQSegQ::LIFODelayDropQueue>(
                        flowId, chrono::duration_cast<decltype(
                                    DRRFQSegQ::LIFODelayDropQueue::maxDelay)>(
                                    streampt.max_latency));
                  },
                  [this, flowId](auto const &filept) {
                    // Never drop a file's pak ts.
                    // Unless we delayed them too long.
                    return std::make_shared<DRRFQSegQ::FIFOBurstQueue>(
                        flowId, _schedctx,
                        [this](auto const &f) {
                          std::lock_guard<decltype(_txmutex)> l(_txmutex);
                          f();
                        },
                        [this]() { _txPhy->continue_frame_maker(); },
                        chrono::duration_cast<chrono::nanoseconds>(
                            filept.transfer_duration),
                        chrono::duration_cast<chrono::nanoseconds>(
                            fsec(options::dll::arq_resend_timeout)),
                        chrono::duration_cast<chrono::nanoseconds>(
                            fsec(options::dll::
                                     burst_max_inter_segment_arrival_duration)),
                        options::dll::new_burst_inter_segment_arrival_ratio);
                  });
            } break;
            case FlowID::Protocol::PSD: {
              return std::make_shared<DRRFQSegQ::DestCoalesceQueue>(flowId);
            } break;
            case FlowID::Protocol::Control: {
              return std::make_shared<DRRFQSegQ::DestCoalesceQueue>(flowId);
            } break;
            default: {
              log::text(
                  (format("unexpected flow protocol %s") % flowId.description())
                      .str());
              return std::make_shared<DRRFQSegQ::DropAllQueue>(flowId);
            } break;
            }
          },
          [this](auto const gnow, auto const round, auto const fid,
                 bool reschedule) {
            if (round > _lastSchedUpdateRound || reschedule) {
              _lastSchedUpdateRound = round;
              this->updateSchedule(gnow);
            }
            std::chrono::nanoseconds debit = 0s, credit = 0s;
            try {
              credit = _quantsched.quantums.at(fid);
            } catch (std::out_of_range) {
            }
            try {
              std::unique_ptr<FlowInfo> &af = _fi.at(fid);
              if (af) {
                debit = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    af->link_info.frame_overhead +
                    fsec((af->link_info.segment_bit_overhead +
                          af->bits_per_segment) /
                         af->link_info.throughput_bps));
              } else {
                // FIXME we should create FlowInfo here too
              }
            } catch (std::out_of_range) {
              // FIXME try to create FlowInfo now
            }
            if (debit <= 0s) {
              credit = 0s;
            }
            return DRRFQSegQ::FlowSchedule{credit, debit};
          })),
      _hmcs(MCS::stringNameToIndex(options::phy::data::header_mcs_name)),
      _txChan(boost::none),
      // TX PHY
      _txPhy(phy_tx), _rxSegmentQueue(rxFrameQueueSize),
      // deframer struct
      _rxDeframers(options::phy::data::payload_demod_nthreads),
      // RX PHY
      _rxPhy(phy_rx::make(
          // header MCS
          MCS::stringNameToIndex(options::phy::data::header_mcs_name),
          options::phy::data::header_demod_nthreads,
          options::phy::data::payload_demod_nthreads,
          //
          // HEADER DEFRAMING
          //
          [this](auto f, auto bits, auto snr, auto, auto rxChainIdx, auto,
                 auto fdd_src) -> bool {
            // we intentionally ignore the parameter f in this call. we are
            // guaranteed that it is nullptr, because this is the header
            // deframer.
            assert(f == nullptr);
            // return value
            bool ret = false;
            // if we are not running, we signal a negative event to the FDD
            if (!_running) {
              fdd_src->notify(nullptr);
              return false;
            }
            // layer 2 detection time
            int64_t const rx_time =
                std::chrono::system_clock::now().time_since_epoch().count();
            // FIXME remove hardcoded constant
            auto const Nn = 108 * 3;
            assert(boost::asio::buffer_size(bits) == Nn);
            try {
              auto const frame = std::make_shared<DFTSOFDMFrame>(
                  MCS::stringNameToIndex(options::phy::data::header_mcs_name),
                  bits);
              NotificationCenter::shared.post(
                  dll::DetectedFrameEvent,
                  dll::DetectedFrameEventInfo{
                      rxChainIdx, frame->sourceNodeID(),
                      frame->destinationNodeID(), frame->payloadMcs(),
                      frame->payloadSymSeqID(), frame->seqNum(),
                      frame->frameID(), frame->numBlocks(false), rx_time, snr});
              ret = true;

              auto arq_feedback = frame->ARQFeedback();
              // notify scheduler of the new ARQ feedback
              if (frame->sourceNodeID() != this->nodeID()) {
                std::unique_lock<decltype(_txmutex)> txl(_txmutex);
                this->_notifyARQFeedback(frame->ARQFeedback(),
                                         std::chrono::steady_clock::now(),
                                         std::chrono::system_clock::now());
                txl.unlock();
                _txPhy->continue_frame_maker();
                // log
                for (auto &v : arq_feedback) {
                  NotificationCenter::shared.post(
                      dll::ReceivedARQFeedbackEvent,
                      dll::ReceivedARQFeedbackEventInfo{frame->frameID(),
                                                        v.flow_uid, v.burst_num,
                                                        v.seq_num});
                }
              }

              // Filter on src and dest NodeID
              if (frame->sourceNodeID() != this->nodeID() &&
                  (frame->destinationNodeID() == this->nodeID() ||
                   frame->destinationNodeID() == AllNodesID)) {
                fdd_src->notify(frame);
              } else {
                // FIXME in this case, we could also set _searching_skip at the
                // FDD to avoid searching through the current frame? --FFS
                fdd_src->notify(nullptr);
              }
            } catch (...) {
              NotificationCenter::shared.post(
                  dll::InvalidFrameHeaderEvent,
                  dll::InvalidFrameHeaderEventInfo{rxChainIdx, rx_time, snr});
              fdd_src->notify(nullptr);
            }
            return ret;
          },
          //
          // PAYLOAD DEFRAMING
          //
          [this](auto frame, auto bits, auto snr, auto noiseVar,
                 auto rxChainIdx, auto resourceIdx, auto) -> bool {
            if (!_running) {
              return false;
            }
            // we intentionally ignore the fdd_src parameter, because the
            // payload deframer does not feed back information to the
            // FDD. However, this deframer needs a frame object.
            assert(frame != nullptr);

            // return value
            bool ret = true;

            // l2 payload decode time
            auto const rx_time = std::chrono::system_clock::now();

            // initialize the deframer object n.b. rx_time needs to be figured
            // out.
            auto rxdf = &_rxDeframers[resourceIdx];
            rxdf->block_number = 0;
            rxdf->frame = frame;
            rxdf->frame_snr = snr;
            auto const blockK = rxdf->frame->readBlock(0, const_buffer()).first;
            assert(rxdf->frame != nullptr);

            // read the bits of the frame
            int64_t numBlocks = rxdf->frame->numBlocks(false);
            int64_t numBlocksValid = 0;
            while (boost::asio::buffer_size(bits) >= blockK) {
              auto const nread =
                  rxdf->frame->readBlock(rxdf->block_number, bits);
              if (nread.second == 0) {
                log::text("WARNING: zero blocks read", __FILE__, __LINE__);
              }
              for (size_t bn = rxdf->block_number;
                   bn < rxdf->block_number + nread.second; ++bn) {
                NotificationCenter::shared.post(
                    dll::ReceivedBlockEvent,
                    dll::ReceivedBlockEventInfo{
                        rxChainIdx, rxdf->frame->sourceNodeID(), nread.first,
                        rxdf->frame->blockIsValid(bn), rxdf->frame_snr,
                        rxdf->frame->frameID(), rxdf->frame->seqNum(),
                        (uint16_t)bn});
                if (!rxdf->frame->blockIsValid(bn)) {
                  ret = false;
                } else {
                  ++numBlocksValid;
                }
              }
              rxdf->block_number = rxdf->block_number + nread.second;
              bits = bits + nread.first * nread.second;
            }

            // write down the number of blocks that were valid vs the number of
            // blocks that were not.
            NotificationCenter::shared.post(
                dll::ReceivedFrameEvent,
                dll::ReceivedFrameEventInfo{
                    .sourceNodeID = rxdf->frame->sourceNodeID(),
                    .destNodeID = rxdf->frame->destinationNodeID(),
                    .frameID = rxdf->frame->frameID(),
                    .numBlocks = numBlocks,
                    .numBlocksValid = numBlocksValid,
                    .rxSuccess = numBlocks == numBlocksValid ? true : false,
                    .snr = snr,
                    .noiseVar = noiseVar});

            // push read segments onto the segment queue
            auto const segments = rxdf->frame->segments();
            auto const backingStore = rxdf->frame->movePayload();
            auto logSegment = [&](auto const &s, auto good) {
              nlohmann::json segmentJson = s;
              int64_t st = s->sourceTime().time_since_epoch().count();
              NotificationCenter::shared.post(
                  dll::ReceivedCompleteSegmentEvent,
                  dll::ReceivedCompleteSegmentEventInfo{
                      .flow = s->flowID(),
                      .sourceNodeID = rxdf->frame->sourceNodeID(),
                      .destNodeID = s->destNodeID(),
                      .seqNum = rxdf->frame->seqNum(),
                      .frameID = rxdf->frame->frameID(),
                      .rxTime = rx_time.time_since_epoch().count(),
                      .sourceTime = st,
                      .description = segmentJson,
                      .queueSuccess = good});
            };
            size_t numBytesRx = 0;
            bool triggerARQFeedback = false;
            for (auto const &s : segments) {
              numBytesRx += s->length();
              bool queueSuccess = true;
              if (s->type() == dll::SegmentType::IPv4 ||
                  s->type() == dll::SegmentType::ARQIPv4) {
                if (!this->_rxSegmentQueue.tryPush(
                        {rxdf->frame->sourceNodeID(), s, backingStore})) {
                  log::text(
                      "OFDM DLL: (MELTDOWN|ERROR|EMERGENCY) RX QUEUE FULL",
                      __FILE__, __LINE__);
                  queueSuccess = false;
                }
                auto arq_ip4_seg =
                    std::dynamic_pointer_cast<net::ARQIP4PacketSegment>(s);
                if (queueSuccess && arq_ip4_seg && arq_ip4_seg->arqDataSet()) {
                  triggerARQFeedback = true;
                  // Notify flow tracker
                  std::lock_guard<decltype(_trackermtx)> l(_trackermtx);
                  auto arq_data = arq_ip4_seg->arqData();
                  _flow_tracker.markReceived(
                      dll::ARQBurstInfo{.flow_uid = arq_ip4_seg->flowUID(),
                                        .burst_num = arq_data.burstNum,
                                        .seq_num = arq_data.seqNum,
                                        .extra = arq_data.arqExtra},
                      arq_ip4_seg->sourceTime(), rx_time,
                      arq_ip4_seg->length());
                }
                logSegment(s, queueSuccess);
              } else if (s->type() == dll::SegmentType::PSD) {
                if (s->destNodeID() == this->nodeID()) {
                  // FIXME HACK
                  NotificationCenter::shared.post(
                      std::hash<std::string>{}("New Rx PSD Segment"),
                      std::make_pair(s, backingStore));
                  logSegment(s, queueSuccess);
                } else {
                  if (!this->_rxSegmentQueue.tryPush(
                          {rxdf->frame->sourceNodeID(), s, backingStore})) {
                    log::text(
                        "OFDM DLL: (MELTDOWN|ERROR|EMERGENCY) RX QUEUE FULL",
                        __FILE__, __LINE__);
                    queueSuccess = false;
                  }
                  logSegment(s, queueSuccess);
                }
              } else if (s->type() == dll::SegmentType::Control) {
                // FIXME HACK
                NotificationCenter::shared.post(
                    std::hash<std::string>{}("New Rx CC Segment"),
                    std::make_pair(s, backingStore));
                logSegment(s, queueSuccess);
              } else {
                log::text("WARNING: Invalid Segment", __FILE__, __LINE__);
              }
            }
            if (triggerARQFeedback) {
              _triggerARQFeedback = true;
              _txPhy->continue_frame_maker();
            }
            return ret;
          })),
      _running(false), _allowSendToSelf(false), _triggerARQFeedback(false),
      _quantsched({{}, true, 0s, 0s, 0s}),
      _lastSchedUpdateRound(std::numeric_limits<int64_t>::min()),
      _schedctx_work(boost::asio::make_work_guard(_schedctx)),
      _schedthread([this] {
        bamradio::set_thread_name("dllsched");
        _schedctx.run();
      }) {
  for (auto &dsn : _dstSeqNum) {
    // start at 0 because nextSeqNum increments before returning
    dsn = 0;
  }
  auto const init_mcs =
      MCS::stringNameToIndex(options::phy::data::initial_payload_mcs_name);
  auto const init_seq = SeqID::stringNameToIndex(
      options::phy::data::initial_payload_symbol_seq_name);
  for (auto &dmcs : _dstInfo) {
    dmcs = {init_mcs, init_seq};
  }
  _specthread = std::thread([this] {
    bamradio::set_thread_name("dllspec");
    return;
    this->scheduleSpeculate();
  });
}

DataLinkLayer::~DataLinkLayer() {
  _schedctx_work.reset();
  _schedctx.stop();
  _schedthread.join();
}

void DataLinkLayer::start() {
  _schedctx.dispatch([this] {
    std::unique_lock<decltype(_txmutex)> l(_txmutex);
    if (_running)
      return;
    _running = true;
    l.unlock();
    _txPhy->start([this] { return prepareFrame(); });
    _runningSignal(true);
  });
}

void DataLinkLayer::stop() {
  _schedctx.dispatch([this] {
    std::unique_lock<decltype(_txmutex)> l(_txmutex);
    if (!_running)
      return;
    _running = false;
    l.unlock();
    _runningSignal(false);
  });
}

bool DataLinkLayer::running() {
  std::lock_guard<decltype(_txmutex)> l(_txmutex);
  return _running;
}

void DataLinkLayer::enableSendToSelf() { _allowSendToSelf = true; }
void DataLinkLayer::disableSendToSelf() { _allowSendToSelf = false; }
bool DataLinkLayer::allowsSendToSelf() const { return _allowSendToSelf; }

void DataLinkLayer::send(dll::Segment::sptr seg,
                         std::shared_ptr<std::vector<uint8_t>> backingStore) {
  checkSend(seg, seg->destNodeID());

  if (!_allowSendToSelf && seg->destNodeID() == nodeID()) {
    throw std::runtime_error("Refusing to send to self.");
  }

  auto const ip4seg = std::dynamic_pointer_cast<net::IP4PacketSegment>(seg);
  if (ip4seg != nullptr) {
    if (ip4seg->packetLength() > mtu()) {
      throw std::runtime_error("Packet length exceeds MTU.");
    }
  }
  // FIXME: else if (seg->length() > mtu()) ...

  _schedctx.post([this, seg, backingStore] {
    {
      std::lock_guard<decltype(_txmutex)> l(_txmutex);
      _txsq->push({seg, backingStore});
    }
    _txPhy->continue_frame_maker();
  });
}

void DataLinkLayer::asyncReceiveFrom(
    boost::asio::mutable_buffers_1 b, NodeID *node,
    std::function<void(net::IP4PacketSegment::sptr)> h) {
  if (!running()) {
    throw std::runtime_error("DLL not running.");
  }

  get_service().work_ios.post([this, b, node, h] {
    // Get the next segment intended for this interface.
    RxSegment rxs;
    NodeID destNodeID;
    do {
      rxs = _rxSegmentQueue.pop();
      destNodeID = rxs.segment->destNodeID();
      // Ensure frame is this interface. Otherwise, discard it and grab next.
    } while (!(destNodeID == nodeID() || destNodeID == AllNodesID));

    if (buffer_size(b) < rxs.segment->length()) {
      throw std::runtime_error("Buffer too small in asyncReceiveFrom");
    }

    mutable_buffer bc = buffer(b);
    for (auto const &rcb : rxs.segment->rawContentsBuffer()) {
      buffer_copy(bc, rcb);
      bc = bc + buffer_size(rcb);
    }

    // FIXME handle control segment here

    net::IP4PacketSegment::sptr seg;
    net::ARQIP4PacketSegment::sptr arq_ip4_seg;
    switch (rxs.segment->type()) {
    case dll::SegmentType::IPv4: {
      seg = std::make_shared<net::IP4PacketSegment>(
          rxs.segment->destNodeID(), buffer(b, rxs.segment->length()));
    } break;
    case dll::SegmentType::ARQIPv4: {
      arq_ip4_seg = std::make_shared<net::ARQIP4PacketSegment>(
          rxs.segment->destNodeID(), buffer(b, rxs.segment->length()));
      seg = arq_ip4_seg;
    } break;
    default: {
      log::doomsday("Unknwon segment type. Unable to receive a segment.");
    }
    }

    if (node) {
      *node = rxs.srcNodeID;
    }

    get_io_service().post([=] { h(seg); });
  });
}

phy_tx::prepared_frame DataLinkLayer::prepareFrame() {
  std::unique_lock<decltype(_txmutex)> l(_txmutex);
  if (!_txChan) {
    return {nullptr, Channel(0, 0, 0)};
  }

  phy_tx::prepared_frame r = {nullptr, *_txChan};

  // Service the segment queue
  std::vector<QueuedSegment> qsv;
  // while (qsv.empty() && _txsq->numReady() > 0) {
  qsv = _txsq->pop(15ms, false, false); // FIXME constant
  //}

  l.unlock();

  // ARQ feedback
  // TODO: do round robin ARQ feedback
  std::unique_lock<decltype(_trackermtx)> tl(_trackermtx);
  auto arq_data = _flow_tracker.getLastSeqNums();
  tl.unlock();

  do {
    if (qsv.empty() && !_triggerARQFeedback) {
      break;
    }

    // Pack segments into a vector.
    std::vector<dll::Segment::sptr> segVec;
    segVec.reserve(qsv.size());
    for (auto const &qs : qsv) {
      segVec.push_back(qs.segment);
    }

    auto const dstNodeID =
        qsv.empty() ? AllNodesID : SegmentQueue::effectiveFrameDestNodeID(qsv);
    auto const dstInfo = getMCSOFDM(dstNodeID);
    auto const seqnum = nextSeqNum(dstNodeID);
    try {
      r.frame = std::make_shared<DFTSOFDMFrame>(
          nodeID(), dstNodeID, segVec, _hmcs, std::get<0>(dstInfo),
          std::get<1>(dstInfo), seqnum,
          std::chrono::steady_clock::now().time_since_epoch().count());
      r.frame->setARQFeedback(arq_data);
      _triggerARQFeedback = false;
    } catch (DFTSOFDMFrame::TooManySegments) {
      // Too many segments.  Requeue the last segment and try again.
      l.lock();
      _txsq->push(qsv.back());
      l.unlock();
      qsv.pop_back();
      r.frame = nullptr;
    } catch (std::exception e) {
      panic("Failed making frame"s + e.what());
    } catch (...) {
      panic("Failed making frame");
    }
  } while (!r.frame);

  return r;
}

uint16_t DataLinkLayer::nextSeqNum(NodeID const nid) {
  std::lock_guard<decltype(_txmutex)> l(_txmutex);
  auto &sn = _dstSeqNum[nid];
  ++sn;
  if (sn == 0 || sn >= (1 << SeqNumBitlength)) {
    // skip zero
    sn = 1;
  }
  return sn;
}

static LinkInfo computeLinkInfo(float const bandwidth, ofdm::MCS mcs,
                                ofdm::SeqID::ID seqid);

// Must hold _txmutex locked while calling
bool DataLinkLayer::updateFlowInfo(std::unique_ptr<FlowInfo> &fi, FlowID fid,
                                   DRRFQSegQ::SchedulerFlowQueue const &sfq,
                                   LinkInfo const &li) const {
  IndividualMandate im;
  if (fid.mandated()) {
    try {
      im = _im.at(fid.flowUID());
    } catch (std::out_of_range) {
      // FIXME handle pending IM how?
      return false;
    }
  } else {
    // FIXME dropqueue!!
    // FIXME fake IM (and synchronize with newFlow cb above)
    // Adding a default pointValue of 1 for non-mandated flows (AllNodes messages are handled differently in this routine's priority calculation logic)
    im = IndividualMandate{1, IndividualMandate::StreamPT{500e2, fsec(4.0)}};
  }
  // The point value for the flow is extracted from the IndividualMandate here and the priority metric is modified using this extracted point value.
  unsigned int point_value = 1;
  if (im.point_value && im.point_value > 1) {
    point_value = im.point_value;
  }
  auto prio = im.visit<int>(
      [](auto const &) { return 50; },
      [](auto const &) { return 50 + options::dll::sched_filept_rel_prio; });
  // Mandated flows (stream or file)
  if (fid.mandated()) {
    prio += point_value * 100;
  }
  // Control messages
  if (sfq.destNodeID() == AllNodesID) {
    prio += 1500;
  }
#warning Bias priority for target flow set
  /* When a flow is so new that it lacks valid stats, use the qual segments
   * to start.
   *
   * qual segments: 661 bytes
   * -8 bytes sourcetime = 653 bytes
   * -8 bytes UDP header = 645 bytes
   * -20 bytes IP header = 625 bytes
   */
  size_t const bpseg =
      8 * (sfq.q->stats().valid() ? sfq.q->stats().maxLength : (size_t)661);
  size_t const bpsegp =
      8 *
      (sfq.q->stats().valid() ? sfq.q->stats().maxPayloadLength : (size_t)653);
  // Compute ratio of time duration of one segment to frame header.
  float const segtoframeeff =
      fsec((li.segment_bit_overhead + bpseg) / li.throughput_bps) /
      li.frame_overhead;
  // Compute number of segments needed to get to at least 10/1
  // segment-to-frameheader efficiency.
  // FIXME 10/1 payload to header constant
  auto numsegspf = (size_t)((10.0f / segtoframeeff) + 1.0f);
  if (sfq.destNodeID() == AllNodesID) {
    // Flows that are broadcast can have a 1/1 segment-to-frameheader
    // efficiency. e.g. control segment flows
    numsegspf = 1;
  }
  // FIXME alpha beta constants
  float const alpha = options::dll::sched_alpha;
  float const beta = options::dll::sched_beta;
  // The new priority metric is a part of the FlowInfo data structure for value/Hz/duty-cycle evaluation in scheduleMaxFlows and scheduleFlows routines.
  FlowInfo newfi{im, li, bpseg, bpseg - bpsegp, numsegspf, alpha, beta, prio};
  if (!fi) {
    fi = std::make_unique<FlowInfo>(newfi);
  } else {
    *fi = newfi;
  }
  return true;
}

void DataLinkLayer::scheduleSpeculate() {
  // Similar to updateSchedule except we iterate over larger/smaller channel
  // bandwidths and MCS pairs to get speculative LinkInfos to compute schedule
  // with.  Consider all flows in current im epoch that have appeared at this
  // node.  Then,

  // updateFlowInfo needs a queue to get destNodeID and flowqueue stats.
  // Don't use actual flowqueue but instead fake it.
  class SpeculativeQueue : public DRRFQSegQ::FlowQueue {
  public:
    SpeculativeQueue(FlowID id, NodeID nextHop)
        : DRRFQSegQ::FlowQueue(id), _nid(nextHop) {}
    void enqueue(QueuedSegment const &, int64_t &, local_clock::time_point) {
      unimplemented();
    }
    bool dequeue(QueuedSegment &, int64_t &, global_clock::time_point,
                 local_clock::time_point) {
      unimplemented();
    }
    NodeID headDestNodeID() const { return _nid; }
    size_t numQueued() const { return 1; }
    size_t bytesQueued() const { return 1; }
    void handleARQInfo(dll::ARQBurstInfo, local_clock::time_point,
                       global_clock::time_point) {}

  private:
    NodeID _nid;
  };

  FlowInfoMap fim;
  std::map<FlowUID, FileFlowProgress> ffpm;
  std::vector<std::unique_ptr<FlowInfo>> tempfi;

  std::unique_lock<std::mutex> l(_txmutex);

  std::map<FlowID, DRRFQSegQ::SchedulerFlowQueue> flowQueues =
      _txsq->flowQueues();

  // TODO ensure cc flow in flowQueues
  // TODO stabilize the flowQueues

  for (auto const &pfp : _epochfid) {
    auto const &fid = pfp.first;
    auto const &pf = pfp.second;

    // fake the sfq
    DRRFQSegQ::SchedulerFlowQueue sfq(
        std::make_shared<SpeculativeQueue>(fid, pf.nextHop), 0);

    tempfi.emplace_back(std::make_unique<FlowInfo>());
    std::unique_ptr<FlowInfo> &fi = tempfi.back();
    /*if (!updateFlowInfo(fi, fid, sfq, li)) {
      continue;
    }*/
  }
}

// Must hold _txmutex locked while calling
void DataLinkLayer::updateLinkInfos(NodeID const nid) {
  if (!_txChan) {
    // effectively disable the channel with large values
    // don't use numeric max,min values because we don't want to overflow/crash
    // when scheduling
    // FIXME better solution than effective disablement
    // FIXME: Is 144kHz the right default bandwidth to be used in this setting?
    _defaultLinkInfo[nid] = {fsec(1e9), 0, 1e-9, 1e-9, 1e-9, 144000};
    return;
  }

  /* WARNING -- we are now lying to the scheduler. Let me explain. We used to do
     this:

     auto const &channel = _channels[_txChanIdx];
     auto const &mcs = MCS::table[_dstInfo[nid].first];
     auto const seqid = _dstInfo[nid].second;
     _defaultLinkInfo[nid] = computeLinkInfo(channel.bandwidth, mcs, seqid);

     But with the "new" bandwidth allocation from August 2019, this creates a
     chicken and egg problem, in that a low bandwidth drives the scheduler to
     not schedule any flows, which in turn drives the bandwidth adaptation to
     not expand the bandwidth. So we are now trying to give the scheduler
     instead of the *current* bandwidth the *maximum possible* bandwidth and let
     any duty cycle problems that result form this be taken care of by the
     gateway. Let's hope this works ¯\_(ツ)_/¯
  */

  auto const &mcs = MCS::table[_dstInfo[nid].first];
  auto const seqid = _dstInfo[nid].second;
  _defaultLinkInfo[nid] = computeLinkInfo(
      decisionengine::Channelization::get_current_max_bandwidth(), mcs, seqid);
}

// Must hold _txmutex locked while calling
void DataLinkLayer::updateSchedule(
    DRRFQSegQ::FlowQueue::global_clock::time_point gnow) {
  // Get active flows.
  auto const &flowQueues = _txsq->flowQueues();
  FlowInfoMap fim;
  std::map<FlowUID, FileFlowProgress> ffpm;
  boost::format fifmt("\n\t%s im %s li %s bpseg %lu bpsegp %lu numseg %lu "
                      "alpha %f beta %f prio %d");
  std::string fistr;
  fistr.reserve(1 << 13);
  for (auto const &fqp : flowQueues) {
    auto const &fid = fqp.first;
    auto const &sfq = fqp.second;
    // Create or update the FlowInfo.
    std::unique_ptr<FlowInfo> &fi = _fi[fid];
    if (!updateFlowInfo(fi, fid, sfq, _defaultLinkInfo[sfq.destNodeID()])) {
      continue;
    }
    // Save/update this flow's info for current IM epoch.
    _epochfid.emplace(fid, PresentFlow{sfq.destNodeID(), sfq.q->stats()});
    // remember this flowid was active (active in scheduler non-purged sense!)
    // in this IM epoch.
    if (fid.mandated()) {
      if (_im.find(fid.flowUID()) != _im.end()) {
        // 1) flow is mandated
        // 2) we know the flow's mandate
        // 3) the mandate is for the current IM epoch
        // -> remember it
        _epochfuids.insert(fid.flowUID());
        // notify CCData
        NotificationCenter::shared.post(
            dll::NewActiveFlowEvent,
            dll::NewActiveFlowEventInfo{fid.flowUID(), fi->bits_per_segment});
      } else {
        // NB: we don't know the flow's mandate yet
      }
    }
    boost::format fifmtc(fifmt);
    fistr += (fifmtc % fid.description() % fi->im.description() %
              fi->link_info.description() % fi->bits_per_segment %
              fi->segment_payload_bit_overhead % fi->min_segments_per_frame %
              fi->alpha % fi->beta % fi->priority)
                 .str();
    // Should we schedule for this mandate? Streams: always try. Files: only if
    // ready, non-expired, and (estaimted or exact) remaining bytes is known.
    if (!fi->im.visit<bool>(
            [](auto const &) {
              // Always try to schedule a stream regardless of its mandated
              // performance thresholds.
              return true;
            },
            [&](auto const &) {
              auto it = ffpm.emplace(fid.flowUID(), FileFlowProgress{}).first;
              // FIXME dropburst how?
              auto const p =
                  std::dynamic_pointer_cast<DRRFQSegQ::FIFOBurstQueue>(sfq.q);
              if (!p) {
                // Maybe it's a HoldQueue? Well, whatever it is, the queue is
                // not a FIFOBurstQueue that tracks the file bursts so we have
                // to bail.
                return false;
              }
              if (p->numQueued() <= 0) {
                return false;
              }
              auto const br = p->headBurstBytesRemaining(gnow);
              if (br < 0) {
                return false;
              }
              it->second =
                  FileFlowProgress{p->headBurstDeadline(), (size_t)(br * 8)};
              return true;
            })) {
      continue;
    }
    fim.emplace(fid, fi.get());
  }
  // Get the schedule for this FlowInfoMap
  auto newsched = scheduleMaxFlows(fim, ffpm, gnow, MaxFlowsSearch::RemoveMinMaxLatencyAndMinValue, true);
  if (newsched != _quantsched) {
    _quantsched = newsched;
    NotificationCenter::shared.post(
        dll::ScheduleUpdateEvent,
        dll::ScheduleUpdateEventInfo{_lastSchedUpdateRound,
                                     _quantsched.quantums, _quantsched.valid,
                                     _quantsched.period, _quantsched.periodlb,
                                     _quantsched.periodub, fistr});
  }
  if (!_quantsched.valid) {
    // FIXME: realloc channel?
    panic("No max flows schedule found!");
  }
}

void DataLinkLayer::setMCSOFDM(NodeID const rxnid, MCS::Name const mcs,
                               SeqID::ID const seqid) {
  _schedctx.dispatch([=] {
    std::lock_guard<decltype(_txmutex)> l(_txmutex);
    auto &di = _dstInfo[rxnid];
    if (di == std::make_pair(mcs, seqid)) {
      return;
    }
    di = {mcs, seqid};
    updateLinkInfos(rxnid);
    updateSchedule(std::chrono::system_clock::now());
  });
}

std::pair<MCS::Name, SeqID::ID> DataLinkLayer::getMCSOFDM(NodeID const rxnid) {
  std::lock_guard<decltype(_txmutex)> l(_txmutex);
  // FIXME floor mcs allnodes
  return _dstInfo[rxnid];
}

void DataLinkLayer::setTxChannel(boost::optional<Channel> chan) {
  _schedctx.dispatch([this, chan] {
    _txChan = chan;
    for (size_t nid = 0; nid < _dstInfo.size(); ++nid) {
      updateLinkInfos((NodeID)nid);
    }
    updateSchedule(std::chrono::system_clock::now());
  });
}

void DataLinkLayer::addIndividualMandates(
    std::map<FlowUID, IndividualMandate> const &imnew) {
  _schedctx.dispatch([=] {
    {
      std::lock_guard<decltype(_trackermtx)> l(_trackermtx);
      _flow_tracker.addIndividualMandates(imnew);
    }
    std::lock_guard<decltype(_txmutex)> l(_txmutex);
    // Purge current mandated flows before updating list.
    std::set<FlowID> topurge;
    for (auto const &fq : _txsq->flowQueues()) {
      if (fq.first.mandated() && imnew.count(fq.first.flowUID()) == 0) {
        // fq is mandated but not in new IM set so forget it.
        topurge.emplace(fq.first);
      }
    }
    _txsq->purgeFlowQueues(topurge);
    // We no longer "add" the mandates but instead forget all previous mandates
    // and only use new set of mandates.
    for (auto const &im : _im) {
      // But we do remember this mandate was previously seen...
      // this way we don't create holdqueues for old mandates
      _imprev[im.first] = im.second;
    }
    // Move to next IM epoch and clear lists. Then, add IMs to list with new
    // epoch tag.
    _epochfid.clear();
    _epochfuids.clear();
    _im = imnew;
    updateSchedule(std::chrono::system_clock::now());
  });
}

phy_tx::sptr DataLinkLayer::tx() const { return _txPhy; }
phy_rx::sptr DataLinkLayer::rx() const { return _rxPhy; }

/// Adding a new bandwidth member to the LinkInfo struct for value/Hz evaluation
LinkInfo computeLinkInfo(float const bandwidth, ofdm::MCS mcs,
                         ofdm::SeqID::ID seqid) {

  // FIXME this all should be computed by a frame...

  float const osymbol_duration_s =
      (ofdm::SeqID::cpLen(seqid) + ofdm::SeqID::symLen(seqid)) / bandwidth;

  float const K = mcs.blockLength * mcs.codeRate;

  // block header efficiency
  float const alpha = (K - 8 * ofdm::DFTSOFDMFrame::BlockHeaderSize) / K;

  // info bits per ofdm symbol
  float const ibpos =
      alpha * ofdm::SeqID::bpos(seqid) * mcs.codeRate.k / mcs.codeRate.n;

  // 5 symbols before payload + estimated xx for padding
  // TODO have multiplier for FlowInfo::bits_per_segment to account for
  // multiple segments per frame and hence increased header-to-payload
  // efficiency. Until then, just lower the frame overhead.  Ugh.
  // FIXME goodput and arq goodput

  // Adding a new bandwidth member to this struct for value/Hz evaluation further down the line...
  return {options::dll::sched_frame_overhead * fsec(osymbol_duration_s), 0, ibpos / osymbol_duration_s,
          options::dll::sched_goodput * ibpos / osymbol_duration_s,
          options::dll::sched_arq_goodput * ibpos / osymbol_duration_s,
          bandwidth};

}

// Must hold _txmutex locked while calling
void DataLinkLayer::_notifyARQFeedback(
    std::vector<dll::ARQBurstInfo> const &feedback,
    std::chrono::steady_clock::time_point feedbackLTime,
    std::chrono::system_clock::time_point feedbackGTime) {
  // FIXME move this inside DRRFQSegQ
  auto const &queues = _txsq->flowQueues();
  // iterate over all feedback messages
  for (auto const &f : feedback) {
    // iterate over all flow queues
    for (auto &q : queues) {
      if (q.first.flowUID() == f.flow_uid) {
        auto &flow_queue = q.second.q;
        assert(flow_queue != nullptr);
        flow_queue->handleARQInfo(f, feedbackLTime, feedbackGTime);
      }
    }
  }
}

} // namespace ofdm
} // namespace bamradio

