// -*- c++ -*-
//  Copyright © 2017-2018 Stephen Larew

#ifndef de80f852cf02408c
#define de80f852cf02408c

#include "bbqueue.h"
#include "dll_types.h"
#include "flowtracker.h"
#include "frame.h"
#include "im.h"
#include "phy.h"
#include "radiocontroller_types.h"
#include "segmentqueue.h"
#include "tun.h"
#include "util.h"

#include <chrono>
#include <mutex>
#include <thread>

#include <boost/asio.hpp>
#include <boost/signals2.hpp>
#include <boost/optional.hpp>

namespace bamradio {

/// Abstract data link layer (layer 2) interface
class AbstractDataLinkLayer {
  typedef boost::signals2::signal<void(bool)> RunningSignal;

public:
  typedef std::shared_ptr<AbstractDataLinkLayer> sptr;
  typedef RunningSignal::slot_type RunningSignalSlotType;

  AbstractDataLinkLayer(std::string const &name, size_t mtu, NodeID n);

  std::string const &name() const;

  /// Maximum transmission unit (MTU) of this node
  virtual size_t mtu() const;
  /// Set the MTU of this node
  virtual void setMtu(size_t mtu);
  /// ID of this node
  virtual NodeID nodeID() const;
  /// Set the ID of this node
  virtual void setNodeID(NodeID id);

  /// Start the underlying DLL and PHY
  virtual void start() = 0;
  /// Stop the underlying DLL and PHY
  virtual void stop() = 0;
  /// Returns true if the underlying DLL and PHY are running
  virtual bool running() = 0;

  boost::signals2::connection observeRunning(RunningSignalSlotType const &slot);

  /// Send a Segment to a node.
  virtual void send(dll::Segment::sptr,
                    std::shared_ptr<std::vector<uint8_t>> backingStore) = 0;
  /// Receive the next Segment from a node.
  virtual void
  asyncReceiveFrom(boost::asio::mutable_buffers_1 b, NodeID *node,
                   std::function<void(net::IP4PacketSegment::sptr)> h) = 0;

protected:
  void checkSend(dll::Segment::sptr s, NodeID node);
  RunningSignal _runningSignal;

private:
  std::string const _name;
  size_t _mtu;
  NodeID _localNodeID;
};

class LoopbackDLL : public AbstractDataLinkLayer,
                    boost::asio::basic_io_object<BackgroundThreadService> {
private:
  BBQueue<std::pair<net::IP4PacketSegment::sptr,
                    std::shared_ptr<std::vector<uint8_t>>>>
      _pkts;
  bool _running;

public:
  LoopbackDLL(std::string const &name, boost::asio::io_service &ios,
              size_t queueSize);
  void start();
  void stop();
  bool running();
  void send(dll::Segment::sptr,
            std::shared_ptr<std::vector<uint8_t>> backingStore);
  void asyncReceiveFrom(boost::asio::mutable_buffers_1 b, NodeID *node,
                        std::function<void(net::IP4PacketSegment::sptr)> h);
};

namespace tun {

/**
 * DataLinkLayer for TUN device
 *
 * Adapts a TUN Device to the AbstractDataLinkLayer interface.
 */
class DataLinkLayer : public AbstractDataLinkLayer,
                      boost::asio::basic_io_object<BackgroundThreadService> {
private:
  Device::sptr _tunDev;

public:
  typedef std::shared_ptr<DataLinkLayer> sptr;

  DataLinkLayer(Device::sptr tunDev, boost::asio::io_service &ios);

  size_t mtu() const;
  void setMtu(size_t mtu);

  void start();
  void stop();
  bool running();

  Device::sptr const &device() const;

  void send(dll::Segment::sptr,
            std::shared_ptr<std::vector<uint8_t>> backingStore);
  void asyncReceiveFrom(boost::asio::mutable_buffers_1 b, NodeID *node,
                        std::function<void(net::IP4PacketSegment::sptr)> h);
};
} // namespace tun

namespace ofdm {

/**
 * DataLinkLayer for OFDM PHY
 *
 * Implements DLL functionality on top of a single-channel half-duplex TDD OFDM
 * PHY.
 */
class DataLinkLayer : public AbstractDataLinkLayer,
                      boost::asio::basic_io_object<BackgroundThreadService> {
private:
  // TX

  /// Segment queue
  DRRFQSegQ::sptr _txsq;

  std::mutex _txmutex;

  /// Next sequence number indexed by destinatiuon NodeID.
  std::array<uint16_t, 1 << (sizeof(NodeID) * 8)> _dstSeqNum;
  /// MCS and OFDM info
  std::array<std::pair<MCS::Name, SeqID::ID>, 1 << (sizeof(NodeID) * 8)>
      _dstInfo;
  MCS::Name const _hmcs;

  // current channel
  boost::optional<Channel> _txChan;

  uint16_t nextSeqNum(NodeID nid);

  phy_tx::prepared_frame prepareFrame();

  phy_tx::sptr const _txPhy;

  // RX

  struct RxSegment {
    NodeID srcNodeID;
    dll::Segment::sptr segment;
    std::shared_ptr<std::vector<uint8_t>> backingStore;
  };
  BBQueue<RxSegment> _rxSegmentQueue;

  struct RxDeframer {
    size_t block_number = 0;
    float frame_snr;
    DFTSOFDMFrame::sptr frame;
  };
  std::vector<RxDeframer> _rxDeframers;

  phy_rx::sptr const _rxPhy;

  bool _running;

  bool _allowSendToSelf;

  // ARQ Feedback
  std::mutex
      _trackermtx; // FIXME: This mutex locks the Tx and Rx threads for very
                   // short period of time. This might affect Tx/Rx performance.
  dll::FlowTracker _flow_tracker;
  bool _triggerARQFeedback;
  void _notifyARQFeedback(std::vector<dll::ARQBurstInfo> const &feedback,
                          std::chrono::steady_clock::time_point feedbackLTime,
                          std::chrono::system_clock::time_point feedbackGTime);

  std::map<FlowUID, IndividualMandate> _im, _imprev;
  std::array<LinkInfo, 1 << (sizeof(NodeID) * 8)> _defaultLinkInfo;
  std::map<FlowID, std::unique_ptr<FlowInfo>> _fi;
  QuantumSchedule _quantsched;
  int64_t _lastSchedUpdateRound;

  // A flow that is present at this DLL
  struct PresentFlow {
    NodeID nextHop;
    DRRFQSegQ::FlowQueue::Stats stats;
  };
  // flows in current epoch of mandates that appeared at this node
  std::map<FlowID, PresentFlow> _epochfid;
  // mandated flows in current epoch of mandates
  std::set<FlowUID> _epochfuids;

  boost::asio::io_context _schedctx;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      _schedctx_work;
  std::thread _schedthread;

  std::thread _specthread;
  void scheduleSpeculate();

  bool updateFlowInfo(std::unique_ptr<FlowInfo> &fi, FlowID fid,
                      DRRFQSegQ::SchedulerFlowQueue const &sfq,
                      LinkInfo const &li) const;

  // precondition: _txmutex is locked.
  void updateLinkInfos(NodeID nid);

  // precondition: _txmutex is locked.
  // new round-robin round (no domax)
  // change in mcs/ofdmseqid
  // channel change
  // mandate change
  void updateSchedule(DRRFQSegQ::FlowQueue::global_clock::time_point gnow);

public:
  typedef std::shared_ptr<DataLinkLayer> sptr;

  DataLinkLayer(std::string const &name, boost::asio::io_service &ios,
                phy_tx::sptr phy_tx, size_t rxFrameQueueSize, size_t mtu,
                NodeID n);
  ~DataLinkLayer();

  void start();
  void stop();
  bool running();

  void enableSendToSelf();
  void disableSendToSelf();
  bool allowsSendToSelf() const;

  void send(dll::Segment::sptr seg,
            std::shared_ptr<std::vector<uint8_t>> backingStore);
  void asyncReceiveFrom(boost::asio::mutable_buffers_1 b, NodeID *node,
                        std::function<void(net::IP4PacketSegment::sptr)> h);

  /// Set the MCS and OFDM sequence for transmitting to a specified receiver.
  void setMCSOFDM(NodeID rxnid, MCS::Name mcs, SeqID::ID seqid);
  std::pair<MCS::Name, SeqID::ID> getMCSOFDM(NodeID rxnid);
  /// Set the Tx channel
  void setTxChannel(boost::optional<Channel> chan);

  void addIndividualMandates(std::map<FlowUID, IndividualMandate> const &im);

  phy_tx::sptr tx() const;
  phy_rx::sptr rx() const;
};
} // namespace ofdm
} // namespace bamradio

#endif
