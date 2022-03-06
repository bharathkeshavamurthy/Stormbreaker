// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#ifndef e147f541c2c5d0eb19
#define e147f541c2c5d0eb19

#include "adaptive_mcs_controller.h"
#include "bbqueue.h"
#include "buffers.h"
#include "cc_controller.h"
#include "cc_data.h"
#include "channelizer2.h"
#include "discrete_channels.h"
#include "dll.h"
#include "fcomplex.h"
#include "gps.h"
#include "notify.h"
#include "psd.h"
#include "radiocontroller_types.h"
#include "sc_sync.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include <gnuradio/blocks/pdu_to_tagged_stream.h>
#include <gnuradio/top_block.h>
#include <gnuradio/uhd/usrp_sink.h>
#include <gnuradio/uhd/usrp_source.h>
#include <uhd/stream.hpp>

#include <boost/asio.hpp>

namespace bamradio {

/// monitor the usrp
class usrp_monitor {
private:
  std::thread _thread;
  std::atomic_bool _running;
  std::string const _name;

public:
  typedef std::shared_ptr<usrp_monitor> sptr;
  usrp_monitor(std::string const &name, ::uhd::usrp::multi_usrp::sptr dev);
  ~usrp_monitor();
};

class AbstractRadioController {
public:
  typedef std::shared_ptr<AbstractRadioController> sptr;

  virtual void start() = 0;
  virtual void stop() = 0;

  virtual ofdm::DataLinkLayer::sptr ofdmDll() const = 0;
  virtual controlchannel::CCData::sptr ccData() const = 0;

  virtual std::vector<Channel> ctrlChannelAlloc() = 0;
};

/// The layer 2 & 1 radio controller. Owns and controls the physical layer
/// chains and the data link layers on top of them.
class RadioController : public AbstractRadioController {
private:
  // chrono
  typedef std::chrono::system_clock::duration Duration;
  typedef std::chrono::system_clock::time_point Timepoint;

  // My SRN ID
  NodeID const _srnid;

  // [0] slot A: OFDM
  // [1] slot B: FSK
  gr::uhd::usrp_source::sptr _usrpSrc[2];
  gr::uhd::usrp_sink::sptr _usrpSnk[2];
  usrp_monitor::sptr _monitor;
  uhd::tx_streamer::sptr _txStreamer;
  int64_t _center_frequency;
  uhd::tx_streamer::sptr _txStreamer_fsk;

  // io_service
  boost::asio::io_service _ios;
  boost::asio::io_service::work *_ios_work;
  std::thread _work_thread; // Single thread

  // USRP Rx
  uhd::rx_streamer::sptr _rxStreamer;
  std::thread _reader;
  std::atomic_bool _reading;
  PinnedComplexRingBuffer::sptr _rx_rb;

  // Channelizer
  size_t const _n_rx_chains;
  std::vector<ofdm::ChannelOutputBuffer::sptr> _chanout;
  size_t _buf_sz; // channel output buffer buffer size
  boost::asio::io_context _channelizer_ios;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      _channelizer_ios_work;
  std::thread _channelizer_thread;
  std::atomic_bool _channelizer_running;
  bam::dsp::Channelizer2 _channelizer;
  bam::dsp::SCSync _synch;
  void _channelizer_synch_work();

  gr::top_block_sptr _tb;

  ofdm::DataLinkLayer::sptr _ofdmDll;
  ofdm::AdaptiveMCSController _adap_mcs;

  controlchannel::CCData::sptr _ccData;
  controlchannel::CCController::sptr _cccontroller;
  psdsensing::PSDSensing::sptr _psd;

  std::vector<NotificationCenter::SubToken> _subTokens;

  // channel allocation timer
  boost::asio::system_timer _timer_chan_update;

  void _changeChannels(
      std::map<NodeID, decisionengine::TransmitAssignment> channels);

public:
  typedef std::shared_ptr<RadioController> sptr;
  RadioController(boost::asio::io_service &net_ios);
  ~RadioController();

  void start();
  void stop();

  ofdm::DataLinkLayer::sptr ofdmDll() const { return _ofdmDll; }
  controlchannel::CCData::sptr ccData() const { return _ccData; }

  std::vector<Channel> ctrlChannelAlloc();
};

} // namespace bamradio

#endif
