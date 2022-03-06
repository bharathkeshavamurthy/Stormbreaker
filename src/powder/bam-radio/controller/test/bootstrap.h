// Bootstrapping experiment
//
// Copyright (c) 2018 Dennis Ogbe

#ifndef de6cad0cb3520a6863bd
#define de6cad0cb3520a6863bd

#include "cc_controller.h"
#include "ctrl_ch.h"
#include "dll.h"
#include "notify.h"

#include <debug.pb.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include <gnuradio/top_block.h>
#include <gnuradio/uhd/usrp_sink.h>
#include <gnuradio/uhd/usrp_source.h>
#include <uhd/stream.hpp>

#include <boost/asio.hpp>
#include <zmq.hpp>

namespace bamradio {

//
// Save options here
//
struct optparams {
  std::string dev_addr;
  std::string server_ip;
  std::string my_ip;
  int server_port;
  int my_port;
  int num_nodes;
  int it_interval_ms;
  double bw;
  double t_slot;
  double center_freq;
  double lo_offset_rx;
  double lo_offset_tx;
  double tx_gain;
  double rx_gain;
};

//
// Fake DLL to make CCController happy
//
class FakeDLL : public AbstractDataLinkLayer {
public:
  FakeDLL();
  bool running();
  void start();
  void stop();
  void send(dll::Segment::sptr,
            std::shared_ptr<std::vector<uint8_t>> backingStore);
  void asyncReceiveFrom(boost::asio::mutable_buffers_1 b, NodeID *node,
                        std::function<void(net::IP4PacketSegment::sptr)> h);
};

//
// Howdy, I'm a boot strapping experiment.
//
class BootstrapExperiment {
public:
  BootstrapExperiment(optparams const &options);
  ~BootstrapExperiment();
  void run();

private:
  optparams const _opt;
  int32_t _my_id;

  std::atomic_bool _synchronizing;
  std::atomic_bool _running;

  // radio
  gr::top_block_sptr _tb;
  gr::uhd::usrp_source::sptr _usrpSrc;
  gr::uhd::usrp_sink::sptr _usrpSnk;
  uhd::tx_streamer::sptr _txStreamer;
  boost::asio::io_context _hop_ioctx;
  std::thread _hop_thread;
  std::atomic_bool _hopping;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      _hop_work;

  // control channel
  controlchannel::ctrl_ch::sptr _ctrl_ch;
  controlchannel::CCController::sptr _cc_ctrl;
  controlchannel::CCData::sptr _cc_data;
  std::shared_ptr<FakeDLL> _fakeDLL;

  // notify
  std::vector<NotificationCenter::SubToken> _st;
  NotificationCenter::SubToken _tuneToken;

  // server comms
  zmq::context_t _ctx;
  zmq::socket_t _srv_in_sock;
  zmq::socket_t _srv_out_sock;

  // work loop
  boost::asio::io_context _ioctx;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      _ioctx_work;

  // timing
  std::chrono::system_clock::time_point _start;

  // work methods
  void _sendmsg(BAMDebugPb::BSCtoS &msg);
  BAMDebugPb::BSStoC _rxmsg();
  void _init_options();
  void _connect();
  void _start_radios();
  void _await_instructions();
  void _report_restart(std::chrono::nanoseconds t);
  void _shutdown();
};

} // namespace bamradio

#endif // de6cad0cb3520a6863bd
