// Bootstrapping experiment
//
// Copyright (c) 2018 Dennis Ogbe

#include "bootstrap.h"
#include "dll_types.h"
#include "events.h"
#include "options.h"
#include "util.h"

#include <algorithm>
#include <iostream>

#include <gnuradio/realtime.h>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#define OPTVAL(val)                                                            \
  boost::program_options::value<decltype(val)>(&val)->required()
#define OPTVAL_DEF(val, default)                                               \
  boost::program_options::value<decltype(val)>(&val)                           \
      ->required()                                                             \
      ->default_value(default)

int main(int argc, char const *argv[]) {
  namespace po = boost::program_options;
  using namespace bamradio;

  // parse options
  optparams opt;
  po::options_description desc("Options");
  desc.add_options()(
      // option
      "help", "help message")(
      // option
      "addr", OPTVAL(opt.dev_addr), "USRP IP address")(
      // option
      "bandwidth", OPTVAL_DEF(opt.bw, 480e3), "Signal bandwidth")(
      // option
      "center-freq", OPTVAL_DEF(opt.center_freq, 915e6), "Center frequency")(
      // option
      "lo-offset-tx", OPTVAL_DEF(opt.lo_offset_tx, -42e6), "LO offset (TX)")(
      // option
      "lo-offset-rx", OPTVAL_DEF(opt.lo_offset_rx, 42e6), "LO offset (RX)")(
      // option
      "tx-gain", OPTVAL_DEF(opt.tx_gain, 0.0), "TX gain")(
      // option
      "rx-gain", OPTVAL_DEF(opt.rx_gain, 0.0), "RX gain")(
      // option
      "num-nodes", OPTVAL(opt.num_nodes), "Number of nodes")(
      // option
      "it-interval-ms", OPTVAL_DEF(opt.it_interval_ms, 500),
      "Pause [ms] between iterations")(
      //  option
      "t-slot", OPTVAL(opt.t_slot), "slot time")(
      // option
      "server-ip", OPTVAL(opt.server_ip), "Server IP address")(
      // option
      "server-port", OPTVAL_DEF(opt.server_port, 6666), "Server port")(
      // option
      "my-ip", OPTVAL(opt.my_ip), "My IP address")(
      // option
      "my-port", OPTVAL_DEF(opt.my_port, 7777), "My port");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (std::exception &ex) {
    if (!vm.count("help")) {
      std::cout << "ERROR: " << ex.what() << "\n" << std::endl;
    }
    // help message
    std::cout << desc << std::endl;
    return EXIT_FAILURE;
  }

  // realtime
  const auto e = gr::enable_realtime_scheduling();
  if (e != gr::RT_OK) {
    std::cout << boost::format("Failed enabling realtime scheduling (%1%)") % e
              << std::endl;
  } else {
    std::cout << "Enabled realtime scheduling" << std::endl;
  }

  // make object, run experiment
  BootstrapExperiment exp(opt);
  exp.run();
}

namespace bamradio {

// Bootstrap Experiment

BootstrapExperiment::BootstrapExperiment(optparams const &options)
    : _opt(options), _hop_work(boost::asio::make_work_guard(_hop_ioctx)),
      _hop_thread([this] { _hop_ioctx.run(); }),
      _fakeDLL(std::make_shared<FakeDLL>()), _ctx(1),
      _srv_in_sock(_ctx, ZMQ_PULL), _srv_out_sock(_ctx, ZMQ_PUSH),
      _ioctx_work(boost::asio::make_work_guard(_ioctx)) {
  // set the options... Ugh
  _init_options();
  // connect to server and start radios
  _connect();
  _start_radios();
}

void BootstrapExperiment::run() {
  _running = true;
  _ioctx.post([this] { _await_instructions(); });
  _ioctx.run();
}

void BootstrapExperiment::_shutdown() {
  if (_running) {
    std::cout << boost::format("[ID: %1%] Shutting down.") % _my_id
              << std::endl;
    _running = false;
    _hopping = false;
    _ioctx_work.reset();
    _st.clear();
    _ioctx.stop();
    _tb->stop();
    _hop_work.reset();
    _hop_ioctx.stop();
    _hop_thread.join();
    _srv_in_sock.close();
    _srv_out_sock.close();
    _ctx.close();
  }
}

void BootstrapExperiment::_init_options() {
  // FIXME: this is the reason why we should not directly mention the options::
  // namespace inside of bam radio blocks... I am hacking this.
  options::phy::control::bandwidth = _opt.bw;
  options::phy::control::sample_rate = _opt.bw;
  options::phy::control::rs_k = 188;
  options::phy::control::num_fsk_points = 8;
  options::phy::control::atten = 1;
  options::phy::control::t_slot = _opt.t_slot;
  options::phy::control::max_delay = 1;
  options::phy::max_n_nodes = _opt.num_nodes;
  options::phy::control::ccsegment_interval = 1;
}

BootstrapExperiment::~BootstrapExperiment() { _shutdown(); }

// wait for commands from server. either shut down or start synchronizing.
void BootstrapExperiment::_await_instructions() {
  namespace cc = controlchannel;

  if (!_running) {
    return;
  }

  // notify server about READY state
  BAMDebugPb::BSCtoS ready;
  ready.set_allocated_ready(new BAMDebugPb::BSCtoSReady());
  ready.mutable_ready()->set_my_id(_my_id);
  _sendmsg(ready);
  auto const ack = _rxmsg();
  if (ack.payload_case() == BAMDebugPb::BSStoC::PayloadCase::kAck) {
    std::cout << boost::format("[ID: %1%] Ready, awaiting instructions.") %
                     _my_id
              << std::endl;
  } else if (ack.payload_case() == BAMDebugPb::BSStoC::PayloadCase::kStop) {
    _shutdown();
    return;
  } else {
    throw std::runtime_error("Should have received Ready ACK (or stop)");
  }
  // wait for server message and figure out what to do next
  auto const msg = _rxmsg();

  if (msg.payload_case() == BAMDebugPb::BSStoC::PayloadCase::kStop) {
    // we are done, shut down
    _shutdown();
    return;
  } else if (msg.payload_case() == BAMDebugPb::BSStoC::PayloadCase::kStart) {
    std::cout << boost::format("[ID: %1%] Starting Synchronization round.") %
                     _my_id
              << std::endl;
    // kill previous cc_data and cc tx
    _cc_data = nullptr;
    _cc_ctrl = nullptr;
    // we want to synchronize, start another round.
    _synchronizing = true;
    // subscribe to netmap notification
    std::vector<bamradio::NodeID> all_control_ids;
    for (auto const &id : msg.start().all_control_ids()) {
      all_control_ids.push_back(id);
    }
    _st.push_back(
        NotificationCenter::shared.subscribe<cc::CCPacketEventInfo>(
            cc::CCPacketEvent, _ioctx,
            [this, all_control_ids](auto ei) {
              if (!this->_running) {
                return;
              }
              if (ei.event_type != cc::CCPacketEventType::CCEVENT_RX) {
                return;
              }
              // write down time in case this was good
              auto const time =
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now() - _start);
              // check whether we are synchronized to the current set of control
              // IDs (we might have heard an old transmission)
              auto const srn_ids = _cc_data->getAllSRNIDs();
              bool good = true;
              for (auto const &id : all_control_ids) {
                if (std::find(begin(srn_ids), end(srn_ids), id) ==
                    end(srn_ids)) {
                  good = false;
                  break;
                }
              }
              if (_synchronizing && good) {
                // we are synchronized. report to server. then go back to
                // awaiting instructions state
                this->_report_restart(time);
              }
            }));
    auto new_control_id = msg.start().your_control_id();
    _cc_data = std::make_shared<cc::CCData>(new_control_id, _opt.t_slot, false);
    _ctrl_ch->set_cc_data(_cc_data);
    _cc_ctrl = controlchannel::CCController::make(
        _cc_data, _fakeDLL,
        [this](auto const &samples, auto nsamp, auto time_spec) -> bool {
          // TODO: check whether the time spec is realizable
          uhd::tx_metadata_t md;
          md.start_of_burst = true;
          md.end_of_burst = false;
          md.has_time_spec = true;
          md.time_spec = time_spec;
          auto const mspp = _txStreamer->get_max_num_samps();
          auto nsent = 0;
          while (nsent < nsamp) {
            nsent += _txStreamer->send(samples.data() + nsent,
                                       std::min(mspp, nsamp - nsent), md, 1.0);
            md.start_of_burst = false;
            md.has_time_spec = false;
          }
          md.end_of_burst = true;
          _txStreamer->send("", 0, md, 1.0);
          // FIXME
          return true;
        });
    _cc_ctrl->start();
    _start = std::chrono::system_clock::now();
  } else {
    throw std::runtime_error("Should have received Start or Stop.");
  }
}

void BootstrapExperiment::_report_restart(std::chrono::nanoseconds t) {
  // unsubscribe
  _st.clear();
  _synchronizing = false;

  // remove cc data from receiver
  _ctrl_ch->set_cc_data(nullptr);

  // report time to server
  BAMDebugPb::BSCtoS msg;
  msg.set_allocated_report(new BAMDebugPb::BSCtoSReport());
  msg.mutable_report()->set_synch_time(static_cast<int64_t>(t.count()));
  msg.mutable_report()->set_my_id(_my_id);
  _sendmsg(msg);

  // wait for acknowledgement from server
  auto const resp = _rxmsg();
  if (resp.payload_case() != BAMDebugPb::BSStoC::PayloadCase::kAck) {
    throw std::runtime_error("Report not Acked.");
  }

  // print output
  std::cout << boost::format("[ID: %1%] Synchronized in %2% ns.") % _my_id %
                   static_cast<int64_t>(t.count())
            << std::endl;

  // wait
  std::this_thread::sleep_for(std::chrono::milliseconds(_opt.it_interval_ms));

  // go back to beginning state
  _ioctx.post([this] { _await_instructions(); });
}

void BootstrapExperiment::_sendmsg(BAMDebugPb::BSCtoS &msg) {
  zmq::message_t m(msg.ByteSizeLong());
  msg.SerializeToArray(m.data(), m.size());
  bool sent = false;
  while (!sent) {
    try {
      sent = _srv_out_sock.send(m);
    } catch (zmq::error_t &ex) {
      if (ex.num() != EINTR)
        throw;
      sent = false;
    }
  }
}

BAMDebugPb::BSStoC BootstrapExperiment::_rxmsg() {
  zmq::message_t m;
  BAMDebugPb::BSStoC msg;
  bool rx = false;
  while (!rx) {
    try {
      rx = _srv_in_sock.recv(&m);
    } catch (zmq::error_t &ex) {
      if (ex.num() != EINTR)
        throw;
      rx = false;
    }
  }
  msg.ParseFromArray(m.data(), m.size());
  return msg;
}

void BootstrapExperiment::_connect() {
  std::cout << boost::format("[ID: ?] Connecting to server...") << std::endl;
  int i = 5;
  while (i-- > 0) {
    try {
      _srv_out_sock.connect(
          (boost::format("tcp://%1%:%2%") % _opt.server_ip % _opt.server_port)
              .str());
      _srv_in_sock.bind(
          (boost::format("tcp://%1%:%2%") % _opt.my_ip % _opt.my_port).str());
      break;
    } catch (std::exception &ex) {
      std::cout << "Connection Failed (" << ex.what() << "), retrying..."
                << std::endl;
    }
  }
  if (i <= 0) {
    throw std::runtime_error("Connection Timeout");
  }
  // set linger to zero so we can safely shut down the program
  int linger = 0;
  _srv_out_sock.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
  _srv_in_sock.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
  // send register message
  BAMDebugPb::BSCtoS msg;
  msg.set_allocated_register_(new BAMDebugPb::BSCtoSReg());
  msg.mutable_register_()->set_my_ip(_opt.my_ip);
  _sendmsg(msg);
  auto const resp = _rxmsg();
  if (resp.payload_case() == BAMDebugPb::BSStoC::PayloadCase::kAck) {
    _my_id = resp.ack().your_id();
    std::cout << boost::format("[ID: %1%] Connected") % _my_id << std::endl;
  } else {
    throw std::runtime_error("Registration error.");
  }
}

void BootstrapExperiment::_start_radios() {
  std::cout << boost::format("[ID: %1%] Connecting Radios") % _my_id
            << std::endl;

  // stream args
  auto stream_args = uhd::stream_args_t("fc32", "sc16");
  stream_args.args = uhd::device_addr_t("underflow_policy=next_burst");
  stream_args.channels = {0};
  auto dev_args = boost::format("addr=%1%,master_clock_rate=%2%") %
                  _opt.dev_addr % "184.32e6";

  // make URSP source (RX)
  _usrpSrc = gr::uhd::usrp_source::make(uhd::device_addr_t(dev_args.str()),
                                        stream_args);
  _usrpSrc->set_gain(_opt.rx_gain);
  _usrpSrc->set_samp_rate(_opt.bw);
  uhd::tune_request_t tr_rx(_opt.center_freq, _opt.lo_offset_rx);
  tr_rx.args = uhd::device_addr_t("mode_n=integer");
  _usrpSrc->set_center_freq(tr_rx);

  // make USRP sink (TX)
  _usrpSnk =
      gr::uhd::usrp_sink::make(uhd::device_addr_t(dev_args.str()), stream_args);
  _usrpSnk->set_gain(_opt.tx_gain);
  _usrpSnk->set_samp_rate(_opt.bw);
  uhd::tune_request_t tr_tx(_opt.center_freq, _opt.lo_offset_tx);
  tr_tx.args = uhd::device_addr_t("mode_n=integer");
  _usrpSnk->set_center_freq(tr_tx);
  _usrpSnk->set_time_now(system_clock_now());
  _txStreamer = _usrpSnk->get_device()->get_tx_stream(stream_args);

  // initialize tx streamer
  uhd::tx_metadata_t beg;
  beg.start_of_burst = true;
  beg.end_of_burst = false;
  beg.has_time_spec = false;
  _txStreamer->send("", 0, beg);
  beg.start_of_burst = false;
  beg.end_of_burst = true;
  _txStreamer->send("", 0, beg);

  // subscribe to tune request
  _hopping = true;
  _tuneToken = NotificationCenter::shared.subscribe<float>(
      controlchannel::CCTuneNotification, _hop_ioctx, [this](auto offset) {
        if (!_hopping) {
          return;
        }
        std::cout << boost::format("[ID: %1%] Hop to %2%") % _my_id % offset
                  << std::endl;
        ::uhd::tune_request_t tr1_tx(_opt.center_freq + offset,
                                     _opt.lo_offset_tx);
        tr1_tx.args = uhd::device_addr_t("mode_n=integer");
        _usrpSnk->set_center_freq(tr1_tx);
        ::uhd::tune_request_t tr1_rx(_opt.center_freq + offset,
                                     _opt.lo_offset_rx);
        tr1_rx.args = uhd::device_addr_t("mode_n=integer");
        _usrpSrc->set_center_freq(tr1_rx);
      });
  // start receiver + top block
  // FIXME figure out these parameters/ make sure they make sense/ see
  // _init_options function
  _ctrl_ch = controlchannel::ctrl_ch::make(nullptr, _opt.num_nodes, 0, _opt.bw,
                                           _opt.t_slot, 1, 8, 188, 600);
  _tb = gr::make_top_block("Bootstrap test");
  _tb->connect(_usrpSrc, 0, _ctrl_ch, 0);
  _tb->start();
  std::cout << boost::format("[ID: %1%] Radios Connected") % _my_id
            << std::endl;
}

// Fake DLL
FakeDLL::FakeDLL() : AbstractDataLinkLayer("fake dll", 1500, 0) {}
bool FakeDLL::running() { return false; }
void FakeDLL::start() {}
void FakeDLL::stop() {}
void FakeDLL::send(dll::Segment::sptr,
                   std::shared_ptr<std::vector<uint8_t>> backingStore) {}
void FakeDLL::asyncReceiveFrom(
    boost::asio::mutable_buffers_1 b, NodeID *node,
    std::function<void(net::IP4PacketSegment::sptr)> h) {}

} // namespace bamradio
