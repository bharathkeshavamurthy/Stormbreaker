/*
 * Control channel Tx & Rx test
 * Copyright (c) 2017 Tomohiro Arakawa <tarakawa@purdue.edu>
 */

#include "../src/cc_data.h"
#include "../src/ctrl_ch.h"
#include "../src/gps.h"
#include "../src/options.h"
#include <boost/format.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <gnuradio/realtime.h>
#include <gnuradio/top_block.h>
#include <gnuradio/uhd/usrp_sink.h>
#include <gnuradio/uhd/usrp_source.h>
#include <iostream>
#include <string>
#include <thread>
#include <uhd/types/time_spec.hpp>
#include <vector>

int main(int argc, char const *argv[]) {
  using namespace bamradio;
  options::init(argc, argv);

  // ios
  // FIXME: just a temporary solution to use ccdata class
  boost::asio::io_service ios;
  auto *ios_work = new boost::asio::io_service::work(ios);
  std::thread work_thread([&] {
    ios.run();
    std::cout << "thread exiting" << std::endl;
  });

  // get options
  double sample_rate = options::phy::control::sample_rate;
  double scale = options::phy::control::atten;
  unsigned int num_fsk_points = options::phy::control::num_fsk_points;
  int rs_k = options::phy::control::rs_k;
  unsigned int N = options::phy::max_n_nodes;
  unsigned int ID = options::phy::control::id;
  double t_slot = options::phy::control::t_slot;

  // get the flowgraph
  auto tb = gr::make_top_block("bam_control");

  // housekeeping
  std::cout << "start uid:gid=" << getuid() << ":" << getgid()
            << "  euid:egid=" << geteuid() << ":" << getegid() << std::endl;
  const auto e = gr::enable_realtime_scheduling();
  if (e != gr::RT_OK) {
    std::cerr << "Failed enabling realtime scheduling (" << e << ")\n";
  }

  // get the radio
  auto sargs = uhd::stream_args_t("fc32", "sc16");
  sargs.args["underflow_policy"] = "next_burst";

  auto usrp_src = gr::uhd::usrp_source::make(
      uhd::device_addr_t(options::phy::uhd_args), sargs);
  auto usrp_sink = gr::uhd::usrp_sink::make(
      uhd::device_addr_t(options::phy::uhd_args), sargs);

  usrp_src->set_center_freq(options::phy::center_freq +
                            options::phy::control::freq_offset);
  usrp_src->set_samp_rate(options::phy::control::sample_rate);
  usrp_sink->set_center_freq(options::phy::center_freq +
                             options::phy::control::freq_offset);
  usrp_sink->set_samp_rate(options::phy::control::sample_rate);
  usrp_sink->set_gain(options::phy::control::tx_gain);

  // Set current time
  timespec realts;
  clock_gettime(CLOCK_REALTIME, &realts);
  usrp_sink->set_time_now(
      uhd::time_spec_t(realts.tv_sec, (double)realts.tv_nsec / 1000000000.0));

  auto info = usrp_sink->get_usrp_info(0);
  for (auto k : info.keys()) {
    std::cout << "\t" << k << "=" << info[k] << std::endl;
  }

  // CC Data
  auto cc_data = std::make_shared<controlchannel::CCData>(
      options::phy::control::id, options::phy::control::t_slot,
      options::collab::gateway);
  // ctrl channel
  auto ctrl_ch = controlchannel::ctrl_ch::make(
      cc_data, N, ID, sample_rate, t_slot, scale, num_fsk_points, rs_k, 600);

  // run the flowgraph
  tb->connect(usrp_src, 0, ctrl_ch, 0);
  tb->connect(ctrl_ch, 0, usrp_sink, 0);

  tb->start(options::phy::max_noutput_items);

  std::thread time_update([=] {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      timespec realts_update;
      clock_gettime(CLOCK_REALTIME, &realts_update);
      usrp_sink->set_time_now(uhd::time_spec_t(
          realts_update.tv_sec, (double)realts_update.tv_nsec / 1000000000.0));
    }
  });

  tb->wait();
  std::cout << "exiting..." << std::endl;
  return 0;
}
