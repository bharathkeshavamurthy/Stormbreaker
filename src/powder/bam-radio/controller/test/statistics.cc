//  Copyright © 2017 Stephen Larew

#define BOOST_TEST_MODULE statistics
#define BOOST_TEST_DYN_LINK

// why did we delete the following line?
#include <boost/test/unit_test.hpp>
#include "../src/statistics.h"
#include "../src/common.h"
#include "../src/dll.h"
#include "../src/events.h"
#include "../src/mac_barrier.h"
#include "../src/options.h"
#include "../src/phy.h"
#include "../src/ippacket.h"
#include "../src/mcs.h"
#include "../src/log.h"

#if __has_include(<gnuradio/analog/noise_source_c.h>)
#include <gnuradio/analog/noise_source_c.h>
#include <gnuradio/blocks/add_cc.h>
#include <gnuradio/blocks/vector_source_c.h>
#else
#include <gnuradio/analog/noise_source.h>
#include <gnuradio/blocks/add_blk.h>
#include <gnuradio/blocks/vector_source.h>
#endif

#include <gnuradio/blocks/copy.h>
#include <gnuradio/blocks/file_sink.h>
#include <gnuradio/blocks/file_source.h>
#include <gnuradio/blocks/null_sink.h>
#include <gnuradio/blocks/rotator_cc.h>
#include <gnuradio/blocks/stream_mux.h>
#include <gnuradio/blocks/tag_debug.h>
#include <gnuradio/blocks/tag_gate.h>
#include <gnuradio/blocks/throttle.h>

#include <gnuradio/filter/fft_filter_ccf.h>
#include <gnuradio/filter/firdes.h>
#include <gnuradio/top_block.h>

#include <array>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>

#include <boost/asio.hpp>
#include <boost/format.hpp>

using boost::format;
using std::cerr;
using std::cout;
using std::endl;
namespace po = boost::program_options;

bool print_stat = true;

void handle_keyboard_interrupt(gr::top_block_sptr tb,
                               boost::asio::io_service &ios,
                               const boost::system::error_code &error,
                               int signal_number) {
  if (!error) {
    print_stat = false;
    cerr << endl << "Shutting down top block & exiting..." << endl;
    tb->stop();
    ios.stop();
  }
}

BOOST_AUTO_TEST_CASE(statistics) {
  using namespace bamradio;

  // Fake arguments.
  std::vector<char const *> argv = {"name",
                                    "--RF.center_freq=5.79e9",
                                    "--RF.rf_bandwidth=20e6",
                                    "--phy.args=addr=0.0.0.0",
                                    "--phy.fftw_wisdom=fftw_wisdom",
                                    "--phy.max-noutput-items=20000",
                                    "--phy_data.freq-offset=5e6",
                                    "--phy_data.tx-gain=20",
                                    "--phy_data.rx-gain=20",
                                    "--phy_data.sample-rate=20e6",
                                    "--phy_data.sync-threshold=0.95",
                                    "--phy_data.subslot-duration=17930",
                                    "--phy_data.num-channels=1",
                                    "--phy_data.rx-frame-queue-size=20",
                                    "--phy_data.tx-frame-queue-size=4",
                                    // ************ Random arguments added by Diyu
                                    "--phy_data.initial-waveform=blabla",
                                    "--test.noise-floor-db=1000",
                                    "--phy_data.tx-segment-queue-size=10",
                                    "--phy_data.guard-band=100e3",
                                    "--phy_control.freq-offset=-3e6",
                                    "--phy_control.tx-gain=20",
                                    "--phy_control.rx-gain=20",
                                    "--phy_control.sample-rate=500e3",
                                    "--phy_control.num_fsk_points=3",
                                    "--phy_control.rs_k=2",
                                    "--phy_control.min_soft_decs=2",
                                    "--phy_control.num_nodes=2",
                                    "--phy_control.id=0",
                                    "--phy_control.t_slot=0",
                                    "--phy_control.atten=0.1",
                                    "--global.verbose=1",
                                    "--global.uid=root",
                                    "--net.tun-iface-prefix=tun",
                                    "--net.tun-ip4=10.20.30.1",
                                    "--net.tun-ip4-netmask=255.255.255.0",
                                    "--net.tun-mtu=1500",
                                    "--psd_sensing.fft_len=128",
                                    "--psd_sensing.mov_avg_len=30",
                                    "--psd_sensing.reset_period=10000",
                                    "--psd_sensing.bin_size=0.2",
                                    "--psd_sensing.sn_gap_bins=30",
                                    "--psd_sensing.empty_bin_items=2",
                                    "--psd_sensing.hist_avg_len=5",
                                    "--psd_sensing.noise_floor_db=-70",
                                    "--psd_sensing.holes_select_mode=0",
                                    "--psd_sensing.snr_threshold=15"};
  // Add in true command line args.
  for (int i = 0; i < boost::unit_test::framework::master_test_suite().argc;
       ++i) {
    argv.push_back(boost::unit_test::framework::master_test_suite().argv[i]);
  }

  // Custom test options.
  double noise_floor_db = 0.0;
  po::options_description testops("test options");
  testops.add_options()("test.noise-floor-db",
                        po::value<double>(&noise_floor_db)->required(),
                        "noise floor (dB)");

  // Init options.
  options::init(argv.size(), &argv[0], &testops);

  // Run StatCenter.
  stats::StatCenter sc;
  log::Logger l;
  l.enableBackend(log::Backend::STDOUT);
  l.enableBackend(log::Backend::JSON, "stat.log", false);
  // print statistics
  bool print_stat = true;
  std::thread stat_printer([&] {
    while (print_stat) {
      sc.publishStatPrintEvent();
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  });


  // Main thread and net IO service.
  boost::asio::io_service ios;
  // generating LOTS of fake traffic
  for (size_t i = 0; i < 5000; ++i) {
    NotificationCenter::shared.post(
        dll::SentFrameEvent,
        dll::SentFrameEventInfo{2, 3, (ofdm::MCS::Name)0, (ofdm::SeqID::ID)666,
                                1000, (uint16_t)(i % 1024), (int64_t)(i + 100000), 10,
                                std::chrono::steady_clock::now().time_since_epoch().count()});

    // assume we detected all frames

    NotificationCenter::shared.post(
        dll::DetectedFrameEvent,
        dll::DetectedFrameEventInfo{
            0, 1, 2, (ofdm::MCS::Name)0, (ofdm::SeqID::ID)666,
            (uint16_t)(i % 1024), 0, 100020 + i, 0, (float)(0.5 * (i % 1024))});
  };
  ios.run();
  return;
}
