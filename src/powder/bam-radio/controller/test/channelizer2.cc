// Channelizer2 tests
//
// Copyright (c) 2018 Dennis Ogbe
//
// This is a standalone program meant to be used with UHD for profiling the
// channelizer's performance.

#include "channelizer2.h"
#include "bandwidth.h"
#include "phy.h"
#include "util.h"

#include <atomic>
#include <thread>

#include <uhd/types/device_addr.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

int main(int argc, char *argv[]) {
  // options
  namespace po = boost::program_options;
  std::string dev_args;
  size_t buf_sz;
  double center_freq;
  size_t num_chan;
  size_t nbatch;
  size_t K;
  int runtime;

  po::options_description desc("Options");
  desc.add_options()(
      // option
      "help", "help message")(
      // option
      "args", po::value<decltype(dev_args)>(&dev_args)->required(),
      "UHD device arguments.")(
      // option
      "buffer-size",
      po::value<decltype(buf_sz)>(&buf_sz)->required()->default_value(1e7),
      "RX buffer size")(
      // option
      "center-frequency",
      po::value<decltype(center_freq)>(&center_freq)
          ->required()
          ->default_value(915e6),
      "Center frequency")(
      // option
      "num-chan",
      po::value<decltype(num_chan)>(&num_chan)->required()->default_value(2),
      "Number of RX channels")(
      // option
      "K", po::value<decltype(K)>(&K)->required()->default_value(16),
      "FFT size multiplier")(
      // option
      "nbatch",
      po::value<decltype(nbatch)>(&nbatch)->required()->default_value(100),
      "Number of batches")(
      // option
      "runtime",
      po::value<decltype(runtime)>(&runtime)->required()->default_value(-1),
      "Runtime of test [sec]");
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

  // the channelizer
  auto N =
      K * (bam::dsp::SubChannel::table()[0].taps.size() - 1); // magic numbers.
  auto channelizer = bam::dsp::Channelizer2::make(
      num_chan, N, nbatch, bam::dsp::SubChannel::table());

  // stream from usrp into a ringbuffer, consume using Channelizer2,
  // Benchmark???
  auto stream_args = uhd::stream_args_t("fc32", "sc16");
  stream_args.args = uhd::device_addr_t("underflow_policy=next_burst");
  stream_args.channels = {0};

  auto usrp = uhd::usrp::multi_usrp::make(
      uhd::device_addr_t((boost::format("%1%,master_clock_rate=%2%") %
                          dev_args % bam::dsp::master_clock_rate)
                             .str()));
  auto tr = uhd::tune_request_t(center_freq, 0);
  usrp->set_rx_rate(bam::dsp::sample_rate);
  usrp->set_rx_freq(tr);
  auto rxStreamer = usrp->get_rx_stream(stream_args);
  auto rx_rb = bamradio::PinnedComplexRingBuffer::make(buf_sz);

  // receiving thread
  std::atomic_bool reading(true);
  std::thread reader([&] {
    bamradio::set_thread_name("sample2buf");
    while (reading) {
      size_t buf_avail = rx_rb->space_avail();
      if (buf_avail < 1) {
        std::this_thread::yield();
        continue;
      }
      // Get raw samples from UHD
      uhd::rx_metadata_t md;
      size_t nproduced = rxStreamer->recv(rx_rb->write_ptr(), buf_avail, md);
      // produce
      rx_rb->produce(nproduced);
    }
  });
  rxStreamer->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);

  // channelizer thread
  std::atomic_bool running(true);
  std::thread cthread([&] {
    bamradio::set_thread_name("channelizer");
    std::vector<size_t> nsamp(num_chan);
    while (running) {
      if (channelizer->load(rx_rb.get())) {
        channelizer->execute();
        channelizer->getNumOutput(nsamp);
      }
    }
  });

  // make sure we leave this test properly.
  boost::asio::io_service ios;
  auto stop = [&](auto, auto) {
    std::cout << "Quitting..." << std::endl;
    reading = false;
    running = false;
    reader.join();
    cthread.join();
    ios.stop();
  };
  boost::asio::deadline_timer to(ios);
  if (runtime > -1) {
    to.expires_from_now(boost::posix_time::seconds(runtime));
    to.async_wait([&](auto &e) { stop(e, 0); });
  }
  boost::asio::signal_set signals(ios, SIGINT);
  signals.async_wait(stop);
  ios.run();
  return 0;
}
