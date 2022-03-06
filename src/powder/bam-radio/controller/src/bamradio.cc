//  Copyright (c) 2017 Dennis Ogbe <dogbe@purdue.edu>
//  Copyright © 2017 Stephen Larew

#include "c2api.h"
#include "de.h"
#include "dll.h"
#include "events.h"
#include "gps.h"
#include "log.h"
#include "net.h"
#include "options.h"
#include "radiocontroller.h"
#include "statistics.h"
#include "tun.h"
#include "watchdog.h"

#include <gnuradio/realtime.h>

#include <cstring>
#include <iostream>
#include <pwd.h>

#include <boost/format.hpp>
#include <zmq.hpp>

int main(int argc, char const *argv[]) {
  using namespace bamradio;
  using namespace boost::asio;

  //
  // Initialize options
  //

  options::init(argc, argv);

  //
  // Enable logging
  //

  log::Logger log;
  // JSON logger
  if (!options::log_json_path.empty()) {
    log.enableBackend(log::Backend::JSON, options::log_json_path,
                      options::log_append);
  }
  // SQLite logger
  if (!options::log_sqlite_path.empty()) {
    log.enableBackend(log::Backend::SQL, options::log_sqlite_path, true);
  }
  // STDOUT
  log.enableBackend(log::Backend::STDOUT);
  // print banner
  if (!Watchdog::shared.haveCrashed()) {
    log::text(log::asciiLogo);
  } else {
    log::text(
        "WARNING: Recovering from crash\n"
        "----------------------------------------------------------------");
    auto start_time = Watchdog::shared.startTime();
    if (start_time) {
      log.setStartTime(*start_time);
    }
  }
  log::text("Logger [Started]");

  //
  // Initialize C2API and Colosseum environment
  //

  log::text("C2API [Creating instance...]");
  zmq::context_t zctx(1);
  c2api::Client c2api(zctx, options::c2api::port, options::c2api::status_path,
                      options::c2api::env_recovery_path,
                      options::c2api::mo_recovery_path);
  c2api.init_env(options::phy::center_freq, options::phy::bandwidth);
  log::text("C2API [Created]");

  //
  // Start the GPS reader
  //

  log::text("GPSReader [Creating instance...]");
  gps::GPSReader gpsReader(6000);
  log::text("GPSReader [Created]");

  //
  // Enable realtime scheduling
  //
#if 0
  const auto e = gr::enable_realtime_scheduling();
  if (e != gr::RT_OK) {
    log::text(
        (boost::format("Failed enabling realtime scheduling (%1%)") % e).str(),
        __FILE__, __LINE__);
  } else {
    log::text("Enabled realtime scheduling", __FILE__, __LINE__);
  }
#endif

  //
  // Create a software defined network (SDN) controller
  //

  log::text("NetController [Creating instance...]");
  auto netCntl = std::make_shared<net::Controller>();
  log::text("NetController [Created]");

  //
  // Create and add the host tun device to the SDN
  //

  // Create tun device
  log::text("TUN device [Creating instance...]");
  auto tunDev = std::make_shared<tun::Device>(
      netCntl->io_service(),
      (boost::format("%1%%%d") % options::net::tun_iface_prefix).str());
  tunDev->setMtu(options::net::tun_mtu);
  tunDev->setAddress(ip::address_v4::from_string(options::net::tun_ip4));
  tunDev->setNetmask(
      ip::address_v4::from_string(options::net::tun_ip4_netmask));

  // Create the DLL and Interface adapters for the tun device
  auto tunDLL =
      std::make_shared<tun::DataLinkLayer>(tunDev, netCntl->io_service());
  auto tunInf = std::make_shared<tun::Interface>(tunDLL);

  tunInf->setUp();
  log::text("TUN device [Created]");

  // Add tun device to the SDN
  netCntl->addInterface(tunInf);

  //
  // Radios
  //

  log::text("RadioController [Creating instance...]");
  auto radioCntl = std::make_shared<RadioController>(netCntl->io_service());
  netCntl->addInterface(
      std::make_shared<net::BasicInterfaceAdapter>(radioCntl->ofdmDll()));
  log::text("RadioController [Created]");

  //
  // Collaboration
  //

  decisionengine::DecisionEngine::sptr de = nullptr;
  if (options::collab::gateway) {
    log::text("DE [Creating instance...]");
    auto const ccParams = collab::CollabClient::ConnectionParams{
        .server_ip = boost::asio::ip::address_v4::from_string(
            (boost::format("172.30.%d.%d") % options::collab::netid %
             options::collab::server_id)
                .str()),
        .client_ip = boost::asio::ip::address_v4::from_string(
            (boost::format("172.30.%d.%d") % options::collab::netid %
             (options::phy::control::id + 100))
                .str()),
        .server_port = options::collab::server_port,
        .client_port = options::collab::client_port,
        .peer_port = options::collab::peer_port,
    };
    auto const hp = psdsensing::PSDSensing::HistParams{
        .bin_size = options::psdsensing::bin_size,
        .empty_bin_thresh = options::psdsensing::empty_bin_items,
        .sn_gap_bins = options::psdsensing::sn_gap_bins,
        .avg_len = options::psdsensing::hist_avg_len,
        .noise_floor = options::psdsensing::noise_floor_db,
    };
    // fixme some of these should be options
    auto const deOpts = decisionengine::DecisionEngine::Options{
        .gatewayID = NodeID(options::phy::control::id),
        .step_period = std::chrono::seconds(5),
        .channel_alloc_delay = std::chrono::seconds(2),
        .cil_broadcast_period = std::chrono::seconds(5),
        .data_tx_gain = options::phy::data::tx_gain,
        .control_tx_gain = options::phy::control::tx_gain,
        .sample_rate = bam::dsp::sample_rate,
        .guard_band = options::phy::data::guard_band,
        .max_wf = options::psdsensing::max_wf,
        .psd_hist_params = hp,
    };
    de =
        decisionengine::DecisionEngine::make(zctx, ccParams, radioCntl, deOpts);
    log::text("DE [Created]");
  }

  //
  // Statistics
  //
  log::text("StatCenter [Creating instance...]");
  stats::StatCenter sc;
  log::text("StatCenter [Created]");

  // when in batch mode, we loop indefinitely until we are told to start
  if (options::batch_mode) {
    log::text("C2API [Starting...]");
    c2api.start();
    log::text("C2API [Started]");
    c2api.updateStatus(c2api::Status::Ready);
    log::text("Waiting for START signal");
    c2api.waitStart();
    log::text("Received START signal");
    c2api.updateStatus(c2api::Status::Active);
  }

  //
  // Start everything
  //

  log::text("StatCenter [Starting...]");
  sc.start();
  log::text("StatCenter [Started]");

  log::text("RadioController [Starting...]");
  radioCntl->start();
  log::text("RadioController [Started]");

  log::text("NetController [Starting...]");
  netCntl->start();
  log::text("NetController [Started]");

  if (de) {
    log::text("DE [Starting...]");
    de->start();
    log::text("DE [Started]");
  }

  // print statistics
  log::text("StatPrinter [Starting...]");
  bool print_stat = true;
  auto stat_printer = std::thread([&] {
    bamradio::set_thread_name("stat_printer");
    while (print_stat) {
      sc.publishStatPrintEvent();
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  });
  log::text("StatCenter [Started]");

  if (options::batch_mode) {
    // in batch mode, block this thread until we are told to stop
    c2api.waitStop();
    c2api.updateStatus(c2api::Status::Stopping);
    // FIXME: putting this here as a stop-gap measure to avoid dabase corruption
    // in the case that the subsequent lines block indefinitely. The real
    // solution to this problem is to make sure that the radio controller shuts
    // down properly. This, as-is, does not log any events associated with the
    // shutdown of the radiocontroller. (But at least it should give us good DB
    // files.)
    log.shutdown();
    radioCntl->stop();
    if (de) {
      de->stop();
    }
    print_stat = false;
    stat_printer.join();
    c2api.updateStatus(c2api::Status::Finished);
  } else {
    // stop the top block on sigint
    boost::asio::io_service ios;
    boost::asio::signal_set signals(ios, SIGINT);
    signals.async_wait([&](auto const e, auto) {
      if (!e) {
        log::text("Shutting down top block & exiting...", __FILE__, __LINE__);
        netCntl.reset();
        log.shutdown();
        radioCntl->stop();
        if (de) {
          de->stop();
        }
        print_stat = false;
        stat_printer.join();
        ios.stop();
      }
    });

    ios.run();
  }
  return 0;
}
