// Test the generic text logger and the doomsday event
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE doomsday
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/events.h"
#include "../src/log.h"
#include "../src/notify.h"

#include <iostream>
#include <thread>

#include "json.hpp"
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>

using namespace bamradio;

// remove a file if exists
void remove(std::string const &fn) {
  boost::filesystem::path p(fn);
  if (boost::filesystem::exists(p)) {
    boost::filesystem::remove(p);
  }
}

BOOST_AUTO_TEST_CASE(doomsday) {
  // run a thread and print DB events
  boost::asio::io_context ios;
  auto work = boost::asio::make_work_guard(ios);
  std::thread printer([&] { ios.run(); });
  std::vector<NotificationCenter::SubToken> st;
  st.push_back(NotificationCenter::shared.subscribe<log::DoomsdayEventInfo>(
      log::DoomsdayEvent, ios, [&](auto ei) {
        nlohmann::json j = ei;
        std::cout << j << std::endl;
        ei.judgement_day();
      }));

  log::doomsday("this is failure", __FILE__, __LINE__);

  std::this_thread::sleep_for(std::chrono::seconds(1));
  work.reset();
  st.clear();
  printer.join();
}

BOOST_AUTO_TEST_CASE(generic) {
  const std::string dbfn("sql1.log");
  const std::string jsfn("json1.log");
  // delete any pre-existing DBs
  remove(dbfn);
  remove(jsfn);

  // create logger and enable db
  log::Logger l;
  l.enableBackend(log::Backend::STDOUT);
  l.enableBackend(log::Backend::JSON, jsfn, true);
  l.enableBackend(log::Backend::SQL, dbfn, false);

  for (int i = 0; i < 5; ++i) {
    log::text("this is some text", __FILE__, __LINE__);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));
}

BOOST_AUTO_TEST_CASE(doomsdaylog) {
  const std::string dbfn("sql2.log");
  const std::string jsfn("json2.log");
  // delete any pre-existing DBs
  remove(dbfn);
  remove(jsfn);

  // create logger and enable db
  log::Logger l;
  l.enableBackend(log::Backend::STDOUT);
  l.enableBackend(log::Backend::JSON, jsfn, true);
  l.enableBackend(log::Backend::SQL, dbfn, false);

  ::bamradio::log::doomsday("bad news");

  l.shutdown();
}

BOOST_AUTO_TEST_CASE(doomsdaylogto) {
  ::bamradio::log::doomsday("bad news timeout");
  std::this_thread::sleep_for(std::chrono::seconds(20));
}
