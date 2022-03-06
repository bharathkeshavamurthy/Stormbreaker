// watchdog test
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE watchdog
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/watchdog.h"

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

// test 0: test the "shared" watchdog, just print (write file manually)
BOOST_AUTO_TEST_CASE(watchdog0) {
  using namespace bamradio;
  auto have_crashed = Watchdog::shared.haveCrashed();
  std::cout << "haveCrashed: " << (have_crashed ? "true" : "false")
            << std::endl;
}

// test 1: no lockfile, should detect that we have not crashed
BOOST_AUTO_TEST_CASE(watchdog1) {
  using namespace bamradio;
  // make sure that we start on clean slate
  boost::filesystem::path p(Watchdog::lockfile);
  if (boost::filesystem::exists(p)) {
    boost::filesystem::remove(p);
  }

  Watchdog w;
  BOOST_REQUIRE(w.haveCrashed() == false);
}

// test 2: lockfile exists, should detect that we have crashed
BOOST_AUTO_TEST_CASE(watchdog2) {
  using namespace bamradio;
  // make sure that we start on clean slate
  boost::filesystem::path p(Watchdog::lockfile);
  if (boost::filesystem::exists(p)) {
    boost::filesystem::remove(p);
  }

  // write the lock file
  std::ofstream of(Watchdog::lockfile);
  of << "DENNIS" << std::endl;
  of.close();

  Watchdog w;
  BOOST_REQUIRE(w.haveCrashed() == true);
}
