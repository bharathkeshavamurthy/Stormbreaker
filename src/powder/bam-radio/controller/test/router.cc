// Router Unit Test
// Copyright (c) 2017-2018 Tomohiro Arakawa

#define BOOST_TEST_MODULE router
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/networkmap.h"
#include "../src/router.h"
#include <iostream>
#include <memory>

BOOST_AUTO_TEST_CASE(flowtracker) {
  using namespace bamradio;
  NetworkMap netmap;
  netmap.setLink(12, 38, 1);
  netmap.setLink(38, 34, 1);

  net::BasicRouter r(10);

  // no route
  r.updateRoutingTable(netmap);

  // there are routes
  netmap.setLink(10, 12, 1);
  r.updateRoutingTable(netmap);
}
