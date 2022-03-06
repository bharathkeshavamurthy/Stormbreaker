//  Copyright © 2017 Stephen Larew

#define BOOST_TEST_MODULE notify
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/notify.h"

BOOST_AUTO_TEST_CASE(notify_simple) {
  using namespace bamradio;
  using namespace boost::asio;

  io_service ios;

  auto const n = std::hash<std::string>{}("test1");

  float result = 0.0f;
  auto t =
      NotificationCenter::shared.subscribe<float>(n, ios, [&result](auto v) {
        BOOST_CHECK(v == 1.0f);
        result = v;
      });

  NotificationCenter::shared.post(n, 1.0f);

  ios.run();

  BOOST_CHECK(result == 1.0f);
}

BOOST_AUTO_TEST_CASE(notify_delete_token) {
  using namespace bamradio;
  using namespace boost::asio;

  io_service ios;

  auto const n = std::hash<std::string>{}("test1");

  float result = 0.0f;
  auto t =
      NotificationCenter::shared.subscribe<float>(n, ios, [&result](auto v) {
        BOOST_CHECK(v == 1.0f);
        result = v;
      });

  t = NotificationCenter::SubToken();
  NotificationCenter::shared.post(n, 1.0f);

  ios.run();

  BOOST_CHECK(result == 0.0f);
}

BOOST_AUTO_TEST_CASE(notify_reset_token) {
  using namespace bamradio;
  using namespace boost::asio;

  io_service ios;

  auto const n = std::hash<std::string>{}("test1");

  float result = 0.0f;
  auto t =
      NotificationCenter::shared.subscribe<float>(n, ios, [&result](auto v) {
        BOOST_CHECK(v == 1.0f);
        result = v;
      });

  t.reset();
  NotificationCenter::shared.post(n, 1.0f);

  ios.run();

  BOOST_CHECK(result == 0.0f);
}

#ifndef NDEBUG
BOOST_AUTO_TEST_CASE(notify_subscribe_in_copy_during_post) {
  using namespace bamradio;
  using namespace boost::asio;

  struct nasty {
    int a;
    io_service ios;
    nasty() = default;
    nasty(int b) : a(b) {}
    nasty(nasty const &n) : a(n.a) {
      NotificationCenter::shared.subscribe<int>(0, ios, [](auto v) {});
    }
  };

  io_service ios;

  auto const n = std::hash<std::string>{}("test1");

  int result = 0;
  auto t =
      NotificationCenter::shared.subscribe<nasty>(n, ios, [&result](auto v) {
        BOOST_CHECK(v.a == 1);
        result = 1;
      });

  try {
  NotificationCenter::shared.post(n, nasty(1));
  } catch (std::runtime_error e) {
  }

  ios.run();

  BOOST_CHECK(result == 0);
}
#endif
