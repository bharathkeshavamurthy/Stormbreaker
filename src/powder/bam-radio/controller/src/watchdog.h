// -*- c++ -*-
//
// Detect whether we are recovering from a crash or not
//
// Copyright (c) Dennis Ogbe

#ifndef b0b046c2de7f30b3694d
#define b0b046c2de7f30b3694d

#include <chrono>
#include <mutex>
#include <string>

#include <boost/optional.hpp>

namespace bamradio {
class Watchdog {
public:
  Watchdog();
  static const std::string lockfile;
  static const std::string start_time_file;

  /// a shared global object to detect whether we have crashed.
  static Watchdog shared;

  /// ask the watchdog whether we have crashed
  bool haveCrashed() const { return _haveCrashed; };
  /// tell the watchdog when the scenario started
  void setStartTime(std::chrono::system_clock::time_point) const;
  /// query the watchdog about when the scenario started -- don't abuse this.
  boost::optional<std::chrono::system_clock::time_point> startTime() const;

  // if I catch anyone calling this outside of a unit test I'm gonna have y'all
  // drop down and do 20
  void debug_set_havecrashed(bool have_crashed) { _haveCrashed = have_crashed; }

private:
  bool _haveCrashed;
  mutable std::mutex _mtx;
};
} // namespace bamradio

#endif // b0b046c2de7f30b3694d
