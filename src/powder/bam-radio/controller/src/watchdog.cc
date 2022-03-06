// -*- c++ -*-
// Copyright (c) Dennis Ogbe

#include "watchdog.h"

#include <fstream>

#include <boost/filesystem.hpp>

namespace bamradio {

const std::string Watchdog::lockfile = ".bamradio_watch";
const std::string Watchdog::start_time_file = ".bamradio_start";

Watchdog::Watchdog()
    : _haveCrashed([] {
        using namespace boost::filesystem;
        path p(lockfile);
        bool have_crashed;
        if (exists(p)) {
          have_crashed = true;
        } else {
          have_crashed = false;
          std::ofstream of(lockfile);
          of << std::chrono::system_clock::now().time_since_epoch().count()
             << std::endl;
          of.close();
        }
        return have_crashed;
      }()) {}

Watchdog Watchdog::shared;

void Watchdog::setStartTime(std::chrono::system_clock::time_point) const {
  using namespace std::chrono;
  std::lock_guard<decltype(_mtx)> l(_mtx);
  std::ofstream of(start_time_file);
  if (of) {
    of << system_clock::now().time_since_epoch().count();
    of.close();
  }
}

boost::optional<std::chrono::system_clock::time_point>
Watchdog::startTime() const {
  using namespace boost::filesystem;
  using namespace std::chrono;
  std::lock_guard<decltype(_mtx)> l(_mtx);
  path p(start_time_file);
  if (!exists(p)) {
    return boost::none;
  }
  std::ifstream ifl(start_time_file);
  int64_t epoch_time;
  ifl >> epoch_time;
  return system_clock::time_point(
      system_clock::time_point::duration(epoch_time));
}

} // namespace bamradio
