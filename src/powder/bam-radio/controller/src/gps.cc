// GPS Reader for SC2
//
// Copyright (c) 2018 Dennis Ogbe

#include "gps.h"
#include "events.h"
#include "notify.h"
#include "util.h"

#include <numeric>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>

namespace bamradio {
namespace gps {

//
// GPS Reader
//

GPSReader::GPSReader(unsigned long port)
    : _port(port),
      _gps("localhost", (boost::format("%lu") % port).str().c_str()),
      _connected(false), _running(true),
      _timer(_ioctx, boost::posix_time::milliseconds(update_interval())) {
  _work_thread = std::thread([this] {
    bamradio::set_thread_name("gps");
    _timer.expires_from_now(boost::posix_time::milliseconds(update_interval()));
    _timer.async_wait([self = this](auto) { self->_run(); });
    _ioctx.run();
  });
}

GPSReader::~GPSReader() {
  _running = false;
  _work_thread.join();
}

bool GPSReader::_connect() {
  if (_gps.stream(WATCH_ENABLE | WATCH_JSON) == NULL) {
    NotificationCenter::shared.post(
        GPSEvent, GPSEventInfo{GPSEventType::TRY_CONNECT_BAD, 0., 0., 0.});
    return false;
  } else {
    NotificationCenter::shared.post(
        GPSEvent, GPSEventInfo{GPSEventType::TRY_CONNECT_GOOD, 0., 0., 0.});
    return true;
  }
}

void GPSReader::_run() {
  if (!_connected) {
    _connected = _connect();
  } else {
    if (_gps.waiting(5000000)) {
      auto gps_data = _gps.read();
      if (gps_data == NULL) {
        NotificationCenter::shared.post(
            GPSEvent, GPSEventInfo{GPSEventType::READ_ERROR, 0., 0., 0.});
      } else if (gps_data->fix.mode < MODE_2D) {
        NotificationCenter::shared.post(
            GPSEvent, GPSEventInfo{GPSEventType::READ_NO_FIX, 0., 0., 0.});
      } else {
        _notify(gps_data);
      }
    } else {
      NotificationCenter::shared.post(
          GPSEvent, GPSEventInfo{GPSEventType::READ_TIMEOUT, 0., 0., 0.});
    }
  }
  // if running, wait for interval milliseconds and repeat
  if (_running) {
    _timer.expires_from_now(boost::posix_time::milliseconds(update_interval()));
    _timer.async_wait([self = this](auto) { self->_run(); });
  }
}

void GPSReader::_notify(struct gps_data_t *g) {
  auto lat = g->fix.latitude;
  auto lon = g->fix.longitude;
  auto alt = g->fix.altitude;
  if (not(std::isfinite(lat) && std::isfinite(lon) && std::isfinite(alt))) {
    NotificationCenter::shared.post(
        GPSEvent, GPSEventInfo{GPSEventType::READ_NO_DATA, 0., 0., 0.});
  } else {
    NotificationCenter::shared.post(
        GPSEvent, GPSEventInfo{GPSEventType::READ_GOOD, lat, lon, alt});
  }
}

} // namespace gps
} // namespace bamradio
