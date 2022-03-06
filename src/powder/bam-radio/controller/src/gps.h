// GPS Reader for SC2
//
// Copyright (c) 2018 Dennis Ogbe

#ifndef e413b361577ac33c6b92a887160bfe81474957d8a
#define e413b361577ac33c6b92a887160bfe81474957d8a

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <libgpsmm.h>

namespace bamradio {
namespace gps {

// A background thread that reads GPS coordinates from gpsd and publishes
// notifications to the notification center
class GPSReader {
public:
  // the update interval of the GPS reader [milliseconds]
  static int update_interval() { return 500; }

  typedef std::shared_ptr<GPSReader> sptr;
  static sptr make(unsigned long port) {
    return std::make_shared<GPSReader>(port);
  }
  GPSReader(unsigned long port);
  ~GPSReader();

protected:
  void _notify(struct gps_data_t *g);
  void _read();
  void _run();
  bool _connect();

  unsigned long const _port;
  gpsmm _gps;
  std::atomic_bool _connected;
  std::atomic_bool _running;

  boost::asio::io_context _ioctx;
  boost::asio::deadline_timer _timer;
  std::thread _work_thread;
};

} // namespace gps
} // namespace bamradio

#endif // e413b361577ac33c6b92a887160bfe81474957d8a
