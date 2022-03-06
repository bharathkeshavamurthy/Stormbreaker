// Logging.
//
// Copyright (c) 2018 Dennis Ogbe
// Copyright (c) 2018 Tomohiro Arakawa
// Copyright (c) 2018 Stephen Larew

#ifndef a51f5a835cb267baabbf
#define a51f5a835cb267baabbf

#include "notify.h"

#include <fstream>
#include <map>
#include <memory>
#include <thread>

#include "json.hpp"
#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <sqlite3.h>

namespace bamradio {
namespace log {

// SQLite wrapper class for BAM Radio events
class database {
public:
  typedef std::shared_ptr<database> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<database>(std::forward<Args>(args)...);
  }
  database();
  database(database const &other) = delete;
  ~database();

  void close();
  void open(std::string const &filename, bool append);
  bool is_open() const;

  //
  // insertion
  //

  // insert the "Start" Line
  bool insert_start(unsigned long long const TimeStamp);

  // DLL Events
  template <typename EventInfo>
  bool insert(EventInfo const &ei, unsigned long long t, unsigned long long ts);
  template <typename EventInfo>
  bool insert_raw(EventInfo const &ei, std::chrono::system_clock::time_point t,
                  std::chrono::steady_clock::time_point ts);

private:
  sqlite3 *db;

  std::map<std::string, sqlite3_stmt *> _stmt;

  int64_t _sql_id; // keep a count of # of SQL statements made
  std::string _filename;

  bool _exec(std::string const &sql);
};

// Log to either JSON file, SQLite database, or stdout. Each backend must be
// enabled explicitly.
enum Backend : uint8_t { JSON = 0, SQL, STDOUT };

class Logger {
public:
  // GNU Radio's grip on us will never let down...
  typedef std::shared_ptr<Logger> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<Logger>(std::forward<Args>(args)...);
  }

  // tors
  Logger();
  Logger(Logger const &other) = delete;
  ~Logger();
  void shutdown();

  // in order to log something, we need to enable at least one backend
  void enableBackend(Backend b, std::string const &filename, bool append);
  void enableBackend(Backend b);
  void disableBackend(Backend b);
  bool backendActive(Backend b) const;

  // set the start time
  void setStartTime(std::chrono::system_clock::time_point t);

protected:
  // worker thread, miscellania
  boost::asio::io_context _ioctx;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      _work;
  std::thread _work_thread;

  // Backend stuff
  std::map<Backend, std::string> _filename;
  database::sptr _db;
  std::ofstream _ofs;

  // notification tokens
  std::vector<NotificationCenter::SubToken> _st;

  // returns the current unix time -- FIXME?
  int64_t _getCurrentTime() const;
  int64_t _getCurrentTimeSteady() const;

  // save the scenario start time
  boost::optional<std::chrono::system_clock::time_point> _scenario_start_time;

  // print a log message
  void printWithTime(std::string const &str);

  // convenience
  void _writeJson(nlohmann::json const &j);
  void _closeFiles();

  template <typename EventInfo>
  void _basic_subscribe(NotificationCenter::Name event);
};

extern const std::string asciiLogo;

} // namespace log
} // namespace bamradio

#endif // a51f5a835cb267baabbf
