//
// collab.h
//

#ifndef e8cf5bcd1048bb60d400d960bd880b917277fd1f5
#define e8cf5bcd1048bb60d400d960bd880b917277fd1f5

#include <cil.pb.h>
#include <registration.pb.h>

#include <boost/asio.hpp>
#include <boost/signals2/signal.hpp>
#include <zmq.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cc_data.h"
#include "json.hpp"
#include "notify.h"
#include "statistics.h"

namespace bamradio {
namespace collab {

//
// 0MQ Polling
//
class Poller {
public:
  Poller(){};
  void add(zmq::socket_t &sock, short events);
  void remove(zmq::socket_t &sock); // costly
  int poll(long timeout);
  bool check_for(zmq::socket_t &sock, short event);
  bool check(zmq::socket_t &sock);

private:
  std::vector<zmq::pollitem_t> _pi;
  std::unordered_map<void *, std::pair<short, short>> _tbl;
};

//
// CIL Message Metadata
//

// the different network types as defined in cil.proto
enum NetworkType {
  Competitor = 0,
  IncumbentPassive,
  IncumbentActive,
  IncumbentDSRC,
  IncumbentJammer,
  Unknown
};

struct Metadata {
  int64_t msg_id;      // bam-internal count
  uint32_t sender_id;  // from CilMessage
  uint64_t msg_count;  // from CilMessage
  NetworkType type;    // from CilMessage
  int64_t seconds;     // from CilMessage
  int64_t picoseconds; // from CilMessage
};

class CollabClient {
public:
  struct ConnectionParams {
    boost::asio::ip::address_v4 server_ip;
    boost::asio::ip::address_v4 client_ip;
    long server_port;
    long client_port;
    long peer_port;
  };
  //
  // instrumentation
  //
  CollabClient(zmq::context_t &ctx, boost::asio::ip::address_v4 server_ip,
               long server_port, boost::asio::ip::address_v4 client_ip,
               long client_port, long peer_port, uint64_t rate_limit = 100);
  CollabClient(zmq::context_t &ctx, ConnectionParams conn, uint64_t rate_limit = 100);
  // shared_ptr convention
  typedef std::shared_ptr<CollabClient> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<CollabClient>(std::forward<Args>(args)...);
  }

  void run();        // blocking, same thread
  void start();      // non-blocking, new thread
  void connect();    // blocking, throws
  void stop();       // blocking
  bool tryConnect(); // blocking, does not throw
  bool connected() { return _state != State::INIT; }

  //
  // connection
  //
  boost::asio::ip::address_v4 client_ip() { return _cl_ip; };
  boost::asio::ip::address_v4 server_ip() { return _srv_ip; };

  //
  // communication
  //
  // broadcast a message to all peers. return msg id, -1 if error
  int64_t broadcast(std::shared_ptr<sc2::cil::CilMessage> msg);
  // send a message to a specific peer. return msg id, -1 if error
  int64_t send(boost::asio::ip::address_v4 peer,
               std::shared_ptr<sc2::cil::CilMessage> msg);

  //
  // handler registration
  //
  void register_handler(
      std::function<void(std::shared_ptr<sc2::cil::Hello>, Metadata md)>
          handler);
  void register_handler(
      std::function<void(std::shared_ptr<sc2::cil::SpectrumUsage>, Metadata md)>
          handler);
  void
  register_handler(std::function<void(std::shared_ptr<sc2::cil::LocationUpdate>,
                                      Metadata md)>
                       handler);
  void register_handler(
      std::function<void(std::shared_ptr<sc2::cil::IncumbentNotify>,
                         Metadata md)>
          handler);
  void register_handler(
      std::function<void(std::shared_ptr<sc2::cil::DetailedPerformance>,
                         Metadata md)>
          handler);

  // TODO:
  // disconnect handlers

  //
  // peers
  //
  std::vector<boost::asio::ip::address_v4> peers() const;
  size_t num_peers() const;

private:
  // housekeeping
  enum class State {
    NO_STATE = 0,
    INIT,
    REGISTER,
    REGISTER_WAITING,
    ACTIVE,
    NUM_STATES
  } _state;
  std::thread _main_thread;
  std::atomic_bool _continue;
  void _run();

  //
  // the connection
  //
  uint32_t _collab_net_id, _server_id, _host_id;
  boost::asio::ip::address_v4 _srv_ip, _cl_ip;
  long _srv_port, _cl_port, _peer_port;

  //
  // server-related thangs
  //
  int32_t _nonce;
  std::chrono::milliseconds _max_keepalive_interval;

  //
  // info about peers
  //
  // FIXME make proper class
  struct Peer {
    std::vector<std::string> dialect;
    // connection
    boost::asio::ip::address_v4 ip;
    zmq::socket_t sock;
    // methods
    Peer(uint32_t ipaddr, zmq::context_t &ctx, long port);
    bool said_hello() { return dialect.size() == 0 ? false : true; }
  };
  std::vector<std::unique_ptr<Peer>> _peers;
  std::mutex _peermtx;
  void _add_peer(uint32_t peer);
  void
  _update_peers(google::protobuf::RepeatedField<unsigned int> const &neighbors);

  //
  // periodic keepalive updates
  //
  std::thread _keepalive_thread;
  void _keepalive(std::chrono::milliseconds interval);
  std::atomic_bool _ka_continue;

  // keep track of messages
  std::atomic<int64_t> _ntx; // number of messages transmitted
  std::atomic<int64_t> _nrx; // number of messages received
  std::atomic_int _msg_count;

  //
  // Rate Limiting
  //
  struct RateLimiter {
    RateLimiter(uint64_t msgps_);
    bool operator()(uint32_t sender);
    std::map<uint32_t, stats::TrafficStat<uint64_t>> r;
    float const msgps;
  } _rlimit;

  //
  // handlers
  //
  // top-level handlers
  void _handle_server(std::shared_ptr<sc2::reg::TellClient> m, int64_t id);
  void _handle_collab(std::shared_ptr<sc2::cil::CilMessage> m, int64_t id);
  // signals
  boost::signals2::signal<void(std::shared_ptr<sc2::cil::Hello>, Metadata md)>
      _handle_hello;
  boost::signals2::signal<void(std::shared_ptr<sc2::cil::SpectrumUsage>,
                               Metadata md)>
      _handle_spectrum_usage;
  boost::signals2::signal<void(std::shared_ptr<sc2::cil::IncumbentNotify>,
                               Metadata md)>
      _handle_incumbent_notify;
  boost::signals2::signal<void(std::shared_ptr<sc2::cil::LocationUpdate>,
                               Metadata md)>
      _handle_location_update;
  boost::signals2::signal<void(std::shared_ptr<sc2::cil::DetailedPerformance>,
                               Metadata md)>
      _handle_detailed_performance;

  //
  // 0MQ related stuff
  //
  zmq::context_t &_ctx;
  zmq::socket_t _srv_in_sock;
  zmq::socket_t _srv_out_sock;
  zmq::socket_t _peer_in_sock;
  Poller _pollin, _pollout;
  long _poll_timeout = 5; // ms
  bool _send(zmq::socket_t &s, std::shared_ptr<google::protobuf::Message> msg);
  std::mutex _sndmtx;

  // logging
  static std::string state2str(State s);
  void register_handler(
      std::function<void(std::shared_ptr<sc2::cil::LocationUpdate>)> handler);
};

//
// Calculate the distance between two points on Earth using the Haversine Method
//
// main formula, all arguments in degrees
double distance_haversine(double xlat, double xlong, double ylat, double ylong);
// for internal location structs
double distance(bamradio::controlchannel::Location x,
                bamradio::controlchannel::Location y);
// for protobuf location structs
double distance(sc2::cil::Location x, sc2::cil::Location y);
double distance(sc2::cil::Location *x, sc2::cil::Location *y);
// timestamp
sc2::cil::TimeStamp *getTimeNow();

} // namespace collab
} // namespace bamradio

#endif // e8cf5bcd1048bb60d400d960bd880b917277fd1f5
