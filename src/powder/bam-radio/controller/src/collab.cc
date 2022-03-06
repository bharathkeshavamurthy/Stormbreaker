//
// collab.cc
//
#define _USE_MATH_DEFINES

#include "collab.h"
#include "events.h"

#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

#include <boost/format.hpp>

namespace bamradio {
namespace collab {

//
// Simple 0MQ poller implementation
//

void Poller::add(zmq::socket_t &sock, short events) {
  if (_tbl.find(static_cast<void *>(sock)) != end(_tbl))
    return;
  _pi.push_back({.socket = static_cast<void *>(sock),
                 .fd = 0,
                 .events = events,
                 .revents = 0});
  _tbl.emplace(static_cast<void *>(sock),
               std::make_pair(_pi.size() - 1, events));
}

void Poller::remove(zmq::socket_t &sock) {
  auto s = _tbl.find(static_cast<void *>(sock));
  if (s == end(_tbl))
    return;
  _pi.erase(begin(_pi) + s->second.first);
  _tbl.clear();
  for (size_t i = 0; i < _pi.size(); ++i) {
    _tbl.emplace(_pi[i].socket, std::make_pair(i, _pi[i].events));
  }
}

int Poller::poll(long timeout) {
  try {
    return zmq::poll(_pi.data(), _pi.size(), timeout);
  } catch (zmq::error_t &ex) {
    if (ex.num() != EINTR)
      throw;
    return 0;
  }
}

bool Poller::check_for(zmq::socket_t &sock, short event) {
  return (_pi[_tbl[static_cast<void *>(sock)].first].revents & event) == event
             ? true
             : false;
}

bool Poller::check(zmq::socket_t &sock) {
  auto p = _tbl[static_cast<void *>(sock)];
  return (_pi[p.first].revents & p.second) == p.second ? true : false;
}

//
// rate limiting
//
CollabClient::RateLimiter::RateLimiter(uint64_t msgps_) : msgps(msgps_) {}

bool CollabClient::RateLimiter::operator()(uint32_t sender) {
  if (r.find(sender) == end(r)) {
    r[sender] = stats::TrafficStat<uint64_t>();
  }
  auto &s = r[sender];
  s.push(1);
  return s.average() > msgps;
}


//
// parse a protobuf message from a zmq socket and try to handle it
//
template <typename T> inline
void try_handle(zmq::socket_t &sock, std::function<void(std::shared_ptr<T>)> handler) {
  zmq::message_t msg;
  sock.recv(&msg);
  auto resp = std::make_shared<T>();
  try {
    if (resp->ParseFromArray(msg.data(), msg.size())) {
      handler(resp);
    }
  } catch (...) {
  }
}

//
// CollabClient implementation
//
CollabClient::CollabClient(zmq::context_t &ctx,
                           boost::asio::ip::address_v4 server_ip,
                           long server_port,
                           boost::asio::ip::address_v4 client_ip,
                           long client_port, long peer_port,
                           uint64_t rate_limit)
    : _state(State::INIT), _continue(false), _srv_ip(server_ip),
      _cl_ip(client_ip), _srv_port(server_port), _cl_port(client_port),
      _peer_port(peer_port), _nonce(0), _ka_continue(false), _ntx(0), _nrx(0),
      _msg_count(1), _rlimit(rate_limit), _ctx(ctx),
      _srv_in_sock(ctx, ZMQ_PULL), _srv_out_sock(ctx, ZMQ_PUSH),
      _peer_in_sock(ctx, ZMQ_PULL) {
  NotificationCenter::shared.post(
      StateChangeEvent,
      StateChangeEventInfo{state2str(State::NO_STATE), state2str(State::INIT)});
}

CollabClient::CollabClient(zmq::context_t &ctx, ConnectionParams conn, uint64_t rate_limit)
   : _state(State::INIT), _continue(false), _srv_ip(conn.server_ip),
      _cl_ip(conn.client_ip), _srv_port(conn.server_port), _cl_port(conn.client_port),
      _peer_port(conn.peer_port), _nonce(0), _ka_continue(false), _ntx(0), _nrx(0),
      _msg_count(1), _rlimit(rate_limit), _ctx(ctx),
      _srv_in_sock(ctx, ZMQ_PULL), _srv_out_sock(ctx, ZMQ_PUSH),
      _peer_in_sock(ctx, ZMQ_PULL) {
  NotificationCenter::shared.post(
      StateChangeEvent,
      StateChangeEventInfo{state2str(State::NO_STATE), state2str(State::INIT)});
}
//
// connect 2 the collab server
//
void CollabClient::connect() {
  if (_state == State::INIT) {
    int i = 5;
    for (; i > 0; --i) {
      try {
        _srv_in_sock.bind(
            (boost::format("tcp://%s:%d") % _cl_ip.to_string() % _cl_port)
                .str());
        _peer_in_sock.bind(
            (boost::format("tcp://%s:%d") % _cl_ip.to_string() % _peer_port)
                .str());
        _srv_out_sock.connect(
            (boost::format("tcp://%s:%d") % _srv_ip.to_string() % _srv_port)
                .str());
        _pollin.add(_srv_in_sock, ZMQ_POLLIN);
        _pollin.add(_peer_in_sock, ZMQ_POLLIN);
        _pollout.add(_srv_out_sock, ZMQ_POLLOUT);
        break;
      } catch (std::exception &ex) {
        NotificationCenter::shared.post(ConnectionEvent, ConnectionEventInfo{false});
      }
    }
    if (i <= 0) {
      throw std::runtime_error("Connection Timeout");
    } else {
      // without exceptions, we are done.
      NotificationCenter::shared.post(ConnectionEvent, ConnectionEventInfo{true});
      NotificationCenter::shared.post(StateChangeEvent,
                                      StateChangeEventInfo{state2str(State::INIT),
                                                           state2str(State::REGISTER)});
      _state = State::REGISTER;
    }
  } else {
    throw std::runtime_error("Connection Timeout");
    return;
  }
}

bool CollabClient::tryConnect() {
  try {
    connect();
    return true;
  } catch (std::runtime_error &e) {
    return false;
  }
}

bool CollabClient::_send(zmq::socket_t &s, std::shared_ptr<google::protobuf::Message>m) {
  std::lock_guard<std::mutex> lock(_sndmtx);
  _pollout.poll(_poll_timeout);
  if (_pollout.check(s)) {
    zmq::message_t msg(m->ByteSizeLong());
    m->SerializeToArray(static_cast<void *>(msg.data()), msg.size());
    bool sent = false;
    while (!sent) {
      try {
        sent = s.send(msg);
      } catch (zmq::error_t &ex) {
        if (ex.num() != EINTR)
          throw;
        sent = false;
      }
    }
    return sent;
  } else {
    NotificationCenter::shared.post(ErrorEvent, ErrorEventInfo{ErrorType::SEND_FAIL, nullptr});
  }
}

//
// Peers
//
CollabClient::Peer::Peer(uint32_t ipaddr, zmq::context_t &ctx, long port)
    : ip(boost::asio::ip::address_v4(ipaddr)),
      sock(zmq::socket_t(ctx, ZMQ_PUSH)) {
  sock.connect((boost::format("tcp://%s:%d") % ip.to_string() % port).str());
}

void CollabClient::_add_peer(uint32_t peer) {
  using namespace sc2;
  std::lock_guard<std::mutex> lock(_peermtx);
  if (peer == _cl_ip.to_ulong())
    return;
  // if the peer is not already in the list, add it
  auto have_peer =
      std::find_if(begin(_peers), end(_peers),
                   [&peer](auto const &p) { return p->ip.to_ulong() == peer; });
  if (have_peer == end(_peers)) {
    auto p = std::make_unique<Peer>(peer, _ctx, _peer_port);
    _pollout.add(p->sock, ZMQ_POLLOUT);
    auto msg = std::make_shared<cil::CilMessage>();
    msg->set_sender_network_id(client_ip().to_ulong());
    msg->set_allocated_timestamp(getTimeNow());
    msg->set_msg_count(_msg_count++);
    msg->set_allocated_hello(new cil::Hello());
    // CilVersion
    msg->mutable_hello()->set_allocated_version(new cil::CilVersion());
    auto cil_version = msg->mutable_hello()->mutable_version();
    cil_version->set_major(3);
    cil_version->set_minor(6);
    cil_version->set_patch(0);
    _ntx++;
    NotificationCenter::shared.post(CollabTxEvent, CollabTxEventInfo{msg, _ntx, {p->ip}});
    NotificationCenter::shared.post(CollabPeerEvent, CollabPeerEventInfo{true, p->ip});
    _send(p->sock, msg);
    _peers.push_back(std::move(p));
  }
}

void CollabClient::_update_peers(
    google::protobuf::RepeatedField<unsigned int> const &neighbors) {
  // add new peers
  for (auto const &p : neighbors)
    _add_peer(p);
  // check whether any peers left and erase them, this is dirty
  {
    std::lock_guard<std::mutex> lock(_peermtx);
    std::vector<size_t> erase;
    for (auto i = 0; i < _peers.size(); ++i) {
      auto const &p = _peers[i];
      auto in_neighbors =
          std::find_if(neighbors.begin(), neighbors.end(),
                       [&p](auto &n) { return p->ip.to_ulong() == n; });
      if (in_neighbors == neighbors.end()) {
        erase.push_back(i);
      }
    }
    for (auto e : erase) {
      _pollout.remove(_peers[e]->sock);
      _peers.erase(begin(_peers) + e);
    }
  }
}

// peer-related
std::vector<boost::asio::ip::address_v4> CollabClient::peers() const {
  std::vector<boost::asio::ip::address_v4> p;
  for (auto const &peer : _peers)
    p.push_back(peer->ip);
  return p;
}
size_t CollabClient::num_peers() const { return _peers.size(); }

void CollabClient::_run() {
  using namespace sc2;
  if (_state != State::REGISTER) {
    // FIXME
    // doomsday("Need to be in REGISTER state before calling run().");
    return;
  }
  while (true) {
    if (!_continue)
      break;
    //
    // check for new messages
    //
    auto have_input = _pollin.poll(_poll_timeout);

    if (_state == State::REGISTER) {
      //
      // send a register message to the server
      //
      auto msg = std::make_shared<reg::TalkToServer>();
      msg->set_allocated_register_(new reg::Register());
      msg->mutable_register_()->set_my_ip_address(_cl_ip.to_ulong());
      _ntx++;
      NotificationCenter::shared.post(ServerTxEvent,
                                      ServerTxEventInfo{msg, _ntx});
      _send(_srv_out_sock, msg);
      _state = State::REGISTER_WAITING;
      NotificationCenter::shared.post(
          StateChangeEvent,
          StateChangeEventInfo{state2str(State::REGISTER),
                               state2str(State::REGISTER_WAITING)});
    } // end state::REGISTER

    if (_state == State::REGISTER_WAITING) {
      //
      // we wait until we have received the INFORM message
      //
      if ((have_input != 0) && _pollin.check(_srv_in_sock)) {
        try_handle<reg::TellClient>(
            _srv_in_sock, [this](auto msg) { this->_handle_server(msg, ++_nrx); });
      }
    } // end State::REGISTER_WAITING

    if (_state == State::ACTIVE) {
      //
      // look for inputs and call handlers
      //
      if (have_input != 0) {
        if (_pollin.check(_srv_in_sock)) {
          // have a server message
          try_handle<reg::TellClient>(
              _srv_in_sock, [this](auto msg) { this->_handle_server(msg, ++_nrx); });
        } else if (_pollin.check(_peer_in_sock)) {
          // have a collab message
          try_handle<cil::CilMessage>(
              _peer_in_sock, [this](auto msg) { this->_handle_collab(msg, ++_nrx); });
        }
      }
    } // end State::ACTIVE
  }
}

//
// periodically send a keepalive message to the server
//
void CollabClient::_keepalive(std::chrono::milliseconds interval) {
  using namespace sc2;
  using namespace std::chrono;
  std::mt19937 mt(system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<> r(0, 1);
  auto rand = [&mt, &r] { return r(mt); };
  do {
    auto start = system_clock::now();
    if (!_ka_continue)
      return;
    auto msg = std::make_shared<reg::TalkToServer>();
    msg->set_allocated_keepalive(new reg::Keepalive());
    msg->mutable_keepalive()->set_my_nonce(_nonce);
    _ntx++;
    NotificationCenter::shared.post(ServerTxEvent, ServerTxEventInfo{msg, _ntx});
    _send(_srv_out_sock, msg);
    auto elapsed = duration_cast<milliseconds>(system_clock::now() - start);
    // 0.75 is a fudge factor from the reference implementation
    auto sleepfor = rand() * (interval - elapsed) * 0.75;
    std::this_thread::sleep_for(sleepfor);
  } while (true);
}

//
// tear down the client
//
void CollabClient::stop() {
  // notify worker threads
  _ka_continue = false;
  _continue = false;
  // send LEAVE message
  auto lmsg = std::make_shared<sc2::reg::TalkToServer>();
  lmsg->set_allocated_leave(new sc2::reg::Leave());
  lmsg->mutable_leave()->set_my_nonce(_nonce);
    _ntx++;
    NotificationCenter::shared.post(ServerTxEvent, ServerTxEventInfo{lmsg, _ntx});
  _send(_srv_out_sock, lmsg);
  // properly destroy worker threads
  _main_thread.join();
  _keepalive_thread.join();
}

//
// start the client
//
// non-blocking, new thread
void CollabClient::start() {
  _continue = true;
  _main_thread = std::thread([this] {
    bamradio::set_thread_name("collabclient_main");
    _run();
  });
}
// blocking, same thread
void CollabClient::run() {
  _continue = true;
  _run();
}

//
// handlers
// note that handlers take ownership of the message
//
void CollabClient::_handle_server(std::shared_ptr<sc2::reg::TellClient> m, int64_t id) {
  NotificationCenter::shared.post(ServerRxEvent, ServerRxEventInfo{m, id});
  if (_state == State::REGISTER_WAITING) {
    //
    // in REGISTER_WAITING state, we should be getting a INFORM message
    //
    if (m->payload_case() == sc2::reg::TellClient::PayloadCase::kInform) {
      auto inform = m->inform();
      _nonce = inform.client_nonce();
      _max_keepalive_interval =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::duration<float>(inform.keepalive_seconds()));
      _update_peers(inform.neighbors());
      // start keepalive thread
      _ka_continue = true;
      _keepalive_thread = std::thread([this]() {
        bamradio::set_thread_name("collabclient_keepalive");
        _keepalive(_max_keepalive_interval);
      });
      NotificationCenter::shared.post(StateChangeEvent, StateChangeEventInfo{state2str(State::REGISTER_WAITING), state2str(State::ACTIVE)});
      _state = State::ACTIVE;
    } else {
      // this might happen when we recover from a crash. in this case, we time
      // out and move to register state.
      NotificationCenter::shared.post(StateChangeEvent, StateChangeEventInfo{state2str(State::REGISTER_WAITING), state2str(State::REGISTER)});
      _state = State::REGISTER;
    }
  } else if (_state == State::ACTIVE) {
    //
    // in ACTIVE state, we get NOTIFY messages from the server
    //
    if (m->payload_case() == sc2::reg::TellClient::PayloadCase::kNotify) {
      auto notify = m->notify();
      _update_peers(notify.neighbors());
    } else {
      // The client state machine has gotten out of sync with the server (We
      // expect only notify messages, but the server sent an inform message.). I
      // am not sure what to do here. The safe thing might be to time out and
      // try to re-register. Let's do this.
      _ka_continue = false;
      _keepalive_thread.join();
      NotificationCenter::shared.post(ErrorEvent, ErrorEventInfo{ErrorType::SERVER_SYNC, nullptr});
      NotificationCenter::shared.post(StateChangeEvent, StateChangeEventInfo{state2str(State::ACTIVE), state2str(State::REGISTER)});
      _state = State::REGISTER;
    }
  }
}

//
// collab message handler dispatch
//
void CollabClient::_handle_collab(std::shared_ptr<sc2::cil::CilMessage> m, int64_t id) {
  using namespace sc2::cil;

  // prevent DoS from peer
  if (_rlimit(m->sender_network_id())) {
    return;
  }

  // FIXME! we are releasing the payload to get the correct handlers
  // dispatched. This means that we have to copy the message before logging
  // it. This should be done differently. In a better implementation, we call
  // the handler for the cilmessage object, not just the payload.
  // I'm sorry guys.
  decltype(m) msg(m->New());
  msg->CopyFrom(*m);
  NotificationCenter::shared.post(CollabRxEvent, CollabRxEventInfo{msg, id});
  auto networkType = [&m] {
    if (m->has_network_type()) {
      switch (m->network_type().network_type()) {
      case sc2::cil::NetworkType::UNKNOWN:
        return collab::NetworkType::Unknown;
      case sc2::cil::NetworkType::COMPETITOR:
        return collab::NetworkType::Competitor;
      case sc2::cil::NetworkType::INCUMBENT_PASSIVE:
        return collab::NetworkType::IncumbentPassive;
      case sc2::cil::NetworkType::INCUMBENT_DSRC:
        return collab::NetworkType::IncumbentDSRC;
      case sc2::cil::NetworkType::INCUMBENT_JAMMER:
        return collab::NetworkType::IncumbentJammer;
      default:
        return collab::NetworkType::Unknown;
      }
    } else {
      return collab::NetworkType::Competitor;
    }
  }();
  Metadata md{
      id,          m->sender_network_id(),   m->msg_count(),
      networkType, m->timestamp().seconds(), m->timestamp().picoseconds()};
  switch (m->payload_case()) {
  case CilMessage::PayloadCase::kHello:
    // SUPPORTED
    _handle_hello(std::shared_ptr<Hello>(m->release_hello()), md);
    break;
  case CilMessage::PayloadCase::kSpectrumUsage:
    // SUPPORTED
    _handle_spectrum_usage(
        std::shared_ptr<SpectrumUsage>(m->release_spectrum_usage()), md);
    break;
  case CilMessage::PayloadCase::kIncumbentNotify:
    // SUPPORTED
    _handle_incumbent_notify(
        std::shared_ptr<IncumbentNotify>(m->release_incumbent_notify()), md);
    break;
  case CilMessage::PayloadCase::kLocationUpdate:
    // SUPPORTED
    _handle_location_update(
        std::shared_ptr<LocationUpdate>(m->release_location_update()), md);
    break;
  case CilMessage::PayloadCase::kDetailedPerformance:
    _handle_detailed_performance(
        std::shared_ptr<DetailedPerformance>(m->release_detailed_performance()), md);
    break;
  default:
    // UNSUPPORTED
    NotificationCenter::shared.post(ErrorEvent, ErrorEventInfo{ErrorType::UNSUPPORTED});
  }
}

//
// communication
//
// broadcast a message to all peers
int64_t CollabClient::broadcast(std::shared_ptr<sc2::cil::CilMessage> msg) {
  std::lock_guard<std::mutex> lock(_peermtx);
  if (_state != State::ACTIVE) {
    NotificationCenter::shared.post(
        ErrorEvent, ErrorEventInfo{ErrorType::NOT_CONNECTED, msg});
    return -1;
  }
  msg->set_msg_count(_msg_count++);
  msg->set_allocated_timestamp(getTimeNow());
  auto nt = msg->mutable_network_type();
  nt->set_network_type(sc2::cil::NetworkType::COMPETITOR);
  _ntx++;
  NotificationCenter::shared.post(CollabTxEvent,
                                  CollabTxEventInfo{msg, _ntx, peers()});
  for (auto const &p : _peers)
    _send(p->sock, msg);
  return _ntx;
}
// send a message to a specific peer
int64_t CollabClient::send(boost::asio::ip::address_v4 peer,
                           std::shared_ptr<sc2::cil::CilMessage> msg) {
  std::lock_guard<std::mutex> lock(_peermtx);
  if (_state != State::ACTIVE) {
    NotificationCenter::shared.post(
        ErrorEvent, ErrorEventInfo{ErrorType::NOT_CONNECTED, msg});
    return -1;
  }
  auto p = std::find_if(begin(_peers), end(_peers),
                        [&peer](auto &p) { return p->ip == peer; });
  if (p != end(_peers)) {
    msg->set_msg_count(_msg_count++);
    msg->set_allocated_timestamp(getTimeNow());
    auto nt = msg->mutable_network_type();
    nt->set_network_type(sc2::cil::NetworkType::COMPETITOR);
    _ntx++;
    NotificationCenter::shared.post(CollabTxEvent,
                                    CollabTxEventInfo{msg, _ntx, {(*p)->ip}});
    _send((*p)->sock, msg);
    return _ntx;
  } else {
    NotificationCenter::shared.post(ErrorEvent,
                                    ErrorEventInfo{ErrorType::NO_PEER, msg});
    return -1;
  }
}

//
// handler registration
//
void CollabClient::register_handler(
    std::function<void(std::shared_ptr<sc2::cil::Hello>, Metadata md)> handler) {
  _handle_hello.disconnect_all_slots();
  _handle_hello.connect(handler);
}
void CollabClient::register_handler(
    std::function<void(std::shared_ptr<sc2::cil::SpectrumUsage>, Metadata md)> handler) {
  _handle_spectrum_usage.disconnect_all_slots();
  _handle_spectrum_usage.connect(handler);
}
void CollabClient::register_handler(
    std::function<void(std::shared_ptr<sc2::cil::LocationUpdate>, Metadata md)>
        handler) {
  _handle_location_update.disconnect_all_slots();
  _handle_location_update.connect(handler);
}
void CollabClient::register_handler(
    std::function<void(std::shared_ptr<sc2::cil::IncumbentNotify>, Metadata md)> handler) {
  _handle_incumbent_notify.disconnect_all_slots();
  _handle_incumbent_notify.connect(handler);
}
void CollabClient::register_handler(
    std::function<void(std::shared_ptr<sc2::cil::DetailedPerformance>, Metadata md)> handler) {
  _handle_detailed_performance.disconnect_all_slots();
  _handle_detailed_performance.connect(handler);
}

std::string CollabClient::state2str(State s) {
  switch(s) {
  case State::NO_STATE:
    return "no_state";
  case State::INIT:
    return "init";
  case State::REGISTER:
    return "register";
  case State::REGISTER_WAITING:
    return "register_waiting";
  case State::ACTIVE:
    return "active";
  case State::NUM_STATES:
    return "¯\\_(ツ)_/¯";
  }
}

//
// Calculate the distance in meters between two points on Earth using the
// Haversine Method
//
double distance_haversine(double xlat, double xlong, double xele,
                          double ylat, double ylong, double yele) {
  // convert an angle in degrees to radians
  auto to_rad = [](double angle) {
    static const double factor = M_PI / 180.0;
    return angle * factor;
  };
  // radius of earth in meters
  static const double R = 6371e3;
  // convert the arguments to radians
  auto phi1 = to_rad(xlat);
  auto phi2 = to_rad(ylat);
  auto delta_phi = to_rad(ylat - xlat);
  auto delta_lambda = to_rad(ylong - xlong);
  // compute the distance
  auto ss1 = std::sin(delta_phi / 2.0);
  ss1 *= ss1;
  auto ss2 = std::sin(delta_lambda / 2.0);
  ss2 *= ss2;
  auto a = ss1 + std::cos(phi1) * std::cos(phi2) * ss2;
  // sin(theta(x,y)/2) == sqrt(a)
  //auto c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
  //return R * c;
  auto ds = (xele - yele)*(xele - yele) + 4 * (R + xele)*(R + yele)*a;
  return std::sqrt(ds);
}
double distance(bamradio::controlchannel::Location x,
                bamradio::controlchannel::Location y) {
  return distance_haversine(x.latitude, x.longitude, x.elevation,
                            y.latitude, y.longitude, y.elevation);
}
double distance(sc2::cil::Location x, sc2::cil::Location y) {
  return distance_haversine(x.latitude(), x.longitude(), x.elevation(),
                            y.latitude(), y.longitude(), y.elevation());
}
double distance(sc2::cil::Location *x, sc2::cil::Location *y) {
  return distance(*x, *y);
}

sc2::cil::TimeStamp *getTimeNow() {
  using namespace sc2;
  auto ts = new cil::TimeStamp();
  // Get time point right now and conver to duration since epoch.
  auto const now = std::chrono::system_clock::now().time_since_epoch();
  // Cast now to seconds duration since epoch.
  auto const seconds = std::chrono::duration_cast<std::chrono::seconds>(now);
  // Subtract off whole seconds and cast to picoseconds duration since epoch.
  auto const picoseconds =
      std::chrono::duration_cast<std::chrono::duration<int64_t, std::pico>>(
          now - seconds);
  ts->set_seconds(seconds.count());
  ts->set_picoseconds(picoseconds.count());
  return ts;
}

} // namespace collab
} // namespace bamradio
