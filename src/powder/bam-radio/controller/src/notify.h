// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#ifndef aa186294d7544448ff
#define aa186294d7544448ff

#include <functional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include <boost/any.hpp>
#include <boost/asio.hpp>
#include <boost/core/noncopyable.hpp>

namespace bamradio {

/// Notifcations via pub-sub pattern.
class NotificationCenter {
public:
  /// Type for a notification name
  typedef std::hash<std::string>::result_type Name;

  static Name makeName(std::string const &s) {
    return std::hash<std::string>{}(s);
  }

  typedef uint64_t SubID;

  class Unsubscriber {
  private:
    NotificationCenter *_nc;
    Name _n;
    SubID _sid;
    friend NotificationCenter;
    Unsubscriber(NotificationCenter *nc, Name n, SubID sid)
        : _nc(nc), _n(n), _sid(sid) {}

  public:
    // Constructors and swap explained by copy-and-swap idiom
    friend void swap(Unsubscriber &l, Unsubscriber &r) {
      using std::swap;
      swap(l._nc, r._nc);
      swap(l._n, r._n);
      swap(l._sid, r._sid);
    }
    Unsubscriber() : _nc(nullptr) {}
    Unsubscriber(Unsubscriber const &) = delete;
    Unsubscriber(Unsubscriber &&u) : Unsubscriber() { swap(*this, u); }
    Unsubscriber &operator=(const Unsubscriber &) = delete;
    Unsubscriber &operator=(Unsubscriber &&u) {
      swap(*this, u);
      return *this;
    }
    ~Unsubscriber() {
      if (_nc) {
        _nc->unsubscribe(_n, _sid);
      }
    }
    void reset() { *this = Unsubscriber(); }
  };

  typedef Unsubscriber SubToken;

  /// Shared notification center object
  static NotificationCenter shared;

  /// Subscribe to notification named n and receive values on ios by calling f.
  ///
  /// Returns a subscription token that must be kept for the lifetime of the
  /// subscription.
  template <typename T>
  SubToken subscribe(Name const &n, boost::asio::io_service &ios,
                     std::function<void(T)> f) {
#ifndef NDEBUG
    std::unique_lock<std::shared_timed_mutex> l(_m, std::chrono::seconds(2));
    if (!l.owns_lock()) {
      throw std::runtime_error("failed to acquire lock (check deadlock)");
    }
#else
    std::lock_guard<std::shared_timed_mutex> l(_m);
#endif
    _subscriptions[n].emplace_back(&ios, boost::any(f), _nextSubID);
    return Unsubscriber(this, n, _nextSubID++);
  }

  /// Post value v under the name n.  Value v is captured by-copy.
  template <typename T> void post(Name const &n, T const &v) const {
    std::shared_lock<std::shared_timed_mutex> l(_m);

    auto const subscribers = _subscriptions.find(n);
    if (subscribers == _subscriptions.end()) {
      return;
    }

    for (auto const &s : subscribers->second) {
      auto const f =
          boost::any_cast<std::function<void(T)>>(std::get<1>(s));
      std::get<0>(s)->dispatch([f, v] { f(v); });
    }
  }

private:
  // TODO: use shared_mutex in c++17 (no timeout needed)
  std::shared_timed_mutex mutable _m;
  SubID _nextSubID;
  typedef std::tuple<boost::asio::io_service *, boost::any, SubID> Subscription;
  std::unordered_map<Name, std::vector<Subscription>> _subscriptions;

  friend Unsubscriber;

  void unsubscribe(Name const &n, SubID i) {
#ifndef NDEBUG
    std::unique_lock<std::shared_timed_mutex> l(_m, std::chrono::seconds(2));
    if (!l.owns_lock()) {
      throw std::runtime_error("failed to acquire lock (check deadlock)");
    }
#else
    std::lock_guard<std::shared_timed_mutex> l(_m);
#endif

    auto const subscribers = _subscriptions.find(n);
    if (subscribers == _subscriptions.end()) {
      throw std::runtime_error("bad name in unsubscribe");
    }

    auto const s =
        std::find_if(subscribers->second.begin(), subscribers->second.end(),
                     [i](auto s) { return std::get<2>(s) == i; });
    if (s == subscribers->second.end()) {
      throw std::runtime_error("bad unsubscribe");
    }
    subscribers->second.erase(s);
  }
};

} // namespace bamradio

#endif
