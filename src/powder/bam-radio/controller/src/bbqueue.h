// -*-c++-*-
//  Copyright © 2017 Stephen Larew

#ifndef f2ff400baff475349e
#define f2ff400baff475349e

#include <condition_variable>
#include <mutex>
#include <queue>

namespace bamradio {

/// A thread-safe, size-limited, humorous FIFO queue.
template <typename T> class BBQueue {
public:
  explicit BBQueue(size_t initial_size) : _available(initial_size) {
    if (!_available) {
      throw std::invalid_argument("initial_size must be greater than zero");
    }
  }

  /// Pop off next item from queue. Block if queue is empty.
  T pop() {
    std::unique_lock<decltype(_mutex)> l(_mutex);
    while (_queue.empty()) {
      _condition.wait(l);
    }
    auto const f = _queue.front();
    _queue.pop();
    ++_available;
    l.unlock();
    _condition.notify_one();
    return f;
  }

  /// Pop an item off the queue without blocking.
  bool tryPopNoBlock(T &f) {
    // Use try_to_lock to avoid blocking.
    std::unique_lock<decltype(_mutex)> l(_mutex, std::try_to_lock);
    if (!l || _queue.empty()) {
      return false;
    }
    f = _queue.front();
    _queue.pop();
    ++_available;
    l.unlock();
    _condition.notify_one();
    return true;
  }

  /// Return true if an item was available and popped off the queue.
  bool tryPop(T &f) {
    std::unique_lock<decltype(_mutex)> l(_mutex);
    if (_queue.empty()) {
      return false;
    }
    f = _queue.front();
    _queue.pop();
    ++_available;
    l.unlock();
    _condition.notify_one();
    return true;
  }

  bool tryPushNoBlock(T const &tee) {
    // Use try_to_lock to avoid blocking.
    std::unique_lock<decltype(_mutex)> l(_mutex, std::try_to_lock);
    if (!l || !_available) {
      return false;
    }
    _queue.push(tee);
    --_available;
    l.unlock();
    _condition.notify_one();
    return true;
  }

  bool tryPush(T const &tee) {
    std::unique_lock<decltype(_mutex)> l(_mutex);
    if (!_available) {
      return false;
    }
    _queue.push(tee);
    --_available;
    l.unlock();
    _condition.notify_one();
    return true;
  }

  void push(T const &tee) {
    std::unique_lock<decltype(_mutex)> l(_mutex);
    while (!_available) {
      _condition.wait(l);
    }
    _queue.push(tee);
    --_available;
    l.unlock();
    _condition.notify_one();
  }

private:
  std::mutex _mutex;
  std::queue<T> _queue;
  std::condition_variable _condition;
  size_t _available;
};
}

#endif
