// Median filter
// Copyright (c) 2018 Tomohiro Arakawa

#ifndef INCLUDED_MEDIAN_H_
#define INCLUDED_MEDIAN_H_

#include <algorithm>
#include <cassert>
#include <chrono>
#include <deque>
#include <numeric>
#include <vector>

namespace bamradio {
namespace stats {
template <class T> class Median {
public:
  Median(float window_sec, T min, T max, size_t nbins)
      : _window_duration(window_sec), _min(min), _max(max), _occurances(nbins) {
    assert(_min < _max);
  };

  void push(T v) {
    auto now = std::chrono::system_clock::now();
    BinIndex const idx = getBin(v);
    _occurances[idx]++;
    _q.push_back(std::make_pair(idx, now));
    _refresh();
  }

  T median() {
    _refresh();
    T const bin_width = (_max - _min) / _occurances.size();
    // sum
    float const sum =
        std::accumulate(_occurances.begin(), _occurances.end(), 0.0f);
    // cumsum
    float cumsum = 0.0f;
    T bin_val = _min + bin_width / 2.0f;
    for (BinIndex idx = 0; idx < _occurances.size(); ++idx) {
      cumsum += _occurances[idx] / sum;
      if (cumsum >= 0.5) {
        return bin_val;
      }
      bin_val += bin_width;
    }
    return bin_val;
  }

  size_t size() const { return _q.size(); }
  void flush() { _q.clear(); }

private:
  typedef size_t BinIndex;
  std::chrono::duration<float> _window_duration;
  typedef std::chrono::time_point<std::chrono::system_clock> TimeStamp;

  T const _min;
  T const _max;
  std::deque<std::pair<BinIndex, TimeStamp>> _q;
  std::vector<size_t> _occurances;

  BinIndex getBin(T v) {
    // Is there a better way of doing this?
    T const bin_width = (_max - _min) / _occurances.size();
    assert(bin_width > 0);
    for (BinIndex idx = 0; idx < _occurances.size(); ++idx) {
      T const bin_max = _min + (idx + 1) * bin_width;
      if (v < bin_max) {
        return idx;
      }
    }
    return _occurances.size() - 1;
  }

  void _refresh() {
    auto now = std::chrono::system_clock::now();
    while (_q.size() > 0) {
      auto v = _q.front();
      auto diff = now - v.second;
      if (diff > _window_duration) {
        assert(_occurances[v.first] > 0);
        _occurances[v.first]--;
        _q.pop_front();
      } else {
        break;
      }
    }
  };
};
} // namespace stats
} // namespace bamradio

#endif
