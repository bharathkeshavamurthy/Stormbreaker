// -*- c++ -*-
// Copyright (c) 2017 Tomohiro Arakawa.

#ifndef ARQBUFFER_H_
#define ARQBUFFER_H_

#include <boost/asio/buffer.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>
#include <chrono>
#include <ctime>
#include <mutex>
#include <tuple>
#include <vector>

namespace bamradio {

struct ARQEntry {
  unsigned int seq_num;
  std::chrono::time_point<std::chrono::system_clock> expiration_time;
  std::shared_ptr<std::vector<uint8_t>> payload;
};

struct seq_num_tag {};
struct expiration_time_tag {};

class ARQBuffer {
 public:
  ARQBuffer(size_t max_buf_size);
  ~ARQBuffer(){};

  typedef boost::multi_index_container<
      ARQEntry,
      boost::multi_index::indexed_by<
          boost::multi_index::ordered_unique<
              boost::multi_index::tag<seq_num_tag>,
              boost::multi_index::member<ARQEntry, unsigned int,
                                         &ARQEntry::seq_num>>,
          boost::multi_index::ordered_unique<
              boost::multi_index::tag<expiration_time_tag>,
              boost::multi_index::member<
                  ARQEntry, std::chrono::time_point<std::chrono::system_clock>,
                  &ARQEntry::expiration_time>>>>
      timed_buffer;

  /**
   * Add packet to buffer
   * @param seq_num Sequence number.
   * @param buf Payload.
   * @param timeout_duration Timeout.
   * @return True when successful. Otherwise returns false.
   */
  bool add(unsigned int seq_num, std::shared_ptr<std::vector<uint8_t>> buf,
           std::chrono::milliseconds timeout_duration);

  /**
   * Remove specified entry from buffer
   * @param seq_num Sequence number.
   * @return True when the entry was found and removed. Otherwise returns false.
   */
  bool remove(unsigned int seq_num);

  /**
   * Get specified segment of packet
   * @param seq_num Sequence number.
   * @return shared_ptr of the payload. Returns nullptr if the entry does not
   * exist.
   */
  std::shared_ptr<std::vector<uint8_t>> get(unsigned int seq_num);

  /**
   * Get all of the sequence numbers
   * @return Vector of sequence numbers.
   * Probably useful for Tx DLL.
   */
  std::vector<unsigned int> getAllSeqNums();

  /**
   * Get sequence numbers of missing entries
   * @return Vector of sequence numbers.
   * Probably useful for Rx DLL.
   */
  std::vector<unsigned int> getMissingSeqNums();

  /// Print buffer contents
  void printDebugMsg();

 private:
  void removeExpiredEntries();
  unsigned int d_max_buffer_size;
  timed_buffer d_buf;
  std::mutex d_m;
};
}

#endif
