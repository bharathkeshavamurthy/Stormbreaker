// -*- c++ -*-
// Copyright (c) 2017 Tomohiro Arakawa.

#include "arqbuffer.h"
#include <iostream>

using namespace boost::multi_index;

namespace bamradio {
ARQBuffer::ARQBuffer(size_t max_buf_size) { d_max_buffer_size = max_buf_size; }

bool ARQBuffer::add(unsigned int seq_num,
                    std::shared_ptr<std::vector<uint8_t>> buf,
                    std::chrono::milliseconds timeout_duration) {
  // Lock
  std::lock_guard<std::mutex> lock(d_m);

  // Remove expired entries
  removeExpiredEntries();

  // Check buffer size
  if (d_buf.size() >= d_max_buffer_size) return false;

  // Add new entry
  ARQEntry ent;
  ent.seq_num = seq_num;
  ent.expiration_time = std::chrono::system_clock::now() + timeout_duration;
  ent.payload = buf;
  d_buf.insert(ent);

  return true;
}

bool ARQBuffer::remove(unsigned int seq_num) {
  // Lock
  std::lock_guard<std::mutex> lock(d_m);

  // Remove expired entries
  removeExpiredEntries();

  // Get a map sorted by sequence number
  auto& m = d_buf.get<seq_num_tag>();

  // Find entry
  auto it = m.find(seq_num);
  if (it == m.end()) return false;

  // Delete
  m.erase(it);
  return true;
}

std::shared_ptr<std::vector<uint8_t>> ARQBuffer::get(unsigned int seq_num) {
  // Lock
  std::lock_guard<std::mutex> lock(d_m);

  // Remove expired entries
  removeExpiredEntries();

  // Get a map sorted by sequence number
  auto& m = d_buf.get<seq_num_tag>();

  // Find entry
  auto it = m.find(seq_num);
  if (it == m.end()) return nullptr;

  return it->payload;
}

std::vector<unsigned int> ARQBuffer::getAllSeqNums() {
  // Lock
  std::lock_guard<std::mutex> lock(d_m);

  // Remove expired entries
  removeExpiredEntries();

  // Get a map sorted by sequence number
  auto& m = d_buf.get<seq_num_tag>();

  // Get all the sequence numbers
  std::vector<unsigned int> seq_nums;
  for (auto it = m.begin(); it != m.end(); ++it)
    seq_nums.push_back(it->seq_num);

  return seq_nums;
}

std::vector<unsigned int> ARQBuffer::getMissingSeqNums() {
  // Lock
  std::lock_guard<std::mutex> lock(d_m);

  // Remove expired entries
  removeExpiredEntries();

  // Get a map sorted by sequence number
  auto& m = d_buf.get<seq_num_tag>();

  std::vector<unsigned int> seq_nums;
  if (m.empty()) return seq_nums;

  // Find missing sequence number
  auto it = m.begin();
  auto last_seq_num = (it++)->seq_num;
  while (it != m.end()) {
    if ((++last_seq_num) != it->seq_num)
      seq_nums.push_back(last_seq_num);
    else
      ++it;
  }
  return seq_nums;
}

void ARQBuffer::removeExpiredEntries() {
  // Get a map sorted by expiration time
  auto& m = d_buf.get<expiration_time_tag>();

  // Remove expired entries
  auto it = m.begin();
  while (it != m.end()) {
    if (it->expiration_time < std::chrono::system_clock::now()) {
      auto del_it = it;
      ++it;
      m.erase(del_it);
    } else
      ++it;
  }
}

void ARQBuffer::printDebugMsg() {
  // Lock
  std::lock_guard<std::mutex> lock(d_m);

  // Get a map sorted by sequence number
  auto& m = d_buf.get<seq_num_tag>();

  // Print
  for (auto it = m.begin(); it != m.end(); ++it) {
    std::time_t t = std::chrono::system_clock::to_time_t(it->expiration_time);
    std::cout << "Seq: " << it->seq_num << " Expiration: " << std::ctime(&t)
              << " Payload size: " << it->payload->size() << std::endl;
  }
}
}
