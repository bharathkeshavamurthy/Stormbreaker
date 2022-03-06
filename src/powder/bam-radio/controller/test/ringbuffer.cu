// -*- c++ -*-
//
// CPU -> GPU ring buffer tests
//
//  Copyright (c) 2018 Dennis Ogbe

#include "buffers.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <string.h>

#define BOOST_TEST_MODULE gpudsp_ringbuffer
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

// since BOOST_TEST does not want to work...
#define DENNIS_ASSERT(expr)                                                    \
  do {                                                                         \
    if (!(expr))                                                               \
      abort();                                                                 \
  } while (0)

// CUDA error checking
#define RBcudaCheckErrors(msg)                                                 \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

//
// helpers
//
template <typename T> std::vector<T> gen_random_int_vec(size_t size) {
  // randomness
  std::mt19937 rd(33);
  std::uniform_int_distribution<int> dis(1, 100);
  std::vector<int> v(size);
  std::generate(begin(v), end(v), [&] { return dis(rd); });
  return v;
}

template <typename T> T rand_int(int min, int max) {
  // randomness
  std::mt19937 rd(33);
  std::uniform_int_distribution<int> dis(min, max);
  return dis(rd);
}

// dumb cuda kernel that demos reading and writing
__global__ void copy2(int *dst, const int *src) {
  dst[threadIdx.x] = src[threadIdx.x];
}

// test single threaded read/write
BOOST_AUTO_TEST_CASE(ringbuffer_st) {
  auto rb = ringbuffer::Ringbuffer<int>::make(10000);

  // generate some random data
  int N = 10; // N times as much as can be held in the buffer
  int nitems = N * rb->nitems();
  auto in = gen_random_int_vec<int>(nitems);

  // randomly produce and consume items (single-threaded)
  std::vector<int> out(in.size());
  int nread = 0;
  int nwritten = 0;
  for (;;) {
    // add a random number of items to the buffer
    auto write = rand_int<int>(1, rb->space_avail());
    memcpy(rb->write_ptr(), &in[nwritten], sizeof(int) * write);
    rb->produce(write);
    nwritten += write;
    // read a random number of items from the buffer
    auto read = rand_int<int>(1, rb->items_avail());
    memcpy(&out[nread], rb->read_ptr(), sizeof(int) * read);
    rb->consume(read);
    nread += read;
    // break at some point
    if (nread > (N - 1) * rb->nitems()) {
      break;
    }
  }

  // compare the input and output
  for (size_t i = 0; i < nread; ++i) {
    DENNIS_ASSERT(in[i] == out[i]);
  }
}

// test single threaded read/write (nocuda)
BOOST_AUTO_TEST_CASE(ringbuffer_st_nocuda) {
  auto rb = ringbuffer::Ringbuffer<
      int, ringbuffer::rb_detail::memfd_nocuda_circbuf<int>>::make(10000);

  // generate some random data
  int N = 10; // N times as much as can be held in the buffer
  int nitems = N * rb->nitems();
  auto in = gen_random_int_vec<int>(nitems);

  // randomly produce and consume items (single-threaded)
  std::vector<int> out(in.size());
  int nread = 0;
  int nwritten = 0;
  for (;;) {
    // add a random number of items to the buffer
    auto write = rand_int<int>(1, rb->space_avail());
    memcpy(rb->write_ptr(), &in[nwritten], sizeof(int) * write);
    rb->produce(write);
    nwritten += write;
    // read a random number of items from the buffer
    auto read = rand_int<int>(1, rb->items_avail());
    memcpy(&out[nread], rb->read_ptr(), sizeof(int) * read);
    rb->consume(read);
    nread += read;
    // break at some point
    if (nread > (N - 1) * rb->nitems()) {
      break;
    }
  }

  // compare the input and output
  for (size_t i = 0; i < nread; ++i) {
    DENNIS_ASSERT(in[i] == out[i]);
  }
}

// detailed housekeeping test
BOOST_AUTO_TEST_CASE(ringbuffer_fill) {
  auto rb = ringbuffer::Ringbuffer<int>::make(10);
  // generate some random data and fill the buffer buffer
  auto in = gen_random_int_vec<int>(rb->space_avail());
  memcpy(rb->write_ptr(), in.data(), in.size() * sizeof(int));
  rb->produce(in.size());
  DENNIS_ASSERT(rb->space_avail() == 0);
  // read some and then fill the buffer again
  auto in2 = gen_random_int_vec<int>(100);
  std::vector<int> out1(in2.size());
  memcpy(out1.data(), rb->read_ptr(), out1.size() * sizeof(int));
  rb->consume(out1.size());
  DENNIS_ASSERT(rb->items_avail() == rb->nitems() - out1.size() - 1);
  for (size_t i = 0; i < in2.size(); ++i) {
    DENNIS_ASSERT(out1[i] == in[i]);
  }
  memcpy(rb->write_ptr(), in2.data(), in2.size() * sizeof(int));
  rb->produce(in2.size());
  DENNIS_ASSERT(rb->space_avail() == 0);
  // the ring buffer should now contain [in2 | last 923 items of in]. if I read
  // 1024 items from it, I should get [last 923 items of in | 100 items of in2].
  std::vector<int> out2(rb->items_avail());
  memcpy(out2.data(), rb->read_ptr(), rb->items_avail() * sizeof(int));
  rb->consume(rb->items_avail());
  DENNIS_ASSERT(rb->space_avail() == rb->nitems() - 1);
  for (size_t i = 0; i < 923; ++i) {
    DENNIS_ASSERT(out2[i] == in[100 + i]);
  }
  for (size_t i = 0; i < in2.size(); ++i) {
    DENNIS_ASSERT(out2[923 + i] == in2[i]);
  }
}

// multi-threaded w/ read/write from CPU
BOOST_AUTO_TEST_CASE(ringbuffer_mt) {
  auto rb = ringbuffer::Ringbuffer<int>::make(10000);

  // generate some random data
  int N = 1000; // N times as much as can be held in the buffer
  int nitems = N * rb->nitems();
  auto in = gen_random_int_vec<int>(nitems);

  // randomly produce and consume items (multi-threaded)
  std::vector<int> out(in.size());
  std::atomic<int> nread(0);
  std::atomic<int> nwritten(0);

  std::atomic_bool producing(true);
  std::thread producer([&] {
    while (producing) {
      // add a random number of items to the buffer
      auto space_avail = rb->space_avail();
      if (space_avail < 1) {
        continue;
      }
      auto write = rand_int<int>(1, space_avail);
      memcpy(rb->write_ptr(), &in[nwritten], sizeof(int) * write);
      rb->produce(write);
      nwritten += write;
    }
  });

  for (;;) {
    // read a random number of items from the buffer
    auto items_avail = rb->items_avail();
    if (items_avail < 1) {
      continue;
    }
    auto read = rand_int<int>(1, items_avail);
    memcpy(&out[nread], rb->read_ptr(), sizeof(int) * read);
    rb->consume(read);
    nread += read;
    // break at some point
    if (nread > (N - 1) * rb->nitems()) {
      producing = false;
      break;
    }
  }
  producer.join();

  // compare the input and output
  for (size_t i = 0; i < nread; ++i) {
    DENNIS_ASSERT(in[i] == out[i]);
  }
  std::cout << "(mt) Read " << nread << " items from the buffer" << std::endl;
}

// multi-threaded w/ writing from the CPU, reading from GPU
BOOST_AUTO_TEST_CASE(ringbuffer_gpuread) {
  auto rb = ringbuffer::Ringbuffer<int>::make(10000);

  // generate some random data
  int N = 10; // N times as much as can be held in the buffer
  int nitems = N * rb->nitems();
  auto in = gen_random_int_vec<int>(nitems);

  // randomly produce and consume items (multi-threaded)
  int nread = 0;
  int nwritten = 0;

  std::atomic_bool producing(true);
  std::thread producer([&] {
    while (producing) {
      // add a random number of items to the buffer
      auto space_avail = rb->space_avail();
      if (space_avail < 1) {
        continue;
      }
      auto write = rand_int<int>(1, space_avail);
      memcpy(rb->write_ptr(), &in[nwritten], sizeof(int) * write);
      rb->produce(write);
      nwritten += write;
    }
  });

  // get max number threads per block
  int tpb;
  cudaDeviceGetAttribute(&tpb, cudaDevAttrMaxThreadsPerBlock, 0);
  RBcudaCheckErrors("dev attr");
  // alloc output array on device
  int *d_out;
  std::vector<int> out(in.size());
  cudaHostRegister(out.data(), in.size() * sizeof(int),
                   0); // write right back to host
  RBcudaCheckErrors("t4: host register");
  d_out = out.data();
  for (;;) {
    // read a random number of items from the buffer
    auto items_avail = rb->items_avail();
    if (items_avail < 1) {
      continue;
    }
    int read = std::min(rand_int<int>(1, items_avail), tpb);
    // n.b. could copy into device memory allocated using cudaMalloc(...) here
    // as well
    copy2<<<1, read>>>(d_out + nread, rb->read_ptr());
    RBcudaCheckErrors("kernel launch");
    cudaThreadSynchronize();
    RBcudaCheckErrors("thread sync");
    rb->consume(read);
    nread += read;
    // break at some point
    if (nread > (N - 1) * rb->nitems()) {
      producing = false;
      break;
    }
  }
  producer.join();
  cudaMemcpy(out.data(), d_out, sizeof(int) * nread, cudaMemcpyDeviceToHost);
  RBcudaCheckErrors("t4: memcpy");
  // compare the input and output
  for (size_t i = 0; i < nread; ++i) {
    DENNIS_ASSERT(in[i] == out[i]);
  }
  std::cout << "(mt gpuread) Read " << nread << " items from the buffer"
            << std::endl;
  cudaHostUnregister(d_out);
  RBcudaCheckErrors("t4: host unregister");
}

// multi-threaded w/ writing from the GPU, reading from CPU
BOOST_AUTO_TEST_CASE(ringbuffer_gpuwrite) {
  auto rb = ringbuffer::Ringbuffer<int>::make(10000);

  // generate some random data
  int N = 10; // N times as much as can be held in the buffer
  int nitems = N * rb->nitems();
  auto in = gen_random_int_vec<int>(nitems);
  cudaHostRegister(in.data(), in.size() * sizeof(int), 0);
  RBcudaCheckErrors("t5: host register");

  // randomly produce and consume items (multi-threaded)
  int nread = 0;
  int nwritten = 0;

  // get max number threads per block
  int tpb;
  cudaDeviceGetAttribute(&tpb, cudaDevAttrMaxThreadsPerBlock, 0);
  RBcudaCheckErrors("dev attr");

  std::atomic_bool producing(true);
  std::thread producer([&] {
    while (producing) {
      // add a random number of items to the buffer
      auto space_avail = rb->space_avail();
      if (space_avail < 1) {
        continue;
      }
      auto write = std::min(tpb, rand_int<int>(1, space_avail));
      copy2<<<1, write>>>(rb->write_ptr(), in.data() + nwritten);
      RBcudaCheckErrors("t5: kernel launch");
      cudaThreadSynchronize();
      RBcudaCheckErrors("t5: thread sync");
      rb->produce(write);
      nwritten += write;
    }
  });

  std::vector<int> out(in.size());
  for (;;) {
    // read a random number of items from the buffer
    auto items_avail = rb->items_avail();
    if (items_avail < 1) {
      continue;
    }
    auto read = rand_int<int>(1, items_avail);
    memcpy(&out[nread], rb->read_ptr(), sizeof(int) * read);
    rb->consume(read);
    nread += read;
    // break at some point
    if (nread > (N - 1) * rb->nitems()) {
      producing = false;
      break;
    }
  }
  producer.join();
  // compare the input and output
  for (size_t i = 0; i < nread; ++i) {
    DENNIS_ASSERT(in[i] == out[i]);
  }
  std::cout << "(mt gpuwrite) Read " << nread << " items from the buffer"
            << std::endl;
  cudaHostUnregister(in.data());
  RBcudaCheckErrors("t5: host unregister");
}
