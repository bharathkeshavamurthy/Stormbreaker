// (c) Tomohiro Arakawa (tarakawa@purdue.edu)

#include "../src/channelizer.h"
#include "buffers.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#define BOOST_TEST_MODULE gpudsp_channelizer
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(channelizer_avg) {
  using namespace bam::dsp;
  size_t const Nsamp = 1000;

  // Generate all-one vector
  std::vector<fcomplex> in_buf(Nsamp);
  // std::fill(in_buf.begin(), in_buf.end(), fcomplex(1, 0));
  for (size_t i = 0; i < Nsamp; ++i) {
    in_buf[i] = fcomplex(i, 0);
  }

  Channelizer channelizer(3, 200);

  {
    // Filter 1
    size_t const ch = 0;
    std::vector<float> taps(100);
    std::fill(taps.begin(), taps.end(), 1);
    channelizer.setTaps(ch, taps);
    channelizer.setDecimationFactor(ch, 5);
    channelizer.setOffsetFreq(ch, 0.2);
  }
  {
    // Filter 2
    size_t const ch = 1;
    std::vector<float> taps(50);
    std::fill(taps.begin(), taps.end(), 1);
    channelizer.setTaps(ch, taps);
    channelizer.setDecimationFactor(ch, 17);
    channelizer.setOffsetFreq(ch, 0);
  }
  {
    // Filter 3
    size_t const ch = 2;
    std::vector<float> taps(30);
    std::fill(taps.begin(), taps.end(), 1);
    channelizer.setTaps(ch, taps);
    channelizer.setDecimationFactor(ch, 21);
    channelizer.setOffsetFreq(ch, 0);
  }

  // run
  std::vector<fcomplex> out_buf(Nsamp);
  size_t nconsumed, outsize;
  nconsumed = channelizer.execute(200, in_buf.data());
  std::cout << "# consumed = " << nconsumed << std::endl;
  outsize = channelizer.getOutput(0, out_buf.data());
  std::cout << "# outsize = " << outsize << std::endl;

  nconsumed = channelizer.execute(200, in_buf.data() + nconsumed);
  std::cout << "# consumed = " << nconsumed << std::endl;
  outsize = channelizer.getOutput(0, out_buf.data() + outsize);
  std::cout << "out2" << std::endl;
  std::cout << "# outsize = " << outsize << std::endl;

  for (size_t i = 0; i < 100; ++i) {
    auto data = out_buf[i];
    std::cout << i << ": " << data.real() << " + j" << data.imag() << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(channelizer_st) {
  using namespace bam::dsp;
  size_t const Nsamp = 5e6;
  size_t const Nchan = 10;
  size_t const nloop = 100;

  // Generate data
  std::vector<fcomplex> in_buf(Nsamp);
  std::fill(in_buf.begin(), in_buf.end(), fcomplex(1, 0));

  Channelizer channelizer(Nchan, Nsamp);

  // Fiter taps
  std::vector<float> filter_taps(500);
  for (size_t i = 0; i < Nchan; ++i) {
    channelizer.setTaps(i, filter_taps);
    channelizer.setDecimationFactor(i, 40 + i);
    channelizer.setOffsetFreq(i, 0.5);
  }

  auto t_begin = std::chrono::steady_clock::now();
  for (size_t i = 0; i < nloop; ++i) {
    size_t nconsumed = channelizer.execute(Nsamp, in_buf.data());
    std::cout << "# consume: " << nconsumed << std::endl;
  }
  auto t_end = std::chrono::steady_clock::now();

  auto t_diff =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin)
          .count();
  std::cout << "Throughput = " << Nsamp * nloop / t_diff << " Msps"
            << std::endl;
}

// Channelizer with ring buffer
// Based on ringbuffer_gpuread testcase (c) Dennis Ogbe
BOOST_AUTO_TEST_CASE(ringbuffer_gpuread) {
  using namespace bam::dsp;

  // parameters
  size_t const nsamp_total = 1e8;
  size_t const nchan = 10;
  size_t const nsamp_gpubuf = 0x1 << 17;
  size_t const nsamp_ringbuf = 1e7;

  // Ring buffer
  auto rb = ringbuffer::Ringbuffer<fcomplex,
                                   ringbuffer::rb_detail::memfd_nocuda_circbuf<
                                       fcomplex>>::make(nsamp_ringbuf);

  std::vector<fcomplex> in(nsamp_ringbuf);

  // generate some random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (size_t i = 0; i < nsamp_ringbuf; ++i) {
    in[i] = fcomplex(dis(gen), dis(gen));
  }

  // produce and consume items (multi-threaded)
  int nread = 0;
  int nwritten = 0;

  // Channelizer
  Channelizer channelizer(nchan, nsamp_gpubuf);

  // Fiter taps
  std::vector<float> filter_taps(500);
  for (size_t i = 0; i < nchan; ++i) {
    channelizer.setTaps(i, filter_taps);
    channelizer.setDecimationFactor(i, 40 + i);
    channelizer.setOffsetFreq(i, 0.5);
  }

  std::atomic_bool producing(true);
  std::thread producer([&] {
    while (producing) {
      auto space_avail = rb->space_avail();
      if (space_avail < 1) {
        continue;
      }
      auto read_size = std::min(space_avail, (ssize_t)nsamp_ringbuf);
      memcpy(rb->write_ptr(), in.data(), sizeof(fcomplex) * read_size);
      rb->produce(read_size);
      nwritten += read_size;
    }
  });

  auto t_begin = std::chrono::steady_clock::now();
  for (;;) {
    auto items_avail = rb->items_avail();
    if (items_avail < nsamp_gpubuf) {
      continue;
    }
    size_t nconsumed = channelizer.execute(nsamp_gpubuf, rb->read_ptr(), true);
    std::cout << "consumed = " << nconsumed << std::endl;

    rb->consume(nconsumed);
    nread += nconsumed;
    // break at some point
    if (nread > nsamp_total) {
      producing = false;
      break;
    }
  }
  auto t_end = std::chrono::steady_clock::now();
  producer.join();

  auto t_diff =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin)
          .count();
  std::cout << "Throughput = " << nsamp_total / t_diff << " Msps" << std::endl;
}
