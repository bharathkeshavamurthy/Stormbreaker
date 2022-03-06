// Median filter unit test
// Copyright (c) 2018 Tomohiro Arakawa

#define BOOST_TEST_MODULE median
#define BOOST_TEST_DYN_LINK
#include "../src/median.h"
#include <array>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <random>
#include <thread>
using namespace bamradio;

BOOST_AUTO_TEST_CASE(median) {
  // data points
  std::array<float, 9> data{{-1, 1.1, 2.3, 4.8, 2.3, 11.6, 10.5, 3.2, 3.1}};

  // median filter
  stats::Median<float> filt(1.0f, 0.0f, 10.0f, 10);
  for (float v : data) {
    filt.push(v);
  }

  float median_val = filt.median();
  std::cout << median_val << std::endl;
  BOOST_REQUIRE_CLOSE(median_val, 3.5, 0.0001);
}

BOOST_AUTO_TEST_CASE(median_ndist) {
  // rng
  std::random_device rd;
  std::mt19937 gen(rd());

  // median filter
  stats::Median<float> filt(1.0f, 0.0f, 10.0f, 10);

  std::normal_distribution<float> d1(4.5, 5);
  for (size_t i = 0; i < 10000; ++i) {
    filt.push(d1(gen));
  }
  float median_val1 = filt.median();
  std::cout << median_val1 << std::endl;
  BOOST_REQUIRE_CLOSE(median_val1, 4.5, 0.0001);

  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::normal_distribution<float> d2(5.5, 2);
  for (size_t i = 0; i < 5000; ++i) {
    filt.push(d2(gen));
  }
  float median_val2 = filt.median();
  std::cout << median_val2 << std::endl;
  BOOST_REQUIRE_CLOSE(median_val2, 5.5, 0.0001);
}
