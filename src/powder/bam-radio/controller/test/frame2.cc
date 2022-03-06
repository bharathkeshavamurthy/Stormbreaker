//  Copyright © 2017 Stephen Larew

#define BOOST_TEST_MODULE ofdm
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/frame.h"
#include "../src/ippacket.h"

#include <array>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>

#include <boost/asio.hpp>
#include <boost/format.hpp>

using boost::format;
using namespace boost::asio;
using std::cout;
using std::cerr;
using std::endl;

bool segmentHaveSameRawContents(bamradio::dll::Segment::sptr l,
                                bamradio::dll::Segment::sptr r) {
  auto const a = l->rawContentsBuffer();
  assert(a.size() > 0);
  auto const b = r->rawContentsBuffer();
  std::vector<uint8_t> rb;
  for (auto const &bb : b) {
    auto const off = rb.size();
    rb.resize(off + buffer_size(bb));
    std::copy_n(buffer_cast<uint8_t const *>(bb), buffer_size(bb), &rb[off]);
  }
  return buffer_size(a) == rb.size() &&
         (memcmp(buffer_cast<uint8_t const *>(a.front()), &rb[0], rb.size()) ==
          0);
}

BOOST_AUTO_TEST_CASE(frame2) {
  using namespace bamradio;

  for (size_t ii = 0; ii < 1000; ++ii) {

  auto const mcs = ofdm::MCS::QAM64_R34_N1944;
  auto const mcss = ofdm::MCS::table[mcs];

  auto const ippp = [](NodeID destNodeID, size_t const n) {
    std::vector<uint8_t> v(n);
    uint8_t va = 0;
    std::generate(begin(v), end(v), [&va] { return va++; });
    auto *p = new net::IPPacket(v); // yeah this is a leak, so sue me
    p->resize(n);
    return std::make_shared<net::IP4PacketSegment>(destNodeID, p->get_buffer(),
                                                   std::chrono::system_clock::now());
  };

  std::vector<dll::Segment::sptr> txSegments{ippp(1, ii == 0 ? 1500 : 20 + rand() % 1480), ippp(1, ii == 0 ? 1500 : 20 + rand() % 1480)};
  if (ii > 0) {
    size_t n = rand() % 5;
    for  (size_t jj = 0; jj < n; ++jj) {
      txSegments.push_back(ippp(1, 20 + rand() % 1480));
    }
  }

  ofdm::DFTSOFDMFrame f(0, 1, txSegments, ofdm::MCS::QPSK_R12_N648, mcs,
                        ofdm::SeqID::ID::P10FULL_128_12_108_QAM64, 23, 0);

  auto const numbits = f.numBlocks(true) * f.blockLength(true) +
                       f.numBlocks(false) * f.blockLength(false);
  std::vector<uint8_t> framebits(numbits);

  {
    mutable_buffer fbb = buffer(framebits);

    for (size_t i = 0; i < f.numBlocks(true); ++i) {
      f.writeBlock(true, i, fbb);
      fbb = fbb + f.blockLength(true);
    }
    for (size_t i = 0; i < f.numBlocks(false); ++i) {
      f.writeBlock(false, i, fbb);
      fbb = fbb + f.blockLength(false);
    }
  }

  // check uncorrupted payload
  {
    const_buffer fbb = buffer(framebits);

    ofdm::DFTSOFDMFrame rf(ofdm::MCS::QPSK_R12_N648, buffer(fbb, 324));

    fbb = fbb + rf.blockLength(true);

    for (size_t i = 0; i < rf.numBlocks(false); ++i) {
      auto const r =
          rf.readBlock(i,
                       buffer(fbb, mcss.blockLength * mcss.codeRate));
      BOOST_TEST(r.first == mcss.blockLength * mcss.codeRate);
      fbb = fbb + rf.blockLength(false);
    }

    auto const rxSegments = rf.segments();
    auto const rxPayload = rf.movePayload();

    auto txsit = begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  srand(3023);

  size_t fuzzit = 0;
  while (fuzzit < 10000) {
    fuzzit++;
    std::cout << fuzzit << std::endl;
    auto framebitsCorrupt = framebits;
    //std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    auto numregions = rand() % 20;
    for (auto i = 0; i < numregions; ++i) {
      auto n = rand() % (1944*2);
      std::generate_n(begin(framebitsCorrupt) + (rand() % (framebitsCorrupt.size()-n)), n, []{ return rand() & 1;});
    }
    const_buffer fbb = buffer(framebitsCorrupt);

    try {
      ofdm::DFTSOFDMFrame rf(ofdm::MCS::QPSK_R12_N648, buffer(fbb, 324));

      fbb = fbb + rf.blockLength(true);

      for (size_t i = 0; i < rf.numBlocks(false); ++i) {
        auto const r =
            rf.readBlock(i,
                         buffer(fbb, mcss.blockLength * mcss.codeRate));
        BOOST_TEST(r.first == mcss.blockLength * mcss.codeRate);
        fbb = fbb + rf.blockLength(false);
      }

      auto const rxSegments = rf.segments();
      auto const rxPayload = rf.movePayload();
    }
    catch (ofdm::DFTSOFDMFrame::BadCRC &e) {
      continue;
    }
    catch (std::logic_error &e) {
      std::cout << e.what() << endl;
      std::abort();
    }
  }
  }
}

