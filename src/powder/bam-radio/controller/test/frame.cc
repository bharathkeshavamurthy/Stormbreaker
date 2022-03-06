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

BOOST_AUTO_TEST_CASE(frame) {
  using namespace bamradio;

  auto const mcs = ofdm::MCS::QPSK_R23_N1944;
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

  std::vector<dll::Segment::sptr> txSegments{ippp(1, 155 * 2 - 8), ippp(1, 155 * 2 - 8),
                                             ippp(2, 300)};

  ofdm::DFTSOFDMFrame f(0, 1, txSegments, ofdm::MCS::QPSK_R12_N648, mcs,
                        ofdm::SeqID::ID::P19FULL_128_12_108_QPSK, 23, 0);

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

  // check corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = ++begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = ++begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0],
                                                     txSegments[2]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (3)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0],
                                                     txSegments[2]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (4)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0],
                                                     txSegments[1]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (5)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 5, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0],
                                                     txSegments[1]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[1],
                                                     txSegments[2]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0],
                                                     txSegments[2]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 5, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0],
                                                     txSegments[1]};

    BOOST_TEST(rxSegments.size() == 2);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 0, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 1, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[2]};

    BOOST_TEST(rxSegments.size() == 1);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 0, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[2]};

    BOOST_TEST(rxSegments.size() == 1);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{txSegments[0]};

    BOOST_TEST(rxSegments.size() == 1);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }
}

BOOST_AUTO_TEST_CASE(frame56) {
  using namespace bamradio;

  auto const mcs = ofdm::MCS::QPSK_R12_N1944;
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

  std::vector<dll::Segment::sptr> txSegments{
      ippp(1, 229 - 8), ippp(1, 229 - 8), ippp(2, 300 - 8), ippp(88, 700 - 8), ippp(22, 203 - 8)};

  ofdm::DFTSOFDMFrame f(0, 1, txSegments, ofdm::MCS::QPSK_R12_N648, mcs,
                        ofdm::SeqID::ID::P19FULL_128_12_108_QPSK, 23, 0);

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

    BOOST_TEST(rxSegments.size() == 5);

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

  // check corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = ++begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = ++begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (3)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (4)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[1], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (5)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 5, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[1], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[1], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 5, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[1], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 0, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 1, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 0, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }
}

BOOST_AUTO_TEST_CASE(frame57) {
  using namespace bamradio;

  auto const mcs = ofdm::MCS::QAM16_R56_N1944;
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

  std::vector<dll::Segment::sptr> txSegments{
      ippp(1, 391 - 8), ippp(1, 391 - 8), ippp(2, 300 - 8), ippp(88, 700 - 8), ippp(22, 203 - 8)};

  ofdm::DFTSOFDMFrame f(0, 1, txSegments, ofdm::MCS::QPSK_R12_N648, mcs,
                        ofdm::SeqID::ID::P10FULL_128_12_108_QAM16, 23, 0);

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

    BOOST_TEST(rxSegments.size() == 5);

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

  // check corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = ++begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = ++begin(txSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(txSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (3)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (4)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[1], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check corrupted payload (5)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 5, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[1], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[1], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 4);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check doubly corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 5, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[1], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (0)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 0, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 1, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (1)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 0, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[2], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }

  // check triply corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 4, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 3, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[0], txSegments[3], txSegments[4]};

    BOOST_TEST(rxSegments.size() == 3);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }
}

BOOST_AUTO_TEST_CASE(frame58) {
  using namespace bamradio;

  auto const mcs = ofdm::MCS::QAM16_R56_N1944;
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

  std::vector<dll::Segment::sptr> txSegments{
      ippp(1, 1500), ippp(1, 1500)};

  ofdm::DFTSOFDMFrame f(0, 1, txSegments, ofdm::MCS::QPSK_R12_N648, mcs,
                        ofdm::SeqID::ID::P10FULL_128_12_108_QAM16, 23, 0);

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

    BOOST_TEST(rxSegments.size() == 2);

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


  // check corrupted payload (2)
  {
    auto framebitsCorrupt = framebits;
    std::fill_n(begin(framebitsCorrupt) + 648 + 1944 * 2, 10, 0);
    const_buffer fbb = buffer(framebitsCorrupt);

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

    std::vector<dll::Segment::sptr> expectedSegments{
        txSegments[1]};

    BOOST_TEST(rxSegments.size() == 1);

    auto txsit = begin(expectedSegments);
    for (auto const &s : rxSegments) {
      BOOST_TEST((txsit != end(expectedSegments)));
      BOOST_TEST((s->type() == (*txsit)->type()));
      BOOST_TEST((s->length() == (*txsit)->length()));
      BOOST_TEST((s->destNodeID() == (*txsit)->destNodeID()));
      BOOST_TEST(segmentHaveSameRawContents(s, *txsit));
      txsit++;
    }
  }
}
