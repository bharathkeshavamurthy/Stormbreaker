// ARQ Segment tests.
//
// Copyright (c) 2018 Dennis Ogbe

#define BOOST_TEST_MODULE arqseg
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "frame.h"
#include "ippacket.h"
#include "mcs.h"
#include "ofdm.h"
#include "segment.h"

#include <algorithm>
#include <chrono>
#include <random>

#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <string.h>

#include <boost/asio.hpp>
#include <boost/format.hpp>

using namespace bamradio;
using addr = boost::asio::ip::address_v4;

// make an IP packet buffer
std::vector<uint8_t> make_ippacket(addr src, addr dst, uint16_t srcPort,
                                   uint16_t dstPort, size_t nbytes,
                                   int seed = 666) {
  // random bytes
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint8_t> rand(0, 255);
  net::IPPacket p(nbytes);
  auto buf = boost::asio::buffer_cast<uint8_t *>(p.get_buffer());
  std::generate_n(buf, nbytes, [&] { return rand(rng); });
  p.resize(nbytes);

  // UDP
  auto const hdr = (struct ip *)buf;
  hdr->ip_hl = 5;
  hdr->ip_v = 4;
  hdr->ip_p = IPPROTO_UDP;
  hdr->ip_tos = 0x00;
  hdr->ip_off = htons(IP_DF);
  auto const ip4hdrlength = hdr->ip_hl * 4;
  auto const l4hdr = (udphdr *)((uint8_t const *)hdr + ip4hdrlength);
  l4hdr->uh_sport = htons(srcPort);
  l4hdr->uh_dport = htons(dstPort);

  // IP
  p.setDstAddr(dst);
  p.setSrcAddr(src);

  return p.get_buffer_vec();
}

// test 1: interface 1 (when I construct as transmitter)

BOOST_AUTO_TEST_CASE(interface1) {
  // make an IP packet
  NodeID destNodeID = 2;
  auto const now = std::chrono::system_clock::now();
  auto const srcAddr = addr::from_string("192.168.101.2");
  auto const dstAddr = addr::from_string(
      (boost::format("192.168.10%1%.2") % (int)destNodeID).str());
  auto const srcPort = 4001;
  auto const dstPort = 5001;
  auto const nbytes = 1500;
  auto const buf = make_ippacket(srcAddr, dstAddr, srcPort, dstPort, nbytes);
  auto const arqData = net::ARQIP4PacketSegment::ARQData{
      .arqExtra = 666, .seqNum = 33, .burstNum = 100};
  auto seg =
      net::ARQIP4PacketSegment(destNodeID, boost::asio::buffer(buf), now);
  BOOST_REQUIRE_EQUAL(seg.arqDataSet(), false);
  seg.setArqData(arqData);
  BOOST_REQUIRE_EQUAL(seg.arqDataSet(), true);

  // call all methods
  BOOST_REQUIRE_EQUAL(seg.length(),
                      nbytes + sizeof(net::ARQIP4PacketSegment::ARQData) +
                          sizeof(std::chrono::nanoseconds::rep));

  BOOST_REQUIRE(seg.type() == dll::SegmentType::ARQIPv4);
  BOOST_REQUIRE(((dll::Segment *)&seg)->type() == dll::SegmentType::ARQIPv4);

  BOOST_REQUIRE_EQUAL(seg.checksummed(), true);
  BOOST_REQUIRE_EQUAL(seg.reliable(), true);
  BOOST_REQUIRE(seg.sourceTime() == now);
  BOOST_REQUIRE(seg.destNodeID() == destNodeID);

  FlowUID expected_flow_uid(dstPort);
  BOOST_REQUIRE(seg.flowUID() == expected_flow_uid);

  std::vector<uint8_t> nothing_vec;
  auto sourceTimeDuration = now.time_since_epoch().count();
  std::vector<boost::asio::const_buffer> expected_rc = {
      boost::asio::buffer(&sourceTimeDuration, sizeof(sourceTimeDuration)),
      boost::asio::buffer(&arqData, sizeof(arqData)), boost::asio::buffer(buf),
      boost::asio::buffer(nothing_vec)};
  auto rc = seg.rawContentsBuffer();
  BOOST_REQUIRE_EQUAL(expected_rc.size(), rc.size());
  for (size_t i = 0; i < rc.size(); ++i) {
    auto p_my = boost::asio::buffer_cast<const void *>(rc[i]);
    auto p_ex = boost::asio::buffer_cast<const void *>(expected_rc[i]);
    auto n = boost::asio::buffer_size(expected_rc[i]);
    BOOST_REQUIRE_EQUAL(0, memcmp(p_my, p_ex, n));
  }

  FlowID expected_flow_id{srcAddr.to_uint(), dstAddr.to_uint(), IPPROTO_UDP,
                          srcPort, dstPort};
  auto flow_id = seg.flowID();
  BOOST_REQUIRE(expected_flow_id == flow_id);

  BOOST_REQUIRE_EQUAL(seg.packetLength(), nbytes);

  auto expected_pc = buf.data();
  auto expected_npc = nbytes;
  auto my_pc = seg.packetContentsBuffer();
  BOOST_REQUIRE_EQUAL(expected_npc, boost::asio::buffer_size(my_pc));
  BOOST_REQUIRE_EQUAL(0, memcmp(boost::asio::buffer_cast<const void *>(my_pc),
                                expected_pc, expected_npc));

  BOOST_REQUIRE_EQUAL(seg.ipHeaderAddress(), (void const *)buf.data());
  BOOST_REQUIRE_EQUAL(seg.protocol(), IPPROTO_UDP);
  auto nnow = std::chrono::system_clock::now();
  BOOST_REQUIRE((nnow - now) == seg.currentDelay(nnow));
  // Intentionallly not testing priority(...) field
  BOOST_REQUIRE_EQUAL(srcAddr, seg.srcAddr());
  BOOST_REQUIRE_EQUAL(dstAddr, seg.dstAddr());
  BOOST_REQUIRE_EQUAL(srcPort, seg.srcPort());
  BOOST_REQUIRE_EQUAL(dstPort, seg.dstPort());

  // setters and getters;
  decltype(srcPort) newSrcPort(6001);
  decltype(dstPort) newDstPort(7001);
  seg.setSrcPort(newSrcPort);
  seg.setDstPort(newDstPort);
  BOOST_REQUIRE_EQUAL(newSrcPort, seg.srcPort());
  BOOST_REQUIRE_EQUAL(newDstPort, seg.dstPort());

  // arq specials
  auto ad = seg.arqData();
  BOOST_REQUIRE_EQUAL(ad.burstNum, arqData.burstNum);
  BOOST_REQUIRE_EQUAL(ad.seqNum, arqData.seqNum);
  BOOST_REQUIRE_EQUAL(ad.arqExtra, arqData.arqExtra);
  decltype(ad) newArqData{.arqExtra = 667, .seqNum = 444, .burstNum = 1};
  seg.setArqData(newArqData);
  auto ad2 = seg.arqData();
  BOOST_REQUIRE_EQUAL(ad2.burstNum, newArqData.burstNum);
  BOOST_REQUIRE_EQUAL(ad2.seqNum, newArqData.seqNum);
  BOOST_REQUIRE_EQUAL(ad2.arqExtra, newArqData.arqExtra);
}

// test 2: interface 2 (when I construct as receiver)

BOOST_AUTO_TEST_CASE(interface2) {
  // make an IP packet
  NodeID destNodeID = 2;
  auto const now = std::chrono::system_clock::now();
  std::chrono::system_clock::duration::rep sourceTimeDuration =
      now.time_since_epoch().count();
  auto const srcAddr = addr::from_string("192.168.101.2");
  auto const dstAddr = addr::from_string(
      (boost::format("192.168.10%1%.2") % (int)destNodeID).str());
  auto const srcPort = 4001;
  auto const dstPort = 5001;
  auto const nbytes = 1500;
  auto const ipbuf = make_ippacket(srcAddr, dstAddr, srcPort, dstPort, nbytes);
  auto const arqData = net::ARQIP4PacketSegment::ARQData{
      .arqExtra = 666, .seqNum = 33, .burstNum = 100};

  // concatenate all data into one vector and create segment
  std::vector<uint8_t> segbuf;
  auto stp = (uint8_t *)&sourceTimeDuration;
  segbuf.insert(end(segbuf), stp, stp + sizeof(sourceTimeDuration));
  auto arqp = (uint8_t *)&arqData;
  segbuf.insert(end(segbuf), arqp, arqp + sizeof(arqData));
  segbuf.insert(end(segbuf), begin(ipbuf), end(ipbuf));

  auto const expected_bufsz = nbytes +
                              sizeof(net::ARQIP4PacketSegment::ARQData) +
                              sizeof(std::chrono::nanoseconds::rep);
  BOOST_REQUIRE_EQUAL(segbuf.size(), expected_bufsz);
  auto seg = net::ARQIP4PacketSegment(destNodeID, boost::asio::buffer(segbuf));

  // call all methods
  BOOST_REQUIRE_EQUAL(seg.length(), expected_bufsz);

  BOOST_REQUIRE(seg.type() == dll::SegmentType::ARQIPv4);
  BOOST_REQUIRE(((dll::Segment *)&seg)->type() == dll::SegmentType::ARQIPv4);

  BOOST_REQUIRE_EQUAL(seg.checksummed(), true);
  BOOST_REQUIRE_EQUAL(seg.reliable(), true);
  BOOST_REQUIRE(seg.sourceTime() == now);
  BOOST_REQUIRE(seg.destNodeID() == destNodeID);

  FlowUID expected_flow_uid(dstPort);
  BOOST_REQUIRE(seg.flowUID() == expected_flow_uid);

  std::vector<uint8_t> nothing_vec;
  std::vector<boost::asio::const_buffer> expected_rc = {
      boost::asio::buffer(segbuf), boost::asio::buffer(nothing_vec)};
  auto rc = seg.rawContentsBuffer();
  BOOST_REQUIRE_EQUAL(expected_rc.size(), rc.size());
  for (size_t i = 0; i < rc.size(); ++i) {
    auto p_my = boost::asio::buffer_cast<const void *>(rc[i]);
    auto p_ex = boost::asio::buffer_cast<const void *>(expected_rc[i]);
    auto n = boost::asio::buffer_size(expected_rc[i]);
    BOOST_REQUIRE_EQUAL(0, memcmp(p_my, p_ex, n));
  }

  FlowID expected_flow_id{srcAddr.to_uint(), dstAddr.to_uint(), IPPROTO_UDP,
                          srcPort, dstPort};
  auto flow_id = seg.flowID();
  BOOST_REQUIRE(expected_flow_id == flow_id);

  BOOST_REQUIRE_EQUAL(seg.packetLength(), nbytes);

  auto expected_pc = ipbuf.data();
  auto expected_npc = nbytes;
  auto my_pc = seg.packetContentsBuffer();
  BOOST_REQUIRE_EQUAL(expected_npc, boost::asio::buffer_size(my_pc));
  BOOST_REQUIRE_EQUAL(0, memcmp(boost::asio::buffer_cast<const void *>(my_pc),
                                expected_pc, expected_npc));

  BOOST_REQUIRE_EQUAL(seg.ipHeaderAddress(),
                      (void const *)(segbuf.data() +
                                     sizeof(net::ARQIP4PacketSegment::ARQData) +
                                     sizeof(std::chrono::nanoseconds::rep)));
  BOOST_REQUIRE_EQUAL(seg.protocol(), IPPROTO_UDP);
  auto nnow = std::chrono::system_clock::now();
  BOOST_REQUIRE((nnow - now) == seg.currentDelay(nnow));
  // Intentionallly not testing priority(...) field
  BOOST_REQUIRE_EQUAL(srcAddr, seg.srcAddr());
  BOOST_REQUIRE_EQUAL(dstAddr, seg.dstAddr());
  BOOST_REQUIRE_EQUAL(srcPort, seg.srcPort());
  BOOST_REQUIRE_EQUAL(dstPort, seg.dstPort());

  // setters and getters;
  decltype(srcPort) newSrcPort(6001);
  decltype(dstPort) newDstPort(7001);
  seg.setSrcPort(newSrcPort);
  seg.setDstPort(newDstPort);
  BOOST_REQUIRE_EQUAL(newSrcPort, seg.srcPort());
  BOOST_REQUIRE_EQUAL(newDstPort, seg.dstPort());

  // arq specials
  auto ad = seg.arqData();
  BOOST_REQUIRE_EQUAL(ad.burstNum, arqData.burstNum);
  BOOST_REQUIRE_EQUAL(ad.seqNum, arqData.seqNum);
  BOOST_REQUIRE_EQUAL(ad.arqExtra, arqData.arqExtra);
  decltype(ad) newArqData{.arqExtra = 667, .seqNum = 444, .burstNum = 1};
  seg.setArqData(newArqData);
  auto ad2 = seg.arqData();
  BOOST_REQUIRE_EQUAL(ad2.burstNum, newArqData.burstNum);
  BOOST_REQUIRE_EQUAL(ad2.seqNum, newArqData.seqNum);
  BOOST_REQUIRE_EQUAL(ad2.arqExtra, newArqData.arqExtra);
}

// test 3: frames

BOOST_AUTO_TEST_CASE(frames) {

  struct segment_under_test {
    net::ARQIP4PacketSegment::sptr seg;
    std::vector<uint8_t> bs;
  };

  // make a frame from a few segments
  auto const nseg = 3;
  NodeID const destNodeID = 2;
  NodeID const sourceNodeID = 1;
  auto const now = std::chrono::system_clock::now();
  auto const srcAddr = addr::from_string("192.168.101.2");
  auto const dstAddr = addr::from_string(
      (boost::format("192.168.10%1%.2") % (int)destNodeID).str());
  auto const srcPort = 4001;
  auto const dstPort = 5001;
  auto const nbytes = 300;
  std::vector<segment_under_test> in_sut;
  std::vector<dll::Segment::sptr> segVec;
  for (size_t i = 0; i < nseg; ++i) {
    auto const buf = make_ippacket(srcAddr, dstAddr, srcPort, dstPort, nbytes);
    auto const arqData = net::ARQIP4PacketSegment::ARQData{
        .arqExtra = 666, .seqNum = (uint16_t)i, .burstNum = 100};
    in_sut.push_back({nullptr, buf});
    auto seg = std::make_shared<net::ARQIP4PacketSegment>(
        destNodeID, boost::asio::buffer(in_sut[i].bs), now);
    seg->setArqData(arqData);
    in_sut[i].seg = seg;
    segVec.push_back(seg);
  }

  // make a frame out of these
  using namespace ofdm;
  auto const hmcs = MCS::Name::QPSK_R12_N648;
  auto const pmcs = MCS::Name::QAM16_R56_N1944;
  auto const pss = SeqID::ID::ZIG_128_12_108_12_QAM16;
  auto frame = std::make_shared<DFTSOFDMFrame>(sourceNodeID, destNodeID, segVec,
                                               hmcs, pmcs, pss, 0, 0);

  // write payload to bit buffer
  auto payl_blockNinfo = 1620;
  std::vector<uint8_t> bv(frame->payloadNumInfoBits());
  auto p = bv.data();
  for (size_t i = 0; i < frame->numBlocks(false); ++i) {
    frame->writeBlock(false, i, boost::asio::buffer(p, payl_blockNinfo));
    p += payl_blockNinfo;
  }

  // copy the payload bytes out
  auto in_payl = frame->movePayload();

  // write header bits
  auto hdr_blockNinfo = 324;
  std::vector<uint8_t> bvh(hdr_blockNinfo * frame->numBlocks(true));
  p = bvh.data();
  for (size_t i = 0; i < frame->numBlocks(true); ++i) {
    frame->writeBlock(true, i, boost::asio::buffer(p, hdr_blockNinfo));
    p += hdr_blockNinfo;
  }

  // construct new frame from header bits
  auto newFrame =
      std::make_shared<DFTSOFDMFrame>(hmcs, boost::asio::buffer(bvh));

  // read payload back into frame (see payload deframer)
  boost::asio::mutable_buffer bvb = boost::asio::buffer(bv);
  auto blockK = newFrame->readBlock(0, boost::asio::const_buffer()).first;
  auto block_number = 0;
  while (boost::asio::buffer_size(bvb) >= blockK) {
    auto nread = newFrame->readBlock(block_number, bvb);
    BOOST_REQUIRE(nread.second != 0);
    for (auto bn = block_number; bn < block_number + nread.second; ++bn) {
      BOOST_REQUIRE(newFrame->blockIsValid(bn));
    }
    block_number += nread.second;
    bvb = bvb + nread.first * nread.second;
  }

  // get segments from newFrame
  auto out_segments = newFrame->segments();
  auto out_payl = newFrame->movePayload();

  // FIXME do more checks (individual segments). I am tired
  BOOST_REQUIRE_EQUAL(out_payl->size(), in_payl->size());
  BOOST_REQUIRE_EQUAL(
      0, memcmp(in_payl->data(), out_payl->data(), in_payl->size()));
}
