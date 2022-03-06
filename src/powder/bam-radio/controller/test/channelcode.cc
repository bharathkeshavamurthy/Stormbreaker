// Channel Code. Run the LDPC decoder in Gaussian Noise and compute bit and
// block error rates.
//
// Copyright (c) 2018 Dennis Ogbe

#include "bam_constellation.h"
#include "fcomplex.h"
#include "mcs.h"

#include "ldpc/yaldpc.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string.h>
#include <thread>

#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "json.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace bamradio;
using MCS = bamradio::ofdm::MCS;
using fcomplex = bamradio::fcomplex;

struct sim_output {
  MCS::Name mcs;
  std::vector<double> ber;
  std::vector<double> bler;
};

void to_json(nlohmann::json &j, sim_output so) {
  auto mod2str = [](auto const &mod) -> const char * {
    switch (mod) {
    case Modulation::BPSK:
      return "bpsk";
    case Modulation::QPSK:
      return "qpsk";
    case Modulation::QAM16:
      return "qam16";
    case Modulation::QAM32:
      return "qam32";
    case Modulation::QAM64:
      return "qam64";
    case Modulation::QAM128:
      return "qam128";
    case Modulation::QAM256:
      return "qam256";
    }
  };
  auto mcs = MCS::table[so.mcs];
  j["mod"] = mod2str(mcs.modulation);
  j["rate"] =
      (boost::format("%1%/%2%") % mcs.codeRate.k % mcs.codeRate.n).str();
  j["blocklength"] = mcs.blockLength;
  j["mcs_index"] = (int)(so.mcs);
  j["ber"] = so.ber;
  j["bler"] = so.bler;
}

int main(int argc, char const *argv[]) {

  // options
  size_t its;
  size_t nthreads;
  size_t max_snr;
  uint64_t ldpc_max_its;
  std::string dectype;
  std::string outname;
  uint64_t rng_seed;

  po::options_description desc("Options");
  desc.add_options()(
      //
      "help", "help message")(
      //
      "its", po::value<decltype(its)>(&its)->required()->default_value(10000),
      "Monte-Carlo iterations")(
      //
      "nthreads",
      po::value<decltype(nthreads)>(&nthreads)->required()->default_value(4),
      "Number of threads")(
      //
      "max-snr",
      po::value<decltype(max_snr)>(&max_snr)->required()->default_value(40),
      "Maximum SNR [dB] (starts from 0)")(
      //
      "dectype",
      po::value<decltype(dectype)>(&dectype)->required()->default_value(
          "serial-c-min-sum"),
      "Decoder type")(
      //
      "outname",
      po::value<decltype(outname)>(&outname)->required()->default_value(
          "bersim"),
      "Output filename prefix")(
      //
      "ldpc-max-its",
      po::value<decltype(ldpc_max_its)>(&ldpc_max_its)
          ->required()
          ->default_value(30),
      "LDPC max.iteratations")(
      //
      "rng-seed",
      po::value<decltype(rng_seed)>(&rng_seed)->required()->default_value(
          std::chrono::system_clock::now().time_since_epoch().count()),
      "RNG seed");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (std::exception &ex) {
    if (!vm.count("help")) {
      std::cout << "ERROR: " << ex.what() << "\n" << std::endl;
    }
    // help message
    std::cout << desc << std::endl;
    return EXIT_FAILURE;
  }
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // validate decoder type
  auto validate_decoder = [&](auto &dectype) {
    if (dectype == "sum-product" || dectype == "min-sum" ||
        dectype == "min-star" || dectype == "serial-c-min-sum") {
      return true;
    } else {
      throw std::runtime_error("Need to select proper decoder type.");
    }
  };
  validate_decoder(dectype);

// define the MCS indices we want to study
#if 0
  // all mcs we theoretically have
  std::vector<MCS::Name> all_mcs = {
      MCS::Name::QPSK_R12_N648,    MCS::Name::QPSK_R23_N648,
      MCS::Name::QPSK_R34_N648,    MCS::Name::QPSK_R56_N648,
      MCS::Name::QAM16_R12_N648,   MCS::Name::QAM16_R23_N648,
      MCS::Name::QAM16_R34_N648,   MCS::Name::QAM16_R56_N648,
      MCS::Name::BPSK_R12_N1296,   MCS::Name::BPSK_R23_N1296,
      MCS::Name::BPSK_R34_N1296,   MCS::Name::BPSK_R56_N1296,
      MCS::Name::QPSK_R12_N1296,   MCS::Name::QPSK_R23_N1296,
      MCS::Name::QPSK_R34_N1296,   MCS::Name::QPSK_R56_N1296,
      MCS::Name::QAM16_R12_N1296,  MCS::Name::QAM16_R23_N1296,
      MCS::Name::QAM16_R34_N1296,  MCS::Name::QAM16_R56_N1296,
      MCS::Name::BPSK_R12_N1944,   MCS::Name::BPSK_R23_N1944,
      MCS::Name::BPSK_R34_N1944,   MCS::Name::BPSK_R56_N1944,
      MCS::Name::QPSK_R12_N1944,   MCS::Name::QPSK_R23_N1944,
      MCS::Name::QPSK_R34_N1944,   MCS::Name::QPSK_R56_N1944,
      MCS::Name::QAM16_R12_N1944,  MCS::Name::QAM16_R23_N1944,
      MCS::Name::QAM16_R34_N1944,  MCS::Name::QAM16_R56_N1944,
      MCS::Name::QAM32_R12_N1944,  MCS::Name::QAM32_R23_N1944,
      MCS::Name::QAM32_R34_N1944,  MCS::Name::QAM32_R56_N1944,
      MCS::Name::QAM64_R12_N1944,  MCS::Name::QAM64_R23_N1944,
      MCS::Name::QAM64_R34_N1944,  MCS::Name::QAM64_R56_N1944,
      MCS::Name::QAM128_R12_N1944, MCS::Name::QAM128_R23_N1944,
      MCS::Name::QAM128_R34_N1944, MCS::Name::QAM128_R56_N1944,
      MCS::Name::QAM256_R12_N1944, MCS::Name::QAM256_R23_N1944,
      MCS::Name::QAM256_R34_N1944, MCS::Name::QAM256_R56_N1944,
  };
#else
  // these are the actually interesting MCS's
  std::vector<MCS::Name> all_mcs = {
      MCS::Name::QPSK_R12_N1944,  MCS::Name::QPSK_R23_N1944,
      MCS::Name::QPSK_R34_N1944,  MCS::Name::QPSK_R56_N1944,
      MCS::Name::QAM16_R12_N1944, MCS::Name::QAM16_R23_N1944,
      MCS::Name::QAM16_R34_N1944, MCS::Name::QAM16_R56_N1944,
      MCS::Name::QAM32_R12_N1944, MCS::Name::QAM32_R23_N1944,
      MCS::Name::QAM32_R34_N1944, MCS::Name::QAM32_R56_N1944,
      MCS::Name::QAM64_R12_N1944, MCS::Name::QAM64_R23_N1944,
      MCS::Name::QAM64_R34_N1944, MCS::Name::QAM64_R56_N1944,
  };
#endif
  auto const simsize = all_mcs.size();

  // we simulate from 0 through 40 dB SNR
  std::vector<double> const snrs = [&] {
    std::vector<double> o;
    o.resize(max_snr + 1);
    int k = 0;
    std::generate_n(begin(o), o.size(), [&k] { return (double)k++; });
    return o;
  }();

  // get some temporary paths and helpers
  auto const tmpdir = [] {
    auto p = fs::temp_directory_path();
    p += "/";
    p += fs::unique_path();
    return p;
  }();
  auto mkpath = [](auto base, auto p) {
    base += p;
    return base;
  };
  fs::create_directories(tmpdir);

  // work dispatcher
  boost::asio::io_context io;

  // output
  auto time2str = [](auto t) {
    auto tt = std::chrono::system_clock::to_time_t(t);
    std::string ret(std::ctime(&tt));
    ret.pop_back();
    return ret;
  };
  auto start = std::chrono::system_clock::now();
  std::cout << "Starting work at " << time2str(start) << "..." << std::endl;

  // helper lambdas part 1
  auto construct_decoder =
      [&](auto &dt, auto const &code) -> yaldpc::Decoder<float, uint8_t>::sptr {
    using llr = float;
    using out = uint8_t;
    if (dt == "sum-product") {
      return yaldpc::SumProductDecoder<llr, out>::make(code, ldpc_max_its, true,
                                                       true);
    } else if (dt == "min-sum") {
      return yaldpc::MinSumDecoder<llr, out>::make(code, ldpc_max_its, true,
                                                   true);
    } else if (dt == "min-star") {
      return yaldpc::MinStarDecoder<llr, out>::make(code, ldpc_max_its, true,
                                                    true);
    } else if (dt == "serial-c-min-sum") {
      return yaldpc::SerialCMinSumDecoder<llr, out>::make(code, ldpc_max_its,
                                                          true, true);
    } else {
      throw std::runtime_error("Need to select proper decoder type.");
    }
  };
  auto make_decoder = [&](auto const &code) {
    return construct_decoder(dectype, code);
  };
  auto make_encoder = [&](auto const &code) {
    // we always use QC LDPC codes, so use this encoder.
    return yaldpc::DDEncoder::make(code);
  };
  auto make_constellation = [&](auto mcs, auto snr) {
    // set a0, a1, a2 to one to make this easier on us. choose nbits from
    // hardcoded in ofdm.cc
    auto const nbits = 6;
    // need a temp path to put wisdom files, delete them right away
    auto tmp_wisdom = mkpath(tmpdir, fs::unique_path());
    // construct a constellation with max_snr = snr and one point for the
    // lut. then set_snr to max_snr+1 to ensure the correct lut lookup table.
    auto constellation = [&]() -> constellation::base<float>::sptr {
      using llr = float;
      switch (mcs.modulation) {
      case Modulation::BPSK:
        return constellation::bpsk<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                              snr, tmp_wisdom.native());
      case Modulation::QPSK:
        return constellation::qpsk<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                              snr, tmp_wisdom.native());
      case Modulation::QAM16:
        return constellation::qam16<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                               snr, tmp_wisdom.native());
      case Modulation::QAM32:
        return constellation::qam32<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                               snr, tmp_wisdom.native());
      case Modulation::QAM64:
        return constellation::qam64<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                               snr, tmp_wisdom.native());
      case Modulation::QAM128:
        return constellation::qam128<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                                snr, tmp_wisdom.native());
      case Modulation::QAM256:
        return constellation::qam256<llr>::make(1.0, 1.0, 1.0, nbits, 1, -0.1,
                                                snr, tmp_wisdom.native());
      }
    }();
    // remove temp wisdom file and return the fresh constellation
    if (fs::exists(tmp_wisdom)) {
      fs::remove(tmp_wisdom);
    }
    if (constellation->bits_per_symbol() != mcs.bitsPerSymbol()) {
      throw std::runtime_error("Problem creating constellation");
    }
    return constellation;
  };

  // launch all work
  std::vector<sim_output> results(simsize);
  for (size_t mcs_idx = 0; mcs_idx < simsize; ++mcs_idx) {
    boost::asio::dispatch(io, [&its, &results, &snrs, &make_decoder,
                               &make_encoder, &make_constellation, &all_mcs,
                               &rng_seed, mcs_idx] {
      // pull out the MCS we are dealing with
      auto mcs = MCS::table[all_mcs[mcs_idx]];

      // create a decoder and encoder
      auto code = mcs.toIEEE802Code();
      auto ecode = yaldpc::expand(code);
      auto enc = make_encoder(code);
      auto dec = make_decoder(ecode);

      // create some random number generators
      std::mt19937 rng1(rng_seed);
      std::mt19937 rng2(rng_seed);

      // create some scratch space and output vectors
      std::vector<uint8_t> raw_bits;
      std::vector<uint8_t> coded_bits;
      std::vector<fcomplex> symbols;
      std::vector<fcomplex> noisy_symbols;
      std::vector<float> llrs;
      std::vector<uint8_t> decoded_bits;
      auto const ninfo = enc->k();
      auto const ncoded = enc->n();
      auto const nsymbols = (decltype(ncoded))std::ceil(
          (float)ncoded / (float)mcs.bitsPerSymbol());
      auto const nllr = nsymbols * mcs.bitsPerSymbol();
      raw_bits.resize(ninfo);
      coded_bits.resize(nllr);
      symbols.resize(nsymbols);
      noisy_symbols.resize(nsymbols);
      llrs.resize(nllr);
      decoded_bits.resize(ninfo);

      std::vector<double> ber;
      std::vector<double> bler;
      ber.reserve(snrs.size());
      bler.reserve(snrs.size());

      // helper lambdas part 2
      auto clear_scratch = [&] {
        auto zerofill = [](auto &vec) { std::fill(begin(vec), end(vec), 0); };
        zerofill(coded_bits);
        zerofill(symbols);
        zerofill(noisy_symbols);
        zerofill(llrs);
        zerofill(decoded_bits);
      };
      auto make_info_bits = [&] {
        std::uniform_int_distribution<uint8_t> dist(0, 1);
        std::generate(begin(raw_bits), end(raw_bits),
                      [&] { return dist(rng1); });
      };
      auto encode = [&] { enc->encode(raw_bits.data(), coded_bits.data()); };
      auto mod = [&](auto constellation) {
        auto const bps = constellation->bits_per_symbol();
        auto bit = coded_bits.data();
        auto op = symbols.data();
        for (size_t i = 0; i < nsymbols; ++i) {
          unsigned int value = 0;
          for (int j = 0; j < bps; ++j) {
            value |= (*bit++ << (bps - 1 - j));
          }
          constellation->map_to_points_and_scale(value, op);
          ++op;
        }
      };
      auto add_noise = [&](auto snr) {
        auto variance = std::pow(10.0, (-1.0) * snr / 10.0);
        std::normal_distribution<float> dist(0.0, sqrt(variance));
        size_t k = 0;
        std::generate(begin(noisy_symbols), end(noisy_symbols), [&] {
          return symbols[k++] + fcomplex(dist(rng2), dist(rng2));
        });
      };
      auto demod = [&](auto constellation, auto snr) {
        auto const bps = constellation->bits_per_symbol();
        auto const sidx = constellation->get_snr_idx(snr + 1.0);
        auto op = llrs.data();
        for (size_t i = 0; i < nsymbols; ++i) {
          constellation->make_soft_decision(noisy_symbols[i], op, sidx);
          op += bps;
        }
      };
      auto decode = [&] { dec->decode(llrs.data(), decoded_bits.data()); };
      auto count_bit_errors = [&] {
        uint64_t nerr = 0;
        for (size_t i = 0; i < raw_bits.size(); ++i) {
          if (raw_bits[i] != decoded_bits[i]) {
            nerr++;
          }
        }
        return nerr;
      };

      // run the simulation for all SNR points
      for (auto const &snr : snrs) {
        auto constellation = make_constellation(mcs, snr);
        uint64_t nbits = 0;
        uint64_t bit_errors = 0;
        uint64_t nblocks = 0;
        uint64_t block_errors = 0;
        for (size_t i = 0; i < its; ++i) {
          clear_scratch();
          make_info_bits();
          encode();
          mod(constellation);
          add_noise(snr);
          demod(constellation, snr);
          decode();
          auto berr = count_bit_errors();
          if (berr > 0) {
            block_errors++;
          }
          bit_errors += berr;
          nbits += dec->k();
          nblocks++;
        }
        ber.push_back(((double)bit_errors) / ((double)nbits));
        bler.push_back(((double)block_errors) / ((double)nblocks));
      }

      // save results
      results[mcs_idx] = {.mcs = all_mcs[mcs_idx], .ber = ber, .bler = bler};
    });
  }

  // wait until all work is done by the threads
  auto nt = std::min(nthreads, simsize);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < nt - 1; ++i) {
    threads.emplace_back([&io] { io.run(); });
  }
  io.run();
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // write down runtime
  auto end = std::chrono::system_clock::now();
  std::cout << "Completed Simulation at " << time2str(end) << "." << std::endl;

  // convert output to JSON and save
  nlohmann::json out;
  nlohmann::json jesults;
  for (auto &result : results) {
    nlohmann::json jesult = result;
    jesults.push_back(jesult);
  }
  out["results"] = jesults;
  out["snrs"] = snrs;
  out["runtime_seconds"] =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  auto oname = boost::format("%1%_%2%_%3%.json") % outname % dectype %
               end.time_since_epoch().count();
  std::ofstream ofile(oname.str());
  ofile << out;
  ofile.close();
  std::cout << "Wrote output data to " << oname << "." << std::endl;

  // remove temp files
  if (fs::exists(tmpdir)) {
    fs::remove_all(tmpdir);
  }

  return 0;
}
