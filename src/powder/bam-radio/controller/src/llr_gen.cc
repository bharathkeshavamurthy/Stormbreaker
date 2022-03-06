// -*- c++ -*-
//
// Copyright (c) 2017 Dennis Ogbe
//
// llr_gen.cc
// A small helper program that generates LLR wisdom for various constellations.
//

//
// Note: This program generates LLR lookup tables for 32-bit floating point
// numbers (C++ type 'float'). The bam_constellation class could be instantiated
// as with fixed-point numbers, but you're gonna have to re-compile this tool
// for fixed numbers if you want to use it.
//

#include "bam_constellation.h"
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  //
  // Welcome message
  //
  std::cout << "\nBAM! Radio LLR Generator\n" << std::endl;

  //
  // lookup table generation parameters
  //
  int precision;
  int snr_npoints;
  float snr_min;
  float snr_max;
  int mod_order;

  //
  // process options
  //
  po::options_description desc("Options");
  desc.add_options()(
      // option
      "help", "help message")(
      // option
      "precision", po::value<int>(&precision)->required(),
      "Precision of the LLR lookup table [bits]")(
      // option
      "npoints", po::value<int>(&snr_npoints)->required(),
      "Number of SNR points to calculate (== number of lookup tables to "
      "generate)")(
      // option
      "min", po::value<float>(&snr_min)->required(), "Minimum SNR [dB]")(
      // option
      "max", po::value<float>(&snr_max)->required(), "Maximum SNR [dB]")(
      // option
      "order", po::value<int>(&mod_order)->required(),
      "Order of the constellation. (1-BPSK, 2-QPSK, 4-16QAM, etc..)");
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

  //
  // make a constellation object, this will write the llr wisdom
  //
  bamradio::constellation::base<float>::sptr constellation;
  switch (mod_order) {
  case 1:
    constellation = bamradio::constellation::bpsk<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "bpsk" % precision %
         snr_min % snr_max % snr_npoints)
            .str());
    break;
  case 2:
    constellation = bamradio::constellation::qpsk<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "qpsk" % precision %
         snr_min % snr_max % snr_npoints)
            .str());
    break;
  case 4:
  case 16:
    constellation = bamradio::constellation::qam16<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "qam16" % precision %
         snr_min % snr_max % snr_npoints)
            .str());
    break;
  case 5:
  case 32:
    constellation = bamradio::constellation::qam32<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "qam32" % precision %
         snr_min % snr_max % snr_npoints)
            .str());
    break;
  case 6:
  case 64:
    constellation = bamradio::constellation::qam64<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "qam64" % precision %
         snr_min % snr_max % snr_npoints)
            .str());
    break;
  case 7:
  case 128:
    constellation = bamradio::constellation::qam128<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "qam128" %
         precision % snr_min % snr_max % snr_npoints)
            .str());
    break;
  case 8:
  case 256:
    constellation = bamradio::constellation::qam256<float>::make(
        1.0, 1.0, 1.0, precision, snr_npoints, snr_min, snr_max,
        (boost::format("%s_%dbit_%.0fto%.0f_%dpts.llrw") % "qam256" %
         precision % snr_min % snr_max % snr_npoints)
            .str());
    break;
  default:
    std::cout << "Valid modulation orders:\n\n"
              << "1 - BPSK\n"
              << "2 - QPSK\n"
              << "4 - 16-QAM\n"
              << "5 - 32-QAM\n"
              << "6 - 64-QAM\n"
              << "7 - 128-QAM\n"
              << "8 - 256-QAM\n"
              << std::endl;
    return EXIT_FAILURE;
  }

  // make sure this does not get compiled out...
  gr_complex p;
  constellation->map_to_points_and_scale(0, &p);
  return EXIT_SUCCESS;
}
