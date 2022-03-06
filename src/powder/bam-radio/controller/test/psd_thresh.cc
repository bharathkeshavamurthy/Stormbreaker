// PSD Thresholding experiments
//
// Copyright (c) 2019 Dennis Ogbe
//
// This is a standalone program that computes thresholded PSDs from real-valued
// PSD data using a variety of techniques and saves the results in a SQLite
// database for plotting. The input is expected to be generated using the
// function `create-psd-db' from src/ai/dev-environment.lisp. The output can be
// plotted using the thresh-psd-plotter.py tool, found in util/
//

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"
#include <sqlite3.h>
#include <volk/volk.h>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "bandwidth.h" // for radio sample rate
#include "events.h"    // for DBLayout
#include "psd.h"       // for histogram thresholding

typedef uint8_t srn_id;

struct freq_band {
  int64_t lower;
  int64_t upper;
};

// quick and dirty sqlite3 wrapper

void sqlerr(int rc) {
  if (rc != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errstr(rc));
  }
}

class database {
public:
  void exec(std::string const &sql) {
    char *err;
    auto rc = sqlite3_exec(dbptr, sql.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
      throw std::runtime_error(std::string(err));
      sqlite3_free(err);
    }
  }

  void exec(sqlite3_stmt *stmt, std::function<int(sqlite3_stmt *)> bind) {
    auto rc = bind(stmt);
    sqlerr(rc);
    sqlite3_step(stmt);
    rc = sqlite3_reset(stmt);
    sqlerr(rc);
  }

  database(std::string const &filename) {
    auto rc = sqlite3_open(filename.c_str(), &dbptr);
    sqlerr(rc);
  }

  ~database() { sqlite3_close(dbptr); }

  sqlite3 *dbptr;
};

// the signature of a thresholded PSD generator is as follows:
// input:
//   raw_psd: a vector of floats containing the non-dB raw values
//   freq_alloc: a map mapping SRN IDs to a lower/upper pair (see the struct
//   above) rf_bandwidth: a number in Hz of the RF bandwidth at this point in
//   time
//                 (use to get control channel offsets)
//
// output:
//   a vector of bytes containing the thresholded bins.
typedef std::function<std::vector<uint8_t>(
    std::vector<float> const &, std::map<srn_id, freq_band> const &, int64_t)>
    thresholder;

std::vector<float> psd_to_dB(std::vector<float> const &raw_psd) {
  std::vector<float> of(raw_psd.size());
  volk_32f_log2_32f(of.data(), raw_psd.data(), of.size());
  volk_32f_s32f_multiply_32f(of.data(), of.data(), 3.010299956639812,
                             of.size());
  return of;
}

std::vector<uint8_t> intv2uintv(std::vector<int8_t> v) {
  std::vector<uint8_t> o(v.size());
  size_t k = 0;
  std::generate(begin(o), end(o), [&] { return v[k++]; });
  return o;
}

// Histogram-based thresholding
using hist_params = bamradio::psdsensing::PSDSensing::HistParams;

std::vector<uint8_t> hist_thresh(std::vector<float> const &raw_psd,
                                 hist_params hp) {
  // run the algorithm
  auto const psddB = psd_to_dB(raw_psd);
  auto const psdThresh =
      bamradio::psdsensing::PSDSensing::thresholdPSD(psddB, hp);
  return intv2uintv(psdThresh);
}

// using libvolk's noise floor estimation
std::vector<uint8_t> volk_thresh(std::vector<float> const &raw_psd,
                                 float spectralExclusionValue,
                                 float thresh_margin_dB) {
  float noise_floor_dB;
  auto const psddB = psd_to_dB(raw_psd);
  volk_32f_s32f_calc_spectral_noise_floor_32f(
      &noise_floor_dB, psddB.data(), spectralExclusionValue, raw_psd.size());
  // std::cout << "volk_thresh: noise_floor = " << noise_floor_dB << " dB."
  //           << std::endl;
  std::vector<uint8_t> o;
  o.reserve(psddB.size());
  for (auto const &bin : psddB) {
    o.push_back(bin > (noise_floor_dB + thresh_margin_dB) ? 1 : 0);
  }
  return o;
}

// estimate the noise floor using libvolk. use this as input to the
// histogram-based thresholder. use empirically determined hist_params
std::vector<uint8_t>
est_noise_floor_hist_thresh(std::vector<float> const &raw_psd) {
  float const spectralExclusionValue = 20.0f;
  float noise_floor_dB;
  auto const psddB = psd_to_dB(raw_psd);
  volk_32f_s32f_calc_spectral_noise_floor_32f(
      &noise_floor_dB, psddB.data(), spectralExclusionValue, raw_psd.size());
  hist_params hp{.bin_size = 0.1,
                 .empty_bin_thresh = 2,
                 .sn_gap_bins = 30,
                 .avg_len = 5,
                 .noise_floor = noise_floor_dB};
  auto const psdThresh =
      bamradio::psdsensing::PSDSensing::thresholdPSD(psddB, hp);
  return intv2uintv(psdThresh);
}

// run a thresholding function but only after the known freq_bands are removed
template <typename threshfun>
std::vector<uint8_t>
thresh_remove_known(std::vector<float> const &raw_data,
                    std::map<srn_id, freq_band> const &bands,
                    int64_t rf_bandwidth, threshfun tfun) {
  // convert offsett freq to bin index
  auto const ofst2bin = [](int64_t ofst, int64_t nbin) {
    auto const clip = [](auto num, auto lo, auto hi) {
      if (num < lo) {
        return lo;
      } else if (num > hi) {
        return hi;
      } else {
        return num;
      }
    };
    auto const in_range = [](auto num, auto lo, auto hi) {
      if (num < lo) {
        return false;
      } else if (num > hi) {
        return false;
      } else {
        return true;
      }
    };
    auto full = bam::dsp::sample_rate;
    auto half = full / 2;
    if (!in_range((double)ofst, -half, half)) {
      throw std::runtime_error("ofst not in range");
    }
    return (size_t)clip(floor((double)nbin * (((double)ofst / full) + 0.5)),
                        0.0, (double)nbin - 1);
  };

  // get all bins occupied by my transmisstions
  std::vector<size_t> occ_bins;
  for (auto const &b : bands) {
    // N.B overestimating a little to cover all bins
    auto const lower_bin = ofst2bin(b.second.lower, raw_data.size()) - 2;
    auto const upper_bin = ofst2bin(b.second.upper, raw_data.size()) + 2;
    for (auto i = lower_bin; i <= upper_bin; ++i) {
      occ_bins.push_back(i);
    }
  }

  // add the control channel bins as well
  auto const cc_bw = 480e3;
  auto const cc_edge_offset = 380e3;
  for (auto const cfreq : {rf_bandwidth / 2 - cc_edge_offset,
                           -1 * (rf_bandwidth / 2 - cc_edge_offset)}) {
    auto const lower_bin = ofst2bin(cfreq - cc_bw / 2, raw_data.size()) - 2;
    auto const upper_bin = ofst2bin(cfreq + cc_bw / 2, raw_data.size()) + 2;
    for (auto i = lower_bin; i <= upper_bin; ++i) {
      occ_bins.push_back(i);
    }
  }

  // compute all bins NOT occupied by me, use only those in the thresholding
  std::sort(begin(occ_bins), end(occ_bins));
  std::vector<size_t> all_bins(raw_data.size());
  std::iota(begin(all_bins), end(all_bins), 0);
  std::vector<size_t> free_bins;
  std::set_difference(cbegin(all_bins), cend(all_bins), cbegin(occ_bins),
                      cend(occ_bins), std::back_inserter(free_bins));

  // prepare the data to send to the thresholder
  std::map<size_t, size_t> reverse_map;
  std::vector<float> in_vector;
  in_vector.reserve(free_bins.size());
  std::size_t k = 0;
  for (auto const &bin : free_bins) {
    in_vector.push_back(raw_data[bin]);
    reverse_map.emplace(k++, bin);
  }

  // run the thresholding algorithm and construct the output
  auto const thresh_data = tfun(in_vector);
  // set to two for known transmissions
  std::vector<uint8_t> out_vector(raw_data.size(), 2);
  for (size_t i = 0; i < thresh_data.size(); ++i) {
    out_vector[reverse_map[i]] = thresh_data[i];
  }

  return out_vector;
}

// TODO: Libvolk but with known allocation removed

// TODO: very simple hard thresholding.

// TODO: (Tomo?) estimate noise floor using always-free band edges, use this
// value to set threshold

// helper functions

void get_next_raw_psd(sqlite3_stmt *st, int &rc, std::vector<float> &psd,
                      srn_id &id, int64_t &step_id) {
  rc = sqlite3_step(st);
  if (rc == SQLITE_DONE) {
    return;
  }

  // col 0: json data, col 1: srn id, col 2: step id

  // grab the JSON array and get a float vector from it
  auto json_sz = sqlite3_column_bytes(st, 0);
  auto json_start = (char *)sqlite3_column_text(st, 0);
  std::string json_str(json_start, json_sz);
  auto json = nlohmann::json::parse(json_str);
  std::vector<float> p = json;
  psd = p;

  // grab the other values and set them
  auto t_srn_id = sqlite3_column_int64(st, 1);
  id = (srn_id)t_srn_id;
  auto t_step_id = sqlite3_column_int64(st, 2);
  step_id = t_step_id;
}

int64_t get_rf_bandwidth(sqlite3_stmt *st, int64_t step_id) {
  sqlerr(sqlite3_bind_int64(st, 1, step_id));
  sqlite3_step(st);
  int64_t t_rfb = sqlite3_column_int64(st, 0);
  sqlerr(sqlite3_reset(st));
  return t_rfb;
}

std::map<srn_id, freq_band> get_freq_alloc(sqlite3_stmt *st, int64_t step_id) {
  sqlerr(sqlite3_bind_int64(st, 1, step_id));
  std::map<srn_id, freq_band> o;
  while (sqlite3_step(st) != SQLITE_DONE) {
    // col 0: lower, col 1: upper, col 2: srn_id
    auto const lower = sqlite3_column_int64(st, 0);
    auto const upper = sqlite3_column_int64(st, 1);
    auto const id = sqlite3_column_int64(st, 2);
    o.emplace((srn_id)id, freq_band{.lower = lower, .upper = upper});
  }
  sqlerr(sqlite3_reset(st));
  return o;
}

int main(int argc, char const *argv[]) {
  //
  // parse options
  //
  namespace po = boost::program_options;
  std::string infile, outfile;
  po::options_description desc("options");
  // clang-format off
  desc.add_options()(
      "help", "help message")(
      "in", po::value<decltype(infile)>(&infile)->required(), "Input SQLite")(
      "out", po::value<decltype(outfile)>(&outfile)->required(),"Output SQLite");
  // clang-format on
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
  // hold all thresholders in a map. the key is the name of the table in the
  // output database.
  //
  std::map<std::string, thresholder> const thresholders = {
      {"psd_db", // convert raw to dB for viz
       [](auto const &psd, auto const &bands, auto rfb) {
         auto const of = psd_to_dB(psd);
         std::vector<uint8_t> o(of.size() * sizeof(*(of.data())));
         std::memcpy(o.data(), of.data(), o.size());
         return o;
       }},
      {"hist_thresh_default", // histogram with default params
       [](auto const &psd, auto const &bands, auto rfb) {
         hist_params hp{.bin_size = 0.2,
                        .empty_bin_thresh = 2,
                        .sn_gap_bins = 30,
                        .avg_len = 5,
                        .noise_floor = -70};
         return hist_thresh(psd, hp);
       }},
      {"hist_thresh_sensitive", // histogram with custom params
       [](auto const &psd, auto const &bands, auto rfb) {
         hist_params hp{.bin_size = 0.1,
                        .empty_bin_thresh = 2,
                        .sn_gap_bins = 30,
                        .avg_len = 5,
                        .noise_floor = -30};
         return hist_thresh(psd, hp);
       }},
      {"hist_no_known", // histogram with custom params + known band removal
       [](auto const &psd, auto const &bands, auto rfb) {
         return thresh_remove_known(psd, bands, rfb, [](auto const &data) {
           return hist_thresh(data, hist_params{.bin_size = 0.1,
                                                .empty_bin_thresh = 2,
                                                .sn_gap_bins = 30,
                                                .avg_len = 5,
                                                .noise_floor = -30});
         });
       }},
      {"hybrid_empirical",
       [](auto const &psd, auto const &bands, auto rfb) {
         return thresh_remove_known(psd, bands, rfb, [](auto const &data) {
           return est_noise_floor_hist_thresh(data);
         });
       }},
      {"volk_thresh_20_10", // libvolk with given exclusion zone and margin
       [](auto const &psd, auto const &bands, auto rfb) {
         return volk_thresh(psd, 20, 10);
       }},
      {"volk_thresh_20_10_no_known",
       [](auto const &psd, auto const &bands, auto rfb) {
         return thresh_remove_known(psd, bands, rfb, [](auto const &data) {
           return volk_thresh(data, 20, 10);
         });
       }}};

  //
  // create the output database and add tables for all the thresholders to it.
  //
  namespace fs = boost::filesystem;
  fs::path opath(outfile);
  if (fs::exists(opath) && fs::is_regular_file(opath)) {
    fs::path newpath(opath);
    newpath += ".old";
    std::cout << "WARNING: found existing file at " << outfile
              << ". Renaming to " << newpath << "." << std::endl;
    fs::rename(opath, newpath);
  }
  std::cout << "Creating output file " << outfile << "...";
  database out_db(outfile);
  std::cout << "Done." << std::endl;
  // keep a map of insert statements
  std::map<std::string, sqlite3_stmt *> o_stmt;
  for (auto const &st : thresholders) {
    auto const table_name = st.first;
    auto layout = bamradio::DBLayout(table_name)
                      .addColumn("data", bamradio::DBLayout::Type::BLOB)
                      .addColumn("srn_id", bamradio::DBLayout::Type::INT)
                      .addColumn("step_id", bamradio::DBLayout::Type::INT);
    out_db.exec(layout.sql());
    sqlite3_stmt *stmt;
    sqlerr(layout.prepare(out_db.dbptr, &stmt, table_name));
    o_stmt.emplace(table_name, stmt);
  }

  //
  // go through the input data base, apply the thresholders to all of the input
  // data, write the result to the output database
  //
  std::cout << "Opening input file " << infile << "...";
  database in_db(infile);
  std::cout << "Done." << std::endl;

  // get all of the statements
  std::cout << "Preparing read statements";
  std::stringstream ss;
  sqlite3_stmt *psd_stmt, *env_stmt, *freq_stmt;
  ss.str("");
  ss.clear();
  ss << "SELECT json, srn_id, step_id from raw_psd";
  sqlerr(sqlite3_prepare_v2(in_db.dbptr, ss.str().c_str(), -1, &psd_stmt,
                            nullptr));
  std::cout << ".";
  ss.str("");
  ss.clear();
  ss << "SELECT rf_bandwidth from env WHERE step_id = ?";
  sqlerr(sqlite3_prepare_v2(in_db.dbptr, ss.str().c_str(), -1, &env_stmt,
                            nullptr));
  std::cout << ".";
  ss.str("");
  ss.clear();
  ss << "SELECT lower, upper, srn_id from freq_alloc WHERE step_id = ?";
  sqlerr(sqlite3_prepare_v2(in_db.dbptr, ss.str().c_str(), -1, &freq_stmt,
                            nullptr));
  std::cout << ".Done." << std::endl;

  // go through the data base and compute outputs
  std::cout << "Extracting and processing data...";
  auto const t_start = std::chrono::system_clock::now();
  int rc = SQLITE_OK;
  int64_t step_id;
  srn_id id;
  std::vector<float> psd;
  while (rc != SQLITE_DONE) {
    // pull out a PSD
    get_next_raw_psd(psd_stmt, rc, psd, id, step_id);
    if (rc == SQLITE_DONE) {
      break;
    }
    auto const rfb = get_rf_bandwidth(env_stmt, step_id);
    auto const freq_alloc = get_freq_alloc(freq_stmt, step_id);
    // process using all thresholders
    for (auto const &st : thresholders) {
      auto const table_name = st.first;
      auto const thresholder = st.second;
      auto const out_data = thresholder(psd, freq_alloc, rfb);
      // insert into out database
      out_db.exec(o_stmt[table_name], [&](auto stmt) {
        int n = 1;
        int rc;
        rc = sqlite3_bind_blob64(stmt, n++, (void *)out_data.data(),
                                 sizeof(*(out_data.data())) * out_data.size(),
                                 SQLITE_TRANSIENT);
        sqlerr(rc);
        rc = sqlite3_bind_int64(stmt, n++, id);
        sqlerr(rc);
        rc = sqlite3_bind_int64(stmt, n++, step_id);
        sqlerr(rc);
        return rc;
      });
    }
  }
  auto const t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - t_start);
  std::cout << "Finished after " << t_msec.count() << "ms." << std::endl;

  //
  // clean up all output and input statements
  //
  std::cout << "Cleaning up...";
  for (auto const &os : o_stmt) {
    sqlite3_finalize(os.second);
  }
  sqlite3_finalize(psd_stmt);
  sqlite3_finalize(env_stmt);
  sqlite3_finalize(freq_stmt);
  std::cout << "Done. Check output file at " << outfile << "." << std::endl;
}
