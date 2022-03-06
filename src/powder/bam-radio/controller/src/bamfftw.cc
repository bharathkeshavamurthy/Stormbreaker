#include "bamfftw.h"
#include "events.h"

// multi-threaded FFTW planning
#ifdef __has_include
#if __has_include(<gnuradio/fft/fft.h>)
#include <gnuradio/fft/fft.h>
#define GUARD_FFTW_PLAN
#endif
#endif

namespace bamradio {
namespace fft {

//
// CPU FFT Interface (FFTW)
//
CPUFFT::CPUFFT(std::vector<size_t> const &sizes, bool forward, int nthreads,
               std::string const &wisdom_filename)
    : _forward(forward) {
#ifdef GUARD_FFTW_PLAN
  gr::fft::planner::scoped_lock lock(gr::fft::planner::mutex());
#endif
  if (fftwf_init_threads() == 0) {
    throw std::runtime_error("fftw_init_threads failed");
  }

  int r = fftwf_import_wisdom_from_filename(wisdom_filename.c_str());
  if (!r) {
    log::text("Precomputed FFTW wisdom was not loaded.", __FILE__, __LINE__);
  }

  fftwf_plan_with_nthreads(nthreads);

  auto maxfftlen = *std::max_element(begin(sizes), end(sizes));
  auto itemp = fftwf_alloc_complex(maxfftlen);
  auto otemp = fftwf_alloc_complex(maxfftlen);

#ifdef NDEBUG // do less exhaustive planning when debugging.
  auto plan_type = FFTW_PATIENT;
#else
  auto plan_type = FFTW_MEASURE;
#endif

  for (auto const &size : sizes) {
    if (_plans.count(size) > 0)
      continue;
    _plans[size] = fftwf_plan_dft_1d(
        size, itemp, otemp, forward ? FFTW_FORWARD : FFTW_BACKWARD, plan_type);
    if (_plans[size] == nullptr) {
      throw std::runtime_error("fftwf_plan_dft_1d failed");
    }
  }
  fftwf_free(itemp);
  fftwf_free(otemp);
  fftwf_export_wisdom_to_filename(wisdom_filename.c_str());
}

CPUFFT::~CPUFFT() {
#ifdef GUARD_FFTW_PLAN
  gr::fft::planner::scoped_lock lock(gr::fft::planner::mutex());
#endif
  for (auto const &p : _plans) {
    fftwf_destroy_plan(p.second);
  }
}

void CPUFFT::execute(size_t size, fcomplex *in, fcomplex *out) const {
  fftwf_execute_dft(_plans.at(size), (fftwf_complex *)in, (fftwf_complex *)out);
}

std::vector<size_t> CPUFFT::sizes() const {
  decltype(sizes()) o;
  for (auto const &p : _plans) {
    o.push_back(p.first);
  }
  std::sort(begin(o), end(o));
  return o;
}

bool CPUFFT::forward() const { return _forward; }
} // namespace fft
} // namespace bamradio
