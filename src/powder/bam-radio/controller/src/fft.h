// -*- c++ -*-
//
// FFTs.
//
// Copyright (c) 2018 Dennis ogbe

#ifndef ecfb5eaadb7a7368f648
#define ecfb5eaadb7a7368f648

#include "fcomplex.h"

#include <complex>
#include <memory>
#include <vector>

namespace bamradio {
namespace fft {

//
// common FFT interface.
//
class FFT {
public:
  typedef std::shared_ptr<FFT> sptr;
  // which: which size FFT to execute (to execute a size-128 forward FFT, call
  // execute(128, in, out, true))
  virtual void execute(size_t size, fcomplex *in, fcomplex *out) const = 0;
  // get a list of the available sizes of this FFT object
  virtual std::vector<size_t> sizes() const = 0;
  // is this FFT forward or inverse?
  virtual bool forward() const = 0;
};
} // namespace fft
} // namespace bamradio

#endif // ecfb5eaadb7a7368f648
