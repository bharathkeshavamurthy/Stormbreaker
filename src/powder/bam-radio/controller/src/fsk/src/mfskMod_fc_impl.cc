/* -*- c++ -*- */
/*
 * Copyright 2014 <+YOU OR YOUR COMPANY+>.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mfskMod_fc_impl.h"
#include <cmath>
#include <gnuradio/expj.h>
#include <gnuradio/io_signature.h>
#include <stdio.h>
#include <stdlib.h>

namespace gr {
namespace bamfsk {

mfskMod_fc::sptr mfskMod_fc::make(double sampleRate, double scale) {
  return gnuradio::get_initial_sptr(new mfskMod_fc_impl(sampleRate, scale));
}

/*
 * The private constructor
 */
mfskMod_fc_impl::mfskMod_fc_impl(double sampleRate, double scale)
    : gr::sync_block("mfskMod_fc", gr::io_signature::make(1, 1, sizeof(float)),
                     gr::io_signature::make(1, 1, sizeof(gr_complex))),
      d_phaseDelta(0), d_sampleRate(sampleRate), _scale(scale, 0.0) {}

/*
 * Our virtual destructor.
 */
mfskMod_fc_impl::~mfskMod_fc_impl() {}

int mfskMod_fc_impl::work(int noutput_items,
                          gr_vector_const_void_star &input_items,
                          gr_vector_void_star &output_items) {
  const float *freqs = (const float *)input_items[0];
  gr_complex *samples = (gr_complex *)output_items[0];

  for (int z = 0; z < noutput_items; z++) {
    d_phaseDelta += freqs[z] / d_sampleRate;

    if (d_phaseDelta > 1) {
      d_phaseDelta = d_phaseDelta - 1;
    }
    if (d_phaseDelta < -1) {
      d_phaseDelta = d_phaseDelta + 1;
    }
    samples[z] = _scale * gr_expj(2 * M_PI * d_phaseDelta);
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace bamfsk */
} /* namespace gr */
