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

#ifndef INCLUDED_BAMFSK_MFSKMOD_FC_IMPL_H
#define INCLUDED_BAMFSK_MFSKMOD_FC_IMPL_H

#include <bamfsk/mfskMod_fc.h>

namespace gr {
namespace bamfsk {

class mfskMod_fc_impl : public mfskMod_fc {
private:
  double d_phaseDelta;
  double d_sampleRate;
  gr_complex _scale;

public:
  mfskMod_fc_impl(double sampleRate, double scale);
  ~mfskMod_fc_impl();

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_MFSKMOD_FC_IMPL_H */
