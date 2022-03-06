/* -*- c++ -*- */
/* 
 * Copyright 2017 Tomohiro Arakawa <tarakawa@purdue.edu>
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

#ifndef INCLUDED_BAMFSK_MFSK_RX_IMPL_H
#define INCLUDED_BAMFSK_MFSK_RX_IMPL_H

#include <bamfsk/mfsk_rx.h>

namespace gr {
  namespace bamfsk {

    class mfsk_rx_impl : public mfsk_rx
    {
     private:
      // Nothing to declare in this block.

     public:
      mfsk_rx_impl(double sample_rate, std::vector<float> pulse, int num_fsk_points, int rs_k, std::vector<uint8_t> preamble, int min_soft_decs, int payload_len);
      ~mfsk_rx_impl();

      // Where all the action really happens
    };

  } // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_MFSK_RX_IMPL_H */

