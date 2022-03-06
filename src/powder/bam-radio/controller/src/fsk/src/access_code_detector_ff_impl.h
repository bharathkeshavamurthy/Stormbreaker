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

#ifndef INCLUDED_FEEDBACK_ACCESS_CODE_DETECTOR_FF_IMPL_H
#define INCLUDED_FEEDBACK_ACCESS_CODE_DETECTOR_FF_IMPL_H

#include <bamfsk/access_code_detector_ff.h>
#include <vector>

#include <gnuradio/gr_complex.h>
#include <gnuradio/fft/fft.h>

namespace gr {
  namespace bamfsk {

    class access_code_detector_ff_impl : public access_code_detector_ff
    {
     private:
		int d_frame_len;
		int d_block_len;
		int d_payload_len;
		double d_upstream_dec_factor;
		std::vector<float> d_pre_bits;
		tag_t d_len_tag;

		bool				d_use_fft;
		std::vector<float>	d_pre_bits_rev;
		int					d_fft_size;
		fft::fft_real_fwd	*d_fft_fwd;
		fft::fft_real_rev	*d_fft_rev;
		gr_complex			*d_fft_preamble;

		uint64_t d_time_offset;
		uint64_t d_time_seconds;
		double d_time_fracs;
		double d_samp_rate;

		void add_time_tag(unsigned int ind);

     public:
      access_code_detector_ff_impl(const int payload_len, 
			  const std::vector<int>pre,
			  const std::string& len_tag_key,
			  const double upstream_dec_factor,
              const bool use_fft);
      ~access_code_detector_ff_impl();

      // Where all the action really happens
      int work(int noutput_items,
	       gr_vector_const_void_star &input_items,
	       gr_vector_void_star &output_items);
    };

  }
} // namespace gr

#endif /* INCLUDED_FEEDBACK_ACCESS_CODE_DETECTOR_FF_IMPL_H */

