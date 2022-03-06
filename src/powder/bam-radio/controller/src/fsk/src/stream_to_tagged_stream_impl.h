/* -*- c++ -*- */
/* 
 * Copyright 2013 Joon Young Kim <kim415@purdue.edu>
 *				  Andrew Balmos <abalmos@purdue.edu>
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

#ifndef INCLUDED_FEEDBACK_STREAM_TO_TAGGED_STREAM_IMPL_H
#define INCLUDED_FEEDBACK_STREAM_TO_TAGGED_STREAM_IMPL_H

#include <bamfsk/stream_to_tagged_stream.h>

namespace gr {
  namespace bamfsk {

	class stream_to_tagged_stream_impl : public stream_to_tagged_stream
	{
	private:
        pmt::pmt_t  d_len_tag_key;  
		size_t d_needed;
		tag_t d_len_tag;

		pmt::pmt_t	d_preamble_time_tag;

	public:
		stream_to_tagged_stream_impl(const std::string& len_tag_key);
		~stream_to_tagged_stream_impl();

		void forecast (int noutput_items, gr_vector_int &ninput_items_required);

		int general_work(int noutput_items,
			gr_vector_int &ninput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
	};

  }
} // namespace gr

#endif /* INCLUDED_FEEDBACK_STREAM_TO_TAGGED_STREAM_IMPL_H */

