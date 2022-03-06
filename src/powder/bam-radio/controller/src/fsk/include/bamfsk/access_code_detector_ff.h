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


#ifndef INCLUDED_FEEDBACK_ACCESS_CODE_DETECTOR_FF_H
#define INCLUDED_FEEDBACK_ACCESS_CODE_DETECTOR_FF_H

#include <bamfsk/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace bamfsk {

    class BAMFSK_API access_code_detector_ff : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<access_code_detector_ff> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of feedback::access_code_detector_ff.
       *
       * To avoid accidental use of raw pointers, feedback::access_code_detector_ff's
       * constructor is in a private implementation
       * class. feedback::access_code_detector_ff::make is the public interface for
       * creating new instances.
       */
      static sptr make(const int payload_len, 
			  const std::vector<int>pre, 
			  const std::string& len_tag_key,
			  const double upstream_dec_factor = 1.0,
              const bool use_fft = true);
    };

  }
} // namespace gr

#endif /* INCLUDED_FEEDBACK_ACCESS_CODE_DETECTOR_FF_H */

