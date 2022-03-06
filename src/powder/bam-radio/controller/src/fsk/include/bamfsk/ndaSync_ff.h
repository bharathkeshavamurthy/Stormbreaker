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


#ifndef INCLUDED_FEEDBACK_NDASYNC_FF_H
#define INCLUDED_FEEDBACK_NDASYNC_FF_H

#include <bamfsk/api.h>
#include <gnuradio/sync_decimator.h>

namespace gr {
  namespace bamfsk {

    /*!
     * \brief <+description of block+>
     * \ingroup feedback
     *
     */
    class BAMFSK_API ndaSync_ff : virtual public gr::sync_decimator
    {
     public:
      typedef boost::shared_ptr<ndaSync_ff> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of feedback::ndaSync_ff.
       *
       * To avoid accidental use of raw pointers, feedback::ndaSync_ff's
       * constructor is in a private implementation
       * class. feedback::ndaSync_ff::make is the public interface for
       * creating new instances.
       */
      static sptr make(int pulseLen, int minSoftDecs);
    };

  } // namespace feedback
} // namespace gr

#endif /* INCLUDED_FEEDBACK_NDASYNC_FF_H */

