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

#ifndef INCLUDED_BAMFSK_MFSKMOD_FC_H
#define INCLUDED_BAMFSK_MFSKMOD_FC_H

#include <bamfsk/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace bamfsk {

/*!
 * \brief <+description of block+>
 * \ingroup bamfsk
 *
 */
class BAMFSK_API mfskMod_fc : virtual public gr::sync_block {
public:
  typedef boost::shared_ptr<mfskMod_fc> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of bamfsk::mfskMod_fc.
   *
   * To avoid accidental use of raw pointers, bamfsk::mfskMod_fc's
   * constructor is in a private implementation
   * class. bamfsk::mfskMod_fc::make is the public interface for
   * creating new instances.
   */
  static sptr make(double sampleRate, double scale);
};

} // namespace bamfsk
} // namespace gr

#endif /* INCLUDED_BAMFSK_MFSKMOD_FC_H */
