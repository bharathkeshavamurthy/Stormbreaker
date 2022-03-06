/* -*- c++ -*- */
// Copyright 2017 Tomohiro Arakawa <tarakawa@purdue.edu>

#ifndef INCLUDED_BAMFSK_CTRL_RECV_IMPL_H
#define INCLUDED_BAMFSK_CTRL_RECV_IMPL_H

#include "cc_data.h"
#include <string>
#include <gnuradio/tagged_stream_block.h>
#include <mutex>
#include <pmt/pmt.h>
#include <uhd/types/time_spec.hpp>

namespace gr {
namespace bamfsk {

class ctrl_recv : virtual public gr::tagged_stream_block {
private:
  std::mutex _m;
  bamradio::controlchannel::CCData::sptr d_cc_data;

public:
  typedef boost::shared_ptr<ctrl_recv> sptr;
  static sptr make(std::string const &length_tag_key, bamradio::controlchannel::CCData::sptr cc_data);
  ctrl_recv(std::string const &length_tag_key, bamradio::controlchannel::CCData::sptr cc_data);
  ~ctrl_recv();
  int calculate_output_stream_length(const gr_vector_int &ninput_items) {
    return 0;
  }

  // Where all the action really happens
  int work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

  void set_cc_data(bamradio::controlchannel::CCData::sptr ccd);

};

} // namespace bamfsk
} // namespace gr

#endif
