// -*- c++ -*-
/* Copyright (c) 2017 Tomohiro Arakawa */

#ifndef CTRL_H_INCLUDED
#define CTRL_H_INCLUDED

#include "cc_data.h"
#include "ctrl_recv.h"
#include <gnuradio/hier_block2.h>

namespace bamradio {
namespace controlchannel {

class ctrl_ch : public gr::hier_block2 {
public:
  typedef boost::shared_ptr<ctrl_ch> sptr;

  static sptr make(CCData::sptr cc_data, unsigned int max_nodes,
                   unsigned int node_num, double sample_rate, double t_slot_sec,
                   double scale, unsigned int num_fsk_points, int rs_k,
                   int min_soft_decs) {
    return gnuradio::get_initial_sptr(
        new ctrl_ch(cc_data, max_nodes, node_num, sample_rate, t_slot_sec,
                    scale, num_fsk_points, rs_k, min_soft_decs));
  }

  void set_cc_data(CCData::sptr ccd);

private:
  ctrl_ch(CCData::sptr cc_data, unsigned int max_nodes, unsigned int node_num,
          double sample_rate, double t_slot_sec, double scale,
          unsigned int num_fsk_points, int rs_k, int min_soft_decs);

  gr::bamfsk::ctrl_recv::sptr d_c_recv;
};
}
}
#endif
