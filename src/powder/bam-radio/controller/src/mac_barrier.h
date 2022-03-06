// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#ifndef aa469ec6d929201606bf88
#define aa469ec6d929201606bf88

#include <gnuradio/sync_block.h>
#include <gnuradio/uhd/usrp_sink.h>
#include <uhd/types/metadata.hpp>
#include "bbqueue.h"


namespace bamradio {
namespace ofdm {

class mac_barrier : public gr::sync_block {
public:
  typedef boost::shared_ptr<mac_barrier> sptr;

  static pmt::pmt_t const delay_port_id;
  static pmt::pmt_t const channel_use_port_id;

  bamradio::BBQueue<::uhd::async_metadata_t> uhd_async_msgs;

  static sptr make(gr::uhd::usrp_sink::sptr usrp_sink);

  ~mac_barrier();

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

  bool start();

  gr::uhd::usrp_sink::sptr usrp_sink() const;

private:
  mac_barrier(gr::uhd::usrp_sink::sptr usrp_sink);

  void handle_chan_use_msg(pmt::pmt_t msg);

  //std::vector<pmt::pmt_t> _time_slots;
  gr::uhd::usrp_sink::sptr _usrp_sink;
  // FIXME: update samp_rate independently of usrp_sink
  double _samp_rate;
  ::uhd::time_spec_t _last_time;
  uint64_t _last_sob;
};

}
}

#endif
