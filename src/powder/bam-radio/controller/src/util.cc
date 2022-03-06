//  Copyright © 2017 Stephen Larew

#include "util.h"

namespace bamradio {

using namespace boost::asio;

BackgroundThreadService::impl::impl(io_service &ios)
    : work(new io_service::work(ios)), work_thread([&] { ios.run(); }) {}
BackgroundThreadService::impl::~impl() {
  work.reset();
  work_thread.join();
}

io_service::id BackgroundThreadService::id;

BackgroundThreadService::BackgroundThreadService(io_service &ios)
    : io_service::service(ios), work_ios() {}

void BackgroundThreadService::construct(implementation_type &i) {
  i = new impl(work_ios);
}

void BackgroundThreadService::destroy(implementation_type &i) {
  delete i;
  i = nullptr;
}

::uhd::time_spec_t system_clock_now(void) {
  return duration_to_uhd_time_spec(
      std::chrono::system_clock::now().time_since_epoch());
}

pmt::pmt_t time_spec_to_pmt(::uhd::time_spec_t const &t) {
  return pmt::make_tuple(pmt::from_uint64(t.get_full_secs()),
                         pmt::from_double(t.get_frac_secs()));
}

::uhd::time_spec_t pmt_to_time_spec(pmt::pmt_t const &p) {
  return ::uhd::time_spec_t(pmt::to_uint64(pmt::tuple_ref(p, 0)),
                            pmt::to_double(pmt::tuple_ref(p, 1)));
}

void set_thread_name(std::string const &name) {
  pthread_setname_np(pthread_self(), name.c_str());
}
} // namespace bamradio
