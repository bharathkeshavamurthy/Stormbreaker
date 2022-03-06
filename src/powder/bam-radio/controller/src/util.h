// -*- c++ -*-
//  Copyright © 2017 Stephen Larew

#ifndef aadd8cbdd5b9c49ad7
#define aadd8cbdd5b9c49ad7

#include <chrono>
#include <memory>
#include <thread>
#include <functional>

#include <boost/asio.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/variant.hpp>

#include <pmt/pmt.h>
#include <uhd/types/time_spec.hpp>

namespace bamradio {

class BackgroundThreadService : public boost::asio::io_service::service {
public:
  struct impl {
    impl(boost::asio::io_service &ios);
    ~impl();
    std::unique_ptr<boost::asio::io_service::work> work;
    std::thread work_thread;
  };

  typedef impl *implementation_type;

  static boost::asio::io_service::id id;

  void shutdown_service() { /* TODO: ? */
  }

  explicit BackgroundThreadService(boost::asio::io_service &ios);

  void construct(implementation_type &);
  void destroy(implementation_type &);

  boost::asio::io_service work_ios;
};

// double_dur_uhd matches precision of uhd::time_spec_t frac_seconds
typedef std::chrono::duration<double, std::chrono::seconds::period> double_dur_uhd;

template <class Rep, class Per>
::uhd::time_spec_t
duration_to_uhd_time_spec(std::chrono::duration<Rep, Per> const &d) {
  auto const secs = std::chrono::duration_cast<std::chrono::seconds>(d);
  auto const fsecs = d - secs;
  return ::uhd::time_spec_t(secs.count(), double_dur_uhd(fsecs).count());
}

template <class Dur>
Dur uhd_time_spec_to_duration(::uhd::time_spec_t const &ts) {
  return std::chrono::duration_cast<Dur>(double_dur_uhd(ts.get_real_secs()));
}

::uhd::time_spec_t system_clock_now(void);
pmt::pmt_t time_spec_to_pmt(::uhd::time_spec_t const &t);
::uhd::time_spec_t pmt_to_time_spec(pmt::pmt_t const &p);

void set_thread_name(std::string const &name);

} // namespace bamradio

namespace boost {

template <typename T1, typename T2, typename R>
class visitor2_ptr_t : public boost::static_visitor<R> {
public:
  typedef std::function<R(T1)> visitor1_t;
  typedef std::function<R(T2)> visitor2_t;

private:
  visitor1_t visitor1_;
  visitor2_t visitor2_;

  typedef typename boost::mpl::eval_if<
      boost::is_reference<T1>, boost::mpl::identity<T1>,
      boost::add_reference<const T1>>::type argument1_fwd_type;
  typedef typename boost::mpl::eval_if<
      boost::is_reference<T2>, boost::mpl::identity<T2>,
      boost::add_reference<const T2>>::type argument2_fwd_type;

public:
  typedef R result_type;

  visitor2_ptr_t(visitor1_t visitor1, visitor2_t visitor2) BOOST_NOEXCEPT
      : visitor1_(visitor1),
        visitor2_(visitor2) {}

  template <typename U> result_type operator()(const U &) const {
    boost::throw_exception(boost::bad_visit());
  }

  result_type operator()(argument1_fwd_type operand) const {
    return visitor1_(operand);
  }
  result_type operator()(argument2_fwd_type operand) const {
    return visitor2_(operand);
  }
};

} // namespace boost

#endif
