//  Copyright © 2017 Stephen Larew

#ifndef aa397a8b8f4e45d658d886
#define aa397a8b8f4e45d658d886

#include "bam_constellation.h"
#include "llr_format.h"
#include <complex>
#include <fstream>
#include <functional>
#include <gnuradio/sync_block.h>
#include <string>
#include <vector>
#include <volk/volk.h>

namespace gr {
namespace bamofdm {

std::vector<std::complex<float>> generate_cazac_seq(const size_t N,
                                                    const size_t M);

template <class T>
void dump_vec(std::string const &filename, const std::vector<T> &vec) {
  std::ofstream out(filename, std::ofstream::binary);
  out.write((char *)vec.data(), sizeof(T) * vec.size());
  out.close();
}

template <class T>
void dump_vec(std::string const &filename, T const *const vec,
              size_t const count) {
  std::ofstream out(filename, std::ofstream::binary);
  out.write((char *)vec, sizeof(T) * count);
  out.close();
}

template <class T>
void dump_vec(std::ofstream &ofs, const std::vector<T> &vec) {
  ofs.write((char *)vec.data(), sizeof(T) * vec.size());
}

template <class T>
void dump_vec(std::ofstream &ofs, T const *const vec, size_t const count) {
  ofs.write((char *)vec, sizeof(T) * count);
}

class lambda_sync_block_base : virtual public sync_block {
public:
  // Make these methods public so the lambda functions can access them.
  using sync_block::add_item_tag;
  using sync_block::get_tags_in_range;
  using sync_block::get_tags_in_window;
};

/// Sync block that calls a lambda expression (1in,0out)
template <typename I>
class lambda_sync_block_10 : public lambda_sync_block_base {
private:
  std::function<ssize_t(lambda_sync_block_10<I> *, I const *,
                        size_t const)> const _work;

  lambda_sync_block_10(
      std::function<void(lambda_sync_block_10<I> *)> const &ctor,
      decltype(_work) const &work)
      : gr::sync_block("lambda_sync_block_10",
                       gr::io_signature::make(1, 1, sizeof(I)),
                       gr::io_signature::make(0, 0, 0)),
        _work(work) {
    ctor(this);
  }

public:
  typedef boost::shared_ptr<lambda_sync_block_10<I>> sptr;

  static sptr make(std::function<void(lambda_sync_block_10<I> *)> const &ctor,
                   decltype(_work) const &work) {
    return gnuradio::get_initial_sptr(new lambda_sync_block_10<I>(ctor, work));
  }

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &) {
    auto const n =
        (int)_work(this, static_cast<I const *>(input_items[0]), noutput_items);
    // std::cout << alias() << " " << n << "/" << noutput_items << std::endl;
    return n;
  }
};

/// Sync block that calls a lambda expression (1in,1out)
template <typename I, typename O>
class lambda_sync_block_11 : public lambda_sync_block_base {
private:
  std::function<ssize_t(lambda_sync_block_11<I, O> *, I const *, O *,
                        size_t const)> const _work;

  lambda_sync_block_11(
      std::function<void(lambda_sync_block_11<I, O> *)> const &ctor,
      decltype(_work) const &work)
      : gr::sync_block("lambda_sync_block_11",
                       gr::io_signature::make(1, 1, sizeof(I)),
                       gr::io_signature::make(1, 1, sizeof(O))),
        _work(work) {
    // Set alignment for VOLK
    const size_t alignment_multiple = volk_get_alignment() / sizeof(O);
    set_alignment(std::max<size_t>(1, alignment_multiple));
    ctor(this);
  }

public:
  typedef boost::shared_ptr<lambda_sync_block_11<I, O>> sptr;

  static sptr
  make(std::function<void(lambda_sync_block_11<I, O> *)> const &ctor,
       decltype(_work) const &work) {
    return gnuradio::get_initial_sptr(
        new lambda_sync_block_11<I, O>(ctor, work));
  }

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) {
    // assert(volk_is_aligned(input_items[0]));
    // assert(volk_is_aligned(output_items[0]));
    return (int)_work(this, static_cast<I const *>(input_items[0]),
                      static_cast<O *>(output_items[0]), noutput_items);
  }
};

/// Sync block that calls a lambda expression (2in,1out)
template <typename I0, typename I1, typename O>
class lambda_sync_block_21 : public lambda_sync_block_base {
private:
  std::function<ssize_t(lambda_sync_block_21<I0, I1, O> *, I0 const *,
                        I1 const *, O *, size_t const)> const _work;

  lambda_sync_block_21(
      std::function<void(lambda_sync_block_21<I0, I1, O> *)> const &ctor,
      decltype(_work) const &work)
      : gr::sync_block("lambda_sync_block_21",
                       gr::io_signature::makev(2, 2, {sizeof(I0), sizeof(I1)}),
                       gr::io_signature::make(1, 1, sizeof(O))),
        _work(work) {
    // Set alignment for VOLK
    const size_t alignment_multiple = volk_get_alignment() / sizeof(O);
    set_alignment(std::max<size_t>(1, alignment_multiple));
    ctor(this);
  }

public:
  typedef boost::shared_ptr<lambda_sync_block_21<I0, I1, O>> sptr;

  static sptr
  make(std::function<void(lambda_sync_block_21<I0, I1, O> *)> const &ctor,
       decltype(_work) const &work) {
    return gnuradio::get_initial_sptr(
        new lambda_sync_block_21<I0, I1, O>(ctor, work));
  }

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) {
    // assert(volk_is_aligned(input_items[0]));
    // assert(volk_is_aligned(input_items[1]));
    // assert(volk_is_aligned(output_items[0]));
    return _work(this, static_cast<I0 const *>(input_items[0]),
                 static_cast<I1 const *>(input_items[1]),
                 static_cast<O *>(output_items[0]), noutput_items);
  }
};

/// Sync block that calls a lambda expression (0in,1out)
template <typename O>
class lambda_sync_block_01 : public lambda_sync_block_base {
private:
  std::function<ssize_t(lambda_sync_block_01<O> *, O *, size_t const)> const
      _work;
  lambda_sync_block_01(
      std::function<void(lambda_sync_block_01<O> *)> const &ctor,
      decltype(_work) const &work)
      : gr::sync_block("lambda_sync_block_01", gr::io_signature::make(0, 0, 0),
                       gr::io_signature::make(1, 1, sizeof(O))),
        _work(work) {

    // Set alignment for VOLK
    const size_t alignment_multiple = volk_get_alignment() / sizeof(O);
    set_alignment(std::max<size_t>(1, alignment_multiple));
    ctor(this);
  }

public:
  typedef boost::shared_ptr<lambda_sync_block_01<O>> sptr;
  static sptr make(std::function<void(lambda_sync_block_01<O> *)> const &ctor,
                   decltype(_work) const &work) {
    return gnuradio::get_initial_sptr(new lambda_sync_block_01<O>(ctor, work));
  }
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) {
    return _work(this, static_cast<O *>(output_items[0]), noutput_items);
  }
};

std::vector<gr_complex>
random_symbols(size_t n, bamradio::constellation::sptr constellation,
               size_t seed = 33);
std::vector<gr_complex> awgn(double snr, std::vector<gr_complex> syms,
                             size_t seed = 33);

bool assert_float_almost_equal(float const &ref, float const &x,
                               double abs_eps = 1e-12, double rel_eps = 1e-6);
bool assert_float_vectors_almost_equal(std::vector<float> const &ref,
                                       std::vector<float> const &x,
                                       double abs_eps = 1e-12,
                                       double rel_eps = 1e-6);
bool assert_complex_almost_equal(gr_complex const &ref, gr_complex const &x,
                                 double abs_eps = 1e-12, double rel_eps = 1e-6);
bool assert_complex_vectors_almost_equal(std::vector<gr_complex> const &ref,
                                         std::vector<gr_complex> const &x,
                                         double abs_eps = 1e-12,
                                         double rel_eps = 1e-6);

} // namespace bamofdm
} // namespace gr

#endif
