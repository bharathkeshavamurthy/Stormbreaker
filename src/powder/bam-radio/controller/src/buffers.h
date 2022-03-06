// -*- c++ -*-
// buffers.h

#ifndef e02a5f6c9fb50951f77cd
#define e02a5f6c9fb50951f77cd

#include "fcomplex.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <system_error>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <boost/format.hpp>
#include <boost/math/common_factor_rt.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <cuda_runtime.h>

// a simple single producer, single comsumer ring buffer implementation that
// works with CPU and CUDA GPU.
//
// Copyright (c) 2018 Dennis Ogbe
//
// Dependencies:
//
// - boost_system

namespace ringbuffer {
//
// detail namespace for the implementation of memory mapping
//
namespace rb_detail {

//
// return L = sizeof(T) * M * pagesize, the smallest number larger than or
// equal to sizeof(T) * nitems that is a multiple (M) of the pagesize
//
// see minimum_buffer_items in buffer.cc of GNU Radio
//
template <typename T> std::size_t next_page_multiple_size(std::size_t nitems) {
  unsigned long page_size = getpagesize();
  size_t min_nitems = page_size / boost::math::gcd(sizeof(T), page_size);
  return nitems % min_nitems == 0
             ? nitems
             : ((nitems / min_nitems) + 1) * min_nitems * sizeof(T); // round up
}

//
// Memory-mapped circular buffer that re-creates the same mapping from the
// viewpoint of the GPU using CUDA. This implementation uses the memfd_create
// syscall and thus works only on Linux.
//
// Use this as a reference interface in case you need to re-implement
//
template <typename T> class memfd_cuda_circbuf {
public:
  memfd_cuda_circbuf(memfd_cuda_circbuf &other) = delete;

  // ctor
  memfd_cuda_circbuf(std::size_t nitems) noexcept(false)
      : _size(rb_detail::next_page_multiple_size<T>(nitems)),
        _nitems(_size / sizeof(T)), _first(NULL), _second(NULL), _fd(0),
        _fd_name((boost::format("bamradio-buffer-%1%") %
                  boost::uuids::to_string(boost::uuids::random_generator()()))
                     .str()) {

    // make sure the requested size is a multiple of the system's page size
    if (_size % getpagesize() != 0) {
      throw std::runtime_error("memfd_cuda_circbuf: Need size to be a multiple "
                               "of system page size.");
    }

    // get an anon file (exists in RAM only) of given size
    _fd = syscall(SYS_memfd_create, _fd_name.c_str(), 0);
    if (_fd == -1) {
      throw std::system_error(errno, std::system_category());
    }
    if (ftruncate(_fd, _size) == -1) {
      throw std::system_error(errno, std::system_category());
    }

    // ask mmap for address at a location where both virtual copies fit
    _first =
        mmap(NULL, 2 * _size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (_first == MAP_FAILED) {
      throw std::system_error(errno, std::system_category());
    }

    // map the buffer at that address
    if (mmap(_first, _size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, _fd,
             0) == MAP_FAILED) {
      throw std::system_error(errno, std::system_category());
    }

    // map the buffer again at the end of the first copy
    _second = static_cast<uint8_t *>(_first) + _size;
    if (mmap(_second, _size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
             _fd, 0) == MAP_FAILED) {
      throw std::system_error(errno, std::system_category());
    }

    // register the slice of memory obtained with CUDA
    cudaHostRegister(_first, 2 * _size, 0);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error((boost::format("CUDA Error (%1%:%2%): %3%") %
                                __FILE__ % __LINE__ % cudaGetErrorString(err))
                                   .str());
    }
  }

  // dtor
  ~memfd_cuda_circbuf() {
    if (munmap(_second, _size) == -1) {
      std::cout << "WARNING (__FILE__:__LINE__): " << strerror(errno)
                << std::endl;
    }
    if (munmap(_first, _size) == -1) {
      std::cout << "WARNING (__FILE__:__LINE__): " << strerror(errno)
                << std::endl;
    }
    if (close(_fd) == -1) {
      std::cout << "WARNING (__FILE__:__LINE__): " << strerror(errno)
                << std::endl;
    }
    cudaHostUnregister(_first);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "WARNING (__FILE__:__LINE__): " << cudaGetErrorString(err)
                << std::endl;
    }
  }

  //
  // public API
  //
  T *first() const { return static_cast<T *>(_first); } // pointer to first copy
  T *second() const { return static_cast<T *>(_second); };
  std::size_t size() const { return _size; }
  std::size_t nitems() const { return _nitems; }

private:
  std::size_t _size;    // size of the buffer in bytes
  std::size_t _nitems;  // number of items
  void *_first;         // address of first copy
  void *_second;        // address of second copy
  int _fd;              // file descriptor of anon file
  std::string _fd_name; // see man memfd_create for details
};

//
// Circular buffer without CUDA dependencies
//
template <typename T> class memfd_nocuda_circbuf {
public:
  memfd_nocuda_circbuf(memfd_nocuda_circbuf &other) = delete;

  // ctor
  memfd_nocuda_circbuf(std::size_t nitems) noexcept(false)
      : _size(rb_detail::next_page_multiple_size<T>(nitems)),
        _nitems(_size / sizeof(T)), _first(NULL), _second(NULL), _fd(0),
        _fd_name((boost::format("bamradio-buffer-%1%") %
                  boost::uuids::to_string(boost::uuids::random_generator()()))
                     .str()) {

    // make sure the requested size is a multiple of the system's page size
    if (_size % getpagesize() != 0) {
      throw std::runtime_error(
          "memfd_nocuda_circbuf: Need size to be a multiple "
          "of system page size.");
    }

    // get an anon file (exists in RAM only) of given size
    _fd = syscall(SYS_memfd_create, _fd_name.c_str(), 0);
    if (_fd == -1) {
      throw std::system_error(errno, std::system_category());
    }
    if (ftruncate(_fd, _size) == -1) {
      throw std::system_error(errno, std::system_category());
    }

    // ask mmap for address at a location where both virtual copies fit
    _first =
        mmap(NULL, 2 * _size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (_first == MAP_FAILED) {
      throw std::system_error(errno, std::system_category());
    }

    // map the buffer at that address
    if (mmap(_first, _size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, _fd,
             0) == MAP_FAILED) {
      throw std::system_error(errno, std::system_category());
    }

    // map the buffer again at the end of the first copy
    _second = static_cast<uint8_t *>(_first) + _size;
    if (mmap(_second, _size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
             _fd, 0) == MAP_FAILED) {
      throw std::system_error(errno, std::system_category());
    }
  }

  // dtor
  ~memfd_nocuda_circbuf() {
    if (munmap(_second, _size) == -1) {
      std::cout << "WARNING (__FILE__:__LINE__): " << strerror(errno)
                << std::endl;
    }
    if (munmap(_first, _size) == -1) {
      std::cout << "WARNING (__FILE__:__LINE__): " << strerror(errno)
                << std::endl;
    }
    if (close(_fd) == -1) {
      std::cout << "WARNING (__FILE__:__LINE__): " << strerror(errno)
                << std::endl;
    }
  }

  //
  // public API
  //
  T *first() const { return static_cast<T *>(_first); } // pointer to first copy
  T *second() const { return static_cast<T *>(_second); };
  std::size_t size() const { return _size; }
  std::size_t nitems() const { return _nitems; }

private:
  std::size_t _size;    // size of the buffer in bytes
  std::size_t _nitems;  // number of items
  void *_first;         // address of first copy
  void *_second;        // address of second copy
  int _fd;              // file descriptor of anon file
  std::string _fd_name; // see man memfd_create for details
};
} // namespace rb_detail

//
// A single consumer, single producer FIFO queue
//
template <typename T, typename buf_impl = rb_detail::memfd_cuda_circbuf<T>>
class Ringbuffer {
public:
  //
  // Constructors etc. Prefer 'make' free function.
  //
  typedef std::shared_ptr<Ringbuffer<T, buf_impl>> sptr;
  static sptr make(std::size_t nitems) {
    return std::make_shared<Ringbuffer<T, buf_impl>>(nitems);
  }
  Ringbuffer(Ringbuffer const &other) = delete;
  Ringbuffer(std::size_t nitems)
      : _cbuf(std::make_unique<buf_impl>(nitems)), _w(0), _r(0) {}

  //
  // public API
  //
  // returns the number of items currently in the buffer
  std::size_t items_avail() const {
    std::lock_guard<decltype(_mtx)> l(_mtx);
    return _index_sub(_w, _r);
  }
  // returns the space left to write to the buffer. if -1 is returned, there is
  // no space available.
  ssize_t space_avail() const {
    return (ssize_t)_cbuf->nitems() - (ssize_t)items_avail() - 1;
  }

  // n.b. no need to lock the mutex in the *_ptr() functions. This is designed
  // to be single producer/single consumer, so producer can only call EITHER
  // write_ptr() OR produce() for example...

  // write pointer
  T *write_ptr() const { return _cbuf->first() + _w; }
  std::size_t write_idx() const { return _w; }
  void produce(std::size_t n) {
    std::lock_guard<decltype(_mtx)> l(_mtx);
    _w = _index_add(_w, n);
  }

  // read pointer
  const T *read_ptr() const { return _cbuf->first() + _r; }
  std::size_t read_idx() const { return _r; }
  void consume(std::size_t n) {
    std::lock_guard<decltype(_mtx)> l(_mtx);
    _r = _index_add(_r, n);
  }

  // misc
  T *base() const { return _cbuf->first(); }
  std::size_t nitems() const { return _cbuf->nitems(); }

private:
  std::unique_ptr<buf_impl> _cbuf; // underlying memory-mapped buffer
  std::size_t _w;                  // write index
  std::size_t _r;                  // read index
  mutable std::mutex _mtx;         // this needs to be thread safe

  // modulo buffer size arithmetic for buffer indices
  std::size_t _index_add(std::size_t a, std::size_t b) const {
    auto s = a + b;
    if (s >= _cbuf->nitems()) {
      s -= _cbuf->nitems();
    }
    assert(s < _cbuf->nitems());
    return s;
  }
  std::size_t _index_sub(std::size_t a, std::size_t b) const {
    ssize_t s = (ssize_t)a - b;
    if (s < 0) {
      s += _cbuf->nitems();
    }
    assert(s >= 0);
    assert(static_cast<size_t>(s) < _cbuf->nitems());
    return static_cast<size_t>(s);
  }
};

} // namespace ringbuffer

// we use these ringbuffers in a variety of places in bam-radio.

namespace bamradio {

// Ringbuffers
typedef ringbuffer::Ringbuffer<
    fcomplex, ringbuffer::rb_detail::memfd_nocuda_circbuf<fcomplex>>
    ComplexRingBuffer;

typedef ringbuffer::Ringbuffer<
    float, ringbuffer::rb_detail::memfd_nocuda_circbuf<float>>
    FloatRingBuffer;

// Ringbuffers in pinned host memory
typedef ringbuffer::Ringbuffer<
    fcomplex, ringbuffer::rb_detail::memfd_cuda_circbuf<fcomplex>>
    PinnedComplexRingBuffer;

typedef ringbuffer::Ringbuffer<float,
                               ringbuffer::rb_detail::memfd_cuda_circbuf<float>>
    PinnedFloatRingBuffer;

namespace ofdm {

class ChannelOutputBuffer {
public:
  typedef std::shared_ptr<ChannelOutputBuffer> sptr;
  template <typename... Args> static sptr make(Args &&... args) {
    return std::make_shared<ChannelOutputBuffer>(std::forward<Args>(args)...);
  }

  ChannelOutputBuffer(size_t n);

  void consume_each(size_t n);
  void produce_each(size_t n);
  size_t items_avail() const;
  ssize_t space_avail() const;

  PinnedComplexRingBuffer::sptr samples;
  PinnedComplexRingBuffer::sptr Pd;
  PinnedFloatRingBuffer::sptr Md;
};

} // namespace ofdm

} // namespace bamradio

#endif // e02a5f6c9fb50951f77cd
