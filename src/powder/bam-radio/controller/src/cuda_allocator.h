// -*- c++ -*-
//
// STL container allocators for CUDA
//
// Copyright (c) 2018 Dennis Ogbe

#ifndef e0c0e2c1abf87d19b940
#define e0c0e2c1abf87d19b940

#include <cuda_runtime.h>

namespace bamradio {

template <class T> struct managed_cuda_allocator {
  managed_cuda_allocator() = default;
  using value_type = T;
  template <class U>
  constexpr managed_cuda_allocator(const managed_cuda_allocator<U> &) noexcept {
  }
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    void *p;
    cudaMallocManaged(&p, n * sizeof(T));
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(p);
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};

template <class T, class U>
bool operator==(const managed_cuda_allocator<T> &,
                const managed_cuda_allocator<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const managed_cuda_allocator<T> &,
                const managed_cuda_allocator<U> &) {
  return false;
}
} // namespace bamradio

#endif // e0c0e2c1abf87d19b940
