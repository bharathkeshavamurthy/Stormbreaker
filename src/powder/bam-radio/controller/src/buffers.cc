// -*- c++ -*-
// buffers.cc

#include "buffers.h"

namespace bamradio {
namespace ofdm {

ChannelOutputBuffer::ChannelOutputBuffer(size_t n)
    : samples(PinnedComplexRingBuffer::make(n)),
      Pd(PinnedComplexRingBuffer::make(n)), Md(PinnedFloatRingBuffer::make(n)) {
}

void ChannelOutputBuffer::consume_each(size_t n) {
  samples->consume(n);
  Pd->consume(n);
  Md->consume(n);
}

void ChannelOutputBuffer::produce_each(size_t n) {
  samples->produce(n);
  Pd->produce(n);
  Md->produce(n);
}

size_t ChannelOutputBuffer::items_avail() const {
  return std::min(
      {samples->items_avail(), Md->items_avail(), Pd->items_avail()});
}

ssize_t ChannelOutputBuffer::space_avail() const {
  return std::min(
      {samples->space_avail(), Md->space_avail(), Pd->space_avail()});
}

} // namespace ofdm
} // namespace bamradio
