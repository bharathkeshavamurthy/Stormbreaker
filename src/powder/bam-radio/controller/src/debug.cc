#include "debug.h"

#include <fstream>
#include <memory>
#include <pmt/pmt.h>

namespace bamradio {
namespace debug {

void ProtoizeOFDMSymbolParams(ofdm::OFDMSymbolParams const &sp,
                              BAMDebugPb::OFDMSymbolParams *psp) {
  using namespace bamradio::ofdm;

  psp->set_symbol_length(sp.symbol_length);
  psp->set_oversample_rate(sp.oversample_rate);
  psp->set_cyclic_prefix_length(sp.cyclic_prefix_length);
  psp->set_postfix_pad(sp.postfixPad);

  psp->set_num_tx_samples(sp.numTXSamples());
  psp->set_num_data_carriers(sp.numDataCarriers());
  psp->set_num_bits(sp.numBits());
  psp->set_num_pilot_carriers(sp.numPilotCarriers());

  auto ndc = sp.numDataCarriers();
  auto dcm = pmt::s32vector_elements(sp.data_carrier_mapping, ndc);
  for (auto i = 0; i < ndc; ++i)
    psp->add_data_carrier_mapping(dcm[i]);
  auto npc = sp.numPilotCarriers();
  auto pcm = pmt::s32vector_elements(sp.pilot_carrier_mapping, npc);
  for (auto i = 0; i < npc; ++i)
    psp->add_pilot_carrier_mapping(pcm[i]);

  for (auto i = 0; i < npc; ++i) {
    auto num = pmt::c32vector_ref(sp.pilot_symbols, i);
    auto fc = psp->add_pilot_symbols();
    fc->set_re(num.real());
    fc->set_im(num.imag());
  }

  if (sp.prefix) {
    auto npre = pmt::length(sp.prefix);
    for (auto i = 0; i < npre; ++i) {
      auto num = pmt::c32vector_ref(sp.prefix, i);
      auto fc = psp->add_prefix();
      fc->set_re(num.real());
      fc->set_im(num.imag());
    }
  }
}

std::vector<char> SerializeFrame(ofdm::DFTSOFDMFrameParams const &frame) {
  std::unique_ptr<BAMDebugPb::DFTSOFDMFrameParams> out(
      new BAMDebugPb::DFTSOFDMFrameParams());

  out->set_num_symbols(frame.numSymbols());
  out->set_num_tx_samples(frame.numTXSamples());
  out->set_num_bits(frame.numBits());
  out->set_dft_spread_length(frame.dft_spread_length);

  for (auto const &symbol : frame.symbols) {
    for (size_t i = 0; i < symbol.first; ++i) {
      auto s = out->add_symbols();
      ProtoizeOFDMSymbolParams(*symbol.second, s);
    }
  }

  std::vector<char> bout(out->ByteSize());
  out->SerializeToArray(bout.data(), bout.size());
  return bout;
}

void DumpFrameParams(std::string const &filename,
                     ofdm::DFTSOFDMFrameParams const &frame) {
  auto bytes = SerializeFrame(frame);
  std::ofstream out(filename, std::ofstream::binary);
  out.write(bytes.data(), bytes.size());
  out.close();
}

} // namespace debug
} // namespace bamradio
