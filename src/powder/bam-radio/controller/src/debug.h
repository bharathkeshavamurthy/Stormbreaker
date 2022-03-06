
#ifndef a9fcd474faa1fabf3611
#define a9fcd474faa1fabf3611

#include <string>
#include <vector>

#include "ofdm.h"
#include <debug.pb.h>

namespace bamradio {
namespace debug {

/// Serialize OFDMSymbolParams to bytes using protobuf
void ProtoizeOFDMSymbolParams(ofdm::OFDMSymbolParams const &sp);
std::vector<char> SerializeFrame(ofdm::DFTSOFDMFrameParams const &frame);
void DumpFrameParams(std::string const &filename,
                     ofdm::DFTSOFDMFrameParams const &frame);

} // namespace debug
} // namespace bamradio

#endif // a9fcd474faa1fabf3611
