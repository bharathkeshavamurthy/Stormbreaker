# compile all of bamradio into a static library that we then re-use for the
# main executable as well as the tests
set(BAMRADIO_SRCS
  # PHY
  "bam_constellation.cc"
  "ofdm.cc"
  "phy.cc"
  "radiocontroller.cc"
  "radiocontroller_types.cc"
  "mcs.cc"
  "mcsv1.cc"
  "adaptive_mcs_controller.cc"
  "bamfftw.cc"
  "bandwidth.cc"
  "buffers.cc"
  # CUDA
  "bamcufft.cu"
  "channelizer.cu"
  "channelizer2.cu"
  "sc_sync.cu"
  # CONTROLCHANNEL
  "cc_data.cc"
  "cc_controller.cc"
  "ctrl_ch.cc"
  "ctrl_recv.cc"
  "networkmap.cc"
  "fsk/src/rs_ccsds.cc"
  # DLL
  "dll.cc"
  "frame.cc"
  "arq.cc"
  "segment.cc"
  "segmentqueue.cc"
  "arqbuffer.cc"
  "dll_types.cc"
  "flowtracker.cc"
  # NET
  "tun.cc"
  "ippacket.cc"
  "interface.cc"
  "net.cc"
  "router.cc"
  # SENSING
  "psd.cc"
  # DECISION ENGINE/COLLAB
  "collab.cc"
  "de.cc"
  "discrete_channels.cc"
  # MISC
  "options.cc"
  "notify.cc"
  "c2api.cc"
  "common.cc"
  "util.cc"
  "gps.cc"
  "statistics.cc"
  "debug.cc"
  "log.cc"
  "events.cc"
  "watchdog.cc"
  "build_info.cc"
  "im.cc"
  )

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR} # needed for protobuf
  ${CUDA_INCLUDE_DIRS}
  )

# generate proto files
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  ${CIL_DIR}/cil.proto
  ${CIL_DIR}/registration.proto
  ${CMAKE_CURRENT_LIST_DIR}/cc_data.proto
  ${CMAKE_CURRENT_LIST_DIR}/debug.proto
  ${CMAKE_CURRENT_LIST_DIR}/log.proto
  ${CMAKE_CURRENT_LIST_DIR}/psd.proto
  )

# generate build info
add_custom_command(
  OUTPUT "build_info.cc"
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/../../util/gen_build_info.sh build_info.cc ${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Generating Build Information"
  VERBATIM
  )

# CMakeListstxt in ai/ builds a static library from our LISP code
add_subdirectory("ai")

# the (legacy) GNU Radio FSK control channel receiver
include_directories("fsk/include")
include_directories("fsk/src")
set(LEGACY_FSK_SRCS
  "fsk/src/access_code_detector_ff_impl.cc"
  "fsk/src/access_code_detector_impl.cc"
  "fsk/src/crc32_bb_impl.cc"
  "fsk/src/drop_header_bb_impl.cc"
  "fsk/src/insertPreamble_bb_impl.cc"
  "fsk/src/mfskMod_fc_impl.cc"
  "fsk/src/mfsk_rx_impl.cc"
  "fsk/src/ndaSync_ff_impl.cc"
  "fsk/src/qa_bamfsk.cc"
  "fsk/src/qa_tsb_chunks_to_symbols_bf.cc"
  "fsk/src/rs_ccsds.cc"
  "fsk/src/rs_ccsds_decode_bb_impl.cc"
  "fsk/src/rs_ccsds_encode_bb_impl.cc"
  "fsk/src/stream_to_tagged_stream_impl.cc"
  "fsk/src/test_bamfsk.cc"
  "fsk/src/tsb_chunks_to_symbols_bf_impl.cc"
  )

# our channel code
set(LDPC_SRCS
  "ldpc/alt_encoder.cpp"
  "ldpc/csr_gf2matrix.cpp"
  "ldpc/dd_encoder.cpp"
  "ldpc/flooding_decoder.cpp"
  "ldpc/gf2matrix.cpp"
  "ldpc/IEEE802_codes.cpp"
  "ldpc/minstar_decoder.cpp"
  "ldpc/minsum_decoder.cpp"
  "ldpc/serial_c_decoder.cpp"
  "ldpc/serial_c_minsum.cpp"
  "ldpc/sumproduct_decoder.cpp"
  "ldpc/systematic_encoder.cpp"
  )

# compile all dependencies into a static library
add_library(bamradio-core STATIC
  ${BAMRADIO_SRCS}
  ${LEGACY_FSK_SRCS}
  ${LDPC_SRCS}
  ${PROTO_SRCS}
  ${PROTO_HDRS})

set_target_properties(bamradio-core
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bamradio-core
  PROPERTIES POSITION_INDEPENDENT_CODE ON)

# bamradio-core dependencies
target_link_libraries(
  bamradio-core
  ${Boost_LIBRARIES}
  ${GNURADIO_ALL_LIBRARIES}
  ${LOG4CPP_LIBRARIES}
  ${UHD_LIBRARIES}
  ${VOLK_LIBRARIES}
  ${FFTW3F_LIBRARIES}
  ${FFTW3F_THREADS_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${ZEROMQ_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${LZ4_LIBRARY}
  ${ECL_LIBRARIES}
  -lpthread
  -lsqlite3
  -lcufft
  -lgps
  )

# compile the bamradio executable
add_executable(bamradio "bamradio.cc")
set_property(TARGET bamradio
  PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bamradio bamradio-core bamradio-lisp)
add_dependencies(bamradio bamradio-core)
install(TARGETS bamradio RUNTIME DESTINATION bin)

#
# misc helper programs
#

# a helper program to generate LLR wisdom
add_executable(llr_gen "llr_gen.cc")
target_link_libraries(llr_gen
  bamradio-core
  ${Boost_LIBRARIES}
  ${GNURADIO_ALL_LIBRARIES}
  )
install(TARGETS llr_gen RUNTIME DESTINATION bin)
