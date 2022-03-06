# test/CMakeLists.txt
#
# bamradio tests
#
# Copyright (c) 2018 Dennis Ogbe

###############################################################################
# SET-UP
###############################################################################

# bamradio tests. tests are defined in a source file in this directory and all
# link to the bamradio library. adding new tests is as easy as invoking the
# following macro
macro(define_bamradio_test NAME SOURCES)
add_executable(${NAME}-tests ${SOURCES})
add_test(NAME ${NAME} COMMAND ${NAME}-tests)
add_dependencies(${NAME}-tests bamradio-core)
add_dependencies(${NAME}-tests bamradio-lisp)
add_dependencies(${NAME}-tests bamradio-test-extra)
set_property(TARGET ${NAME}-tests
  PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${NAME}-tests
  bamradio-core bamradio-test-extra bamradio-lisp)
endmacro(define_bamradio_test)

# for extra test code that is not used in the core library, we compile another
# static library
set(BAMRADIO_TEST_EXTRA_SRCS
  # legacy phy
  legacy_phy.cc
  ../src/common.cc
  # common test stuff
  test_extra.cc
  )
add_library(bamradio-test-extra STATIC
  ${BAMRADIO_TEST_EXTRA_SRCS})
set_target_properties(bamradio-test-extra
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bamradio-test-extra
  PROPERTIES POSITION_INDEPENDENT_CODE ON)
# bamradio-core dependencies
target_link_libraries(
  bamradio-test-extra
  ${Boost_LIBRARIES}
  ${GNURADIO_ALL_LIBRARIES}
  ${LDPC_STATIC_LIBRARIES}
  ${LOG4CPP_LIBRARIES}
  ${UHD_LIBRARIES}
  ${VOLK_LIBRARIES}
  ${FFTW3F_LIBRARIES}
  ${FFTW3F_THREADS_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${GR_BAMFSK_LIBRARIES}
  ${ZEROMQ_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${ECL_LIBRARIES}
  -lpthread
  -lsqlite3
  -lcufft
  -lgps
  )

# as a (small) hack, we hardcode the path to the auto-generated protobuf
# headers here. This is only about headers. The compiled code we just pull from
# the static library.
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../src")
# we also want the header files from the src directory to be visible. makes
# this a little easier on everyone
include_directories("${CMAKE_CURRENT_LIST_DIR}/../src")
# for the FSK headers...
include_directories("${CMAKE_CURRENT_LIST_DIR}/../src/fsk/include")

###############################################################################
# TEST DEFINITIONS
###############################################################################

# control channel
define_bamradio_test(bamcc controlchannel.cc)
define_bamradio_test(bamradio-cc-tx cc_tx.cc)
# c2api
define_bamradio_test(c2api c2api.cc)
define_bamradio_test(bamradio-bootstrap bootstrap.cc)
# notification framework
define_bamradio_test(bamradio-notify notify.cc)
# framing
define_bamradio_test(bamradio-frame frame.cc)
define_bamradio_test(bamradio-frame2 frame2.cc)
define_bamradio_test(bamradio-arqseg arqseg.cc)
# gps
define_bamradio_test(bamradio-gps gps.cc)
# ofdm phy
define_bamradio_test(bamradio-ofdm-tx tx.cc)
define_bamradio_test(bamradio-sc-sync sc_sync.cc)
define_bamradio_test(bamradio-channelizer2 channelizer2.cc)
define_bamradio_test(bamradio-channelizer2-unit channelizer2_unit.cc)
define_bamradio_test(scalefac scalefac.cc)
define_bamradio_test(channelcode channelcode.cc)
# channelizer
define_bamradio_test(dsp-channelizer-benchmark channelizer_benchmark.cc)
define_bamradio_test(dsp-channelizer-poc channelizer_poc.cu)
# ring buffer
define_bamradio_test(dsp-ringbuffer ringbuffer.cu)
# statistics
define_bamradio_test(bamradio-statistics statistics.cc)
define_bamradio_test(bamradio-db db.cc)
# segment queue
define_bamradio_test(bamradio-segmentqueue segmentqueue.cc)
# individual mandate
define_bamradio_test(bamradio-im im.cc)
# watchdog (crash detect)
define_bamradio_test(bamradio-watchdog watchdog.cc)
# doomsday events
define_bamradio_test(bamradio-doomsday doomsday.cc)
# ccdata test with multiple nodes
define_bamradio_test(bamradio-cc-multinodes cc_multinodes.cc)
# freq alloc
# define_bamradio_test(bamradio-freq-alloc freq_alloc_test.cc)
# flow tracker
define_bamradio_test(bamradio-flowtracker flowtracker.cc)
# router
define_bamradio_test(bamradio-router router.cc)
# de
define_bamradio_test(bamradio-de de.cc)
# median filter
define_bamradio_test(bamradio-median median.cc)
# lisp
define_bamradio_test(bamradio-lisp lisp.cc)
# psd thresholding
define_bamradio_test(psd-thresh psd_thresh.cc)
