########################################################################
# Project setup
########################################################################
# prevent in-tree build.
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Prevented in-tree build. This is bad practice.")
endif(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

cmake_minimum_required(VERSION 3.8)

project(bamradio LANGUAGES CXX CUDA)

enable_testing()

# export the compiler commands (-I,-Wall, etc.) for YouCompleteMe & rtags
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# we are modern gentlemen and ladies
set(CMAKE_CXX_STANDARD 14)

# compiler flags
# enable warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Werror")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

# we want position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

########################################################################
# Find libraries
########################################################################

find_package(Boost "1.58" REQUIRED COMPONENTS system program_options
	thread unit_test_framework filesystem serialization)

set(GR_REQUIRED_COMPONENTS RUNTIME BLOCKS DIGITAL FFT UHD PMT FILTER ANALOG)
find_package(Gnuradio "3.8.0" REQUIRED)

find_package(UHD REQUIRED)
find_package(Log4cpp REQUIRED)
find_package(Volk REQUIRED)

find_package(FFTW3f REQUIRED)
find_library(FFTW3F_THREADS_LIBRARIES
  NAMES fftw3f_threads libfftw3f_threads
  HINTS ${PC_FFTW3F_LIBDIR})
if(FFTW3F_THREADS_LIBRARIES-NOTFOUND)
	message(FATAL_ERROR "Could not find FFTW3f thread library")
endif()

# https://cmake.org/cmake/help/v3.3/module/FindProtobuf.html
# https://cmake.org/cmake/help/v3.7/module/FindProtobuf.html
find_package(Protobuf REQUIRED)

find_package(ZeroMQ 4.0 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(GPSD REQUIRED libgps)
message(STATUS "GPSD_LIBRARIES: ${GPSD_LIBRARIES}")

# find Embeddable Common Lisp
include(src/ai/FindECL.cmake)

# this is still not optimal
find_package(CUDA REQUIRED)
# <2018-09-18 Tue> this is *still* not optimal. BUT, if, say, you wanted to use
# cuda-gdb to debug your cuda kernels, you need to add this line. of course
# YMMV, but i just don't know anymore...
#
# keep commented out unless you're single stepping CUDA code. and if you have
# to do that (debugging cuda kernels that is)... then godspeed.
#
# string(APPEND CMAKE_CUDA_FLAGS "-gencode arch=compute_60,code=sm_60, -g -G")

# Compression library LZ4
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)

# The CIL sources
set(CIL_DIR ${CMAKE_CURRENT_LIST_DIR}/../cil/proto
  CACHE PATH "Location of the CIL protocol definitions.")

########################################################################
# Setup the include and linker paths
########################################################################

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${LDPC_INCLUDE_DIR}
  ${GR_BAMFSK_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${GNURADIO_ALL_INCLUDE_DIRS}
  ${UHD_INCLUDE_DIRS}
  ${LOG4CPP_INCLUDE_DIRS}
  ${VOLK_INCLUDE_DIRS}
  ${FFTW3F_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIRS}
  ${ZeroMQ_INCLUDE_DIRS}
  ${GPSD_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${LZ4_INCLUDE_DIR}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${GNURADIO_ALL_LIBRARY_DIRS}
)

########################################################################
# subdirectories
########################################################################
# source files for the executable
add_definitions(-DBOOST_ASIO_DISABLE_EPOLL)
add_subdirectory(src)
add_subdirectory(test)
