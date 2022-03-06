# ai/CMakelists.txt
#
# build the ai library.
#
# compiles the LISP code + dependencies to static libraries that need to be
# linked to the final executable
#
# Copyright (c) 2019 Dennis Ogbe

# make sure that this lines up with the LISP package name
set(LISP_PACKAGE_NAME "bam-radio")

# all LISP sources, just like the usual
set(LISP_SOURCES
  "package.lisp"
  "time.lisp"
  "debug.lisp"
  "data.lisp"
  "decision-engine.lisp"
  )

# copy the lisp sources to the binary dir
set(LISP_SYSTEM_DEFINITION "bam-radio.asd")
set(LISP_COMPILE_SCRIPT "compile.lisp")
foreach(FILENAME ${LISP_SOURCES} ${LISP_SYSTEM_DEFINITION} ${LISP_COMPILE_SCRIPT})
  configure_file(${FILENAME} ${FILENAME} COPYONLY)
endforeach(FILENAME)

# copy the dependencies to the binary dir
file(COPY "lisp-deps" DESTINATION .)

# add custom command + target that compiles the code using the ECL compiler
set(LISP_STATIC_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LISP_PACKAGE_NAME}--all-systems.a)
add_custom_target(compile-lisp
  COMMAND ecl -norc --load ${LISP_COMPILE_SCRIPT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

# create an "imported" library target
add_library(bamradio-lisp STATIC IMPORTED GLOBAL)
add_dependencies(bamradio-lisp compile-lisp)
set_property(TARGET bamradio-lisp PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# tell the build system about the location of the file
set_property(TARGET bamradio-lisp
   PROPERTY IMPORTED_LOCATION ${LISP_STATIC_LIB_FILE})
