#!/bin/bash
#
# generate some build information. This file creates the build_info c++ header
# and source dependency
#
# Copyright (c) 2018 Dennis Ogbe

set -e

# (0) argument parsing. We assume that the working directory is the output directory!
[[ "$#" != 2 ]] && \
  echo "Need to specify output file and path to src/ directory" && \
  exit 1
OUTCC="$1"
BASEDIR=$(readlink -f "$2") # bam-radio/controller/src
CILDIR=$(cd "$BASEDIR"/../../cil && pwd)

# (1) get the commit hash and build time
if (cd "$BASEDIR" && git status 2>&1 >/dev/null); then
  COMMIT=$(cd "$BASEDIR" && git rev-parse HEAD)
elif  [[ -f "$HOME/ver" ]]; then
  # special case when building in container
  COMMIT=$(cat "$HOME/ver")
else
  COMMIT="(unknown commit)"
fi
TIME=$(date)

# (2) get the proto definitions
CILPROTO=$(sed -e 's/"/\\"/g' -e '/^ *$/d;s/.*/"&\\n"/' "$CILDIR"/proto/cil.proto)
REGPROTO=$(sed -e 's/"/\\"/g' -e '/^ *$/d;s/.*/"&\\n"/' "$CILDIR"/proto/registration.proto)
CCDATAPROTO=$(sed -e 's/"/\\"/g' -e '/^ *$/d;s/.*/"&\\n"/' "$BASEDIR"/cc_data.proto)
LOGPROTO=$(sed -e 's/"/\\"/g' -e '/^ *$/d;s/.*/"&\\n"/' "$BASEDIR"/log.proto)

# (3) write the output file
cat <<EOF > "$OUTCC"
#include "build_info.h"
namespace buildinfo {
const std::string commithash = "$COMMIT";
const std::string buildtime = "$TIME";
const std::string cilproto = $CILPROTO;
const std::string regproto = $REGPROTO;
const std::string ccdataproto = $CCDATAPROTO;
const std::string logproto = $LOGPROTO;
} // namespace buildinfo
EOF

# (4) copy the header file in the build directory
# This kind of sucks, but it works, oh well ¯\_(ツ)_/¯
HEADERFILE="$BASEDIR"/build_info.h
cp "$HEADERFILE" "$(pwd)"/.

exit 0
