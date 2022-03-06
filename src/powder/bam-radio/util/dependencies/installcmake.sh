#!/usr/bin/env bash

set -e

[ $# -ge 3 ] || { echo "usage: $(basename $0) version patch prefix" >&2; exit 1; }

VERSION=$1
PATCH=$2
FVERS=$VERSION.$PATCH
PREFIX=$3

[ -d $PREFIX ] && { echo "prefix already exists"; sleep 5; }

WORKDIR="$(mktemp -d)"
function finish {
  rm -rf "$WORKDIR"
}
trap finish EXIT

cd "$WORKDIR"

curl -L "https://cmake.org/files/v$VERSION/cmake-$FVERS.tar.gz" | pax -r -z
cd cmake-$FVERS
./bootstrap --prefix=$PREFIX
make -j8
make install || { echo "make install failed. Is the prefix writable? Fix and run make install again."; bash; }
