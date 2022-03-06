#!/usr/bin/env bash

set -e

[ $# -ge 4 ] || { echo "usage: $(basename $0) vM vm vp prefix" >&2; exit 1; }

PREFIX=$4

[ -d $PREFIX ] && { echo "prefix already exists"; sleep 5; }

WORKDIR="$(mktemp -d)"
function finish {
  rm -rf "$WORKDIR"
}
trap finish EXIT

cd "$WORKDIR"

curl -L "https://dl.bintray.com/boostorg/release/$1.$2.$3/source/boost_$1_$2_$3.tar.bz2" | pax -r -j
mkdir build
cd boost_$1_$2_$3
echo "using gcc : 6 : /usr/bin/g++-6 ; " >> tools/build/src/user-config.jam
./bootstrap.sh --prefix=$PREFIX
./b2 --prefix=$PREFIX --build-dir=$(pwd)/../build --layout=tagged -d+2 install
