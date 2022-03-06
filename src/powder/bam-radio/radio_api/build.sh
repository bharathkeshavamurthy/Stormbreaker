#!/bin/bash
#
# build the bam-radio code at SRN startup
#
# Dennis Ogbe <dogbe@purdue.edu>

# the code comes in as xz'd tarball named "/root/radio_api/radio.conf"
# containing the following directory structure:
#
# - bam-radio.tar -- the bam-radio source directory as we know it, stripped and tar'd
# - bam-radio.tar.sha256 -- sha256 sum of bam-radio.tar
# - radio.conf -- the actual radio configuration file we'd like to use
# - radio.conf.sha256 -- sha256 sum of radio.conf
# - ver -- plain text file containing the commit hash of the code

set -e

# constants and paths
BUNDLE="/root/radio_api/radio.conf"
TMPD=$(mktemp -d)
BAMRADIO_DIR="/root/bam-radio" # this is where we put the code
if [[ -d "$BAMRADIO_DIR" ]]; then
  rm -rf "$BAMRADIO_DIR";
fi
mkdir -p "$BAMRADIO_DIR"

# extract the bundle into a temp directory
tar -C "$TMPD" -xJf "$BUNDLE"

# verify checksums
if ! (cd "$TMPD" && sha256sum -c "bam-radio.tar.sha256"); then
  echo "WARNING (bam-radio.tar): SHA-256 does not match!!"
fi
if ! (cd "$TMPD" && sha256sum -c "radio.conf.sha256"); then
  echo "WARNING (radio.conf): SHA-256 does not match!!"
fi

# move everything where it needs to go
cp "$TMPD/ver" "/root/ver"
cp "$TMPD/radio.conf" "/root/radio.conf"
tar -C "$BAMRADIO_DIR" -xf "$TMPD/bam-radio.tar"
rm -rf "$TMPD"

# compile the code
mkdir -p "$BAMRADIO_DIR/build"
(cd "$BAMRADIO_DIR/build" && \
   export CUDACXX=/usr/local/cuda/bin/nvcc &&  \
   cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../controller \
   && make -j30 bamradio llr_gen)
# run the llr_gen script
mods=( 2 4 5 6 7 8 )
for order in "${mods[@]}"; do
	echo "Generating LLR table of order $order..."
	(cd "/root" && "$BAMRADIO_DIR/build/src/llr_gen" \
		              --precision=6 \
		              --npoints=20 \
		              --min=0 \
		              --max=40 \
		              --order=$order)
done
# make the symlink
ln -s "$BAMRADIO_DIR/build/src/bamradio" "/root/bamradio"
# copy the built binary to logs
cp "$BAMRADIO_DIR/build/src/bamradio" "/logs/bamradio"
# prepare for coredumps
rm -rf "/tmp/cores"
mkdir -p "/logs/cores"
ln -s "/logs/cores" "/tmp/cores"
# if we've made it until here, we are good
exit 0
