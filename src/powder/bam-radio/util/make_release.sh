#!/bin/bash
#
# package a specific commit or branch into a bam-radio release for colosseum.
#
# Dennis Ogbe <dogbe@purdue.edu>

# this script takes two optional inputs, one mandatory input, and produces one
# output file. see the print_usage function for more inputs

DEBUG=false

# bounce when bad things happen
set -e

# this is how to use this script
print_usage() {
  echo "Usage: $0 -i <branch-or-commit> [-c <config-file>] [-n <name>]"
  echo ""
  echo "  The required -i switch determines the branch or commit to pull from"
  echo "  The optionsl -n switch lets you give this release a name"
  echo "  The optional -c switch lets you specify a config file. Default is radio-default.conf"
  echo ""
  echo "  Example: $0 -i master -c /home/user/radio-test.conf -o mytest"
  echo "  Will produce the file $workdir/mytest-$commithash.tar.xz"
}

# parse command line
BRANCH=""
CONFIG=""
NAME=""
while [[ "$1" != "" ]]; do
  case "$1" in
    "-i" | "--input")
      shift
      BRANCH="$1"
      ;;
    "-c" | "--config")
      shift
      CONFIG="$1"
      ;;
    "-n" | "--name")
      shift
      NAME="$1"
      ;;
    "-h" | "--help")
      print_usage
      exit 0
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
  shift
done

# check required arguments
if [[ -z "$BRANCH" ]]; then
  echo "ERROR: BRANCH not specified (-i). Aborting"
  echo ""
  print_usage
  exit 1
fi

# check whether config file exists (if specd)
if ! [[ "$CONFIG" = "" ]]; then
  if ! [[ -r "$CONFIG" ]]; then
    echo "ERROR: Cannot read $CONFIG. Aborting."
    exit 1
  fi
fi

# debug output
if [[ $DEBUG = true ]]; then
  echo "BRANCH: $BRANCH"
  echo "CONFIG: $CONFIG"
  echo "NAME: $NAME"
fi

# make temporary workspaces
REPO=$(mktemp -d)
ARCHIVE=$(mktemp -d)

# clone bam-radio into temp & massage contents
BAM_RADIO_REPO="ssh://git@cloudradio.ecn.purdue.edu/bam-wireless/bam-radio.git" # change if this changes
git clone "$BAM_RADIO_REPO" "$REPO"
(cd "$REPO" && git checkout "$BRANCH" && git submodule init && git submodule update)
(cd "$REPO/controller/src/ai/lisp-deps" && git submodule init && git submodule update)
COMMIT=$(cd "$REPO" && git rev-parse HEAD)
# copy the config file out
if [[ "$CONFIG" = "" ]]; then
  cp "$REPO/radio_api/radio_default.conf" "$ARCHIVE/radio.conf"
else
  cp "$CONFIG" "$ARCHIVE/radio.conf"
fi
# delete stuff we don't need before packaging
(cd "$REPO" && \
   rm -rf .git .gitignore .gitmodules && \
   rm -rf radio-api) # this already exists in the bare image

# tar, hash, add version info
(cd "$REPO" && tar -cf "$ARCHIVE/bam-radio.tar" *)
(cd "$ARCHIVE" && sha256sum "bam-radio.tar" > "bam-radio.tar.sha256")
(cd "$ARCHIVE" && sha256sum "radio.conf" > "radio.conf.sha256")
echo "$COMMIT" > "$ARCHIVE/ver"

# package it up and write output
if [[ "$NAME" = "" ]]; then
  OUTNAME="$PWD/bam-radio-$(echo -n $COMMIT | cut -b-12).tar.xz"
else
  OUTNAME="$PWD/$NAME-$(echo -n $COMMIT | cut -b-12).tar.xz"
fi
(cd "$ARCHIVE" && tar -cJf "$OUTNAME" *)
echo "Wrote output to $OUTNAME."

# clean up
if [[ $DEBUG = true ]]; then
  echo "REPO: $REPO"
  echo "ARCHIVE: $ARCHIVE"
else
  rm -rf "$REPO"
  rm -rf "$ARCHIVE"
fi

# fin
