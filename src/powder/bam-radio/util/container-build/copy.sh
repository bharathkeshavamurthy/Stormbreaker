#!/bin/bash

#
# skippy a file to the colosseum
#

[[ "$#" != 2 ]] && \
    echo "Usage: $0 <file> <dest-path>" && \
    echo "" && \
    echo "Where <dest-path> is anything after /share/nas/bamwireless" && \
    exit 1

! [[ -f "$1" ]] && \
    echo "File no found: $1" && \
    exit 1

scp -r "$1" sc2-lz:/share/nas/bamwireless"$2"
