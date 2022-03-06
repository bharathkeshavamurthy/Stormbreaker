#!/bin/bash
# download all logs of a reservation. this assumes that .ssh/config has
# "sc2-lz" properly set-up. Uses rsync.
# Copyright (c) 2018 Dennis Ogbe
set -e
# parse args
print_usage() {
  echo "Usage: $0 <reservation ID> <output directory>"
}
[[ "$#" -ne "2" ]] && print_usage && exit 1
RESID="$1"
OUTDIR=$(readlink -f "$2")
# get candidates
readarray FOLDERS < <(ssh sc2-lz "find /share/nas/bamwireless" -type 'd' -maxdepth 1 | awk -v "resid=$RESID" '$0 ~ resid')
echo $FOLDERS
[[ -z "$FOLDERS" ]] && \
  echo "Reservation $RESID not found. Aborting" && \
  exit 1
# make output
mkdir -p "$OUTDIR"
# download all files
for FOLDER in "${FOLDERS[@]}"; do
  rsync -ravz "sc2-lz:$FOLDER" "$OUTDIR"
done
# download corresponding colosseum mandates and environment files
for jsonName in freeplay batch_input; do
  fileName=$(find "$OUTDIR" -name "${jsonName}.json")
  if [ -e "$fileName" ]; then
    rfScenario=$(cat "$fileName" | python3 -c "import sys, json; print(json.load(sys.stdin)['RFScenario'])")
    rsync -raz "sc2-lz:/share/nas/common/scenarios/${rfScenario}" "$OUTDIR"
    break
  fi
done
# decompress compressed files
if ! [[ -x "$(command -v unxz)" ]]; then
  echo 'Error: unxz command is not installed. Aborting.'
  exit 1
fi
find "$OUTDIR" -name '*.xz' -execdir unxz -v --threads=0 '{}' ';'
chmod a+x -R "$OUTDIR"
echo "$OUTDIR"
