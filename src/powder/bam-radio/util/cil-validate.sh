#!/bin/bash

# this is a total kludge and thus no support or guarantees.

# Copyright (c) 2019 Dennis Ogbe

# the current version of the validation scenario is 7026
# the current version of the CIL tool is 3.4.1

# $1 is the path to a fetched scenario, i.e. ~/data/dl/RESERVATION-28391432
# you need a running docker CIL setup and everything. see [1]

# [1] https://gitlab.com/darpa-sc2-phase3/CIL/blob/master/doc/CIL-Validation-Procedure.md

set -e

SCENARIO_NUM="7026"
EVENT_TAG="3.4.1"
BAMW_IMG_NAME="bare"

if [[ $# != 1 ]]; then
  echo "Need the path to the reservation directory."
  exit 1
else
  DIRNAME=$(readlink -f "$1")
fi

WORKDIR="$(pwd)"/"$(basename $DIRNAME)"

# going off of [1]

# (3) download CIL PCAPs -- we assume we downloaded everything using our fetch tool into DIRNAME
LOG_DIR="$DIRNAME"

# find the PCAP file of our gateway
pfile=$(cd "$LOG_DIR" && \
          find . -name "*bamwireless-${BAMW_IMG_NAME}*colbr*" -exec du -b {} + \
            | sort -n -r | head -n1 | cut -f2)
pfile=${pfile#"./"}

# (4) download mandate files
MANDATE_SRC="colosseum:/share/nas/common/scenarios/$SCENARIO_NUM/Mandated_Outcomes/"
MANDATE_DIR="$WORKDIR/mandates"
if [[ ! -d "$MANDATE_DIR" ]]; then
  mkdir -p "$MANDATE_DIR"
  (cd "$MANDATE_DIR" && rsync -rav "$MANDATE_SRC" "$MANDATE_DIR")
fi

# (4.1) download env files
ENVIRONMENT_SRC="colosseum:/share/nas/common/scenarios/$SCENARIO_NUM/Environment/"
ENVIRONMENT_DIR="$WORKDIR/environment"
if [[ ! -d "$ENVIRONMENT_DIR" ]]; then
  mkdir -p "$ENVIRONMENT_DIR"
  (cd "$ENVIRONMENT_DIR" && rsync -rav "$ENVIRONMENT_SRC" "$ENVIRONMENT_DIR")
fi

# (5) traffic files
TRAFFIC_DIR="$DIRNAME"/traffic_logs

# (6) RF Start time
RF_START_TIME=$(cat "$DIRNAME"/"Inputs"/"rf_start_time.json" | cut -d'.' -f1)

# we are set up, run the darn thang
CILOUT="$WORKDIR/cil-val.txt"
SCOREOUT="$WORKDIR/score-val.txt"

# (7) validate CIL messages
docker run --rm -it \
       -v "$LOG_DIR":/common_logs \
       registry.gitlab.com/darpa-sc2-phase3/cil/cil-tool:"$EVENT_TAG" \
       ciltool cil-checker --src-auto \
       --startup-grace-period=20 --match-duration=630 \
       --match-start-time "$RF_START_TIME" \
       /common_logs/"$pfile" | tee "$CILOUT"

# (8) Validate Scoring -- this currently seems to crash
docker run --rm -it \
       -v "$LOG_DIR":/common_logs \
       -v "$MANDATE_DIR":/mandates \
       -v "$ENVIRONMENT_DIR":/environment \
       registry.gitlab.com/darpa-sc2-phase3/cil/cil-tool:"$EVENT_TAG" \
       ciltool perf-checker --src-auto \
       /common_logs/"$pfile" \
       --common-logs /common_logs --mandates /mandates --environment /environment \
  | tee "$SCOREOUT"

echo ""
echo "Done. Check output in"
echo "  $CILOUT"
echo "and"
echo "  $SCOREOUT"
