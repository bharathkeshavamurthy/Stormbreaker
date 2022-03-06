#!/usr/bin/env bash
# v0.5.0
# stop.sh - This script is called by the Colosseum to tell the radio the match is ending.
# No input is accepted.
# STDOUT and STDERR may be logged, but the exit status is always checked.
# The script should return 0 to signify successful execution.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source "$DIR/zmq.sh"

# for debug message
echo_time() {
  date +"%c $*"
}

#check if there is an input argument
case $# in
  0)   #zero inputs
    # logging
    echo_time "Stop command received" >> "/logs/c2api.log"
    # notify bam-radio
    send_msg "{\"BAM_INTERNAL_MAGIC\":\"DRAGON ENERGY\", \"CMD\": \"STOP\"}"
    # disable the systemd service
    systemctl stop bamradio
    # save systemd logs
    journalctl -u bamradio > /logs/bamradio-service.log
    journalctl -u bamradio-build > /logs/bamradio-build-service.log
    # copy logs
    cp /media/ramdisk/log_sqlite.db /logs/log_sqlite.db
    xz -0 --threads=0 /logs/log_sqlite.db
    #exit successfully
    exit 0
    ;;
  *)   #one or more input arguments (example error case)
    #example error exit code:
    exit 64 #exit with an error
    ;;
esac
