#!/usr/bin/env bash
# v0.5.0
# start.sh - This script is called by the Colosseum to tell the radio the match is starting.
# No input is accepted.
# STDOUT and STDERR may be logged, but the exit status is always checked.
# The script should return 0 to signify successful execution.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source "$DIR/zmq.sh"

# for debug message
echo_time() {
  date +"%c $*"
}

#check if there is an input argument for radio configuration
case $# in
  0)   #zero inputs
    #example default radio start:
    send_msg "{\"BAM_INTERNAL_MAGIC\":\"DRAGON ENERGY\", \"CMD\": \"START\"}"
    #logging
    echo_time "Start command received" >> "/logs/c2api.log"
    #exit successfully
    exit 0
    ;;
  1)   #one input argument
    #example using input as radio config specifier:
    #/path/to/radio/main /path/to/radio/configs/$1
    exit 64 #exit with an error - should not have gotten input
    ;;
  *)   #more than one input argument (example error case)
    #example error exit code:
    exit 64 #exit with an error
    ;;
esac
