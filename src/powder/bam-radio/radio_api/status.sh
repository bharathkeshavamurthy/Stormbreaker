#!/usr/bin/env bash
# v0.5.0
# status.sh - This script is called by the Colosseum to check on the radio state.
# No input is accepted.
# Output should be given by way of STDOUT as a serialized JSON dictionary.
# The dictionary must contain a STATUS key with a state string as a value.
# The dictionary may contain an INFO key with more detailed messages for the team.
# STDERR may be logged, but the exit status is always checked.
# The script should return 0 to signify successful execution.

# for debug message
echo_time() {
  date +"%c $*"
}

#check if there is an input argument for error exit example
if [ $# -ne 0 ]
then exit 64 #exit with an error
fi

STATE_DATA=$(<"/root/radio_api/status.txt")

#put the state in a serialized dictionary
OUTPUT="{\"STATUS\":\"$STATE_DATA\",\"INFO\":\"BAM! Wireless\"}"

#print to STDOUT
echo $OUTPUT

#logging
echo_time "Status check: $STATE_DATA" >> "/logs/c2api.log"

#exit good
exit 0
