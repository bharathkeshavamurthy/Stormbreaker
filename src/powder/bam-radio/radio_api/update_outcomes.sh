#!/usr/bin/env bash
# update_outcomes.sh - This script is called by the Colosseum to tell the radio to update the configured outcomes.
# No input is accepted.
# STDOUT and STDERR may be logged, but the exit status is always checked.
# The script should return 0 to signify successful execution.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source "$DIR/zmq.sh"
send_msg "$(</root/radio_api/mandated_outcomes.json)"

#this means it is running
echo "[`date`] Ran update_outcomes.sh" >> /logs/run.log
echo "[`date`] New mandated outcomes:" >> /logs/mo.log
cat /root/radio_api/mandated_outcomes.json >> /logs/mo.log
echo "" >> /logs/mo.log
exit 0
