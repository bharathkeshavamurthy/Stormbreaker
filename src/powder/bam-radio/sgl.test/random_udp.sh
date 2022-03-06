#!/bin/bash
# random_udp.sh
#   Usage:
#     ./random_udp.sh <dest host> <dest port>

[ $# -ge 2 ] ||
  { echo "usage: $(basename $0) <dest host> <dest port>" >&2; exit 1; }

while true;
do
  # generate some random data between 50-250 characters
  nchars=1472 #$(($RANDOM%200 + 50))
  random="$(dd if=/dev/urandom bs=$nchars count=1 2> /dev/null)"

  # send the random chars by netcat to the host and port 
  echo -n $random | nc -w0 -u $1 $2 

  # determine a random time from 0.001s to 0.004s
  t=$(($RANDOM%3 + 1))
  sleep $(($t/5000))
done 

