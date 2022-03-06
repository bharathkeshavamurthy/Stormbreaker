#!/usr/bin/env bash
#
# send zmq messages using pyzmq
#
# Copyright (c) 2018 Dennis Ogbe

# see options.cc for hardcoded port number
send_msg() {
  PORT=9999
  echo "$1" | /usr/bin/python3 -c "import zmq, sys; \
ctx = zmq.Context(); \
sock = ctx.socket(zmq.PUSH); \
sock.connect(\"tcp://localhost:$PORT\"); \
sock.send_string(sys.stdin.read())"
}
