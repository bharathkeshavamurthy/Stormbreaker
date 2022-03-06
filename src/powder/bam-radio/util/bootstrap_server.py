#!/usr/bin/env python3
#
# Server for bootstrapping experiment. Use with test/bootstrap.cc
#
# Copyright (c) 2018 Dennis Ogbe

import zmq
import argparse
import json
import importlib
import sys
import os
import time
import random


def parse_args():
    """Parse command line arguments and return options."""
    parser = argparse.ArgumentParser(description="Options")
    # IP and port
    parser.add_argument("-addr", type=str, help="IP address", required=True)
    parser.add_argument("-port", type=int, help="Port number", default=6666)
    parser.add_argument(
        "-cl-port", type=int, help="Client port number", default=7777)
    # number of rounds
    parser.add_argument(
        "-rounds", type=int, help="Number of rounds", required=True)
    # timeouts
    parser.add_argument(
        "-connection-timeout",
        type=float,
        help="Connection timeout",
        default=5000.0)
    parser.add_argument(
        "-synch-timeout",
        type=float,
        help="Synchronization timeout",
        default=100.0)
    # output file name
    parser.add_argument(
        "-output", type=str, help="Output file name", default="bs.json")
    # path to proto definition file
    parser.add_argument(
        "-proto",
        type=str,
        help="Path to proto definition file",
        required=True)
    # number of nodes
    parser.add_argument(
        "-num-nodes", type=int, help="Number of nodes", required=True)
    # random number seed
    parser.add_argument("-seed", type=int, help="RNG seed", default=33)
    return parser.parse_args()


class BootStrapServer():
    """The server for the BAM! Wireless Bootstrap environment"""

    def __init__(self, opts):
        """Initialize the server."""
        print("[Server] Starting Bootstrap Experiment Server with options:")
        print(opts)
        # variable init
        self.opts = opts
        self.data = list()
        self.ready_rx = list()
        self.ip2id = dict()
        self.last_control_ids = set()
        self.all_control_ids = set(range(0, 128))
        # ZMQ
        self.ctx = zmq.Context.instance()
        self.in_sock = self.ctx.socket(zmq.PULL)
        self.in_sock.bind("tcp://{:s}:{:d}".format(self.opts.addr,
                                                   self.opts.port))
        self.out_sock = dict()
        # load proto definitions
        pdir = os.path.dirname(os.path.abspath(self.opts.proto))
        sys.path.append(pdir)
        self.proto = importlib.import_module("debug_pb2")
        sys.path.remove(pdir)

    def ids(self):
        return self.out_sock.keys()

    def recv(self):
        """Blocking receive the next message"""
        rx = self.in_sock.recv()
        return self.proto.BSCtoS.FromString(rx)

    def send_start(self, control_id, id, control_ids):
        parent = self.proto.BSStoC()
        parent.start.your_id = id
        parent.start.your_control_id = control_id
        parent.start.all_control_ids[:] = control_ids
        buf = parent.SerializeToString()
        self.out_sock[id].send(buf)

    def send_stop(self, id):
        parent = self.proto.BSStoC()
        parent.stop.your_id = id
        buf = parent.SerializeToString()
        self.out_sock[id].send(buf)

    def send_ack(self, id):
        parent = self.proto.BSStoC()
        parent.ack.your_id = id
        buf = parent.SerializeToString()
        self.out_sock[id].send(buf)

    def wait_register(self):
        """Wait until all nodes are registered."""
        print("[Server] Waiting for {:d} nodes to register...".format(
            self.opts.num_nodes))
        tick = time.time()
        id = 0
        while ((time.time() - tick < self.opts.connection_timeout)
               and (len(self.out_sock) < self.opts.num_nodes)):
            # wait for incoming
            msg = self.recv()
            if msg.HasField("register"):
                # register this node
                ip = msg.register.my_ip
                if ip in self.ip2id:
                    self.send_ack(self.ip2id[ip])
                    continue
                sock = self.ctx.socket(zmq.PUSH)
                sock.connect("tcp://{:s}:{:d}".format(ip, self.opts.cl_port))
                self.out_sock[id] = sock
                self.send_ack(id)
                print("[Server] Registered node {:d} (addr: {:s})".format(
                    id, ip))
                self.ip2id[ip] = id
                id += 1
            elif msg.HasField("ready"):
                # add to ready list
                nid = msg.ready.my_id
                if nid not in self.ready_rx:
                    self.ready_rx.append(nid)
                    print("[Server] node {:d} is READY".format(nid))
                self.send_ack(nid)
            else:
                raise RuntimeError(
                    "I should be receiving either REGISTER or READY messages..."
                )
        if len(self.out_sock) == self.opts.num_nodes:
            print("[Server] Success. Registered {:d} nodes.".format(
                len(self.out_sock)))
        else:
            raise RuntimeError("Register timeout")

    def wait_ready(self):
        print("[Server] Waiting for {:d} nodes to get ready...".format(
            self.opts.num_nodes))
        while (len(self.ready_rx) < self.opts.num_nodes):
            msg = self.recv()
            if msg.HasField("ready"):
                id = msg.ready.my_id
                if id not in self.ready_rx:
                    self.ready_rx.append(id)
                    print("[Server] node {:d} is READY".format(id))
                self.send_ack(id)
            elif msg.HasField("register"):
                ip = msg.register.my_ip
                if ip in self.ip2id:
                    self.send_ack(self.ip2id[ip])
            else:
                raise RuntimeError(
                    "I should be receiving either REGISTER or READY messages..."
                )
        print("[Server] Success. {:d} nodes are in READY state.".format(
            self.opts.num_nodes))

    def generate_new_control_ids(self):
        new = random.sample(self.all_control_ids - self.last_control_ids,
                            len(self.ids()))
        self.last_control_ids = set(new)
        return new

    def run(self):
        """Run opts.rounds iterations of the experiment and save the results."""
        print("[Server] Running Bootstrap Experiment...")
        iteration = 0
        while iteration < self.opts.rounds:
            # wait for every node to be ready
            self.wait_ready()
            # clear the ready list for the next round
            self.ready_rx = list()
            # run this iteration
            print("[Server] Iteration {:d}".format(iteration))
            control_ids = self.generate_new_control_ids()
            id2control = {k: v for k, v in zip(self.ids(), control_ids)}
            itdata = list()
            # send start messages to all nodes
            for id in self.ids():
                self.send_start(id2control[id], id, control_ids)
            # receive incoming messages until I receive all "sync" messages
            resp_count = 0
            while resp_count < len(self.ids()):  # FIXME might add timeout here
                msg = self.recv()
                if msg.HasField("report"):
                    id = msg.report.my_id
                    time = msg.report.synch_time
                    itdata.append({
                        "id": id,
                        "control_id": id2control[id],
                        "synch_time": time
                    })
                    self.send_ack(id)
                    resp_count += 1
                elif msg.HasField("ready"):
                    id = msg.ready.my_id
                    self.ready_rx.append(id)
                    self.send_ack(id)
                else:
                    raise RuntimeError(
                        "I should be receiving either REPORT or READY messages..."
                    )
            # save data and move on to next iteration
            self.data.append(itdata)
            iteration += 1
        # we are done, send stop messages and shut down
        for id in self.ids():
            self.send_stop(id)
        self.save_data()
        print("[Server] Done running {:d} iterations".format(self.opts.rounds))

    def save_data(self):
        """Save the collected data."""
        print("[Server] Saving data...")
        with open(self.opts.output, "w") as f:
            json.dump(self.data, f)
        print("[Server] Saved data to {:s}".format(self.opts.output))


if __name__ == "__main__":
    opts = parse_args()
    random.seed(opts.seed)
    srv = BootStrapServer(opts)
    srv.wait_register()
    try:
        srv.run()
    except KeyboardInterrupt as e:
        srv.save_data()
