#!/usr/bin/env python3
#
# Combine the logs of one Colosseum run
#
# Copyright (c) 2018 Dennis Ogbe


import sqlite3
import os
import sys
import configparser
import re
import json
import pickle
import calendar
import argparse
import subprocess
from collections import OrderedDict
from io import StringIO
from functools import partial
from datetime import datetime

# flush stream after every call to print(...)
print = partial(print, flush=True)

#
# (-2) functions
#
def print_use():
    print("""
Usage: {:s} <data-dir>
    where <data-dir> points to the directory given to 'fetchall.sh'.
""".format(sys.argv[0]))


def getColumnDict(cursor: sqlite3.Cursor, table: str, keep_id: bool) -> OrderedDict:
    cols = cursor.execute("PRAGMA table_info('{:s}');".format(table)).fetchall()
    out = OrderedDict()
    for column in cols:
        colid, colname, coltype, notnull, default, pk = column
        if (colname == "id") and not keep_id:
            continue  # we don't need this one
        out[colname] = {
            "type": coltype,
            "notnull": True if notnull == 1 else False,
            "default": default,
            "pk": True if pk == 1 else False,
        }
    return out


# determine scenario
def scenario_name(id) -> str:
    if id is None:
        return "Unknown Scenario"
    scenario_map = {
        7013: "PE2 Alleys of Austin w/Points - 5 Team",
        7026: "PE2 Passive Incumbent w/Points - 5 Team",
        7047: "PE2 A Slice of Life w/Points - 5 Team",
        7065: "Payline (2 Stage)",
        7074: "PE2 Jammers w/Points - 5 Team",
        7087: "Wildfire w/Scores",
        8101: "Trash Compactor",
        8204: "Nowhere to Run Baby w/Scores",
        8302: "Active Incumbent w/Scores",
        8401: "Alleys of Austin Variant",
        8411: "Alleys of Austin Variant",
        8901: "Temperature Rising w/Scores, 12-50% threshold",
        8911: "Temperature Rising w/Scores, 62-100% threshold",
        9971: "-50dBFS Test",
        9972: "-70dBFS Test",
        9973: "-90dBFS Test",
        9988: "SCE Qualification"
    }

    deprecation_map = {
        2100: 7025,
        2157: 7025,
        2058: 7025,
        2059: 7025,
        7024: 7025,
        4057: 7046,
        4058: 7046,
        4059: 7046,
        7041: 7046,
        9502: 7013,
        9557: 7013,
        1058: 7013,
        7012: 7013,
        7071: 7073,
        7081: 7086,
        7085: 7086,
        8300: 8301,
        9901: 9971,
        9902: 9972,
        9903: 9973,
        9975: 9988,
        9977: 9988,
        9980: 9988,
        7025: 7026,
        7046: 7047,
        7064: 7065,
        7073: 7074,
        7086: 7087,
        8100: 8101,
        8203: 8204,
        8301: 8302,
        8400: 8401,
        8410: 8411,
        8910: 8911,
    }
    if id in scenario_map:
        return scenario_map[id]
    elif id in deprecation_map:
        return scenario_map[deprecation_map[id]]
    else:
        return "Unknown Scenario"


def get_restime(traffic_log_dir: str):
    # pull date and time of reservation (FIXME, see get_colosseum_rate(...))
    dt = None
    date_re = r"(?P<date>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}.\d{6})"
    tlogd = os.listdir(traffic_log_dir)
    if len(tlogd) > 0:
        with open(os.path.join(traffic_log_dir, tlogd[0]), "r") as f:
            m = re.search(date_re, f.readline())
            if m:
                dt = datetime.strptime(m.group("date"), '%Y-%m-%d_%H:%M:%S.%f')
    return dt


#
# (-1) output
#
print("BAM! Wireless log processor")


#
# (0) command line args
#

parser = argparse.ArgumentParser()
parser.add_argument("directory",
                    type=str,
                    help="Path to the directory to merge. This directory usually contains the RESERVATION-XXX folder. The merge script will write the full.db file here. Use fetchall.sh to create the correct directory structure."
                    )
parser.add_argument("-n", "--no-deliv-rate", action="store_true", help="Don't compute colosseum rate.")
parser.add_argument("-ns", "--no-scoring", action="store_true", help="Don't run scoring tool.")
args = parser.parse_args()
dpath = os.path.abspath(os.path.expanduser(args.directory))
do_colo_rate = not args.no_deliv_rate
do_scoring = not args.no_scoring
if not os.path.exists(dpath):
    print("data path does not exist.")
    exit(1)

outfile = os.path.join(dpath, "full.db")
srns = list()
srcfiles = list()
matchconf = None
batchinput = None
coloconf = None
config = None
traffic_log_dir = None
resid = None
restime = None
freeplay = False
for root, dirs, files in os.walk(dpath, followlinks=True):
    if "radio.conf" in files:
        config = configparser.ConfigParser(strict=False)
        try:
            config.read(os.path.join(root, "radio.conf"))
            dbname = config.get("global", "log_sqlite_path")
        except:
            # This usually means that this radio is not our SRN
            continue
        srcfiles.append(os.path.join(root, os.path.basename(dbname)))
        match = re.search("-srn([0-9]+)-", root)
        srns.append(int(match.group(1)))
    if "match_conf.json" in files:
        with open(os.path.join(root, "match_conf.json"), "r") as fp:
            matchconf = json.load(fp)
    if "freeplay.json" in files:
        with open(os.path.join(root, "freeplay.json"), "r") as fp:
            matchconf = json.load(fp)
        freeplay = True
    if "batch_input.json" in files:
        with open(os.path.join(root, "batch_input.json"), "r") as fp:
            batchinput = json.load(fp)
    if "colosseum_config.ini" in files:
        coloconf = configparser.ConfigParser(strict=False)
        coloconf.read(os.path.join(root, "colosseum_config.ini"))
    if "traffic_logs" in root:
        traffic_log_dir = os.path.abspath(root)
    # pull reservation id
    if "RESERVATION" in root:
        m = re.search("RESERVATION-(?P<resid>[0-9]+)", root)
        if m:
            resid = int(m.group("resid"))

restime = get_restime(traffic_log_dir)
assert(len(srns) == len(srcfiles))
assert(len(srns) > 0)
if not freeplay and matchconf is not None:
    if len(matchconf["node_to_srn_mapping"]) != len(srns):
        print("WARNING: No DB found for some SRNs!")
        print("  This is most likely because other Teams' SRNs participated in this reservation.")
        print("  Make sure to verify this yourself.")
else:
    pass  # TODO
# move old DB over
if os.path.exists(outfile):
    os.replace(outfile, outfile + ".old")

scenario_id = None
if batchinput is not None:
    scenario_id = batchinput["RFScenario"]
else:
    scenario_id = matchconf["RFScenario"]

#
# (1) prep. open the first table and gather some information
#
conn = sqlite3.connect(srcfiles[0])
cur = conn.cursor()
table_names = [r[0] for r in
               cur.execute(
                   "SELECT name FROM sqlite_master WHERE type=?;", ('table',)
               ).fetchall()]
columns = OrderedDict()
for name in table_names:
    columns[name] = getColumnDict(cur, name, False)
conn.commit()
conn.close()
print("Beginning merge of {:d} tables from {:d} SRNs..."
      .format(len(table_names), len(srns)))

#
# (2) create an output database
#
print("Creating output database {:s}...".format(outfile), end="")
outc = sqlite3.connect(outfile)
cout = outc.cursor()
# info tables
sinfo_name = "ScenarioInfo"
info_table_cols = OrderedDict()
info_table_cols["radioconf"] = "TEXT"
info_table_cols["coloconf"] = "TEXT"
info_table_cols["matchconf"] = "TEXT"
info_table_cols["batchinput"] = "TEXT"
info_table_cols["resID"] = "INT"
info_table_cols["date"] = "TEXT"
info_table_cols["ScenarioName"] = "TEXT"
info_table_cols["ScenarioID"] = "INT"
cmd = "CREATE TABLE IF NOT EXISTS {}({});".format(
    sinfo_name,
    ",".join("{} {}".format(k, v) for k, v in info_table_cols.items())
)
cout.execute(cmd)
cmd = "INSERT INTO {}({}) VALUES({})".format(
    sinfo_name, ",".join(str(k) for k in info_table_cols.keys()),
    ",".join(["?"] * len(info_table_cols)))

def get_ini(cp):
    sio = StringIO()
    cp.write(sio)
    st = sio.getvalue()
    sio.close()
    return st
vals = (
    get_ini(config),
    get_ini(coloconf),
    json.dumps(matchconf) if matchconf is not None else "None",
    json.dumps(batchinput) if batchinput is not None else "None",
    resid if resid is not None else 0,
    str(restime.ctime()) if restime is not None else "Unknown Reservation Time",
    scenario_name(scenario_id),
    scenario_id if scenario_id is not None else 0,
)
cout.execute(cmd, vals)
print("Done.")

# write some informational JSON for quick access
with open(os.path.join(dpath, "info.json"), "w") as jf:
    json.dump({
        "Reservation Number": resid,
        "Time": str(restime.ctime()),
        "Scenario ID": scenario_id if scenario_id is not None else "Unknown",
        "Scenario Name": scenario_name(scenario_id),
        "Freeplay": freeplay
    }, jf, indent=2)

#
# (3) uninteresting merges: for these, we simply add an SRN column
#
print("Merging tables...")
srnid_colname = "srnID"
skip = [
    "SentFrame",
    "DetectedFrame",
    "ReceivedFrame",
    "ReceivedBlock",
    "SentSegment",
    "ReceivedCompleteSegment",
    "ModulationEvent",
    "FlowQuantum"
]
conns = {}  # dictionary that holds DB connections
for table, cols in columns.items():
    if table in skip:
        continue
    print(" * Merging {:s} table...".format(table), end="")
    # put together the create_table ... string + SRN ID
    cmd = "CREATE TABLE IF NOT EXISTS {:s} (".format(table)
    cmd += ", ".join(["{:s} {:s}".format(k, v["type"]) for k, v in cols.items()])
    cmd += ", {} INT".format(srnid_colname)
    cmd += ", id INTEGER PRIMARY KEY"
    cmd += ")"
    cout.execute(cmd)
    # insert the "merged" data into the DB
    for srn, srcfile in zip(srns, srcfiles):
        try:
            conn = conns[srn]
        except KeyError:
            conn = sqlite3.connect(srcfile)
            conns[srn] = conn
        cur = conn.cursor()
        mycols = ", ".join(cols.keys())
        data = cur.execute("SELECT {} from {}".format(mycols, table))
        mynewcols = ", ".join([mycols, srnid_colname])
        ph = ",".join(["?"] * (len(cols.items()) + 1))
        cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(table, mynewcols, ph)
        cout.executemany(cmd, [d + (srn,) for d in data])
    print("Done.")
for srn, conn in conns.items():
    conn.commit()

#
# (4) Process Frame tables
#
# we create a temporary in-memory database for the "Frame" table with the
# frameID field. in the final output, we drop the frameid field.
print(" * Reading Frames...", end="")
tempdb = sqlite3.connect(":memory:")
tempcur = tempdb.cursor()
# now "merge" the SentFrame table into the temp database like above
cmd = "CREATE TABLE IF NOT EXISTS SentFrame ("
cmd += ", ".join(["{:s} {:s}".format(k, v['type'])
                  for k, v in columns["SentFrame"].items()])
cmd += ", id INTEGER PRIMARY KEY"
cmd += ")"
tempcur.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["SentFrame"].keys())
    data = cur.execute("SELECT {:s} from SentFrame".format(mycols))
    ph = ",".join(["?"] * len(columns["SentFrame"].items()))
    cmd = "INSERT INTO SentFrame({:s}) VALUES ({:s})".format(mycols, ph)
    tempcur.executemany(cmd, data)
tempdb.commit()
print("Done.")

# now keep a large dict mapping (srcNodeID,frameID) to the "id" field. this
# will become useful later.
print(" * Indexing Frames...", end="")
fcols = getColumnDict(tempcur, "SentFrame", True)
fdict = {}
data = tempcur.execute("SELECT * from SentFrame")
for raw_d in data:
    d = {k: v for k, v in zip(fcols.keys(), raw_d)}
    fdict[(d["srcNodeID"], d["frameID"])] = d["id"]
print("Done.")

# the "Frame" table contains all frames that were ever *created*. This is
# similar to the SentFrame table, but we drop the frameID field
print(" * Creating Frame table...", end="")
del fcols["frameID"]
del fcols["id"]
cmd = "CREATE TABLE IF NOT EXISTS Frame ("
cmd += ", ".join(["{:s} {:s}".format(k, v['type']) for k, v in fcols.items()])
cmd += ", id INTEGER PRIMARY KEY"
cmd += ")"
cout.execute(cmd)
mycols = ", ".join(fcols.keys())
data = tempcur.execute("SELECT {} from SentFrame".format(mycols))
ph = ",".join(["?"] * len(fcols.items()))
cmd = "INSERT INTO Frame({:s}) VALUES ({:s})".format(mycols, ph)
cout.executemany(cmd, data)
print("Done.")
for srn, conn in conns.items():
    conn.commit()

# the table "FrameDetect" contains all Detection events and references back to
# the 'Frame' table
print(" * Creating FrameDetect table...", end="")
fd_cols = ["snr", "channelIdx", "rxTime", "numBlocks", srnid_colname, "frame"]
fd_types = ["REAL", "INT", "INT", "INT", "INT", "INT NOT NULL"]
assert(len(fd_cols) == len(fd_types))
fd_name = "FrameDetect"
cmd = "CREATE TABLE IF NOT EXISTS {:s} (".format(fd_name)
cmd += ", ".join(" ".join(x) for x in zip(fd_cols, fd_types))
cmd += ", id INTEGER PRIMARY KEY"
cmd += ", FOREIGN KEY(frame) REFERENCES Frame(id)"
cmd += ")"
cout.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["DetectedFrame"].keys())
    data = cur.execute("SELECT {} from DetectedFrame".format(mycols))
    outdata = list()
    for i, raw_d in enumerate(data):
        # find the corresponding frame
        d = {k: v for k, v in zip(columns["DetectedFrame"].keys(), raw_d)}
        try:
            fid = fdict[(d['srcNodeID'], d['frameID'])]
            outdata.append((d["snr"], d["channelIdx"], d["rxTime"], d["numBlocks"], srn, fid))
        except KeyError:
            print("\nWARNING (FrameDetect) this should not happen! (fid not found): {}".format(d))
    # now write the data to the output database
    ph = ",".join(["?"] * len(fd_cols))
    mycols = ", ".join(fd_cols)
    cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(fd_name, mycols, ph)
    cout.executemany(cmd, outdata)
print("Done.")
for srn, conn in conns.items():
    conn.commit()

# the table "FrameRx" contains information about successful decode
print(" * Creating FrameRx table...", end="")
fr_cols = ["rxSuccess", "snr", "noiseVar", "time", srnid_colname, "frame"]
fr_types = ["INT", "REAL", "REAL", "INT", "INT", "INT NOT NULL"]
assert(len(fr_cols) == len(fr_types))
fr_name = "FrameRx"
cmd = "CREATE TABLE IF NOT EXISTS {:s} (".format(fr_name)
cmd += ", ".join(" ".join(x) for x in zip(fr_cols, fr_types))
cmd += ", id INTEGER PRIMARY KEY"
cmd += ", FOREIGN KEY(frame) REFERENCES Frame(id)"
cmd += ")"
cout.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["ReceivedFrame"].keys())
    data = cur.execute("SELECT {} from ReceivedFrame".format(mycols))
    outdata = list()
    for i, raw_d in enumerate(data):
        # find the corresponding frame
        d = {k: v for k, v in zip(columns["ReceivedFrame"].keys(), raw_d)}
        try:
            fid = fdict[(d['srcNodeID'], d['frameID'])]
            outdata.append((d["rxSuccess"], d["snr"], d["noiseVar"], d["time"], srn, fid))
        except KeyError:
            print("\nWARNING (FrameRx) this should not happen! (fid not found): {}".format(d))
    # now write the data to the output database
    ph = ",".join(["?"] * len(fr_cols))
    mycols = ", ".join(fr_cols)
    cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(fr_name, mycols, ph)
    cout.executemany(cmd, outdata)
print("Done.")
for srn, conn in conns.items():
    conn.commit()

#
# (5) Process Segment tables
#

# the "Segment" table contains all segments that were ever *created*. Similar
# to the frame table, we mangle the data to refer to the frame by its rowID
# instead of the (srcNodeID,frameID) pair
print(" * Reading Segments...", end="")
cmd = "CREATE TABLE IF NOT EXISTS SentSegment ("
cmd += ", ".join(["{:s} {:s}".format(k, v['type'])
                  for k, v in columns["SentSegment"].items()])
cmd += ", id INTEGER PRIMARY KEY"
cmd += ")"
tempcur.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["SentSegment"].keys())
    data = cur.execute("SELECT {:s} from SentSegment".format(mycols))
    ph = ",".join(["?"] * len(columns["SentSegment"].items()))
    cmd = "INSERT INTO SentSegment({:s}) VALUES ({:s})".format(mycols, ph)
    tempcur.executemany(cmd, data)
tempdb.commit()
print("Done.")

print(" * Indexing Segments...", end="")
scols = getColumnDict(tempcur, "SentSegment", True)
sdict = {}  # sequence number + srcNodeID + frameID maps to a Tx segment.
data = tempcur.execute("SELECT * from SentSegment")
for raw_d in data:
    d = {k: v for k, v in zip(scols.keys(), raw_d)}
    sdict[(d["srcNodeID"], d["sourceTime"])] = d["id"]
print("Done.")

print(" * Creating Segment table...", end="")
seg_cols = [
    "srcNodeID", "dstNodeID", "seqNum", "sourceTime", "description", "nbytes",
    "type", "frame"
]
seg_types = ["INT", "INT", "INT", "INT", "TEXT", "INT", "INT", "INT NOT NULL"]
assert(len(seg_cols) == len(seg_types))
seg_name = "Segment"
cmd = "CREATE TABLE IF NOT EXISTS {:s} (".format(seg_name)
cmd += ", ".join(" ".join(x) for x in zip(seg_cols, seg_types))
cmd += ", id INTEGER PRIMARY KEY"
cmd += ", FOREIGN KEY(frame) REFERENCES Frame(id)"
cmd += ")"
cout.execute(cmd)
mycols = ", ".join(scols.keys())
data = tempcur.execute("SELECT {:s} from SentSegment".format(mycols))
outdata = list()
for raw_d in data:
    d = {k: v for k, v in zip(scols.keys(), raw_d)}
    try:
        fid = fdict[(d["srcNodeID"], d["frameID"])]
        outdata.append(
            (d["srcNodeID"], d["dstNodeID"], d["seqNum"], d["sourceTime"],
             d["description"], d["nbytes"], d["type"], fid))
    except KeyError:
        print("\nWARNING (Segment) this should not happen! (fid not found): {}".format(d))
ph = ",".join(["?"] * len(seg_cols))
mycols = ", ".join(seg_cols)
cmd = "INSERT INTO Segment({:s}) VALUES ({:s})".format(mycols, ph)
cout.executemany(cmd, outdata)
print("Done.")

# the "SegmentRx" table contains all received segment events and refers back to
# the 'Segment' table, which refers back to the 'Frame' table
print(" * Creating SegmentRx table...", end="")
sr_cols = ["rxTime", "queueSuccess", srnid_colname, "segment"]
sr_types = ["INT", "INT", "INT", "INT NOT NULL"]
assert(len(sr_cols) == len(sr_types))
sr_name = "SegmentRx"
cmd = "CREATE TABLE IF NOT EXISTS {:s} (".format(sr_name)
cmd += ", ".join(" ".join(x) for x in zip(sr_cols, sr_types))
cmd += ", id INTEGER PRIMARY KEY"
cmd += ", FOREIGN KEY(segment) REFERENCES Segment(id)"
cmd += ")"
cout.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["ReceivedCompleteSegment"].keys())
    data = cur.execute("SELECT {:s} from ReceivedCompleteSegment".format(mycols))
    outdata = list()
    for raw_d in data:
        d = {k: v for k, v in zip(columns["ReceivedCompleteSegment"].keys(), raw_d)}
        try:
            sid = sdict[(d["srcNodeID"], d["sourceTime"])]
            outdata.append((d["rxTime"], d["queueSuccess"], srn, sid))
        except KeyError:
            print("\nWARNING (SegmentRx) this should not happen! (sid not found): {}".format(d))
    ph = ",".join(["?"] * len(sr_cols))
    mycols = ", ".join(sr_cols)
    cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(sr_name, mycols, ph)
    cout.executemany(cmd, outdata)
print("Done.")

#
# (6) Process ReceivedBlock tables
#
print(" * Creating BlockRx table...", end="")
br_cols = ["numBits", "isValid", "blockNumber", "time", "timeSteady", srnid_colname, "frame"]
br_types = ["INT", "INT", "INT", "INT", "INT", "INT", "INT NOT NULL"]
assert(len(br_cols) == len(br_types))
br_name = "BlockRx"
cmd = "CREATE TABLE IF NOT EXISTS {:s} (".format(br_name)
cmd += ", ".join(" ".join(x) for x in zip(br_cols, br_types))
cmd += ", id INTEGER PRIMARY KEY"
cmd += ", FOREIGN KEY(frame) REFERENCES Frame(id)"
cmd += ")"
cout.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["ReceivedBlock"].keys())
    data = cur.execute("SELECT {:s} from ReceivedBlock".format(mycols))
    outdata = list()
    for raw_d in data:
        d = {k: v for k, v in zip(columns["ReceivedBlock"].keys(), raw_d)}
        try:
            fid = fdict[(d["srcNodeID"], d["frameID"])]
            outdata.append((d["numBits"], d["valid"], d["blockNumber"],
                            d["time"], d["timeSteady"], srn, fid))
        except KeyError:
            print("\nWARNING (BlockRx) this should not happen! (fid not found): {}".format(d))
    ph = ",".join(["?"] * len(br_cols))
    mycols = ", ".join(br_cols)
    cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(br_name, mycols, ph)
    cout.executemany(cmd, outdata)
print("Done.")

#
# (7) Process ModulationEvent tables
#
print(" * Creating ModulationEvent table...", end="")
me_cols = ["t_code_ns", "t_mod_ns", "t_spread_ns", "t_map_ns", "t_shift_ns",
           "t_cp_ns", "t_mix_ns", "t_scale_ns", "t_stream_ns", "time",
           srnid_colname, "frame"]
me_types = ["INT"] * (len(me_cols) - 1) + ["INT NOT NULL"]
assert(len(me_cols) == len(me_types))
me_name = "ModulationEvent"
cmd = "CREATE TABLE IF NOT EXISTS {} (".format(me_name)
cmd += ", ".join(" ".join(x) for x in zip(me_cols, me_types))
cmd += ", id INTEGER PRIMARY KEY"
cmd += ", FOREIGN KEY(frame) REFERENCES Frame(id)"
cmd += ")"
cout.execute(cmd)
for srn in srns:
    conn = conns[srn]
    cur = conn.cursor()
    mycols = ", ".join(columns["ModulationEvent"].keys())
    data = cur.execute("SELECT {:s} from ModulationEvent".format(mycols))
    outdata = list()
    for raw_d in data:
        d = {k: v for k, v in zip(columns["ModulationEvent"].keys(), raw_d)}
        try:
            fid = fdict[(d["srcNodeID"], d["frameID"])]
            outdata.append(tuple(d[col] for col in me_cols[:-2]) + (srn, fid,))
        except KeyError:
            print("\nWARNING (ModulationEvent) this should not happen! (fid not found): {}".format(d))
    ph = ",".join(["?"] * len(me_cols))
    mycols = ", ".join(me_cols)
    cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(me_name, mycols, ph)
    cout.executemany(cmd, outdata)
print("Done.")

#
# (8) Process traffic logs
#

# this is a modified version of Diyu Yang's code to plot the colosseum rate
def get_colosseum_rate(srns, path, tstart, tend, winsize) -> dict:
    # helper functions
    print('our srns: ',srns)
    def utc(rawtime):
        """convert input timestamp into unix time in millisec"""
        datetime1 = datetime.strptime(rawtime, '%Y-%m-%d_%H:%M:%S.%f')
        timeInSeconds = calendar.timegm(datetime1.utctimetuple())
        return timeInSeconds * 1000

    def filt(a, b, func):
        return [b[i] for (i, val) in enumerate(a) if func(val)]
    # parse all files
    in_dict = dict()
    date_re = r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}.\d{6}"
    pattern_f = re.compile("flow>(\d{4})")
    srn_f = re.compile("dst>\d{3}.\d{3}.(\d{3})")
    rec_p = re.compile("({:s}) RECV".format(date_re))
    sent_p = re.compile("sent>({:s})".format(date_re))
    size_p = re.compile("size>(\d{3})")
    for filename in os.listdir(path):
        if "listen" not in filename:
            continue
        with open(os.path.join(path, filename), "r") as f:
            for line in f:
                srn = srn_f.findall(line)
                if len(srn) is 0:
                    continue
                srn = int(srn[0])-100
                if srn not in srns:
                    continue
                flow_list = pattern_f.findall(line)
                time_sent_list = sent_p.findall(line)
                time_rec_list = rec_p.findall(line)
                size_list = size_p.findall(line)
                if len(flow_list) == 0:
                    continue
                flow_num = int(flow_list[0])
                t_sent = utc(time_sent_list[0])
                t_rec = utc(time_rec_list[0])
                pac_size = int(size_list[0])
                if flow_num not in in_dict:
                    in_dict[flow_num] = {
                        "time_rec": list(),
                        "time_send": list(),
                        "pac_size": list()
                    }
                in_dict[flow_num]["time_rec"].append(t_rec)
                in_dict[flow_num]["time_send"].append(t_sent)
                in_dict[flow_num]["pac_size"].append(pac_size)

    # compute output dictionary
    out_dict = dict()
    winspeed = int((tend-tstart)/1000)
    time = list(range(0, int(tend - tstart), winspeed))
    rate = [0] * len(time)
    for k in in_dict.keys():
        out_dict[k] = {"time": list(), "rate": list()}
        cur_win = in_dict[k]["time_rec"]
        cur_pac_size = in_dict[k]["pac_size"]
        for t in range(int(tstart), int(tend), winspeed):
            out_dict[k]["time"].append(t-tstart)
            num_pack = float(sum(filt(cur_win, cur_pac_size,
                                      lambda x: ((x >= t) and (x < t + winsize)))) * 8 / 1000)
            out_dict[k]["rate"].append(float(num_pack / winsize))
        rate = [a + b for a, b in zip(rate, out_dict[k]["rate"])]

    # save the total throughput as special case. then we are done
    out_dict[-1] = {"time": time, "rate": rate}
    return out_dict

# get some constants for the above function
def get_start_time():
    cout.execute("select MIN(time) from C2APIEvent where txt='START'")
    return cout.fetchone()[0] / 1000000
def get_stop_time():
    cout.execute("select MAX(time) from C2APIEvent where txt='STOP'")
    return cout.fetchone()[0] / 1000000

if do_colo_rate:
    print(" * Processing Colosseum Traffic Logs...", end="")
    tstart = get_start_time()
    tstop = get_stop_time()
    wsize = 5000  # FIXME hardcoded

    # first create the Colosseum throughput table and set up the input command
    th_table_name = "ColosseumRate"
    th_table_cols = OrderedDict([
        ("flowID", "INT"),
        ("data", "BLOB")
    ])
    cmd = "CREATE TABLE IF NOT EXISTS {}({});".format(
        th_table_name,
        ",".join("{} {}".format(k, v) for k, v in th_table_cols.items())
    )
    cout.execute(cmd)
    cmd = "INSERT INTO {}({}) VALUES({})".format(
        th_table_name, ",".join(str(k) for k in th_table_cols.keys()),
        ",".join(["?"] * len(th_table_cols)))
    # now insert the computed throughput curves
    crate = get_colosseum_rate(srns, traffic_log_dir, tstart, tstop, wsize)
    cout.executemany(cmd, [(k, sqlite3.Binary(pickle.dumps(v))) for k, v in crate.items()])
    print("Done.")

#
# (9) scoring tool
#
# Assumes that 'scoringtool' has been installed. For installation instructions see
# https://gitlab.com/darpa-sc2-phase3/CIL/tree/master/tools/scoringtool
#
if do_scoring:
    dirs = os.listdir(dpath)
    for dir in dirs:
        if "RESERVATION" in dir:
            resname = dir
            break
    output = subprocess.check_output([
        "scoringtool", "scoring-checker",
        "--common-logs", os.path.join(dpath, resname),
        "--mandates", os.path.join(dpath, str(scenario_id), "Mandated_Outcomes"),
        "--environment", os.path.join(dpath, str(scenario_id), "Environment"),
        "--output-format", "darpa"])
    all_scores = json.loads(output.decode("utf-8"))
    # save json
    with open(os.path.join(dpath, "score.json"), "w") as jf:
        json.dump(all_scores, jf, indent=2)
    # save results to db
    print(" * Creating MatchScores table...", end="")
    ms_cols = ["mp_index", "team_id", "mp_score"]
    ms_types = ["INT", "TEXT", "INT"]
    assert(len(ms_cols) == len(ms_types))
    ms_name = "MatchScores"
    cmd = "CREATE TABLE IF NOT EXISTS {} (".format(ms_name)
    cmd += ", ".join(" ".join(x) for x in zip(ms_cols, ms_types))
    cmd += ", id INTEGER PRIMARY KEY"
    cmd += ")"
    cout.execute(cmd)
    gw_dict = dict()
    teamscore_dict = dict()
    if freeplay:
        # get gateway SRNID and individual scores
        for node in matchconf["NodeData"]:
            if node["isGateway"] == True:
                gw_dict[node["team_no"]] = node["RFNode_ID"]
        for teamname, scores in all_scores.items():
            if type(scores) is dict and "IndividualMatchScore" in scores:
                teamscore_dict[int(teamname)] = scores["IndividualMatchScore"]

    outdata = list()
    for teamname, scores in all_scores.items():
        if type(scores) is dict and "IndividualMPScore" in scores:
            idx = 0
            for score in scores["IndividualMPScore"]:
                idx += 1
                if freeplay:
                    teamname_str = "Team {} (Score:{})".format(teamname, teamscore_dict[int(teamname)])
                else:
                    teamname_str = "Team {}".format(teamname)
                outdata.append((idx, teamname_str, score))
    ph = ",".join(["?"] * len(ms_cols))
    mycols = ", ".join(ms_cols)
    cmd = "INSERT INTO {:s}({:s}) VALUES ({:s})".format(ms_name, mycols, ph)
    cout.executemany(cmd, outdata)
    print("Done.")

#
# (end) close all database connections
#
print("Finishing up...", end="")
for srn, conn in conns.items():
    conn.commit()
    conn.close()
outc.commit()
outc.close()
tempdb.close()
print("Done.")

# that's all folks
print("Wrote output to {:s}.".format(outfile))
