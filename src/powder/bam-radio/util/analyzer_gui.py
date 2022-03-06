#!/usr/bin/env python3

# Log Analyzer GUI
# Copyright (c) 2018 Tomohiro Arakawa
import _pickle as cPickle
import json
from pylab import frange
import re
import math
import contextlib
import sys
import sqlite3
import numpy as np
import csv
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import subprocess
from distutils.spawn import find_executable
import pathlib
import tempfile
import importlib
import copy
from ipaddress import ip_address
import inspect
from configparser import ConfigParser
from collections import OrderedDict, namedtuple
import json
import pickle
import re
import socket
import struct

try:
    from qtconsole.rich_ipython_widget import RichJupyterWidget
    from qtconsole.inprocess import QtInProcessKernelManager
    from IPython.lib import guisupport
    consoleReady = True
except ImportError as e:
    print("WARNING: Could not import qtconsole")
    consoleReady = False

# Global variables
phy_sample_rate = 46.08e6

class PHYLinkPlot(QWidget):
    def __init__(self, analyzer, tabwidget, ntab):
        QWidget.__init__(self)

        self.hbox = QHBoxLayout()
        self.lvbox = QVBoxLayout()
        rvbox = QVBoxLayout()

        # Link list
        self.linkList = QStandardItemModel()
        self.linkListWidget = QListView()
        self.linkListWidget.setModel(self.linkList)
        self.lvbox.addWidget(QLabel("Links"))
        self.lvbox.addWidget(self.linkListWidget)

        # Plot pane
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        rvbox.addWidget(toolbar)
        rvbox.addWidget(self.canvas)

        # plumbing
        lvboxWidget = QWidget()
        lvboxWidget.setLayout(self.lvbox)
        lvboxWidget.setMaximumWidth(150)
        rvboxWidget = QWidget()
        rvboxWidget.setLayout(rvbox)
        self.hbox.addWidget(lvboxWidget)
        self.hbox.addWidget(rvboxWidget)
        self.setLayout(self.hbox)

        # variables
        self.analyzer = analyzer
        self.tabIndex = ntab
        # connect slots
        self.connectRefresh(tabwidget)

    def connectRefresh(self, tabwidget):
        tabwidget.currentChanged.connect(self.refreshOnTabChange)
        self.linkList.itemChanged.connect(self.refreshView)

    def refreshOnTabChange(self, index):
        if self.tabIndex == index:
            self.refreshView()

    def refreshView(self, *args, **kwargs):
        if self.dataReady():
            self.refreshPlot()
        else:
            if self.isConnected():
                if self.loadData():
                    self.refreshView()

    def isConnected(self):
        return self.analyzer.conn is not None

    def dataReady(self):
        raise NotImplementedError()

    def refreshPlot(self):
        raise NotImplementedError()

    def loadData(self):
        raise NotImplementedError()

# Frame detection and Error rate ##############################################

# a "Link" is a srcNodeID, dstNodeID pair
PHYLink = namedtuple("PHYLink", ["srcNodeID", "dstNodeID"])
FrameTx = namedtuple("FrameTx", ["id", "txTime", "detected", "rxSuccess"])
BlockTx= namedtuple("BlockTx", ["txTime", "valid"])

def PHYLink2Str(link):
    return "{:d} => {:d}".format(link.srcNodeID, link.dstNodeID)

# helper class to compute error and detection rates for frames and segments
class FERCompute(object):
    def __init__(self, analyzer):
        # database connection
        self.analyzer = analyzer
        # stuff
        self.links = []  # a "link" here is a (src, dst) pair
        self.start_time_ms = self.analyzer.get_start_time()
        self.stop_time_ms = self.analyzer.get_stop_time()
        # set up caches
        self.frames = None
        pass

    def get_links(self):
        cur = self.analyzer.conn.cursor()
        links = cur.execute("select srcNodeID,dstNodeID from Frame group by srcNodeID,dstNodeID")
        self.links = [PHYLink(srcNodeID=src, dstNodeID=dst) for src, dst in links]

    # return a dictionary that maps a (srcNodeID, dstNodeID) named tuple to a
    # list of (id, txTime, detectionTime, success) named tuple of frames of
    # this "link"
    def get_frame_dict(self) -> OrderedDict:
        cur = self.analyzer.conn.cursor()
        out = OrderedDict()
        self.get_links()
        for link in self.links:
            out[link] = list()

        # get all detected frames
        print("(FER) Getting detected frames...", end="", flush=True)
        dl = ["srcNodeID", "dstNodeID", "id", "txTime", "rxSuccess"]
        df = cur.execute(
            "select Frame.srcNodeID, Frame.dstNodeID, Frame.id, Frame.txTime, FrameRx.rxSuccess "
            "from Frame inner join FrameRx "
            "on Frame.dstNodeID = FrameRx.srnID "
            "where Frame.id = FrameRx.frame"
        )
        for f in df:
            frame = {k: v for k, v in zip(dl, f)}
            link = PHYLink(srcNodeID=frame["srcNodeID"], dstNodeID=frame["dstNodeID"])
            out[link].append(FrameTx(
                id=frame["id"],
                txTime=frame["txTime"] / 1000000.,  # convert to ms
                detected=True,
                rxSuccess=True if frame["rxSuccess"] == 1 else False
            ))
        print("Done.", flush=True)

        # get all undetected frames (sqlite caches previous query, so this should be fast enough)
        print("(FER) Getting undetected frames...", end="", flush=True)
        ul = ["srcNodeID", "dstNodeID", "id", "txTime"]
        uf = cur.execute(
            "select Frame.srcNodeID, Frame.dstNodeID, Frame.id, Frame.txTime "
            "from Frame where id not in "
            "(select Frame.id from Frame inner join FrameRx "
            "on Frame.dstNodeID = FrameRx.srnID where Frame.id = FrameRx.frame)"
        )
        for f in uf:
            frame = {k: v for k, v in zip(ul, f)}
            link = PHYLink(srcNodeID=frame["srcNodeID"], dstNodeID=frame["dstNodeID"])
            out[link].append(FrameTx(
                id=frame["id"],
                txTime=frame["txTime"] / 1000000.,
                detected=False,
                rxSuccess=False
            ))
        print("Done.", flush=True)
        # sort frames and return
        for link in self.links:
            out[link].sort(key=lambda frame: frame.txTime)
        return out

    # sliding window computation of frame detection and error rate lies here
    def get_fdr_fer(self, windowSizeMS):
        if self.frames is None:
            self.frames = self.get_frame_dict()
        out = dict()
        print("(FER) Computing sliding window ({:.1f} ms) average...".format(windowSizeMS), end="", flush=True)
        for link, frames in self.frames.items():
            out[link] = {"fdr": list(), "fer": list(), "time": list()}
            for idx in range(len(frames)):
                # start time for this window
                start_time = frames[idx].txTime
                # find all indices to average over
                end_idx = idx
                while (frames[end_idx].txTime - start_time) < windowSizeMS:
                    end_idx += 1
                    # corner case that we hit the end of frames list
                    if end_idx >= len(frames) - 1:
                        end_idx -= 1
                        break
                # sum the number of detected frames and number of frames in this window
                ndetect = 0.0
                nerr = 0.0
                nframes = 0.0
                if end_idx == idx:  # only one frame in this window
                    ndetect = 1.0 if frames[idx].detected else 0.0
                    nerr = 1.0 if (frames[idx].detected is True and frames[idx].rxSuccess is False) else 0.0
                    nframes = 1.0
                else:
                    for i in range(idx, end_idx + 1):
                        ndetect += 1.0 if frames[i].detected else 0.0
                        nerr += 1.0 if (frames[i].detected is True and frames[i].rxSuccess is False) else 0.0
                        nframes += 1
                out[link]["fdr"].append(ndetect/nframes)
                out[link]["fer"].append(nerr/nframes)
                out[link]["time"].append(start_time - self.start_time_ms)
        print("Done.", flush=True)
        return out


class FrameTab(PHYLinkPlot):
    def __init__(self, analyzer, tabwidget, ntab):
        super().__init__(analyzer, tabwidget, ntab)
        self.windowSize = 1*1000  # in milliseconds
        self.ferCompute = None
        self.fdr_fer = None

    def dataReady(self):
        return self.fdr_fer is not None

    def loadData(self):
        self.ferCompute = FERCompute(self.analyzer)
        self.fdr_fer = self.ferCompute.get_fdr_fer(self.windowSize)
        # initialize the list
        for link in sorted(self.fdr_fer.keys(), key=lambda l: l.srcNodeID):
            if link.dstNodeID == 255:
                continue
            item = QStandardItem()
            item.setCheckable(True)
            item.setSelectable(False)
            item.setData(link)
            item.setText("{:d} => {:d}".format(link.srcNodeID, link.dstNodeID))
            self.linkList.appendRow(item)

    # given current selection, plot the corresponding data
    def refreshPlot(self):
        # this is a split plot. top: detection bottom: error
        self.figure.clear()
        fdr_ax, fer_ax = self.figure.subplots(nrows=2)
        # loop through all items and plot the selected ones
        for idx in range(self.linkList.rowCount()):
            item = self.linkList.item(idx)
            if item.checkState() == Qt.Checked:
                link = item.data()
                time = self.fdr_fer[link]["time"]
                fdr = self.fdr_fer[link]["fdr"]
                fer = self.fdr_fer[link]["fer"]
                lab = "{:d} => {:d}".format(link.srcNodeID, link.dstNodeID)
                fdr_ax.scatter(time, fdr, s=0.5, label=lab)
                fdr_ax.set_ylabel("FDR")
                fer_ax.scatter(time, fer, s=0.5, label=lab)
                fer_ax.set_ylabel("FER")
                fer_ax.set_xlabel("Time [ms]")
        # format the figure
        fdr_ax.legend()
        fer_ax.legend()
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

class BLERCompute(object):
    def __init__(self, analyzer):
        # database connection
        self.analyzer = analyzer
        # stuff
        self.links = []  # a "link" here is a (src, dst) pair
        self.start_time_ms = self.analyzer.get_start_time()
        self.stop_time_ms = self.analyzer.get_stop_time()
        # set up caches
        self.blocks = None
        pass

    def get_links(self):
        cur = self.analyzer.conn.cursor()
        links = cur.execute("select srcNodeID,dstNodeID from Frame group by srcNodeID,dstNodeID")
        self.links = [PHYLink(srcNodeID=src, dstNodeID=dst) for src, dst in links]

    # return a dictionary that maps a (srcNodeID, dstNodeID) named tuple to a
    # list of (id, txTime, detectionTime, success) named tuple of frames of
    # this "link"
    def get_block_dict(self) -> OrderedDict:
        cur = self.analyzer.conn.cursor()
        out = OrderedDict()
        self.get_links()
        for link in self.links:
            out[link] = list()

        # get all detected blocks
        print("(BLER) Getting detected blocks...", end="", flush=True)
        dl = ["srcNodeID", "dstNodeID","isValid", "txTime"]
        df = cur.execute(
            "select Frame.srcNodeID, Frame.dstNodeID, BlockRx.isValid, BlockRx.time from Frame inner join BlockRx on Frame.dstNodeID = BlockRx.srnID where Frame.id = BlockRx.frame group by frame.id"
        )
        for f in df:
            block = {k: v for k, v in zip(dl, f)}
            link = PHYLink(srcNodeID=block["srcNodeID"], dstNodeID=block["dstNodeID"])
            out[link].append(BlockTx(
                txTime=block["txTime"] / 1000000.,  # convert to ms
                valid=block['isValid']
            ))
        print("Done.", flush=True)
        # sort blocks and return
        for link in self.links:
            out[link].sort(key=lambda block: block.txTime)
        return out

    # sliding window computation of frame detection and error rate lies here
    def get_bler(self, windowSizeMS):
        if self.blocks is None:
            self.blocks = self.get_block_dict()
        out = dict()
        print("(BLER) Computing sliding window ({:.1f} ms) average...".format(windowSizeMS), end="", flush=True)
        for link, blocks in self.blocks.items():
            out[link] = {"bler": list(), "time": list()}
            for idx in range(len(blocks)):
                # start time for this window
                start_time = blocks[idx].txTime
                # find all indices to average over
                end_idx = idx
                while (blocks[end_idx].txTime - start_time) < windowSizeMS:
                    end_idx += 1
                    if end_idx >= len(blocks) - 1:
                        end_idx -= 1
                        break
                # sum the number of valid blocks  and number of total blocks in this window
                nvalid = 0.0
                nblocks = 0.0
                if end_idx == idx:  # only one frame in this window
                    nvalid += blocks[idx].valid
                    nblocks += 1
                else:
                    for i in range(idx, end_idx + 1):
                        nvalid += blocks[i].valid
                        nblocks += 1
                bler_rate=1.-nvalid/nblocks
                out[link]["bler"].append(bler_rate)
                out[link]["time"].append(start_time - self.start_time_ms)
        print("Done.", flush=True)
        return out



class BlockTab(PHYLinkPlot):
    def __init__(self, analyzer, tabwidget, ntab):
        super().__init__(analyzer, tabwidget, ntab)
        self.windowSize = 1*1000  # in milliseconds
        self.blerCompute = None
        self.bler = None
        self.button = QPushButton("Export")
        self.lvbox.addWidget(self.button)
        self.button.clicked.connect(self.export)

    def dataReady(self):
        return self.bler is not None

    def loadData(self):
        self.blerCompute = BLERCompute(self.analyzer)
        self.bler = self.blerCompute.get_bler(self.windowSize)
        # initialize the list
        for link in sorted(self.bler.keys(), key=lambda l: l.srcNodeID):
            if link.dstNodeID == 255:
                continue
            item = QStandardItem()
            item.setCheckable(True)
            item.setSelectable(False)
            item.setData(link)
            item.setText("{:d} => {:d}".format(link.srcNodeID, link.dstNodeID))
            self.linkList.appendRow(item)

    # given current selection, plot the corresponding data
    def refreshPlot(self):
        # this is a split plot. top: detection bottom: error
        self.figure.clear()
        bler_ax = self.figure.subplots(nrows=1)
        # loop through all items and plot the selected ones
        for idx in range(self.linkList.rowCount()):
            item = self.linkList.item(idx)
            if item.checkState() == Qt.Checked:
                link = item.data()
                time = self.bler[link]["time"]
                bler_plot = self.bler[link]["bler"]
                lab = "{:d} => {:d}".format(link.srcNodeID, link.dstNodeID)
                bler_ax.scatter(time, bler_plot, s=0.5, label=lab)
                bler_ax.set_ylabel("BLER")
                bler_ax.set_xlabel("Time [ms]")
        # format the figure
        bler_ax.legend()
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    # fixme generalize in superclass
    def export(self):
        if not self.dataReady():
            return False
        filename, _filt = QFileDialog.getSaveFileName(self, "Select Output file", ".", 'JavaScript Object Notation (*.json);;All Files (*.*)')
        out = list()
        for k, v in self.bler.items():
            out.append({"src": k.srcNodeID, "dst": k.dstNodeID, "time": v["time"], "data": v["bler"]})
        with open(filename, "w") as of:
            json.dump(out, of)

# SNR #########################################################################

class SNRTab(PHYLinkPlot):
    def __init__(self, analyzer, tabwidget, ntab, dname="snr"):
        super().__init__(analyzer, tabwidget, ntab)
        self.dname = dname
        self.data = None
        self.button = QPushButton("Export")
        self.lvbox.addWidget(self.button)
        self.button.clicked.connect(self.export)

    def dataReady(self):
        return self.data is not None

    def loadData(self):
        st = self.analyzer.get_start_time()
        cur = self.analyzer.conn.cursor()
        self.links = [PHYLink(srcNodeID=src, dstNodeID=dst) for src, dst in
                      cur.execute("select srcNodeID, dstNodeID from Frame group by srcNodeID,dstNodeID")]
        o1 = OrderedDict()
        for link in self.links:
            o1[link] = list()
        print("({}) Computing {} curves...".format(self.dname, self.dname), end="", flush=True)
        cols = ["srcNodeID", "dstNodeID", "time", self.dname]
        df = cur.execute(
            "select Frame.srcNodeID, Frame.dstNodeID, FrameRx.time, FrameRx.{:s} "
            "from Frame inner join FrameRx on Frame.id = FrameRx.frame".format(self.dname)
        )
        for f in df:
            frame = {k: v for k, v in zip(cols, f)}
            link = PHYLink(srcNodeID=frame["srcNodeID"], dstNodeID=frame["dstNodeID"])
            o1[link].append((frame[self.dname], frame["time"] / 1000000. - st))
        self.data = OrderedDict()
        # sort by tx time and save
        for link in self.links:
            if link.dstNodeID == 255:
                continue
            d = sorted(o1[link], key=lambda frame: frame[1])
            self.data[link] = {self.dname: [dd[0]  for dd in d],
                               "time": [dd[1] for dd in d]}
            item = QStandardItem()
            item.setCheckable(True)
            item.setSelectable(False)
            item.setData(link)
            item.setText("{:d} => {:d}".format(link.srcNodeID, link.dstNodeID))
            self.linkList.appendRow(item)
        print("Done", flush=True)

    def refreshPlot(self):
        self.figure.clear()
        ax = self.figure.subplots()
        for idx in range(self.linkList.rowCount()):
            item  = self.linkList.item(idx)
            if item.checkState() == Qt.Checked:
                link = item.data()
                time = self.data[link]["time"]
                snr = self.data[link][self.dname]
                lab = "{:d} => {:d}".format(link.srcNodeID, link.dstNodeID)
                ax.scatter(time, snr, s=0.5, label=lab)
        ax.legend()
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    def export(self):
        if not self.dataReady():
            return False
        filename, _filt = QFileDialog.getSaveFileName(self, "Select Output file", ".", 'JavaScript Object Notation (*.json);;All Files (*.*)')
        out = list()
        for k, v in self.data.items():
            out.append({"src": k.srcNodeID, "dst": k.dstNodeID, "time": v["time"], "data": v[self.dname]})
        with open(filename, "w") as of:
            json.dump(out, of)

# Channel Allocation ##########################################################

class AllocTab(QWidget):
    def __init__(self, analyzer, tabwidget, ntab):
        QWidget.__init__(self)

        self.hbox = QHBoxLayout()
        lvbox = QVBoxLayout()
        rvbox = QVBoxLayout()

        # SRN List
        self.srnList = QStandardItemModel()
        self.srnListWidget = QListView()
        self.srnListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.srnListWidget.setModel(self.srnList)
        self.srnListWidget.setMaximumHeight(30)
        # FIXME I do not need this right now
        # lvbox.addWidget(QLabel("SRN List"))
        # lvbox.addWidget(self.srnListWidget)

        # Plot pane
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        lvbox.addWidget(toolbar)
        lvbox.addWidget(self.canvas)

        # Info field
        self.infoField = QTableView()
        self.infoModel = QStandardItemModel()
        # self.infoField.setReadOnly(True)
        self.infoField.setMaximumHeight(500)
        self.infoField.setModel(self.infoModel)
        lvbox.addWidget(QLabel("Event Info"))
        lvbox.addWidget(self.infoField)

        # Event Log
        rvbox.addWidget(QLabel("Event Log"))
        self.eventFilterCheckBox = QCheckBox("Show failed attempts")
        self.eventFilterCheckBox.setCheckState(Qt.Checked)
        rvbox.addWidget(self.eventFilterCheckBox)
        self.eventLog = QStandardItemModel()
        self.eventLogWidget = QListView()
        self.eventLogWidget.setModel(self.eventLog)
        self.eventLogWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        rvbox.addWidget(self.eventLogWidget)

        # GUI plumbing
        lvboxWidget = QWidget()
        lvboxWidget.setLayout(lvbox)
        rvboxWidget = QWidget()
        rvboxWidget.setLayout(rvbox)
        rvboxWidget.setMaximumWidth(200)
        self.hbox.addWidget(lvboxWidget)
        self.hbox.addWidget(rvboxWidget)

        # a progress bar for children
        self.progressView = QVBoxLayout()
        hb = QHBoxLayout()
        hbw = QWidget()
        hbw.setLayout(hb)
        self.progressButton = QPushButton("Process data")
        self.progressBar = QProgressBar()
        hb.addWidget(self.progressButton)
        hb.addWidget(self.progressBar)
        self.progressButton.clicked.connect(self.loadEverything)
        self.progressView.addWidget(hbw)
        self.setLayout(self.progressView)

        # other member vars
        self.analyzer = analyzer
        self.tabwidget = tabwidget
        self.channelAllocEvents = list()
        self.channelAllocUpdateEvents = list()
        self.bars = dict()
        self.plotData = list()
        self.waveforms = None
        self.bandwidth = None
        self.ccdatamod = None
        self.logmod = None

        # connect slots
        self.ntab = ntab
        self.connectRefresh()

    def connectRefresh(self):
        self.tabwidget.currentChanged.connect(self.refreshOnTabChange)
        self.srnList.dataChanged.connect(self.refreshView)
        self.eventFilterCheckBox.stateChanged.connect(self.refreshView)
        self.eventLogWidget.selectionModel().selectionChanged.connect(self.refreshView)

    def refreshOnTabChange(self, index):
        if self.ntab == index:
            self.refreshView()

    def refreshView(self, *args, **kwargs):
        if self.dataReady():
            self.refreshPlot()
            self.refreshEventLog()
            self.refreshEventInfo()

    def loadEverything(self, *args, **kwargs):
        if self.isConnected():
            self.loadEvents()
            QWidget().setLayout(self.progressView)
            self.setLayout(self.hbox)
            self.refreshView()

    def loadEvents(self):
        # load proto modules first
        cur = self.analyzer.conn.cursor()
        try:
            self.ccdatamod = load_proto_from_string(
                get_proto_string_from_db("ccdataproto", cur), "CCDataPb"
            )
            self.logmod = load_proto_from_string(
                get_proto_string_from_db("logproto", cur), "BAMLogPb"
            )
        except Exception as e:
            self.showError(str(e))
            return False
        # load all Events and do some housekeeping
        self.channelAllocEvents = self.readTable("ChannelAlloc")
        self.channelAllocUpdateEvents = self.readTable("ChannelAllocUpdate")
        self.waveforms = self.loadWaveforms()
        self.bandwidth = self.getBandwidth()
        self.processEvents()
        self.initEventLog()
        return self.initPlot()

    def readTable(self, table):
        cur = self.analyzer.conn.cursor()
        # get column names first
        colnames = list()
        ret = cur.execute("PRAGMA table_info('{}')".format(table))
        for column in ret:
            colid, colname, coltype, notnull, default, pk = column
            colnames.append(colname)
        # now read and parse all proto messages
        out = list()
        data = cur.execute("SELECT * from {}".format(table))
        for raw_data in data:
            d = {k: v for k, v in zip(colnames, raw_data)}
            raw_bytes = d["data"]
            if table == "ChannelAlloc":
                m = self.logmod.ChannelAllocEventInfo()
            elif table == "ChannelAllocUpdate":
                m = self.ccdatamod.ChannelParamsUpdateInfo()
            else:
                raise RuntimeError("Incorrect Message selected.")
            m.ParseFromString(raw_bytes)
            d["msg"] = m
            del d["data"]
            out.append(d)
        return out

    def loadWaveforms(self):
        cur = self.analyzer.conn.cursor()
        radioconf = load_ini_from_si("radioconf", cur)
        srnid = cur.execute(
            "select srnID from Waveform limit 1;").fetchall()[0][0]
        # get column names # FIXME DRY
        colnames = list()
        ret = cur.execute("PRAGMA table_info('{}')".format("Waveform"))
        for column in ret:
            colid, colname, coltype, notnull, default, pk = column
            colnames.append(colname)
        ret = cur.execute(
            "select * from Waveform where srnID={:d}".format(srnid)).fetchall()
        # FIXME PLEASE
        srate = phy_sample_rate
        out = {}  # dict mapping waveformID to bandwidth
        for rd in ret:
            d = {k: v for k, v in zip(colnames, rd)}
            out[d["waveformID"]] = srate * d["edge"]
        return out

    def getBandwidth(self):
        cur = self.analyzer.conn.cursor()
        coloconf = load_ini_from_si("coloconf", cur)
        bw = coloconf.get("RF", "rf_bandwidth")
        return float(bw)

    def processEvents(self):
        # FIXME this might not be what we want to do... this is very
        # inefficient (at least it only runs once)
        for event in self.channelAllocEvents:
            event.update({"updates": list()})
            if event["msg"].ofdm_params_update_id == 0:
                event.update({"success": False})
            else:
                event.update({"success": True})

        def getIndexOfAlloc(updateID):
            for idx, alloc in enumerate(self.channelAllocEvents):
                if alloc["msg"].ofdm_params_update_id == updateID:
                    return idx
            else:
                return None

        for event in self.channelAllocUpdateEvents:
            idx = getIndexOfAlloc(event["msg"].channel_last_update)
            if idx is not None:
                self.channelAllocEvents[idx]["updates"].append(event)

    def initEventLog(self):
        colors = {
            "green": (140,183,9),
            "red": (245,114,10)
        }
        # make everything relative to scenario start time
        start_time = self.analyzer.get_start_time() / 1000
        for ei in self.channelAllocEvents:
            reltime = ei["time"]/1000000000. - start_time
            ei.update({"reltime": reltime})
        self.channelAllocEvents.sort(key=lambda x: x["reltime"])
        for ei in self.channelAllocEvents:
            item = QStandardItem()
            item.setCheckable(False)
            item.setSelectable(True)
            item.setData(ei)
            # color code and label based on success of allocation
            if ei["success"] is False:
                item.setText("[{:.1f}] {:s}".format(ei["reltime"], "Failure"))
                item.setBackground(QColor(*colors["red"]))
            else:
                item.setText("[{:.1f}] {:s}".format(ei["reltime"], "Success"))
                item.setBackground(QColor(*colors["green"]))
            self.eventLog.appendRow(item)

    def initPlot(self):
        self.figure.clear()
        ax = self.figure.subplots()
        start_time_ms = self.analyzer.get_start_time()
        stop_time_ms = self.analyzer.get_stop_time()
        # filter out all successful events. they are already sorted
        tmp = list()
        for i, event in enumerate(self.channelAllocEvents):
            if event["success"] is True:
                tmp.append({**event, **{"index": i}})
        # exit early if we do not have successful channel allocation events.
        if len(tmp) == 0:
            ax.text(0.5, 0.5, 'No successful events.', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()
            return False
        # compute the duration of each allocation
        for i, event in enumerate(tmp[:-1]):
            rel_start_ms = event["time"]/1000000. - start_time_ms
            rel_end_ms = tmp[i+1]["time"]/1000000. - start_time_ms
            event.update({"rel_start_ms": rel_start_ms,
                          "rel_end_ms": rel_end_ms})
        # need to do separate for last
        tmp1 = tmp[-1]["time"]/1000000. - start_time_ms
        tmp[-1].update({"rel_start_ms": tmp1,
                        "rel_end_ms": stop_time_ms - start_time_ms})
        # assemble the plot data
        for event in tmp:
            # the slot -> srn mapping
            #slot2srn = {k: v for k, v in zip(event["msg"].tx_slots,
            #                                 event["msg"].node_ids)}
            slot2srn = {k: v for k, v in zip(range(0, len(event["msg"].node_ids)),
                                             event["msg"].node_ids)}
            # extract the bandwidths and offsets
            try:
                update = event["updates"][0]["msg"]
                alloc = dict()
                for slot, info in enumerate(update.channel_info):
                    alloc[slot2srn[slot]] = {
                        "offset": info.center_offset_hz,
                        "bandwidth": self.waveforms[info.waveform_id]
                    }
                self.plotData.append({
                    "start": event["rel_start_ms"],
                    "stop": event["rel_end_ms"],
                    "width": event["rel_end_ms"] - event["rel_start_ms"],
                    "alloc": alloc,
                    "index": event["index"]  # use this to map back to the ChannelAllocEvent
                })
            except IndexError as e:
                continue
            except KeyError as e:
                # I don't know what this means
                continue
        # alternate between two colors
        colors = [(85/255,98/255,112/255),
                  (78/255,205/255,196/255)]
        self.progressBar.setRange(0, len(self.plotData))
        yabsmax = 0
        for i, d in enumerate(self.plotData):
            self.progressBar.setValue(i)
            color = colors[i % 2]
            for _, a in d["alloc"].items():
                offset = a["offset"] / 1e6  # Hz->MHz
                if abs(offset) > 1e9:
                    continue
                bw = a["bandwidth"] / 1e6   # Hz->MHz
                yabsmax = max(yabsmax, abs(offset+bw/2), abs(offset-bw/2))
                container = ax.barh(y=offset,
                                    width=d["width"] / 1e3,  # ms->s
                                    height=bw,
                                    left=d["start"] / 1e3,   # ms->s
                                    align="center",
                                    color=color,
                                    picker=True)
                # pull out the artist and save the index into the plotdata
                # dict. use to later look up the data that belongs to this
                # allocation
                self.bars[container.patches[0]] = d["index"]
        self.figure.tight_layout(pad=0)
        yabsmax = max(yabsmax, 12.5)
        ax.set_ylim(-yabsmax, yabsmax)
        ax.invert_yaxis()
        ax.set_ylabel('Freq (MHz)')
        ax.set_xlabel('Time (s)')
        self.canvas.mpl_connect("pick_event", lambda x: self.updateOnPick(x))
        self.progressBar.setValue(len(self.plotData))
        return True

    def updateOnPick(self, event):
        # when we click on a bar in the plot, select the corresponding
        # ChannelAllocEvent and display it
        if isinstance(event.artist, Rectangle):
            try:
                idx = self.bars[event.artist]
                # FIXME find all other bars with the same index and highlight
                # them?
                self.updateSelectedEvent(idx)
            except KeyError as e:
                pass

    def refreshPlot(self):
        pass

    def updateSelectedEvent(self, idx):
        # change the selectedevent to the event idx. This is a little bit of a
        # hack but oh well.
        self.eventLogWidget.setCurrentIndex(self.eventLog.index(idx, 0))

    def refreshEventInfo(self):
        if (len(self.eventLogWidget.selectedIndexes()) <= 0):
            return
        self.infoModel.clear()
        si = self.eventLogWidget.selectedIndexes()[0]
        event = self.eventLog.item(si.row()).data()
        if event["success"] is False:
            pass  # FIXME ?
        else:
            rows = []
            slot2srn = {k: v for k, v in zip(event["msg"].tx_slots,
                                             event["msg"].node_ids)}
            slots = sorted(event["msg"].tx_slots)
            srns = [" ".join(["SRN", str(slot2srn[i])]) for i in slots]
            # add a row for slots
            rows.append("Slot")
            self.infoModel.appendRow([QStandardItem(str(s)) for s in slots])
            # add a row for target
            rows.append("BW needed [MHz]")
            self.infoModel.appendRow([QStandardItem(str(bw.bandwidth_needed/1000000.))
                                      for bw in event["msg"].target])
            # add rows for allocation
            try:
                update = event["updates"][0]["msg"]
                self.infoModel.appendRow([QStandardItem(str(self.waveforms[wf.waveform_id]/1000000.))
                                          for wf in update.channel_info])
                rows.append("BW alloc. [MHz]")
                self.infoModel.appendRow([QStandardItem(str(f.center_offset_hz/1000000.))
                                          for f in update.channel_info])
                rows.append("Center freq. [MHz]")
            except IndexError as e:
                pass
            self.infoModel.setHorizontalHeaderLabels(srns)
            self.infoModel.setVerticalHeaderLabels(rows)


    def refreshEventLog(self):
        showFailed = True if self.eventFilterCheckBox.checkState() == Qt.Checked else False
        for i in range(self.eventLog.rowCount()):
            ei = self.eventLog.item(i).data()
            if (ei["success"] is False) and (showFailed is False):
                self.eventLogWidget.setRowHidden(i, True)
            else:
                self.eventLogWidget.setRowHidden(i, False)

    def isConnected(self):
        return self.analyzer.conn is not None

    def dataReady(self):
        return len(self.channelAllocEvents) > 0

    def showError(self, msg: str):
        hb = QHBoxLayout()
        hb.addWidget(QLabel("ERROR: {:s}".format(msg)))
        QWidget().setLayout(self.hbox)
        self.setLayout(hb)

class MiscTab(object):
    """This tab lets you easily plot any data that does not warrant its own tab. Simply add a new plotter() class as a member here."""
    def __init__(self, analyzer, tabwidget):
        self.analyzer = analyzer
        self.widget = QWidget()
        hbox = QHBoxLayout()
        lbox = QVBoxLayout()
        rbox = QVBoxLayout()

        # plot selection list
        self.plotList = QStandardItemModel()
        self.plotListWidget = QListView()
        self.plotListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.plotListWidget.setModel(self.plotList)
        lbox.addWidget(QLabel("Available Plots"))
        lbox.addWidget(self.plotListWidget)

        # the figure and nav toolbar
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self.widget)
        rbox.addWidget(toolbar)
        rbox.addWidget(self.canvas)

        # get plotters ready
        self.initPlotters()
        self.currentPlotter = None

        # GUI plumbing
        lboxWidget = QWidget()
        lboxWidget.setLayout(lbox)
        lboxWidget.setMaximumWidth(200)
        rboxWidget = QWidget()
        rboxWidget.setLayout(rbox)
        hbox.addWidget(lboxWidget)
        hbox.addWidget(rboxWidget)
        self.widget.setLayout(hbox)

        # signals
        tabwidget.tabBarClicked.connect(self.refresh)
        self.plotListWidget.selectionModel().selectionChanged.connect(self.refresh)

    def refresh(self, *args, **kwargs):
        # get selection and set currentPlotter
        if len(self.plotListWidget.selectedIndexes()) > 0:
            si = self.plotListWidget.selectedIndexes()[0].row()
            self.currentPlotter = self.plotList.item(si).data()
        # redraw
        if (self.currentPlotter is not None) and self.isConnected():
            self.figure.clear()
            self.currentPlotter.plot(self.figure, self.analyzer.conn)
            self.canvas.draw()
        else:
            self.figure.clear()
            ax = self.figure.subplots()
            ax.text(0.5, 0.5, 'No database loaded.', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()

    def isConnected(self):
        return self.analyzer.conn is not None

    def initPlotters(self):
        for member in dir(MiscTab):
            if not member.startswith("__"):
                p = getattr(self, member)
                if inspect.isclass(p):
                    item = QStandardItem()
                    item.setData(p())
                    item.setText(p.__name__)
                    item.setSelectable(True)
                    item.setCheckable(False)
                    self.plotList.appendRow(item)

    # Custom plotters go down here ======================

    class PHYDelay:
        """Read the statistics on PHY Delay and plot the total delay for each SRN over the entire match."""
        def plot(self, fig, db):
            if self.data is None:
                self.loadData(db)
            assert(self.data is not None)
            np = len(self.data)
            si = 1
            for srn, data in self.data.items():
                ax = fig.add_subplot(np, 1, si)
                ax.plot(data[self.timekey], data[self.queuekey], label="UHD send()")
                ax.plot(data[self.timekey], data[self.datakey], label="Processing")
                ax.set_ylabel("SRN {:d}".format(srn))
                ax.set_xlabel("Time [{}]".format(self.timeunits))
                ax.legend()
                ax.grid()
                si += 1;
            fig.suptitle("Total PHY Tx Processing/Queueing delay [{}] vs. Match time"
                         .format(self.timeunits))

        def __init__(self):
            self.data = None
            self.datakey = "sum"
            self.queuekey = "t_stream"
            self.timekey = "reltime"
            self.timeunits = "ms"
            # divider from nanoseconds (raw data) to time_units
            self.time_unit_dict = {"ms": 1000000.}

        def loadData(self, db):
            cur = db.cursor()
            st = cur.execute("select MIN(time) from C2APIEvent where txt='START'").fetchone()[0]
            srns = [r[0] for r in set(cur.execute("select srnID from ModulationEvent"))]
            self.data = dict();
            for srn in srns:
                # compute the total processing time
                r= cur.execute("SELECT " + "+".join([
                    "t_code_ns", "t_mod_ns", "t_spread_ns", "t_map_ns", "t_shift_ns",
                    "t_cp_ns", "t_mix_ns", "t_scale_ns"
                ]) + ",t_stream_ns,time" + " from ModulationEvent where srnID = {:d}".format(srn))
                self.data[srn] = dict();
                self.data[srn][self.datakey] = list()
                self.data[srn][self.timekey] = list()
                self.data[srn][self.queuekey] = list()
                for d in r:
                    s, q, t = d;
                    proctime = s/self.time_unit_dict[self.timeunits];
                    queuetime = q/self.time_unit_dict[self.timeunits];
                    reltime = (t - st)/self.time_unit_dict[self.timeunits];
                    self.data[srn][self.datakey].append(proctime)
                    self.data[srn][self.queuekey].append(queuetime)
                    self.data[srn][self.timekey].append(reltime)

    class TxSegment:
        def __init__(self):
            self.data = None
        
        def get_sliding_window_sum(self, data, t_in, t_out, t_interval):
            data_out = [0] * len(t_out)
            in_idx = 0
            if len(t_in) <= 1:
                return data_out
            for out_idx in range(len(t_out)):
                while t_in[in_idx] < t_out[out_idx]:
                    in_idx += 1
                    if in_idx >= len(data):
                        in_idx -= 1
                        break
                i = in_idx - 1
                sum_window = 0
                while i >= 0 and (t_out[out_idx] - t_in[i]) < t_interval:
                    sum_window += data[i]
                    i -= 1
                data_out[out_idx] = sum_window
            return data_out
        
        def loadData(self, db):
            self.data = dict()
            cur = db.cursor()
            # Get SRN IDs
            srns = [r[0] for r in set(cur.execute("SELECT DISTINCT srnID as d_srnid FROM Start ORDER BY d_srnid ASC"))]
            # time
            t1_origin = cur.execute("select MIN(time) from C2APIEvent where txt='START'").fetchone()[0] / 1000000
            t1_end = cur.execute("select MAX(time) from C2APIEvent where txt='STOP'").fetchone()[0] / 1000000

            self.t_out = np.linspace(0., (t1_end - t1_origin) / 1000, 1000)
            # For each SRN, compute Tx rate
            for srnid in srns:
                data = cur.execute("SELECT Frame.txTime, Segment.nbytes "
                        "FROM Segment JOIN Frame ON Segment.frame=Frame.id "
                        "WHERE Segment.srcNodeID={:d}".format(srnid))
                rows = cur.fetchall()
                t = [] # msec
                rate = [] # Mbits
                for row in rows:
                    t.append(row[0]/1000000)
                    rate.append(8 * row[1] / 1000000)  # rate in Mbps
                rate_sum = self.get_sliding_window_sum(np.array(rate), (np.array(t) - t1_origin) / 1000, self.t_out, 1)
                self.data[srnid] = rate_sum

        def plot(self, fig, db):
            if self.data is None:
                self.loadData(db)
            assert(self.data is not None)
            np = len(self.data)
            si = 1
            for srn, data in self.data.items():
                ax = fig.add_subplot(np, 1, si)
                ax.plot(self.t_out, data)
                ax.set_ylabel("SRN {:d}".format(srn))
                ax.set_xlabel("Time (s)")
                ax.grid()
                si += 1;
            fig.suptitle("Segment Tx Rate vs. Match time")

    # End custom plotters  ==============================


# https://stackoverflow.com/a/12375397
class QIPythonWidget(RichJupyterWidget):
    """ Convenience class for a live IPython console widget. We can replace the standard banner using the customBanner argument"""
    def __init__(self,customBanner=None,*args,**kwargs):
        if not customBanner is None: self.banner=customBanner
        super(QIPythonWidget, self).__init__(*args,**kwargs)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt().exit()
        self.exit_requested.connect(stop)

    def pushVariables(self,variableDict):
        """ Given a dictionary containing name / value pairs, push those variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)
    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()
    def printText(self,text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)
    def executeCommand(self,command):
        """ Execute a command in the frame of the console widget """
        self._execute(command,False)

# interactive python console to manipulate and vizualize more data
class ConsoleTab(object):
    def __init__(self, analyzer, main_window, tabwidget, parent=None):
        self.analyzer = analyzer
        self.mainWindow = main_window
        self.running = False

        self.widget = QWidget()
        self.box = QHBoxLayout()
        self.label = QLabel("Console not running, load a log file first!")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)
        self.box.addWidget(self.label)
        self.widget.setLayout(self.box)

        tabwidget.tabBarClicked.connect(self.comeInFocus)

    def isConnected(self):
        return self.analyzer.conn is not None

    def comeInFocus(self, *args, **kwargs):
        if (not self.running) and consoleReady and self.isConnected():
            try:
                console = QIPythonWidget("BAM! Wireless Logs.\nUse the sql() command to execute queries against the loaded dadabase.\n\n")
                self.box.removeWidget(self.label)
                self.box.addWidget(console)
                # add access to variables to console
                def execSql(s):
                    """Execute a SQL query against the loaded database."""
                    cur = self.analyzer.conn.cursor()
                    r = cur.execute(s)
                    return r.fetchall()
                console.pushVariables({
                    "analyzer": self.analyzer,
                    "db": self.analyzer.conn,
                    "mainWindow": self.mainWindow,
                    "sql": execSql
                })
                self.running = True
            except Exception as e:
                self.label.setText(str(e))


def load_proto_from_string(proto: str, modname: str):
    """Load the proto definition from 'proto', compile it, and load the python
    module 'modname' from it. Returns the python module."""
    protopath = find_executable("protoc")
    if protopath is None:
        raise RuntimeError("No Protobuf compiler ('protoc') fount on PATH.")
    with tempfile.TemporaryDirectory() as tempdir:
        fn = "{:s}.proto".format(modname)
        # write .proto file
        outpath = pathlib.Path(tempdir) / fn
        with outpath.open("w") as f:
            f.write(proto)
        # run the compiler
        ret = subprocess.run([str(pathlib.Path(protopath).resolve()),
                              "--python_out={}".format(outpath.parent.resolve()),
                              "-I{}".format(outpath.parent.resolve()),
                              str(outpath.resolve())])
        if ret.returncode != 0:
            raise RuntimeError(
                "Error compiling protobuf: {}".format(outpath.resolve())
            )
        # import the module
        sys.path.append(str(pathlib.Path(tempdir)))
        mod = importlib.import_module("{:s}_pb2".format(modname))
        # clean up
        sys.path.remove(str(pathlib.Path(tempdir)))
        return mod


def get_proto_string_from_db(protoname: str, cur: sqlite3.Cursor) -> str:
    # FIXME add some error protection
    return cur.execute(
        "SELECT {:s} from BuildInfo LIMIT 1;".format(protoname)
    ).fetchall()[0][0]


def load_ini_from_si(name: str, cur: sqlite3.Cursor) -> ConfigParser:
    ret = cur.execute(
        "SELECT {:s} from ScenarioInfo limit 1;".format(name)).fetchall()[0][0]
    p = ConfigParser()
    p.read_string(ret)
    return p


def load_json_from_si(name: str, cur: sqlite3.Cursor) -> dict:
    ret = cur.execute(
        "SELECT {:s} from ScenarioInfo limit 1;".format(name)).fetchall()[0][0]
    return json.loads(ret)

# Collaboration visualizer
# Copyright (c) 2018 Dennis Ogbe
class CollabTab(object):
    def __init__(self, analyzer, tabwidget, tab_index, parent=None):
        # I need the analyzer to have access to the database
        self.analyzer = analyzer
        # save my tab index so I can react to focus changing on me
        self.tab_index = tab_index
        # we have four different message types
        self.ServerTx = "ServerTx"
        self.ServerRx = "ServerRx"
        self.CollabTx = "CollabTx"
        self.CollabRx = "CollabRx"
        # color code messages
        self.msgColors = {
            self.ServerTx: (243,156,29),
            self.ServerRx: (245,114,10),
            self.CollabTx: (170,219,18),
            self.CollabRx: (140,183,9)
        }

        # widgets
        self.tabwidget = tabwidget
        self.widget = QWidget()
        hbox = QHBoxLayout()
        lbox = QVBoxLayout()
        rbox = QVBoxLayout()

        # message list
        self.msgList = QStandardItemModel()
        self.msgListWidget = QListView()
        self.msgListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.msgListWidget.setModel(self.msgList)
        lbox.addWidget(QLabel("Message List"))
        lbox.addWidget(self.msgListWidget)

        # filters
        self.msgFilterModel = QStandardItemModel()
        self.peerFilterModel = QStandardItemModel()
        vbox2 = QVBoxLayout()
        vbox2.addWidget(QLabel("Filters"))
        fbox = QHBoxLayout()
        fboxWidget = QWidget()
        fboxWidget.setLayout(fbox)
        self.msgFilterWidget = QListView()
        self.msgFilterWidget.setModel(self.msgFilterModel)
        fbox.addWidget(self.msgFilterWidget)
        self.peerFilterWidget = QListView()
        self.peerFilterWidget.setModel(self.peerFilterModel)
        fbox.addWidget(self.peerFilterWidget)
        vbox2.addWidget(fboxWidget)
        vbox2Widget = QWidget()
        vbox2Widget.setLayout(vbox2)
        vbox2Widget.setMaximumHeight(120)
        rbox.addWidget(vbox2Widget)

        # Message view
        self.msgView = QTextEdit()
        rbox.addWidget(QLabel("Message View"))
        rbox.addWidget(self.msgView)
        self.initMsgView()

        # Event Log
        rrbox = QVBoxLayout()
        rrbox.addWidget(QLabel("Event Log"))
        self.eventLog = QStandardItemModel()
        self.eventLogWidget = QListView()
        self.eventLogWidget.setModel(self.eventLog)
        rrbox.addWidget(self.eventLogWidget)

        # add all boxes
        lboxWidget = QWidget()
        lboxWidget.setLayout(lbox)
        lboxWidget.setMinimumWidth(400)
        lboxWidget.setMaximumWidth(400)
        rboxWidget = QWidget()
        rboxWidget.setLayout(rbox)
        rrboxWidget = QWidget()
        rrboxWidget.setLayout(rrbox)
        rrboxWidget.setMaximumWidth(600)
        rrboxWidget.setMinimumWidth(400)
        hbox.addWidget(lboxWidget)
        hbox.addWidget(rboxWidget)
        hbox.addWidget(rrboxWidget)
        self.hbox = hbox
        self.widget.setLayout(self.hbox)

        # data
        self.messages = []
        self.connevent = None
        self.statechange = None
        self.peerevent = None
        self.errorevent = None

        # save the modules as member of this class, we need them to parse from
        # binary blob
        self.cilmod = None
        self.regmod = None

        self.currentMsg = None;

        self.connect_refresh()

    def initMsgView(self):
        self.msgView.setFontPointSize(15)
        self.msgView.setFontWeight(40)
        self.msgView.setReadOnly(True)

    def initFilters(self):
        # list all peers
        for p in self.peerevent:
            if p["isAdd"] == 1:
                ip = ip_address(p["ip"])
                item = QStandardItem()
                item.setData(ip)
                item.setText(str(ip))
                item.setCheckState(Qt.Checked)
                item.setCheckable(True)
                item.setSelectable(False)
                if not self.peerFilterModel.findItems(item.text()):
                    self.peerFilterModel.appendRow(item)
        # list all message types
        for t in (self.ServerTx, self.ServerRx, self.CollabTx, self.CollabRx):
            item = QStandardItem()
            item.setText(t)
            item.setData(t)
            item.setCheckState(Qt.Checked)
            item.setCheckable(True)
            item.setBackground(QColor(*self.msgColors[t]))
            item.setSelectable(False)
            self.msgFilterModel.appendRow(item)

    def initMsgList(self):
        self.messages.sort(key=lambda x: x["time"])
        start_time = self.analyzer.get_start_time()/1000
        for i, msg in enumerate(self.messages):
            item = QStandardItem()
            item.setData(msg)
            reltime = msg["time"]/1000000000. - start_time
            ptype = msg["msg"].WhichOneof("payload")
            if ptype is None:
                ptype = "no payload"
            msgDesc = "[{:.1f}] {:s}".format(reltime, ptype)
            item.setText(msgDesc)
            item.setBackground(QColor(*self.msgColors[msg["type"]]))
            item.setCheckable(False)
            self.msgList.appendRow(item)

    def refreshMsgList(self):
        # (1) get filters
        msg1 = []
        activeFilters = []
        for i in range(self.msgFilterModel.rowCount()):
            item = self.msgFilterModel.item(i)
            if item.checkState() == Qt.Checked:
                activeFilters.append(item.text())
        activePeers = []
        for i in range(self.peerFilterModel.rowCount()):
            item = self.peerFilterModel.item(i)
            if item.checkState() == Qt.Checked:
                activePeers.append(item.data())
        # (2) filter active indices
        for i in range(self.msgList.rowCount()):
            msg = self.msgList.item(i).data()
            self.msgListWidget.setRowHidden(i, False)
            if msg["type"] not in activeFilters:
                self.msgListWidget.setRowHidden(i, True)
            else:
                if self.CollabRx in msg["type"]:
                    if ip_address(msg["msg"].sender_network_id) not in activePeers:
                        self.msgListWidget.setRowHidden(i, True)
                if self.CollabTx in msg["type"]:
                    if msg["dstIP"] is None:
                        continue
                    if (msg["broadcast"] == 0) and (ip_address(msg["dstIP"]) not in activePeers):
                        self.msgListWidget.setRowHidden(i, True)
        # (3) change currently selected message to None if it was filtered
        if len(self.msgListWidget.selectedIndexes()) > 0:
            si = self.msgListWidget.selectedIndexes()[0]
            if self.msgListWidget.isRowHidden(si.row()):
                self.currentMsg = None
            else:
                self.currentMsg = self.msgList.item(si.row()).data()

    def initEventLog(self):
        self.eventLogWidget.setFocusPolicy(Qt.NoFocus) # remove focus rectangle
        # different types of events
        connectionAttempt = "ca"
        stateChange = "sc"
        peer = "p"
        error = "e"
        # make everything relative to scenario start time
        start_time = self.analyzer.get_start_time() / 1000
        # compile all messages
        log = []
        for ei in self.connevent:
            reltime = ei["time"]/1000000000. - start_time
            msg = "Connection attempt: " + "Success" if ei["success"] > 0 else "Fail"
            log.append({"reltime": reltime, "msg": msg, "type": connectionAttempt})
        for ei in self.statechange:
            reltime = ei["time"]/1000000000. - start_time
            msg = "State change from {} to {}.".format(ei["fromState"].upper(), ei["toState"].upper())
            log.append({"reltime": reltime, "msg": msg, "type": stateChange})
        for ei in self.peerevent:
            reltime = ei["time"]/1000000000. - start_time
            msg = "{} peer {}".format("Added" if ei["isAdd"] > 0 else "Removed",
                                      str(ip_address(ei["ip"])))
            log.append({"reltime": reltime, "msg": msg, "type": peer})
        for ei in self.errorevent:
            reltime = ei["time"]/1000000000. - start_time
            msg = "Error: {}".format(ei["type"])
            log.append({"reltime": reltime, "msg": msg, "type": error})
        # sort and write to the model
        log.sort(key = lambda x: x["reltime"])
        for l in log:
            item = QStandardItem()
            item.setData(l)
            item.setText("[{:.1f}] {:s}".format(l["reltime"], l["msg"]))
            item.setCheckable(False)
            item.setSelectable(False)
            # FIXME: Maybe color code this?
            self.eventLog.appendRow(item)

    # connect refresh signals. basically on any change
    def connect_refresh(self):
        self.tabwidget.currentChanged.connect(self.refreshOnTabChange)
        self.msgFilterModel.dataChanged.connect(self.refresh_view)
        self.peerFilterModel.dataChanged.connect(self.refresh_view)
        self.msgListWidget.selectionModel().selectionChanged.connect(self.refresh_view)

    def is_connected(self):
        return self.analyzer.conn is not None

    def data_ready(self):
        return len(self.messages) > 0

    def load_messages(self):
        # load the proto modules
        cur = self.analyzer.conn.cursor()
        try:
            self.cilmod = load_proto_from_string(
                get_proto_string_from_db("cilproto", cur), "cil")
            self.regmod = load_proto_from_string(
                get_proto_string_from_db("regproto", cur), "registration")
        except Exception as e:
            self.showError(str(e))
            return False
        # load all messages and save as parameters
        regtx = self.getMessages("CollabServerTx")
        regrx = self.getMessages("CollabServerRx")
        ciltx = self.getMessages("CollabCILTx")
        cilrx = self.getMessages("CollabCILRx")
        for m in regtx:
            m.update({"type": self.ServerTx})
            self.messages.append(m)
        for m in regrx:
            m.update({"type": self.ServerRx})
            self.messages.append(m)
        for m in ciltx:
            m.update({"type": self.CollabTx})
            self.messages.append(m)
        for m in cilrx:
            m.update({"type": self.CollabRx})
            self.messages.append(m)
        self.connevent = self.readTable("CollabConnectionAttempt")
        self.statechange = self.readTable("CollabStateChange")
        self.peerevent = self.readTable("CollabPeerEvent")
        self.errorevent = self.readTable("CollabError")
        # initialize fields
        self.initFilters()
        self.initEventLog()
        self.initMsgList()
        self.refreshMsgList()
        return True

    def readTable(self, table):
        cur = self.analyzer.conn.cursor()
        # get column names first
        colnames = list()
        ret = cur.execute("PRAGMA table_info('{}')".format(table))
        for column in ret:
            colid, colname, coltype, notnull, default, pk = column
            colnames.append(colname)
        # now read and parse all proto messages
        out = list()
        data = cur.execute("SELECT * from {}".format(table))
        for raw_data in data:
            d = {k:v for k,v in zip(colnames, raw_data)}
            out.append(d)
        return out

    def getMessages(self, table):
        out = list()
        for i, dd in enumerate(self.readTable(table)):
            d = copy.copy(dd)
            if table == "CollabServerTx":
                m = self.regmod.TalkToServer()
            elif table == "CollabServerRx":
                m = self.regmod.TellClient()
            else:
                m = self.cilmod.CilMessage()
            try:
                m.ParseFromString(d["msg"])
                d["msg"] = m
                out.append(d)
            except Exception as e:
                print("WARNING: Failed to parse message {} of table {}".format(i, table))
        return out

    def refresh_view(self, *args, **kwargs):
        if self.data_ready():
            self.refreshMsgList()
            if self.currentMsg:
                text = str(self.currentMsg["msg"])
                s = "sender_network_id: "
                pat = re.compile(s + "[0-9]+\n")
                m = pat.match(text)
                if m:
                    netid = int(m.group()[len(s):])
                    text = "sender_network_id: {}\n".format(ip_address(netid)) + text[len(m.group()):]
                self.msgView.setPlainText(text)
        else:
            # try to load data
            if self.is_connected():
                if self.load_messages():
                    self.refresh_view()

    # refresh triggers
    def refreshOnTabChange(self, index):
        if index == self.tab_index:
            self.refresh_view()

    # if something goes wrong, print an error message
    def showError(self, msg):
        label = QLabel()
        hbox = QHBoxLayout()
        label.setText("ERROR: {:s}".format(msg))
        hbox.addWidget(label)
        QWidget().setLayout(self.hbox)
        self.hbox = hbox
        self.widget.setLayout(self.hbox)


# plot all transmitted spectrumusage messages
class SpectrumUsageTab(QWidget):
    def __init__(self, analyzer, collabTab, tabwidget, ntab):
        super().__init__(None)
        self.analyzer = analyzer
        self.ntab = ntab
        self.tabwidget = tabwidget
        self.collabTab = collabTab
        self.done = False

        self.vbox = QVBoxLayout()

        # a progress bar
        self.progressView = QVBoxLayout()
        hb = QHBoxLayout()
        hbw = QWidget()
        hbw.setLayout(hb)
        self.progressButton = QPushButton("Process data")
        self.progressBar = QProgressBar()
        hb.addWidget(self.progressButton)
        hb.addWidget(self.progressBar)
        self.progressButton.clicked.connect(self.loadEverything)
        self.progressView.addWidget(hbw)
        self.setLayout(self.progressView)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbox.addWidget(toolbar)
        self.vbox.addWidget(self.canvas)

        # self.connectRefresh()
        self.historical = []
        self.future = []

    def plot_data(self, data, ax):
        #ax.set_facecolor("blue")
        # get the start time
        start_time = self.analyzer.get_start_time()
        def tc(t):
            fpico = t.seconds * 1e12
            tpico = t.picoseconds + fpico
            return tpico/(1e9)
        for i, vox in enumerate(data):
            self.progressBar.setValue(self.progress)
            # get dimensions of box to draw
            relstart = (tc(vox.time_start) - start_time)/1000.
            relstop = (tc(vox.time_end) - start_time)/1000.
            freqstart = vox.freq_start
            freqstop = vox.freq_end
            duty_cycle = vox.duty_cycle.value
            # set the color
            width = relstop - relstart
            height = freqstop - freqstart
            center = freqstop - height/2
            # draw the box
            ax.barh(y=center, width=width, height=height, left=relstart, align="center",
                    color=self.cmap(duty_cycle),
                    edgecolor="black",
                    linewidth=0.5)
            self.progress = self.progress + 1

    def loadEverything(self):
        # make sure collab tab has loaded data
        if not self.collabTab.data_ready():
            if not self.collabTab.load_messages():
                return
        # get all transmitted spectrum usage messages
        for msg in self.collabTab.messages:
            if ((msg["type"] == self.collabTab.CollabTx) and
                (msg["msg"].WhichOneof("payload") == "spectrum_usage")):
                su = msg["msg"].spectrum_usage
                for svoxusg in su.voxels:
                    if svoxusg.measured_data == True:
                        self.historical.append(svoxusg.spectrum_voxel)
                    else:
                        self.future.append(svoxusg.spectrum_voxel)
        # plot the data
        self.progressBar.setRange(0, len(self.historical) + len(self.future))
        self.progress = 0
        # future
        self.cmap = get_cmap("viridis")
        fax = self.figure.add_subplot(2, 1, 1)
        self.plot_data(self.future, fax)
        fax.set_title("future")
        fax.invert_yaxis()
        # historical
        hax = self.figure.add_subplot(2, 1, 2, sharex=fax)
        self.plot_data(self.historical, hax)
        hax.set_title("historical")
        hax.invert_yaxis()
        # update the view
        QWidget().setLayout(self.progressView)
        self.setLayout(self.vbox)
        # draw the canvas
        # TODO: add standalone colorbar from self.cmap
        self.canvas.draw()
        self.done = True


class PSDTab(object):
    def __init__(self, analyzer, tabwidget, parent=None):
        self.analyzer = analyzer
        self.running = False

        # big H box
        hbox = QHBoxLayout()
        left_vbox = QVBoxLayout()
        # PSD source
        left_vbox.addWidget(QLabel("PSD Source"))
        radio_box = QHBoxLayout()
        self.psd_source_widget = QRadioButton("local")
        self.psd_source_widget.setChecked(True)
        self.psd_source_widget.toggled.connect(self.redraw_psd)
        radio_box.addWidget(self.psd_source_widget)
        radio_box.addWidget(QRadioButton("gateway"))
        radio_box_widget = QWidget()
        radio_box_widget.setLayout(radio_box)
        left_vbox.addWidget(radio_box_widget)
        # SRN IDs
        left_vbox.addWidget(QLabel('SRNs'))
        self.srns_list_widget = QListWidget()
        self.srns_list_widget.currentItemChanged.connect(self.redraw_psd)
        left_vbox.addWidget(self.srns_list_widget)
        # left V box
        left_vbox_widget = QWidget()
        left_vbox_widget.setLayout(left_vbox)
        left_vbox_widget.setMaximumWidth(300)
        hbox.addWidget(left_vbox_widget)
        # Canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        # right V box
        right_vbox = QVBoxLayout()
        figure_toolbar = NavigationToolbar(self.canvas, tabwidget)
        right_vbox.addWidget(figure_toolbar)
        right_vbox.addWidget(self.canvas)
        hbox.addLayout(right_vbox)

        self.widget = QWidget()
        self.widget.setLayout(hbox)

        self.psd_dict = { "local": {}, "gateway": {} }
        self.time_dict = { "local": {}, "gateway": {} }
        self.valid_dict = { "local": {}, "gateway": {} }

    def isConnected(self):
        return self.analyzer.conn is not None

    def redraw_psd(self):
        srnid = self.srns[self.srns_list_widget.currentRow()]
        source = "local" if self.psd_source_widget.isChecked() else "gateway"
        if self.time_dict[source][srnid].size == 0:
            return
        img_psd_array = self.psd_dict[source][srnid]
        t_end = self.time_dict[source][srnid][-1] - self.time_dict[source][srnid][0];
        t_end /= 1e3;  # ms to s
        self.figure.clear()
        if len(img_psd_array.shape) == 2:
            ax = self.figure.add_subplot(1, 1, 1)
            img_alpha = np.tile(np.invert(self.valid_dict[source][srnid]).astype(float), [img_psd_array.shape[0], 1])
            shape = img_alpha.shape
            img_mask = np.zeros([shape[0], shape[1], 4])
            img_mask[:,:,3] = img_alpha * .8
            extent = [0, t_end , (phy_sample_rate/2)/1e6, -(phy_sample_rate/2)/1e6]
            im = ax.imshow(img_psd_array, aspect='auto', interpolation='none', extent=extent)
            self.figure.colorbar(im, ax=ax)
            ax.imshow(img_mask, aspect='auto', interpolation='none', extent=extent)
            ax.set_ylabel('Freq (MHz)')
            ax.set_xlabel('Time (s)')
        self.canvas.draw()

    def process_psd(self, t_array, psd_array):
        if t_array.size <= 1 or psd_array.size == 0:
            return t_array, psd_array, np.array([])
        t_array = t_array / 1e6  # convert to ms
        t_start = self.analyzer.get_start_time()
        t_diff = np.diff(t_array).astype(int)  # round to int ms
        lst = list(t_diff)
        t_interval = max(set(lst), key=lst.count)
        if t_interval < 6:
            t_interval = 6
        t_end = t_array[-1]
        t_range = np.arange(t_start, t_end, t_interval)
        psd = []
        valid = []
        for t in t_range:
            idx = np.searchsorted(t_array, t, side='left')
            if idx > 0 and (idx == len(t_array) or abs(t - t_array[idx-1]) < abs(t - t_array[idx])):
                idx -= 1
            psd.append(psd_array[idx])
            valid.append(abs(t - t_array[idx]) <= 1000)  # valid if within 1000 ms
        return np.array(t_range), np.transpose(np.array(psd)), np.array(valid)

    def load_data(self):
        self.srns_list_widget.clear()
        self.srns = self.analyzer.get_srnIDs()
        for source in ["local", "gateway"]:
            for srnid in self.srns:
                t, psd_array = self.analyzer.get_PSD(srnid, source)
                t1, psd1, valid = self.process_psd(t, psd_array)
                self.time_dict[source][srnid] = t1
                self.psd_dict[source][srnid] = psd1
                self.valid_dict[source][srnid] = valid
        # update SRN list view
        for srnid in self.srns:
            item = QListWidgetItem()
            item.setText(str(srnid))
            self.srns_list_widget.addItem(item)


class CRateTab(PHYLinkPlot):
    """A plot of the data rate delivered to MGEN by us. """
    def __init__(self, analyzer, tabwidget, ntab):
        super().__init__(analyzer, tabwidget, ntab)
        self.data = None

    def dataReady(self):
        return self.data is not None

    def loadData(self):
        try:
            self.data = {k: pickle.loads(bytes(v)) for k, v in
                         self.analyzer.conn.cursor().execute(
                             "select * from ColosseumRate").fetchall()}
        except sqlite3.OperationalError:
            self.data = None
            return

        def make_item(k, name):
            item = QStandardItem()
            item.setCheckable(True)
            item.setSelectable(False)
            item.setData(k)
            item.setText(name)
            return item

        self.linkList.appendRow(make_item(-1, "Sum"))
        for k in self.data.keys():
            if k == -1:
                continue
            self.linkList.appendRow(make_item(k, str(k)))

    def refreshPlot(self):
        self.figure.clear()
        ax = self.figure.subplots()
        for idx in range(self.linkList.rowCount()):
            item = self.linkList.item(idx)
            if item.checkState() == Qt.Checked:
                k = item.data()
                time = self.data[k]["time"]
                rate = self.data[k]["rate"]
                lab = str(k) if k != -1 else "Sum"
                ax.plot(time, rate, label=lab)
        ax.legend()
        self.figure.tight_layout(pad=0)
        self.canvas.draw()


class MCSTab(PHYLinkPlot):
    """A plot of MCS per SRN pair"""
    def __init__(self, analyzer, tabwidget, ntab):
        super().__init__(analyzer, tabwidget, ntab)
        self.data = None
        self.max_mcs = 44
        self.dkey = "data"
        self.timekey = "reltime"
        self.timeunits = "ms"
        self.time_unit_dict = {"ms": 1000000.}

    def dataReady(self):
        return self.data is not None

    def loadData(self):
        cur = self.analyzer.conn.cursor()
        st = cur.execute("select MIN(time) from C2APIEvent where txt='START'").fetchone()[0]
        srns = [r[0] for r in set(cur.execute("select srnID from ModulationEvent"))]
        allNodesID = 255
        data = dict()
        for srn in srns:
            # get the MCS indices for all frames that were not control frames
            r = cur.execute(
                "SELECT payloadMCS, time, dstNodeID from Frame where srcNodeID = {:d} and dstNodeID <> {:d}"
                .format(srn, allNodesID))
            for dd in r:
                d = {k: v for k, v in zip(["payloadMCS", "time", "dstNodeID"], dd)}
                reltime = (d["time"] - st)/self.time_unit_dict[self.timeunits]
                link = PHYLink(srcNodeID=srn, dstNodeID=d["dstNodeID"])
                try:
                    data[link][self.dkey].append(d["payloadMCS"])
                    data[link][self.timekey].append(reltime)
                except KeyError:
                    data[link] = {self.dkey: list(), self.timekey: list()}
                    data[link][self.dkey].append(d["payloadMCS"])
                    data[link][self.timekey].append(reltime)
        self.data = OrderedDict(sorted(data.items(), key=lambda x: x[0].srcNodeID))
        for k in self.data.keys():
            item = QStandardItem()
            item.setCheckable(True)
            item.setCheckState(Qt.Unchecked)
            item.setSelectable(False)
            item.setData(k)
            item.setText(PHYLink2Str(k))
            self.linkList.appendRow(item)
        self.refreshPlot()

    def refreshPlot(self):
        self.figure.clear()
        ax = self.figure.subplots()
        for idx in range(self.linkList.rowCount()):
            item = self.linkList.item(idx)
            if item.checkState() == Qt.Checked:
                k = item.data()
                ax.scatter(self.data[k][self.timekey], self.data[k][self.dkey],
                           label=PHYLink2Str(k), marker=".")
        ax.set_yticks(np.arange(0, self.max_mcs + 1, 1))
        ax.grid(alpha=0.25)
        self.figure.legend(ncol=int(len(self.data)/5), loc="lower right")
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

class MCSReqTab(PHYLinkPlot):
    """A plot of MCS per SRN pair"""
    def __init__(self, analyzer, tabwidget, ntab):
        super().__init__(analyzer, tabwidget, ntab)
        self.data = None
        self.max_mcs = 44
        self.dkey = "data"
        self.timekey = "reltime"
        self.timeunits = "ms"
        self.time_unit_dict = {"ms": 1000000.}

    def dataReady(self):
        return self.data is not None

    def loadData(self):
        cur = self.analyzer.conn.cursor()
        st = cur.execute("select MIN(time) from C2APIEvent where txt='START'").fetchone()[0]
        srns = [r[0] for r in set(cur.execute("select srnID from MCSDecisionEvent"))]
        allNodesID = 255
        data = dict()
        for srn in srns:
            # get the MCS indices for all frames that were not control frames
            r = cur.execute(
                "SELECT payloadMCS, time, srnID from MCSDecisionEvent where txNodeID = {:d}"
                .format(srn))
            for dd in r:
                d = {k: v for k, v in zip(["payloadMCS", "time", "txNodeID"], dd)}
                reltime = (d["time"] - st)/self.time_unit_dict[self.timeunits]
                link = PHYLink(srcNodeID=srn, dstNodeID=d["txNodeID"])
                try:
                    data[link][self.dkey].append(d["payloadMCS"])
                    data[link][self.timekey].append(reltime)
                except KeyError:
                    data[link] = {self.dkey: list(), self.timekey: list()}
                    data[link][self.dkey].append(d["payloadMCS"])
                    data[link][self.timekey].append(reltime)
        self.data = OrderedDict(sorted(data.items(), key=lambda x: x[0].srcNodeID))
        for k in self.data.keys():
            item = QStandardItem()
            item.setCheckable(True)
            item.setCheckState(Qt.Unchecked)
            item.setSelectable(False)
            item.setData(k)
            item.setText(PHYLink2Str(k))
            self.linkList.appendRow(item)
        self.refreshPlot()

    def refreshPlot(self):
        self.figure.clear()
        ax = self.figure.subplots()
        for idx in range(self.linkList.rowCount()):
            item = self.linkList.item(idx)
            if item.checkState() == Qt.Checked:
                k = item.data()
                ax.scatter(self.data[k][self.timekey], self.data[k][self.dkey],
                           label=PHYLink2Str(k), marker=".")
        ax.set_yticks(np.arange(0, self.max_mcs + 1, 1))
        ax.grid(alpha=0.25)
        self.figure.legend(ncol=int(len(self.data)/5), loc="lower right")
        self.figure.tight_layout(pad=0)
        self.canvas.draw()


class NetTab(QWidget):
    def __init__(self, analyzer, tabwidget, parent=None):
        QWidget.__init__(self)
        self.analyzer = analyzer

        left_vbox = QVBoxLayout()

        # flows
        self.flows_model = QStandardItemModel()
        self.flows_view = QTableView()
        self.flows_view.setModel(self.flows_model)
        left_vbox.addWidget(QLabel('Flows'))
        left_vbox.addWidget(self.flows_view)
        self.flows_model.setHorizontalHeaderLabels(["Flow UID", "Src SRN", "Dst SRN"])
        self.flows_view.verticalHeader().hide()
        self.flows_view.resizeColumnsToContents()

        select_buttons_hbox = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.selectFlowAll)
        select_none_button = QPushButton("Select None")
        select_none_button.clicked.connect(self.selectFlowNone)
        select_buttons_hbox.addWidget(select_all_button)
        select_buttons_hbox.addWidget(select_none_button)
        left_vbox.addLayout(select_buttons_hbox)

        # data types
        self.data_types_model = QStandardItemModel()
        self.data_types_view = QListView()
        self.data_types_view.setModel(self.data_types_model)
        left_vbox.addWidget(QLabel('Data Types'))
        left_vbox.addWidget(self.data_types_view)
        for type_name in ["Offered data rate", "Delivered data rate", "Latency"]:
            item = QStandardItem(type_name)
            item.setCheckState(Qt.Unchecked)
            item.setCheckable(True)
            self.data_types_model.appendRow(item)

        # aggregate checkbox
        self.aggregate_checkbox = QCheckBox('Show aggregate statistics')
        left_vbox.addWidget(self.aggregate_checkbox)

        # plot button
        show_button = QPushButton("Plot")
        show_button.clicked.connect(self.plot)
        left_vbox.addWidget(show_button)

        # figure canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # right V box
        right_vbox = QVBoxLayout()
        figure_toolbar = NavigationToolbar(self.canvas, self)
        right_vbox.addWidget(figure_toolbar)
        right_vbox.addWidget(self.canvas)

        # left V box
        left_vbox_widget = QWidget()
        left_vbox_widget.setLayout(left_vbox)
        left_vbox_widget.setMaximumWidth(300)

        # big H box
        hbox = QHBoxLayout()
        hbox.addWidget(left_vbox_widget)
        hbox.addLayout(right_vbox)
        self.setLayout(hbox)

        self.data = None
        self.t = None
        self.avg_window = 1.0


    def selectFlowAll(self):
        for row_idx in range(self.flows_model.rowCount()):
            item = self.flows_model.item(row_idx, 0)
            item.setCheckState(Qt.Checked)

    def selectFlowNone(self):
        for row_idx in range(self.flows_model.rowCount()):
            item = self.flows_model.item(row_idx, 0)
            item.setCheckState(Qt.Unchecked)

    def loadData(self):
        # get all flow UIDs
        cur = self.analyzer.conn.cursor()
        cur.execute(
            "SELECT DISTINCT dstPort as uid FROM RouteDecision ORDER BY uid ASC"
        )
        self.flow_uids = [row[0] for row in cur.fetchall()]

        # Get flow info
        cur.execute(
            "SELECT DISTINCT(dstPort), srcIP, dstIP FROM RouteDecision"
        )
        res = cur.fetchall()
        uid_src_map = dict()
        uid_dst_map = dict()
        def ipstr2srn(addr):
            return (struct.unpack("!I", socket.inet_aton(addr))[0] >> 8 & 0xFF) - 100
        for row in res:
            uid_src_map[row[0]] = ipstr2srn(row[1])
            uid_dst_map[row[0]] = ipstr2srn(row[2])

        for uid in self.flow_uids:
            flow = QStandardItem(str(uid))
            flow.setData(uid)
            flow.setCheckable(True)
            self.flows_model.appendRow([flow,
                QStandardItem(str(uid_src_map[uid])),QStandardItem(str(uid_dst_map[uid]))])

    def compute(self):
        self.data = dict()
        self.data[0] = dict()
        self.data[1] = dict()
        for uid in self.flow_uids:
            # Compute offered rate
            t, flow_rate_raw = self.analyzer.get_offered_rate(uid)
            flow_rate = self.analyzer.get_sliding_window_sum(
                flow_rate_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
            self.data[0][uid] = flow_rate
            # Compute delivered rate
            t, flow_rate_raw = self.analyzer.get_delivered_rate(uid)
            flow_rate = self.analyzer.get_sliding_window_sum(
                flow_rate_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
            self.data[1][uid] = flow_rate

    def plot(self):
        # generate time vector
        t1_origin = self.analyzer.get_start_time()
        t1_end = self.analyzer.get_stop_time()
        self.t = np.linspace(0, (t1_end - t1_origin) / 1000, 2 * int((t1_end - t1_origin) / 1000))

        # get selected flow UIDs
        selected_uids = []
        for row_idx in range(self.flows_model.rowCount()):
            item = self.flows_model.item(row_idx, 0)
            if item.checkState() == Qt.Checked:
                selected_uids.append(item.data())

        # want aggregate data?
        aggregate = self.aggregate_checkbox.isChecked()

        if aggregate:
            # prepare data
            n_subplots = 0
            plot_data = list()
            for type_idx in range(self.data_types_model.rowCount()):
                item = self.data_types_model.item(type_idx, 0)
                if item.checkState() == Qt.Checked:
                    n_subplots += 1
                    # offered traffic
                    if type_idx == 0:
                        # Compute delivered rate
                        t, flow_rate_raw = self.analyzer.get_offered_rate(selected_uids)
                        flow_rate = self.analyzer.get_sliding_window_sum(
                            flow_rate_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
                        # generate plot
                        subplot_data = dict()
                        subplot_data["xlabel"] = "Time (s)"
                        subplot_data["ylabel"] = "Offered Traffic (Mbps)"
                        subplot_data["dataset"] = {'Aggregate' : flow_rate}
                        plot_data.append(subplot_data)
                    # delivered traffic
                    elif type_idx == 1:
                        # Compute delivered rate
                        t, flow_rate_raw = self.analyzer.get_delivered_rate(selected_uids)
                        flow_rate = self.analyzer.get_sliding_window_sum(
                            flow_rate_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
                        # generate plot
                        subplot_data = dict()
                        subplot_data["xlabel"] = "Time (s)"
                        subplot_data["ylabel"] = "Delivered Traffic (Mbps)"
                        subplot_data["dataset"] = {'Aggregate' : flow_rate}
                        plot_data.append(subplot_data)
                    # latency
                    elif type_idx == 2:
                        # Compute latency
                        t, flow_latency_raw = self.analyzer.get_latency(selected_uids)
                        flow_rate = self.analyzer.get_sliding_window_avg(
                            flow_latency_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
                        # generate plot
                        subplot_data = dict()
                        subplot_data["xlabel"] = "Time (s)"
                        subplot_data["ylabel"] = "Latency (ms)"
                        subplot_data["dataset"] = {'Aggregate' : flow_rate}
                        plot_data.append(subplot_data)
        else:
            if self.data is None:
                self.data = dict()
            for uid in selected_uids:
                # do not recompute statistics
                if uid in self.data:
                    continue
                self.data[uid] = list()
                # Compute offered rate
                t, flow_rate_raw = self.analyzer.get_offered_rate(uid)
                flow_rate = self.analyzer.get_sliding_window_sum(
                    flow_rate_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
                self.data[uid].append(flow_rate)
                # Compute delivered rate
                t, flow_rate_raw = self.analyzer.get_delivered_rate(uid)
                flow_rate = self.analyzer.get_sliding_window_sum(
                    flow_rate_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
                self.data[uid].append(flow_rate)
                # Compute latency
                t, flow_latency_raw = self.analyzer.get_latency(uid)
                flow_latency = self.analyzer.get_sliding_window_avg(
                    flow_latency_raw, (t - t1_origin) / 1000, self.t, self.avg_window)
                self.data[uid].append(flow_latency)

            # prepare data
            n_subplots = 0
            plot_data = list()
            for type_idx in range(self.data_types_model.rowCount()):
                item = self.data_types_model.item(type_idx, 0)
                if item.checkState() == Qt.Checked:
                    n_subplots += 1
                    # offered traffic
                    if type_idx == 0:
                        subplot_data = dict()
                        subplot_data["xlabel"] = "Time (s)"
                        subplot_data["ylabel"] = "Offered Traffic (Mbps)"
                        subplot_datapts = dict()
                        for uid in selected_uids:
                            subplot_datapts[uid] = self.data[uid][0]
                        subplot_data["dataset"] = subplot_datapts
                        plot_data.append(subplot_data)
                    # delivered traffic
                    elif type_idx == 1:
                        subplot_data = dict()
                        subplot_data["xlabel"] = "Time (s)"
                        subplot_data["ylabel"] = "Delivered Traffic (Mbps)"
                        subplot_datapts = dict()
                        for uid in selected_uids:
                            subplot_datapts[uid] = self.data[uid][1]
                        subplot_data["dataset"] = subplot_datapts
                        plot_data.append(subplot_data)
                    # latency
                    elif type_idx == 2:
                        subplot_data = dict()
                        subplot_data["xlabel"] = "Time (s)"
                        subplot_data["ylabel"] = "Latency (ms)"
                        subplot_datapts = dict()
                        for uid in selected_uids:
                            subplot_datapts[uid] = self.data[uid][2]
                        subplot_data["dataset"] = subplot_datapts
                        plot_data.append(subplot_data)


        # reset figure
        self.figure.clear()

        # plot data
        nplot = len(plot_data)
        si = 1
        for subplot_data in plot_data:
            ax = self.figure.add_subplot(nplot, 1, si)
            for key, data in subplot_data["dataset"].items():
                ax.plot(self.t, data)
            ax.set_ylabel(subplot_data["ylabel"])
            ax.set_xlabel(subplot_data["xlabel"])
            si += 1

        self.figure.tight_layout()
        # refresh canvas
        self.canvas.draw()
        

class Window(QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setWindowTitle('BAM! Wireless Log Analyzer')

        # Log analyzer
        self.analyzer = LogAnalyzer()

        # flow uids
        self.flow_uid_list_model = QStandardItemModel()

        # data types
        self.data_type_list_model = QStandardItemModel()
        for type_name in [
                "Offered data rate",
                "Delivered data rate",
                "Latency",
                "Individual mandates (reported)",
                "Score (reported)",
                "Score (scoring tool)",
                "Individual mandates (computed)",
                "Individual mandates (StatCenter)",
        ]:
            item = QStandardItem(type_name)
            item.setCheckState(Qt.Unchecked)
            item.setCheckable(True)
            self.data_type_list_model.appendRow(item)

        self.create_menu()
        self.create_main_frame()

    def create_main_frame(self):
        # tabs
        self.tabs = QTabWidget()

        # === Create Net tab ===
        left_vbox = QVBoxLayout()
        # Flow UIDs
        flow_uid_list_view = QListView()
        left_vbox.addWidget(QLabel('Flow UIDs'))
        left_vbox.addWidget(flow_uid_list_view)
        flow_uid_list_view.setModel(self.flow_uid_list_model)
        # data type
        data_type_list_view = QListView()
        left_vbox.addWidget(QLabel('Data Types'))
        left_vbox.addWidget(data_type_list_view)
        data_type_list_view.setModel(self.data_type_list_model)
        # plot button
        show_button = QPushButton("&Plot")
        show_button.clicked.connect(self.plot)
        left_vbox.addWidget(show_button)
        # figures
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        # right V box
        right_vbox = QVBoxLayout()
        figure_toolbar = NavigationToolbar(self.canvas, self)
        right_vbox.addWidget(figure_toolbar)
        right_vbox.addWidget(self.canvas)
        # left V box
        left_vbox_widget = QWidget()
        left_vbox_widget.setLayout(left_vbox)
        left_vbox_widget.setMaximumWidth(300)
        # big H box
        hbox = QHBoxLayout()
        hbox.addWidget(left_vbox_widget)
        hbox.addLayout(right_vbox)

        ## Tabs
        ntab = 0

        # === add NET tab ==
        self.netTab = NetTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.netTab, "NET (new)")
        ntab += 1
        
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "NET")
        self.tab1.setLayout(hbox)
        ntab += 1

        # === add ColosseumRate tab ===
        self.CRateTab = CRateTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.CRateTab, "NET-DELIVER")
        ntab += 1

        # === add FRAME tab ===
        self.frameTab = FrameTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.frameTab, "FRAME")
        ntab += 1
        
        # === add FRAME tab ===
        self.blockTab = BlockTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.blockTab, "BLOCK")
        ntab += 1


        # === add SNR tab ===
        self.snrTab = SNRTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.snrTab, "SNR")
        ntab += 1

        self.noisevarTab = SNRTab(self.analyzer, self.tabs, ntab, "noiseVar")
        self.tabs.addTab(self.noisevarTab, "NOISEVAR")
        ntab += 1

        # === add FREQALLOC tab ===
        self.allocTab = AllocTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.allocTab, "FREQALLOC")
        ntab += 1

        # === add PSD tab ==
        self.psdTab = PSDTab(self.analyzer, self.tabs)
        self.tabs.addTab(self.psdTab.widget, "PSD")
        ntab += 1

        # === add collab tab ==
        self.collabTab = CollabTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.collabTab.widget, "COLLAB")
        ntab += 1

        # === add spectrum usage tab ==
        self.spectrumUsageTab = SpectrumUsageTab(self.analyzer, self.collabTab, self.tabs, ntab)
        self.tabs.addTab(self.spectrumUsageTab, "SPECTRUM-USAGE")
        ntab += 1

        # === add MCS tab ===
        self.mcsTab = MCSTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.mcsTab, "MCS")
        ntab += 1

        # === add MCS tab ===
        self.mcsReqTab = MCSReqTab(self.analyzer, self.tabs, ntab)
        self.tabs.addTab(self.mcsReqTab, "MCS-REQ")
        ntab += 1

        # === add misc tab ==
        self.miscTab = MiscTab(self.analyzer, self.tabs)
        self.tabs.addTab(self.miscTab.widget, "MISC")
        ntab += 1

        # === add console tab ==
        self.consoleTab = ConsoleTab(self.analyzer, self, self.tabs)
        self.tabs.addTab(self.consoleTab.widget, "CONSOLE")
        ntab += 1

        # === Add Tabs to the Window ==
        self.setCentralWidget(self.tabs)

    def create_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        openButton = QAction('Open SQLite DB', self)
        openButton.triggered.connect(self.load_file)
        file_menu.addAction(openButton)

        aboutButton = QAction('About', self)
        aboutButton.triggered.connect(self.show_about)
        file_menu.addAction(aboutButton)

        exitButton = QAction('Exit', self)
        exitButton.triggered.connect(self.close)
        file_menu.addAction(exitButton)

    def load_file(self, filename=None):
        filename, _filt = QFileDialog.getOpenFileName(
            self, 'Open a data file', '.',
            'SQLite DB files (*.db);;All Files (*.*)')
        if filename:
            self.analyzer.connect_sqlite(filename)
            self.analyzer.load_mandates()
            flow_uids = self.analyzer.get_flows()
            for uid in flow_uids:
                item = QStandardItem(uid)
                item.setText(str(uid))
                item.setCheckState(Qt.Unchecked)
                item.setCheckable(True)
                self.flow_uid_list_model.appendRow(item)
            self.psdTab.load_data()
            self.netTab.loadData()
            # change title to something informational
            self.setTitle()

    def setTitle(self):
        info = self.analyzer.get_res_info()
        title_str = "Reservation {} ({}) - {} ({}) - BAM! Radio @ {}".format(
            info["resid"], info["restime"], info["scname"], info["scid"], info["hash"])
        self.setWindowTitle(title_str)

    def plot(self):
        # get selected flow UIDs
        selected_uids = []
        for row in range(self.flow_uid_list_model.rowCount()):
            idx = self.flow_uid_list_model.index(row, 0)
            checked = self.flow_uid_list_model.data(
                idx, Qt.CheckStateRole) == QVariant(Qt.Checked)
            flow_uid = self.flow_uid_list_model.data(idx)
            if checked:
                selected_uids.append(flow_uid)

        # count how many subplots we need
        n_subplots = 0
        for row in range(self.data_type_list_model.rowCount()):
            idx = self.data_type_list_model.index(row, 0)
            checked = self.data_type_list_model.data(
                idx, Qt.CheckStateRole) == QVariant(Qt.Checked)
            if checked:
                n_subplots += 1

        subplot_idx = 1
        avg_window = 1
        t1_origin = self.analyzer.get_start_time()
        duration = self.analyzer.get_stop_time() - t1_origin
        t_out = np.linspace(0., duration / 1000, 1000)

        # reset figure
        self.figure.clear()

        idx = self.data_type_list_model.index(0, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            if len(selected_uids) > 0:
                for uid in selected_uids:
                    t1, flow_rate_raw = self.analyzer.get_offered_rate(uid)
                    flow_rate_sum = self.analyzer.get_sliding_window_sum(
                        flow_rate_raw, (t1 - t1_origin) / 1000, t_out, avg_window)
                    ax.plot(t_out, np.array(flow_rate_sum) / avg_window, label=uid)
                    ax.legend()
            else:
                t1, flow_rate_raw = self.analyzer.get_offered_rate()
                flow_rate_sum = self.analyzer.get_sliding_window_sum(
                    flow_rate_raw, (t1 - t1_origin) / 1000, t_out, avg_window)
                ax.plot(t_out, np.array(flow_rate_sum) / avg_window, label='All')
                ax.legend()
            ax.set_ylabel('Rate (Mbps)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Offered Traffic Rate')
            subplot_idx += 1

        idx = self.data_type_list_model.index(1, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            if len(selected_uids) > 0:
                for uid in selected_uids:
                    tman,rateman = self.analyzer.get_mandate_rate(int(uid))
                    ax.step(tman,rateman,linestyle='--',where='post',label='mandate_'+str(uid))
                    t1, flow_rate_raw = self.analyzer.get_delivered_rate(uid)
                    flow_rate_sum = self.analyzer.get_sliding_window_sum(
                        flow_rate_raw, (t1 - t1_origin) / 1000, t_out, avg_window)
                    ax.plot(t_out, np.array(flow_rate_sum) / avg_window, label=uid)
                    ax.legend()
            else:
                t1, flow_rate_raw = self.analyzer.get_delivered_rate()
                flow_rate_sum = self.analyzer.get_sliding_window_sum(
                    flow_rate_raw, (t1 - t1_origin) / 1000, t_out, avg_window)
                ax.plot(t_out, np.array(flow_rate_sum) / avg_window, label='All')
                ax.legend()
            ax.set_ylabel('Rate (Mbps)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Delivered Traffic Rate')
            subplot_idx += 1
        

        idx = self.data_type_list_model.index(2, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            if len(selected_uids) > 0:
                for uid in selected_uids:
                    tman,latencyman = self.analyzer.get_mandate_latency(int(uid))
                    ax.step(tman,latencyman,linestyle='--',where='post',label='mandate_'+str(uid))
                    t1, flow_latency_raw = self.analyzer.get_latency(uid)
                    flow_rate_sum = self.analyzer.get_sliding_window_avg(
                        flow_latency_raw, (t1 - t1_origin) / 1000, t_out,
                        1)
                    ax.plot(t_out, np.array(flow_rate_sum), label=uid)
                    ax.legend()
            else:
                t1, flow_latency_raw = self.analyzer.get_latency()
                flow_rate_sum = self.analyzer.get_sliding_window_avg(
                    flow_latency_raw, (t1 - t1_origin) / 1000, t_out,
                    avg_window)
                ax.plot(t_out, np.array(flow_rate_sum), label='All')
                ax.legend()
            ax.set_ylabel('Latency (msec)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Packet Latency')
            subplot_idx += 1

        idx = self.data_type_list_model.index(3, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            mandates_map = self.analyzer.get_individual_mandates("mandates")
            for type in ["us", "peer", "incumbent"]:
                for netid, d in mandates_map.items():
                    if d["type"] != type:
                        continue
                    t = np.array(d["time"]) / 1e9 - t1_origin / 1e3
                    mandates = np.array(d["mandates_achieved"])
                    label = "{} ({})".format(ip_address(netid), d["type"])
                    ax.plot(t, mandates, label=label)
            ax.legend()
            ax.set_ylabel('# Mandates')
            ax.set_xlabel('Time (s)')
            ax.set_title('Individual Mandates Achieved (reported)')
            subplot_idx += 1

        idx = self.data_type_list_model.index(4, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            scores_map = self.analyzer.get_individual_mandates("score")
            for type in ["us", "peer", "incumbent"]:
                for netid in sorted(list(scores_map)):
                    d = scores_map[netid]
                    if d["type"] != type:
                        continue
                    t = np.array(d["time"]) / 1e9 - t1_origin / 1e3
                    score = np.array(d["score"])
                    threshold = np.array(d["threshold"])
                    label = "{} ({})".format(ip_address(netid), d["type"])
                    color = next(ax._get_lines.prop_cycler)['color']
                    ax.plot(t, score, label=label, color=color)
                    ax.plot(t, threshold, color=color, linestyle=':')
            ax.legend()
            ax.set_ylabel('Score')
            ax.set_xlabel('Time (s)')
            ax.set_title('Score (reported)')
            subplot_idx += 1

        idx = self.data_type_list_model.index(5, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            scores_map = self.analyzer.get_scores_tool()
            for type in ["us", "peer", "ensemble"]:
                for teamname in sorted(list(scores_map), key=lambda k: scores_map[k]["network_id"]):
                    d = scores_map[teamname]
                    if d["type"] != type:
                        continue
                    t = np.array(d["time"])
                    score = np.array(d["score"])
                    label = "{} ({})".format(teamname, ip_address(d["network_id"])
                                             if d["type"] != "ensemble" else "")
                    linestyle = ':' if type == "ensemble" else '-'
                    ax.plot(t, score, label=label, linestyle=linestyle)
            ax.legend()
            ax.set_ylabel('Score')
            ax.set_xlabel('Time (s)')
            ax.set_title('Score (scoring tool)')
            subplot_idx += 1

        # compute sum of selected individual mandates achieved based on database 
        idx = self.data_type_list_model.index(6, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            tout = frange(0,duration/1000.)
            num_man = np.array([0]*len(tout))
            if len(selected_uids) > 0:
                for uid in selected_uids:
                    print('computing mandate plot for flow ',uid,'...')
                    if 'file_transfer_deadline_s' in self.analyzer.mandate[int(uid)].keys():
                        print('file transfer flow detected')
                        num_man += self.analyzer.compute_mandate_file(int(uid),tout)
                    else:
                        num_man += self.analyzer.compute_mandates(int(uid))
                ax.plot(tout,num_man, label='Selected flows')
                print('Done.')
            else:
                for uid in self.analyzer.mandate.keys():
                    print('computing mandate plot for flow ',uid,'...')
                    if 'file_transfer_deadline_s' in self.analyzer.mandate[int(uid)].keys():
                        num_man += self.analyzer.compute_mandate_file(int(uid),tout)
                        print('file transfer flow detected')
                    else:
                        num_man += self.analyzer.compute_mandates(int(uid))
                ax.plot(tout,num_man, label='All')
                ax.legend()
                print('Done.')
            ax.set_ylabel('# Mandates')
            ax.set_xlabel('Time (s)')
            ax.set_title('Individual Mandates Achieved (computed)')
            subplot_idx += 1
        
        idx = self.data_type_list_model.index(7, 0)
        if self.data_type_list_model.data(idx, Qt.CheckStateRole) == QVariant(
                Qt.Checked):
            ax = self.figure.add_subplot(n_subplots, 1, subplot_idx)
            t1, nims = self.analyzer.get_nachieved_ims()
            ax.plot(t1, nims, label='All')
            ax.legend()
            ax.set_ylabel('# Mandates')
            ax.set_xlabel('Time (s)')
            ax.set_title('Individual Mandates Achieved (StatCenter)')
            subplot_idx += 1
 
        self.figure.tight_layout()
        # refresh canvas
        self.canvas.draw()

    def show_about(self):
        QMessageBox.about(
            self, "About",
            "BAM! Wireless Log Analyzer. (c) 2018 Tomohiro Arakawa.")


class LogAnalyzer(object):
    # Constants for link-flow & flow-link mapping
    # DST Port Replacement String
    DST_PORT_REPLACEMENT_STRING = '${dstPort}'

    # NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID = Usually 255...I believe I shouldn't care for this link
    NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID = '${NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID}'

    # Non-relevant node for this visualization
    NON_RELEVANT_NODE = 255

    # Source SRN ID Replacement String
    SRC_NODE_ID_REPLACEMENT_STRING = '${SRC_SRN_ID}'

    # Next Hop SRN ID Replacement String
    NEXT_HOP_NODE_ID_REPLACEMENT_STRING = '${NEXT_HOP}'

    # Flow UIDs DB Query
    FLOW_UIDS_DB_QUERY = 'SELECT DISTINCT dstPort as Flow_UID FROM RouteDecision ORDER BY Flow_UID ASC'

    # Link Path Per Flow DB Query - Forward
    LINK_PATH_DB_QUERY_PER_FORWARD_FLOW = 'SELECT DISTINCT srnID,nextHop FROM RouteDecision ' \
                                          ' WHERE dstPort = ${dstPort} AND action = "Forward" ORDER BY time ASC'

    # Link Path Per Flow DB Query - WriteToTUN
    # TODO: Is this required for functional and/or performance analysis?
    LINK_PATH_DB_QUERY_PER_WRITE_TO_TUN_FLOW = 'SELECT DISTINCT srnID,nextHop FROM RouteDecision WHERE dstPort = ' \
                                               '${dstPort} AND action = "WriteToTun" ORDER BY time ASC'

    # SRNs DB Query
    SRNS_DB_QUERY = 'SELECT DISTINCT srnID FROM RouteDecision ORDER BY srnID ASC'

    # LIST OF LINKS DB Query
    LIST_OF_LINKS_DB_QUERY = 'SELECT DISTINCT srcNodeID,dstNodeID FROM Frame WHERE ' \
                             'srcNodeID != ${NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID} AND ' \
                             'dstNodeID != ${NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID}'

    GET_ASSOCIATED_FLOWS = 'SELECT DISTINCT dstPort FROM RouteDecision WHERE srnID = ${SRC_SRN_ID} ' \
                           'AND nextHop = ${NEXT_HOP}'

    def __init__(self, filename=None):
        self.conn = None
        self.mandate = {}
        # DB File Path
        self.db_file_path = ''
        # Data structures for link-flow & flow-link mapping
        # A dictionary mapping the links used for every flow
        self.links_per_flow = {}
        # A dictionary mapping the flows over each link
        self.flows_per_link = {}
    def create_connection(self, db_file):
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Exception as e:
            print(e)
        return None

    def connect_sqlite(self, filename=None):
        self.db_file_path = filename
        self.conn = self.create_connection(filename)
    
    def load_mandates(self):
        # parse json text from MandatedOutcomeUpdate
        cur = self.conn.cursor()
        cur.execute(
            "SELECT json, time FROM MandatedOutcomeUpdate ORDER BY time ASC"
        )
        rows = cur.fetchall()
        # each row is a tuple
        for row in rows:
          man_list = json.loads(row[0])
          t=int(row[1])
          for flow_dict in man_list:
              uid = flow_dict['flow_uid']
              if uid not in self.mandate.keys():
                  self.mandate[uid]={}
              for req in flow_dict['requirements'].keys():
                  if req not in self.mandate[uid].keys():
                      self.mandate[uid][req]=flow_dict['requirements'][req]
                      self.mandate[uid]['time']=t
                      self.mandate[uid]["holdsec"]=flow_dict["hold_period"]
        fd = open('sql.dat','wb')
        cPickle.dump(self.mandate,fd)

    def get_start_time(self):
        # return start time in milliseconds
        cur = self.conn.cursor()
        cur.execute("select MIN(time) from C2APIEvent where txt='START'")
        return cur.fetchone()[0] / 1000000

    def get_stop_time(self):
        # return stop time in milliseconds
        cur = self.conn.cursor()
        cur.execute("select MAX(time) from C2APIEvent where txt='STOP'")
        return cur.fetchone()[0] / 1000000

    def get_srnIDs(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT srnID as d_srnid FROM Start ORDER BY d_srnid ASC")
        rows = cur.fetchall()
        srns = []
        for row in rows:
            srns.append(row[0])
        return srns

    def get_flows(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT dstPort as uid FROM RouteDecision ORDER BY uid ASC"
        )
        rows = cur.fetchall()
        flows = []
        for row in rows:
            flows.append(row[0])
        return flows

    ###################################################################################################################

    # https://cloudradio.ecn.purdue.edu/redmine/issues/37: Link-Flow & Flow-Link Mapping Feature

    # An auto-closing, auto-committing, cleaner DB access method
    def db_query(self, statement):
        with contextlib.closing(sqlite3.connect(self.db_file_path)) as connection:
            with connection:
                with contextlib.closing(connection.cursor()) as cursor:
                    result = cursor.execute(statement).fetchall()
        return result

    # Get the flows
    def get_flow_uids(self):
        # Return an array of the Flow_UIDs
        return [dst_port[0] for dst_port in self.db_query(self.FLOW_UIDS_DB_QUERY)]

    # Get the links
    def get_links(self):
        links_collection = []
        links = self.db_query(
            self.LIST_OF_LINKS_DB_QUERY.replace(self.NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID, str(self.NON_RELEVANT_NODE)))
        for link in links:
            links_collection.append('[' + str(link[0]) + '\u2192' + str(link[1]) + ']')
        return links_collection

    # Get the links for every flow
    def get_link_mappings_for_flows(self):
        links_mapped_to_flows = {}
        flows = self.get_flow_uids()
        for flow in flows:
            dev_string = ''
            links = self.db_query(
                self.LINK_PATH_DB_QUERY_PER_FORWARD_FLOW.replace(self.DST_PORT_REPLACEMENT_STRING, str(flow)))
            for link in links:
                dev_string += '[' + str(link[0]) + '\u2192' + str(link[1]) + ']' + ', '
            links_mapped_to_flows[flow] = dev_string[:len(dev_string) - 2]
        # Set the value to the instance variable in case some one wants to use this
        self.links_per_flow = links_mapped_to_flows
        # Also, return the map like a conventional method
        return links_mapped_to_flows

    # Get the flows over individual links
    def get_flow_mappings_for_links(self):
        flow_mappings = {}
        links = self.db_query(
            self.LIST_OF_LINKS_DB_QUERY.replace(self.NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID,
                                                str(self.NON_RELEVANT_NODE)))
        for link in links:
            dev_string = ''
            associated_flows = self.db_query(
                self.GET_ASSOCIATED_FLOWS.replace(self.SRC_NODE_ID_REPLACEMENT_STRING, str(link[0])).replace(
                    self.NEXT_HOP_NODE_ID_REPLACEMENT_STRING, str(link[1])))
            for flow in associated_flows:
                dev_string += str(flow[0]) + ', '
            flow_mappings[link] = dev_string[:len(dev_string) - 2]
        # Set the value to the instance variable in case some one wants to use this
        self.flows_per_link = flow_mappings
        # Also, return the map like a conventional method
        return flow_mappings

    # Get the links for every flow for table visualization
    def get_link_mappings_for_flows_for_visualization(self):
        links_mapped_to_flows = {}
        flows = self.get_flow_uids()
        for flow in flows:
            dev_string = ''
            links = self.db_query(
                self.LINK_PATH_DB_QUERY_PER_FORWARD_FLOW.replace(self.DST_PORT_REPLACEMENT_STRING, str(flow)))
            for link in links:
                dev_string += '[' + str(link[0]) + '\u2192' + str(link[1]) + ']' + ', '
            links_mapped_to_flows[str(flow)] = dev_string[:len(dev_string) - 2]
        # Set the value to the instance variable in case some one wants to use this
        self.links_per_flow = links_mapped_to_flows
        # Also, return the map like a conventional method
        return links_mapped_to_flows

    # Get the flows over individual links for table visualization
    def get_flow_mappings_for_links_for_table_visualization(self):
        flow_mappings = {}
        links = self.db_query(
            self.LIST_OF_LINKS_DB_QUERY.replace(self.NON_RELEVANT_LINK_SRC_OR_DST_NODE_ID,
                                                str(self.NON_RELEVANT_NODE)))
        for link in links:
            link_string = '[' + str(link[0]) + '\u2192' + str(link[1]) + ']'
            dev_string = ''
            associated_flows = self.db_query(
                self.GET_ASSOCIATED_FLOWS.replace(self.SRC_NODE_ID_REPLACEMENT_STRING, str(link[0])).replace(
                    self.NEXT_HOP_NODE_ID_REPLACEMENT_STRING, str(link[1])))
            for flow in associated_flows:
                dev_string += str(flow[0]) + ', '
            flow_mappings[link_string] = dev_string[:len(dev_string) - 2]
        # Set the value to the instance variable in case some one wants to use this
        self.flows_per_link = flow_mappings
        # Also, return the map like a conventional method
        return flow_mappings

    ###################################################################################################################

    def flow2link2flow(self):
        cur = self.conn.cursor()
        ftl = cur.execute("select distinct dstPort,srnID,nextHop from RouteDecision where nextHop < 253 order by dstPort asc").fetchall()
        flow2link = dict()
        link2flow = dict()
        for elem in ftl:
            flow2link[elem[0]] = list()
            link2flow[(elem[1], elem[2])] = list()
        for elem in ftl:
            flow2link[elem[0]].append((elem[1], elem[2]))
            link2flow[(elem[1], elem[2])].append(elem[0])
        return flow2link, link2flow

    def flow2endpt(self):
        def pull_srnid_ip(ip):
            return int(ip.split(".")[2]) - 100
        cur = self.conn.cursor()
        fte = cur.execute("select distinct dstPort,srcIP,dstIP from RouteDecision order by dstPort asc;")
        o = dict()
        for elem in fte:
            o[elem[0]] = (pull_srnid_ip(elem[1]), pull_srnid_ip(elem[2]))
        return o

    def get_sliding_window_sum(self, data, t_in, t_out, t_interval):
        data_out = [0] * len(t_out)
        in_idx = 0
        if len(t_in) <= 1:
            return data_out
        for out_idx in range(len(t_out)):
            while t_in[in_idx] < t_out[out_idx]:
                in_idx += 1
                if in_idx >= len(data):
                    in_idx -= 1
                    break
            i = in_idx - 1
            sum_window = 0
            while i >= 0 and (t_out[out_idx] - t_in[i]) < t_interval:
                sum_window += data[i]
                i -= 1
            data_out[out_idx] = sum_window
        return data_out

    def get_sliding_window_avg(self, data, t_in, t_out, t_interval):
        data_out = [0] * len(t_out)
        in_idx = 0
        if len(t_in) <= 1:
            return data_out
        for out_idx in range(len(t_out)):
            while t_in[in_idx] < t_out[out_idx]:
                in_idx += 1
                if in_idx >= len(data):
                    in_idx -= 1
                    break
            i = in_idx - 1
            sum_window = 0
            nitems = 0
            while i >= 0 and (t_out[out_idx] - t_in[i]) < t_interval:
                sum_window += data[i]
                i -= 1
                nitems += 1
            if nitems > 0:
                data_out[out_idx] = sum_window / nitems
            else:
                data_out[out_idx] = 0
        return data_out

    def get_offered_rate(self, flow_uid=None):
        cur = self.conn.cursor()
        if flow_uid is None:
            cur.execute(
                "SELECT time/1000000 AS time_sec, SUM(packetLength - 28) FROM RouteDecision WHERE srcNodeID=254 GROUP BY time_sec"
            )
        elif type(flow_uid) is list:
            cur.execute(
                "SELECT time/1000000 AS time_sec, SUM(packetLength - 28) FROM RouteDecision WHERE (srcNodeID=254 AND dstPort in (" + ",".join(str(v) for v in flow_uid) + ")) GROUP BY time_sec")
        else:
            cur.execute(
                "SELECT time/1000000 AS time_sec, SUM(packetLength - 28) FROM RouteDecision WHERE (srcNodeID=254 AND dstPort=%s) GROUP BY time_sec"
                % flow_uid)

        rows = cur.fetchall()
        rate = []
        t = []

        for row in rows:
            t.append(row[0])
            rate.append(8 * row[1] / 1000000)  # rate in Mbps

        return np.array(t), np.array(rate)
    
    def get_mandate_rate(self, uid):
        if 'min_throughput_bps' in self.mandate[int(uid)].keys():
            rateman = [self.mandate[int(uid)]['min_throughput_bps']/1000000.]*2
            tman = [self.mandate[int(uid)]['time']/1000000., self.get_stop_time()]
            tman = (np.array(tman) - self.get_start_time()) / 1000.
            rateman=np.array(rateman)
        else:
            tman=np.array([])
            rateman=np.array([])
        return tman,rateman
    
    def get_mandate_latency(self, uid):
        if 'max_latency_s' in self.mandate[int(uid)].keys():
            rateman = [self.mandate[int(uid)]['max_latency_s'] * 1000.]*2
            tman = [self.mandate[int(uid)]['time']/1000000.,self.get_stop_time()] 
            tman = (np.array(tman) - self.get_start_time())/1000.
            rateman=np.array(rateman)
        else:
            tman=np.array([])
            rateman=np.array([])
        return tman,rateman
    
    def compute_mandates(self, uid):
        tout = frange(0.,(self.get_stop_time() - self.get_start_time())/1000.)
        manout = np.array([0]*len(tout))
        tman,rateman=self.get_mandate_rate(uid)
        if rateman.size==0:
            return manout
        time,packets=self.drop_packet(uid) # return values are arrays
        holdsec=self.mandate[uid]['holdsec']
        rman=rateman[0]
        t_hold_start = tman[0]
        for i, t in enumerate(tout):
            if t < tman[0]:
                continue
            r=np.sum(packets[np.where((time>t) & (time<=t+1))])
            if r < rman:
                t_hold_start = t
            elif t-t_hold_start > holdsec:
                manout[i] += 1
        return manout

    def get_nachieved_ims(self):
        cur = self.conn.cursor()
        # hack warning
        cur.execute(
            "SELECT time/1000000000 AS time_sec, SUM(nAchievedIMs) FROM AchievedIMsUpdate GROUP BY time_sec"
        )
        rows = cur.fetchall()
        sumims = []
        t = []
        for row in rows:
            t.append(row[0] - self.get_start_time() / 1000)
            sumims.append(row[1])  # rate in Mbps
        return np.array(t), np.array(sumims)

    def get_delivered_rate(self, flow_uid=None):
        cur = self.conn.cursor()
        if flow_uid is None:
            cur.execute(
                "SELECT time/1000000 AS time_sec, SUM(packetLength - 28) FROM RouteDecision WHERE action='WriteToTun' GROUP BY time_sec"
            )
        elif type(flow_uid) is list:
            cur.execute(
                "SELECT time/1000000 AS time_sec, SUM(packetLength - 28) FROM RouteDecision WHERE (action='WriteToTun' AND dstPort in (" + ",".join(str(v) for v in flow_uid) + ")) GROUP BY time_sec")
        else:
            cur.execute(
                "SELECT time/1000000 AS time_sec, SUM(packetLength - 28) FROM RouteDecision WHERE (action='WriteToTun' AND dstPort=%s) GROUP BY time_sec"
                % flow_uid)
        rows = cur.fetchall()
        rate = []
        t = []

        for row in rows:
            t.append(row[0])
            rate.append(8 * row[1] / 1000000)  # rate in Mbps

        return np.array(t), np.array(rate)
    
    # added by Diyu to plot mandated throughput

    def get_latency(self, flow_uid=None):
        cur = self.conn.cursor()
        if flow_uid is None:
            cur.execute(
                "SELECT time/1000000 AS time_sec, AVG(time - sourceTime) FROM RouteDecision WHERE action='WriteToTun' GROUP BY time_sec"
            )
        elif type(flow_uid) is list:
            cur.execute(
                "SELECT time/1000000 AS time_sec, AVG(time - sourceTime) FROM RouteDecision WHERE (action='WriteToTun' AND dstPort in (" + ",".join(str(v) for v in flow_uid) + ")) GROUP BY time_sec")
        else:
            cur.execute(
                "SELECT time/1000000 AS time_sec, AVG(time - sourceTime) FROM RouteDecision WHERE (action='WriteToTun' AND dstPort=%s) GROUP BY time_sec"
                % flow_uid)
        rows = cur.fetchall()
        delay = []
        t = []

        for row in rows:
            t.append(row[0])
            delay.append(row[1] / 1000000)  # latency in msec

        return np.array(t), np.array(delay)
    
    # given a flow uid, this function drops all packets that do not meet min_latency mandate
    def drop_packet(self, flow_uid):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT sourcetime/1000000 AS time_sec, time - sourceTime, packetLength FROM RouteDecision WHERE (action='WriteToTun' AND dstPort=%s)"
                % flow_uid)
        rows = cur.fetchall()
        packets = []
        time = []
        latencyman = self.mandate[flow_uid]['max_latency_s'] * 1000.
        for row in rows:
            t=(row[0] - self.get_start_time()) /1000.
            d=row[1] / 1000000.
            r=8 * row[2] / 1000000. # Mbits
            if d < latencyman:
                time.append(t)
                packets.append(r)  # latency in msec
        return np.array(time),np.array(packets)

    def compute_mandate_file(self, uid, measure_period):
        manout = np.array([0]*len(measure_period))
        cur = self.conn.cursor()
        #select packets from dstPort
        cur.execute(
            "SELECT sourceTime/1000000 AS time_sec, (time - sourceTime)/1000000 FROM RouteDecision WHERE (action = 'WriteToTun' AND dstPort=%s)"
                % uid)
        rows = cur.fetchall()
        delay = []
        srctime = []
        for row in rows:
            srctime.append(row[0]) 
            delay.append(row[1])
        srctime=(np.array(srctime) - self.get_start_time()) / 1000. # timestamp in sec
        delay=np.array(delay) # actual latency in msec
        # select packets from source 
        cur.execute(
            "SELECT sourceTime/1000000 AS time_sec FROM RouteDecision WHERE (action = 'Forward' AND dstPort=%s AND srcNodeID=254)"
                % uid)
        rows = cur.fetchall()
        srctime_src = []
        for row in rows:
            srctime_src.append(row[0]) 
        srctime_src=(np.array(srctime_src) - self.get_start_time()) / 1000. # timestamp in sec
        #print(srctime_src)
        deadline = self.mandate[uid]['file_transfer_deadline_s'] * 1000. # mandated latency in msec
        holdsec=self.mandate[uid]['holdsec']
        # first iteration: mark all measurement periods that satisfy IM without considering holdsec
        for i,t in enumerate(measure_period):
            file_pac_idx = np.where((srctime>t) & (srctime<=t+1))
            file_pac_idx_src = np.where((srctime_src>t) & (srctime_src<=t+1))
            #print(t,np.shape(file_pac_idx_src)[1])
            if np.shape(file_pac_idx_src)[1]==0:
                manout[i] = manout[i-1]
            elif np.shape(file_pac_idx_src)[1] > np.shape(file_pac_idx)[1]:
                #print(t,'pakcets dropped')
                manout[i] = 0
            elif all(e<deadline for e in delay[file_pac_idx]):
                #print(t,'mandate met')
                manout[i] = 1
        # second iteration: holdsec
        holdcount = 0
        for i, m in enumerate(manout):
            holdcount += 1
            if m == 0:
                holdcount = 0
            elif holdcount > holdsec:
                manout[i] += 1

        return manout//2
                

    def get_PSD(self, srn_id, source = "local"):
        table_name, column_name = ("PSDUpdateEvent", "srnID") if source == "local" else ("PSDRxEvent", "srcNodeID")
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT time_ns, psd FROM {} WHERE {}={} ORDER BY time_ns ASC"
                .format(table_name, column_name, srn_id))
            rows = cur.fetchall()
        except:
            rows = []
        psd = []
        t = []
        for row in rows:
            t.append(row[0])
            psd_vec = np.fromstring(row[1], dtype='<f4')
            psd.append(psd_vec)
        psd_db = 10 * np.log10(np.array(psd))
        psd_db = np.clip(psd_db, -55, None)
        return np.array(t), psd_db

    def get_individual_mandates(self, option):
        # dirtiest hack ever
        cur = self.conn.cursor()
        cilmod = load_proto_from_string(
            get_proto_string_from_db("cilproto", cur), "cil")
        table = "CollabCILTx"
        colnames = list()
        ret = cur.execute("PRAGMA table_info('{}')".format(table))
        for column in ret:
            colid, colname, coltype, notnull, default, pk = column
            colnames.append(colname)
        out1 = list()
        data = cur.execute("SELECT * from {}".format(table))
        for raw_data in data:
            d = {k:v for k,v in zip(colnames, raw_data)}
            out1.append(d)
        out2 = list()
        for i, dd in enumerate(out1):
            d = copy.copy(dd)
            m = cilmod.CilMessage()
            try:
                m.ParseFromString(d["msg"])
                d["msg"] = m
                out2.append(d)
            except Exception as e:
                print("WARNING: Failed to parse message {} of table {}".format(i, table))
        ciltx = out2
        table = "CollabCILRx"
        colnames = list()
        ret = cur.execute("PRAGMA table_info('{}')".format(table))
        for column in ret:
            colid, colname, coltype, notnull, default, pk = column
            colnames.append(colname)
        out1 = list()
        data = cur.execute("SELECT * from {}".format(table))
        for raw_data in data:
            d = {k:v for k,v in zip(colnames, raw_data)}
            out1.append(d)
        out2 = list()
        for i, dd in enumerate(out1):
            d = copy.copy(dd)
            m = cilmod.CilMessage()
            try:
                m.ParseFromString(d["msg"])
                d["msg"] = m
                out2.append(d)
            except Exception as e:
                print("WARNING: Failed to parse message {} of table {}".format(i, table))
        cilrx = out2
        # ciltx, cilrx are ready
        messages = []
        for m in ciltx:
            m.update({"type": "us"})
            messages.append(m)
        for m in cilrx:
            if "INCUMBENT" in str(m["msg"].network_type).upper():
                m.update({"type": "incumbent"})
            else:
                m.update({"type": "peer"})
            messages.append(m)
        messages.sort(key=lambda x: x["time"])
        # messages are ready
        if option == "mandates":
            mandates_map = {}
            for m in messages:
                msg = m["msg"]
                if msg.WhichOneof("payload") != "detailed_performance":
                    continue
                id = msg.sender_network_id
                if id not in mandates_map:
                    mandates_map[id] = {"type": m["type"], "time": [], "mandates_achieved": []}
                if str(msg.detailed_performance.mandates) != "[]":
                    mandates_map[id]["time"].append(m["time"])
                    mandates_map[id]["mandates_achieved"].append(msg.detailed_performance.mandates_achieved)
            return mandates_map
        else:
            scores_map = {}
            for m in messages:
                msg = m["msg"]
                if msg.WhichOneof("payload") != "detailed_performance":
                    continue
                id = msg.sender_network_id
                if id not in scores_map:
                    scores_map[id] = {"type": m["type"], "time": [], "score": [], "threshold": []}
                if str(msg.detailed_performance.mandates) != "[]":
                    scores_map[id]["time"].append(m["time"])
                    scores_map[id]["score"].append(msg.detailed_performance.total_score_achieved)
                    scores_map[id]["threshold"].append(msg.detailed_performance.scoring_point_threshold)
            return scores_map

    # get scores computed by the scoring tool
    def get_scores_tool(self):
        cur = self.conn.cursor()
        data = cur.execute("SELECT * from MatchScores")
        scores_map = {}
        for d in data:
            teamname = d[1]
            if teamname not in scores_map:
                type = "ensemble" if teamname == "ensemble" else "peer"
                scores_map[teamname] = {"type": type, "time": [], "score": []}
            scores_map[teamname]["time"].append(d[0])
            scores_map[teamname]["score"].append(d[2])
        # get and process reported scores and try to match up network ids
        reported = self.get_individual_mandates("score")
        t = list(scores_map.values())[0]["time"]
        t_start = self.get_start_time()
        for d in reported.values():
            y = np.interp(t, np.array(d["time"])/1e9 - t_start/1e3, d["score"])
            d["time"] = t
            d["score"] = y
            del d["threshold"]
        def calculate_error(a, b):
            (a,b) = (np.array(a).astype(float), np.array(b).astype(float))
            c = abs(a) + abs(b)
            c = np.array([i if i != 0 else 1 for i in c])
            return sum(((a-b)/c)**2)
        errors = {}
        for (name, id) in [(a,b) for a in scores_map.keys() if a != "ensemble" for b in reported.keys()]:
            errors[(name, id)] = calculate_error(scores_map[name]["score"], reported[id]["score"])
        # try to minimize total error using greedy algorithm
        pairing = []
        while errors:
            p = min(errors, key=errors.get)
            pairing.append(p)
            errors = {key: errors[key] for key in errors if key[0] != p[0] and key[1] != p[1]}
        for (name, id) in pairing:
            scores_map[name]["network_id"] = id
            scores_map[name]["type"] = reported[id]["type"]
        for d in scores_map.values():
            if "network_id" not in d.keys():
                d["network_id"] = 0
        return scores_map

    # obtain information about this reservation from the db
    def get_res_info(self):
        cur = self.conn.cursor()
        h = cur.execute("select commithash from BuildInfo").fetchone()[0][:12]
        try:
            scinfo = cur.execute("select ScenarioID,ScenarioName from ScenarioInfo").fetchone()
            scid = str(scinfo[0])
            scname = scinfo[1]
        except:
            scid = "Unknown ID"
            scname = "Unknown Scenario"
        try:
            resinfo = cur.execute("select resID,date from ScenarioInfo").fetchone()
            resid = str(resinfo[0])
            restime = resinfo[1]
        except:
            resid = "[Unknown ID]"
            restime = "Unknown time"
        return {
            "hash": h,
            "scid": scid,
            "scname": scname,
            "resid": resid,
            "restime": restime
        }

    # export flows from specified stages to spreadsheet. to export stages 1 and 2
    # to a spreadsheet 'mysheet.xlsx' without computing the actual number of MPs,
    # you would do:
    #
    #   analyzer.export_sheets('mysheet', False, 1, 2, form='xlsx')
    #
    # Warning: printing the number of IMs achieved seemingly takes a LONG time.
    def export_sheets(self, fn: str, do_im: bool, *args, form="ods"):
        # try to load the necessary py modules
        try:
            pex = importlib.import_module("pyexcel")
        except ImportError as e:
            err = "Failed to import pyexcel module ({}). Try 'pip install pyexcel pyexcel-xlsx pyexcel-ods3'".format(str(e))
            raise ImportError(err)
        if form not in ("ods", "xlsx"):
            raise RuntimeError("can only export .ods of .xlsx files")
        # pull the stages to export from the optional positional arguments
        stages = list(args)
        if len(stages) == 0:
            raise RuntimeError("No stages specified.")
        # get all mandated outcome updates from the database
        cur = self.conn.cursor()
        srns = self.get_srnIDs()
        mo_updates = [json.loads(u[0]) for u in
                      cur.execute("select json from MandatedOutcomeUpdate where srnID={:d} order by time asc".format(srns[-1]))]
        if max(stages) > len(mo_updates):
            raise RuntimeError("Need to specify stage numbers that actually happened.")
        # load flow-to-link mappings
        f2l, l2f = self.flow2link2flow()
        f2e = self.flow2endpt()
        # pull all information for a stage
        def get_flow_info(stage):
            print("Exporting stage {:d}...".format(stage), flush=True)
            if do_im:
                print("Computing IMs will block & take a long time.", flush=True)
            ims = mo_updates[stage-1]
            # different columns
            acol = ["flow_uid", "goal_set", "goal_type", "hold_period"]
            fcol = ["file_transfer_deadline_s"]
            scol = ["max_latency_s", "min_throughput_bps"]
            lcol = ["link", "hops"]
            imcol = ["# MPs achieved"] if do_im else []
            # initialize output data
            sf = list()
            ss = list()
            # write column headers
            sf.append(acol + fcol + lcol + imcol)
            ss.append(acol + scol + lcol + imcol)
            # see calculate_mandates(...) idk why this is needed
            t1_origin = self.get_start_time()
            duration = self.get_stop_time() - t1_origin
            tout = frange(0, duration/1000.)
            # write a row for each flow
            for im in ims:
                if "file_transfer_deadline_s" in im["requirements"]:
                    col = fcol
                    o = sf
                    num = np.sum(self.compute_mandate_file(im["flow_uid"], tout)) if do_im else None
                else:
                    col = scol
                    o = ss
                    num = np.sum(self.compute_mandates(im["flow_uid"])) if do_im else None
                o.append([im[key] for key in acol] +
                         [im["requirements"][key] for key in col] +
                         [str(f2e[im["flow_uid"]]), str(f2l[im["flow_uid"]])] +
                         ([int(num)] if do_im else []))
            return sf, ss
        out = dict()
        # add the general information about the run
        info = self.get_res_info()
        out["Info"] = [
            ["Reservation ID", "Time", "Scenario ID", "Scenario Name", "Commit"],
            [info["resid"], info["restime"], info["scid"], info["scname"], info["hash"]]]
        # loop over stages and fill the output spread sheets
        for stage in stages:
            sheetprefix = "stg{:d}".format(stage)
            sf, ss = get_flow_info(stage)
            out["{}_fileIM".format(sheetprefix)] = sf
            out["{}_streamIM".format(sheetprefix)] = ss
        # save as spreadsheet file and we are done
        dfn = "{:s}.{:s}".format(fn, form)
        pex.save_book_as(bookdict=out, dest_file_name=dfn)
        print("Wrote output to {:s}.".format(dfn))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
