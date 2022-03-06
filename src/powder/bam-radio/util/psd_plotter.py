#!/usr/bin/env python3
#
# PSD plotter. Use with `create-psd-db' and test/psd_thresh.cc
#
# Copyright (c) 2019 Dennis Ogbe

import os
import sys
import argparse
import numpy as np
import sqlite3

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# Global variables
phy_sample_rate = 46.08e6
# constants
HIGH = 10
LOW = 0
UNDEF = -10
MY = -5

# one tab per thresholder
class PSDTab(QWidget):
    def __init__(self, app, tabname, real):
        QWidget.__init__(self)
        self.app = app
        self.tabname = tabname
        self.plotReal = real
        self.data = None
        self.loadData()
        self.initGUI()

    def loadData(self):
        if self.plotReal:
            self.loadRealData()
        else:
            self.loadThreshData()

    def loadRealData(self):
        cur = self.app.db.cursor()
        self.data = {}
        for srn in self.app.allSRNs:
            rows = cur.execute("select data, step_id from {} where srn_id = {}".format(self.tabname, srn))
            step2data = {step: np.fromstring(data, dtype="float32") for data, step in rows}
            psdDim = len(step2data[list(step2data.keys())[0]])
            img = np.zeros((psdDim, self.app.lastStep + 1))
            for step in range(self.app.lastStep + 1):
                if step in step2data:
                    img[:, step] = np.clip(step2data[step], -55, None)
                else:
                    img[:, step] = np.full(psdDim, -55)
            self.data.update({srn: img})

    def loadThreshData(self):
        cur = self.app.db.cursor()
        self.data = {}
        for srn in self.app.allSRNs:
            rows = cur.execute("select data, step_id from {} where srn_id = {}".format(self.tabname, srn))
            step2data = {step: np.fromstring(data, dtype="uint8") for data, step in rows}
            psdDim = len(step2data[list(step2data.keys())[0]])
            img = np.zeros((psdDim, self.app.lastStep + 1))
            for step in range(self.app.lastStep + 1):
                if step in step2data:
                    m1 = np.where(step2data[step] == 1, HIGH, step2data[step])
                    m2 = np.where(m1 == 0, LOW, m1)
                    m3 = np.where(m2 == 2, MY, m2) # also show "our" transmissions if marked.
                    img[:, step] = m3
                else:
                    img[:, step] = np.full(psdDim, UNDEF)
            self.data.update({srn: img})

    def initGUI(self):
        # Layout
        self.setMinimumSize(800, 600)
        self.app.tabWidget.addTab(self, self.tabname)
        self.vbox = QVBoxLayout()
        self.srnFilterModel = QStandardItemModel()
        self.srnFilterWidget = QListView()
        self.srnFilterWidget.setModel(self.srnFilterModel)
        self.srnFilterWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.srnFilterWidget.setFlow(QListView.LeftToRight)
        self.srnFilterWidget.setMaximumHeight(25)
        self.vbox.addWidget(self.srnFilterWidget)
        self.figure = plt.figure(figsize=(100, 6))
        self.canvas = FigureCanvas(self.figure)
        self.vbox.addWidget(NavigationToolbar(self.canvas, self))
        self.vbox.addWidget(self.canvas)
        self.setLayout(self.vbox)

        # SRN filters
        for srn in self.app.allSRNs:
            item = QStandardItem(srn)
            item.setText(str(srn))
            item.setData(srn)
            item.setSelectable(True)
            item.setCheckable(False)
            self.srnFilterModel.appendRow(item)

        # refresh
        self.srnFilterWidget.selectionModel().selectionChanged.connect(self.drawPlot)

    def drawPlot(self):
        if not self.data:
            return
        if not (len(self.srnFilterWidget.selectedIndexes()) > 0):
            return
        self.figure.clear()
        srn = self.srnFilterModel.item(self.srnFilterWidget.selectedIndexes()[0].row()).data()
        ax = self.figure.subplots()
        extent = [0, self.app.lastStep, (phy_sample_rate/2)/1e6, -(phy_sample_rate/2)/1e6]
        if self.plotReal:
            im = ax.imshow(self.data[srn], interpolation="none", extent=extent, aspect="auto")
        else:
            im = ax.imshow(self.data[srn], interpolation="none", extent=extent, aspect="auto",
                           vmin=UNDEF, vmax=HIGH)
        self.figure.colorbar(im, ax=ax)
        self.figure.tight_layout(pad=0)
        self.canvas.draw()


# the main application
class PSDPlotter(QMainWindow):
    def __init__(self, args):
        super(PSDPlotter, self).__init__(None)
        self.setWindowTitle("BAM! Wireless PSD Plotter")

        # create db connection
        self.allSRNs = None
        self.lastStep = None
        self.allPSDs = None
        self.db = sqlite3.connect(args.infile)
        self.initData()

        # initialize tabs
        self.tabWidget = QTabWidget()
        self.tabs = []
        for tabname in self.allPSDs:
            if tabname == "psd_db":
                self.tabs.append(PSDTab(self, tabname, True))
            else:
                self.tabs.append(PSDTab(self, tabname, False))

        # that's it, we are set up
        self.setCentralWidget(self.tabWidget)
        self.db.close()

    def initData(self):
        cur = self.db.cursor()
        cur.execute("select max(step_id) from psd_db")
        self.lastStep = cur.fetchall()[0][0]
        self.allSRNs = list(map(lambda x: x[0],
                                cur.execute("select srn_id from psd_db group by srn_id").fetchall()))
        self.allSRNs.sort()
        self.allPSDs = list(map(lambda x: x[0],
                                cur.execute("select tbl_name from sqlite_master").fetchall()))


def do_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input file")
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unparsed_args = do_args()
    app = QApplication(sys.argv[:1] + unparsed_args)
    main = PSDPlotter(args)
    main.show()
    sys.exit(app.exec_())


