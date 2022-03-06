#!/usr/bin/env python3
#
# Decision Engine visualizer
#
# Copyright (c) 2019 Dennis Ogbe

import json
import os
import sys
import argparse
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from ipaddress import ip_address

# Global variables
phy_sample_rate = 46.08e6

def do_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input file")
    return parser.parse_known_args()

class SpectrumTab(QWidget):
    def __init__(self, app):
        QWidget.__init__(self)
        self.app = app

        # add myself to the main app
        self.app.tabs.addTab(self, "Spectrum")
        self.tabnumber = self.app.ntab
        self.app.ntab += 1

        # a simple vbox/hbox layout
        self.setMinimumSize(800, 600)
        self.hbox = QHBoxLayout() # main
        self.vbox = QVBoxLayout()
        self.vboxWidget = QWidget()
        self.vboxWidget.setMinimumWidth(800)
        self.vboxWidget.setLayout(self.vbox)
        self.listVBox = QVBoxLayout()
        self.listVBoxWidget = QWidget()
        self.listVBoxWidget.setLayout(self.listVBox)
        self.listVBoxWidget.setMaximumWidth(200)

        # SRN/Team selection list
        self.srnFilterModel = QStandardItemModel()
        self.srnFilterWidget = QListView()
        self.srnFilterWidget.setModel(self.srnFilterModel)
        self.srnFilterWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listVBox.addWidget(QLabel("SRNs (our network)"))
        self.listVBox.addWidget(self.srnFilterWidget)

        self.teamFilterModel = QStandardItemModel()
        self.teamFilterWidget = QListView()
        self.teamFilterWidget.setModel(self.teamFilterModel)
        self.teamFilterWidget.setSelectionMode(QAbstractItemView.NoSelection)
        self.listVBox.addWidget(QLabel("Collaborators"))
        self.listVBox.addWidget(self.teamFilterWidget)

        # overlay selection list
        self.overlayFilterModel = QStandardItemModel()
        self.overlayFilterWidget = QListView()
        self.overlayFilterWidget.setFlow(QListView.LeftToRight)
        self.overlayFilterWidget.setSelectionMode(QAbstractItemView.NoSelection)
        self.overlayFilterWidget.setMaximumHeight(30)
        self.overlayFilterWidget.setModel(self.overlayFilterModel)
        lab1 = QLabel("Show data")
        lab1.setMaximumHeight(25)
        self.vbox.addWidget(lab1)
        self.vbox.addWidget(self.overlayFilterWidget)

        # plot window
        self.figure = plt.figure(figsize=(100, 6))
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbox.addWidget(toolbar)
        self.vbox.addWidget(self.canvas)

        # initialize GUI
        self.connectRefresh()
        self.initFilters()
        self.hbox.addWidget(self.listVBoxWidget)
        self.hbox.addWidget(self.vboxWidget)
        self.setLayout(self.hbox)

        # constants
        self.HIGH = 10
        self.LOW = 0
        self.UNDEF = -10

        # pre-process data
        self.preprocessPSD()

    def initFilters(self):
        # the overlay filters
        for filt in ("Thresholded PSD", "Spectrum Usage", "Channelization"):
            if (filt == "Spectrum Usage") and not "spectrum-usage" in self.app.data.keys():
                continue
            item = QStandardItem(filt)
            item.setText(filt)
            item.setCheckState(Qt.Unchecked)
            item.setCheckable(True)
            item.setSelectable(False)
            self.overlayFilterModel.appendRow(item)
        # the team/srn filters
        for srn in self.app.data["thresh-psd"].keys():
            item = QStandardItem(srn)
            item.setText(srn)
            item.setData(srn)
            item.setSelectable(True)
            item.setCheckable(False)
            self.srnFilterModel.appendRow(item)
        if "spectrum-usage" in self.app.data.keys():
            for team in self.app.data["spectrum-usage"].keys():
                ip = ip_address(int(team, 10))
                item = QStandardItem()
                item.setText(str(ip))
                item.setData(team)
                item.setSelectable(False)
                item.setCheckable(True)
                item.setCheckState(Qt.Unchecked)
                self.teamFilterModel.appendRow(item)

    def connectRefresh(self):
        self.srnFilterWidget.selectionModel().selectionChanged.connect(self.drawPlot)
        self.overlayFilterModel.dataChanged.connect(self.drawPlot)
        self.teamFilterModel.dataChanged.connect(self.drawPlot)

    def preprocessPSD(self):
        # we need all steps
        allSteps = self.app.allSteps()
        allStepTimes = self.app.allStepTimes()
        self.psdData = {"step-id" : allSteps,
                        "time" : allStepTimes}
        # preprocess data for all SRNs
        for srn in self.app.data["thresh-psd"].keys():
            # save PSD data as image
            psdDim = len(self.app.data["thresh-psd"][srn]["thresh-psd"][0])
            img = np.zeros((psdDim, len(allSteps)))
            # map the step number to the thresholded psd data
            step2data = {step: np.array(data) for step, data in
                         zip(self.app.data["thresh-psd"][srn]["step-id"],
                             self.app.data["thresh-psd"][srn]["thresh-psd"])}
            # now for every step, we add the processed PSD to the output image
            for i, step in enumerate(allSteps):
                if step in step2data:
                    rawData = step2data[step]
                    # map 1 to HIGH and 0 to LOW
                    m1 = np.where(rawData == 1, self.HIGH, rawData)
                    psdData = np.where(m1 == 0, self.LOW, m1)
                else:
                    psdData = np.full(psdDim, self.UNDEF)
                img[:, i] = psdData
            # add the image to the output
            self.psdData.update({srn: img})

    def drawPlot(self):
        # re-draw the plot if overlay selection changed
        activeOverlays = []
        for i in range(self.overlayFilterModel.rowCount()):
            item = self.overlayFilterModel.item(i)
            if item.checkState() == Qt.Checked:
                activeOverlays.append(item.text())
        self.figure.clear()
        if "Thresholded PSD" in activeOverlays:
            self.plotThreshPSD()
        if "Spectrum Usage" in activeOverlays:
            self.plotSpectrumUsage()
        if "Channelization" in activeOverlays:
            self.plotChannelization()
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    def plotThreshPSD(self):
        # fixme how to do this?
        if not (len(self.srnFilterWidget.selectedIndexes()) > 0):
            return
        srn = self.srnFilterModel.item(self.srnFilterWidget.selectedIndexes()[0].row()).text()
        ax = self.figure.subplots()
        extent = [self.psdData["time"][0], self.psdData["time"][-1], (phy_sample_rate/2)/1e6, -(phy_sample_rate/2)/1e6]
        im = ax.imshow(self.psdData[srn], interpolation="none", extent=extent, aspect="auto",
                       vmin=self.UNDEF,vmax=self.HIGH)

    def plotSpectrumUsage(self):
        print("implement plot spectrum usage")

    def plotChannelization(self):
        print("implement plot channelization")


class SRNMetricTab(QWidget):
    def __init__(self, app, datakey):
        QWidget.__init__(self)
        self.app = app
        self.datakey = datakey;

        # add to tab
        self.app.tabs.addTab(self, datakey)
        self.tabnumber = self.app.ntab
        self.app.ntab += 1

        # initialize data and GUI
        self.preprocessData()
        self.initGUI()

    def drawPlot(self):
        self.figure.clear()
        ax = self.figure.subplots()
        for idx in range(self.srnFilterModel.rowCount()):
            item = self.srnFilterModel.item(idx)
            if item.checkState() == Qt.Checked:
                y = self.data[item.text()][self.datakey]
                x = self.data[item.text()]["time"]
                if x and y:
                    ax.plot(x, y, label=item.text())
        ax.legend()
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    def preprocessData(self):
        self.data = {}
        for srn, data in self.app.data[self.datakey].items():
            data["time"] = [t - self.app.start_t_sec for t in data["time"]]
            self.data[srn] = data

    def initGUI(self):
        self.vbox = QVBoxLayout()
        # a checkbox list for SRNs
        lab1 = QLabel("SRNs")
        lab1.setMaximumHeight(25)
        self.vbox.addWidget(lab1)
        self.srnFilterModel = QStandardItemModel()
        self.srnFilterWidget = QListView()
        self.srnFilterWidget.setModel(self.srnFilterModel)
        self.srnFilterWidget.setSelectionMode(QAbstractItemView.NoSelection)
        self.srnFilterWidget.setMaximumHeight(25)
        self.srnFilterWidget.setFlow(QListView.LeftToRight)
        self.vbox.addWidget(self.srnFilterWidget)

        # plot window
        self.figure = plt.figure(figsize=(100, 6))
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbox.addWidget(toolbar)
        self.vbox.addWidget(self.canvas)

        # connect slots, initialize
        self.initList()
        self.srnFilterModel.dataChanged.connect(self.drawPlot)

        # done
        self.setLayout(self.vbox)

    def initList(self):
        for srn in sorted(self.app.data[self.datakey], key=lambda x: int(x)):
            item = QStandardItem(srn)
            item.setText(srn)
            item.setData(srn)
            item.setSelectable(False)
            item.setCheckable(True)
            item.setCheckState(Qt.Unchecked)
            self.srnFilterModel.appendRow(item)



class DEAnalyzer(QMainWindow):
    def __init__(self, args, parent=None):
        super(DEAnalyzer, self).__init__(parent)
        self.setWindowTitle('BAM! Wireless Decision Engine Viz')

        # load data
        self.data = None
        with open(args.infile, "r") as ifp:
            self.data = json.load(ifp)
        self.start_t_sec = self.data["environments"]["time"][0]
        if self.start_t_sec == 33333333:
            self.start_t_sec = self.data["environments"]["time"][1]

        # set up tabs
        self.tabs = QTabWidget()
        self.ntab = 0

        # add viz tabs
        self.spectrumTab = SpectrumTab(self)
        self.dutyCycleTab = SRNMetricTab(self, "duty-cycle")
        self.metricTab = SRNMetricTab(self, "active-mandate-metric")

        # add all tabs to window
        self.setCentralWidget(self.tabs)

    def allSteps(self):
        return np.array(self.data["environments"]["step-id"])

    def allStepTimes(self):
        return np.array(self.data["environments"]["time"]) - self.start_t_sec


if __name__ == '__main__':
    args, unparsed_args = do_args()
    app = QApplication(sys.argv[:1] + unparsed_args)
    main = DEAnalyzer(args)
    main.show()
    sys.exit(app.exec_())
