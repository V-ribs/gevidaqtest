# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:17:17 2019

@author: lhuismans

Part of this code was derived from:
    https://github.com/sidneycadot/pyqt-and-graphing/blob/master/PyQtGraphing.py

"""
import logging
import math
import os
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import Icons, StylishQT
from ..NIDAQ.constants import MeasurementConstants
from ..NIDAQ.DAQoperator import DAQmission
from .patchclamp import (
    PatchclampSealTest,
    PatchclampSealTest_currentclamp,
    PatchclampSealTest_hold,
    PatchclampSealTest_zap,
)

# Setting graph settings
"""
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('useOpenGL', True)
pg.setConfigOption('leftButtonPan', False)
"""
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")
pg.setConfigOption("useOpenGL", True)
pg.setConfigOption("leftButtonPan", False)


class SlidingWindow(pg.PlotWidget):
    """SlidingWindow gives access to the windowSize most recent values that were appended to it.
    Since the class inherits the pg.PlotWidget it can be added to the UI as a plot widget with a
    sliding window. It is not yet verified that this will work. However it is worth a try ;).
    """

    def __init__(self, windowSize, title, unit, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # Call the pg.PlotWidget so this class has the same behaviour as the PlotWidget object
        self.data = np.array([])
        self.n = 0
        self.windowSize = windowSize
        self.setTitle(title=title)
        self.setLabel("left", units=unit)
        self.setLabel("bottom", text="20 ms")

        self.pen = QPen()
        self.pen.setColor(QColor(145, 255, 244))
        # a pen with width 0 will always be drawn 1px wide
        self.pen.setWidth(0)
        self.pen.setStyle(Qt.DashLine)

        # call plot, so it is not needed to call this in the UI.
        # However, you can still change the pen variables in the UI.
        self.plotData = self.plot(pen=self.pen)

    def append_(self, values):
        """Append new values to the sliding window."""
        lenValues = len(values)
        if self.n + lenValues > self.windowSize:
            # Buffer is full so make room.
            copySize = self.windowSize - lenValues
            self.data = self.data[-copySize:]
            self.n = copySize

        self.data = np.append(self.data, values)

        self.n += lenValues

    def updateWindow(self):
        """Get a window of the most recent 'windowSize' samples (or less if not available)."""
        self.plotData.setData(self.data)


class PatchclampSealTestUI(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saving_dir = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Patch clamp\seal_test"  # TODO hardcoded path

        # === Initiating patchclamp class ===
        self.sealTest = PatchclampSealTest()
        self.sealTest.measurementThread.measurement.connect(
            self.handleMeasurement
        )  # Connecting to the measurement signal

        self.holdTest = PatchclampSealTest_hold()
        self.holdTest.measurementThread_hold.measurement.connect(
            self.handleMeasurement
        )  # Connecting to the measurement signal

        self.currentclampTest = PatchclampSealTest_currentclamp()
        self.currentclampTest.measurementThread_currentclamp.measurement.connect(
            self.handleMeasurement
        )  # Connecting to the measurement signal

        self.zapfunction = PatchclampSealTest_zap()
        # === GUI ===
        self.setFixedHeight(770)
        self.setWindowTitle("Patchclamp Seal Test")

        with Icons.Path("off.png") as path:
            self.ICON_RED_LED = QPixmap(path)
        with Icons.Path("on.png") as path:
            self.ICON_GREEN_LED = QPixmap(path)
        self.is_sealtesting = False

        # === Gains ===
        gainContainer = StylishQT.roundQGroupBox(title="Gains")
        # gainContainer.setFixedWidth(320)
        gainLayout = QGridLayout()

        gainLayout.addWidget(QLabel("Input Voltage"), 0, 0)
        self.inVolGainList = QComboBox()
        self.inVolGainList.addItems(["1/10", "1", "1/50"])
        gainLayout.addWidget(self.inVolGainList, 1, 0)

        gainLayout.addWidget(QLabel("Output Voltage"), 0, 1)
        self.outVolGainList = QComboBox()
        self.outVolGainList.addItems(["10", "2", "5", "1", "20", "50", "100"])
        gainLayout.addWidget(self.outVolGainList, 1, 1)

        gainLayout.addWidget(QLabel("Output Current"), 0, 2)
        self.outCurGainList = QComboBox()
        self.outCurGainList.addItems(["1", "2", "5", "10", "20", "50", "100"])
        gainLayout.addWidget(self.outCurGainList, 1, 2)

        gainLayout.addWidget(QLabel("Probe"), 0, 3)
        self.probeGainList = QComboBox()
        self.probeGainList.addItems(["100M\u03a9", "10G\u03a9"])
        gainLayout.addWidget(self.probeGainList, 1, 3)

        gainContainer.setLayout(gainLayout)

        # === Wavesettings ===
        WavesettingsContainer = StylishQT.roundQGroupBox(title="Wave settings")
        # WavesettingsContainer.setFixedWidth(320)
        WavesettingsContainerLayout = QGridLayout()

        WavesettingsContainerLayout.addWidget(QLabel("Voltage step(mV)"), 0, 0)
        self.DiffVoltagebox = QSpinBox(self)
        self.DiffVoltagebox.setMaximum(2000)
        self.DiffVoltagebox.setMinimum(-2000)
        self.DiffVoltagebox.setValue(10)
        self.DiffVoltagebox.setSingleStep(10)
        WavesettingsContainerLayout.addWidget(self.DiffVoltagebox, 0, 1)

        WavesettingsContainerLayout.addWidget(
            QLabel("Voltage baseline(mV)"), 0, 2
        )
        self.LowerVoltagebox = QSpinBox(self)
        self.LowerVoltagebox.setMaximum(2000)
        self.LowerVoltagebox.setMinimum(-2000)
        self.LowerVoltagebox.setValue(-10)
        self.LowerVoltagebox.setSingleStep(10)
        WavesettingsContainerLayout.addWidget(self.LowerVoltagebox, 0, 3)

        WavesettingsContainer.setLayout(WavesettingsContainerLayout)

        # === Membrane potential ===
        Vm_measureContainer = StylishQT.roundQGroupBox(title="Vm measurement")
        # Vm_measureContainer.setFixedWidth(320)
        Vm_measureContainerLayout = QGridLayout()

        Vm_measureContainerLayout.addWidget(
            QLabel("Clamping current(pA)"), 0, 0
        )
        self.clampingcurrentbox = QSpinBox(self)
        self.clampingcurrentbox.setMaximum(2000)
        self.clampingcurrentbox.setMinimum(-2000)
        self.clampingcurrentbox.setValue(0)
        self.clampingcurrentbox.setSingleStep(100)
        Vm_measureContainerLayout.addWidget(self.clampingcurrentbox, 0, 1)

        self.membraneVoltLabel = QLabel("Vm: ")
        Vm_measureContainerLayout.addWidget(self.membraneVoltLabel, 0, 2)

        self.VmstartButton = StylishQT.runButton()
        self.VmstartButton.clicked.connect(lambda: self.measure_currentclamp())
        Vm_measureContainerLayout.addWidget(self.VmstartButton, 0, 3)

        self.VmstopButton = StylishQT.stop_deleteButton()
        self.VmstopButton.setEnabled(False)
        self.VmstopButton.clicked.connect(
            lambda: self.stopMeasurement_currentclamp()
        )
        Vm_measureContainerLayout.addWidget(self.VmstopButton, 0, 4)

        Vm_measureContainer.setLayout(Vm_measureContainerLayout)

        # === zap ===
        zapContainer = StylishQT.roundQGroupBox(title="ZAP")
        # zapContainer.setFixedWidth(320)
        zapContainerLayout = QGridLayout()

        self.zapiconlabel = QLabel()
        with Icons.Path("zap.jpg") as path:
            self.zapiconlabel.setPixmap(QPixmap(path))
        zapContainerLayout.addWidget(self.zapiconlabel, 0, 0)

        self.zapButton = QPushButton("ZAP!")
        self.zapButton.setStyleSheet(
            "QPushButton {color:white;background-color: blue; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
            "QPushButton:pressed {color:black;background-color: red; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
            "QPushButton:hover:!pressed {color:blue;background-color: white; border-style: outset;border-radius: 10px;border-width: 2px;font: bold 14px;padding: 6px}"
        )
        self.zapButton.setShortcut("z")
        zapContainerLayout.addWidget(self.zapButton, 0, 6)
        self.zapButton.clicked.connect(self.zap)

        zapContainerLayout.addWidget(QLabel("ZAP voltage(V)"), 0, 2)
        self.zapVbox = QDoubleSpinBox(self)
        self.zapVbox.setMaximum(1)
        self.zapVbox.setMinimum(-2000)
        self.zapVbox.setValue(1)
        self.zapVbox.setSingleStep(0.1)
        zapContainerLayout.addWidget(self.zapVbox, 0, 3)

        zapContainerLayout.addWidget(QLabel("ZAP duration(us)"), 0, 4)
        self.zaptimebox = QSpinBox(self)
        self.zaptimebox.setMaximum(200000000)
        self.zaptimebox.setMinimum(-200000000)
        self.zaptimebox.setValue(200)
        self.zaptimebox.setSingleStep(200)
        zapContainerLayout.addWidget(self.zaptimebox, 0, 5)

        zapContainer.setLayout(zapContainerLayout)

        # === Hold Vm ===
        HoldContainer = StylishQT.roundQGroupBox(title="Set Vm")
        HoldContainerLayout = QGridLayout()

        HoldContainerLayout.addWidget(QLabel("Holding voltage(mV):"), 1, 0)
        self.HoldingList = QSpinBox(self)
        self.HoldingList.setMaximum(200000000)
        self.HoldingList.setMinimum(-200000000)
        self.HoldingList.setValue(0)
        self.HoldingList.setSingleStep(10)
        HoldContainerLayout.addWidget(self.HoldingList, 1, 1)

        # self.iconlabel = QLabel(self)
        # self.iconlabel.setPixmap(self.ICON_RED_LED)
        # HoldContainerLayout.addWidget(self.iconlabel, 1, 1)

        self.holdingbutton = QPushButton("Hold")
        # self.holdingbutton.setCheckable(True)
        # self.holdingbutton.toggle()

        # self.holdingbutton.clicked.connect(lambda: self.btnstate())
        self.holdingbutton.clicked.connect(lambda: self.hold())
        # self.holdingbutton.clicked.connect(self.setGreenlight)
        HoldContainerLayout.addWidget(self.holdingbutton, 1, 2)

        # self.stopandholdbutton = QPushButton("Stop and Hold")
        # self.stopandholdbutton.clicked.connect(lambda: self.stopMeasurement())
        # self.stopandholdbutton.clicked.connect(lambda: self.measure_hold())
        # self.stopandholdbutton.clicked.connect(self.setGreenlight)
        # HoldContainerLayout.addWidget(self.stopandholdbutton, 0, 2)

        HoldContainer.setLayout(HoldContainerLayout)

        # === Control ===
        controlContainer = StylishQT.roundQGroupBox(title="Control")
        controlContainer.setFixedWidth(350)
        controlLayout = QGridLayout()

        """
        self.stopholdingbutton = QPushButton("Stop holding")
        self.stopholdingbutton.clicked.connect(lambda: self.stophold())
        self.stopholdingbutton.clicked.connect(self.setRedlight)
        controlLayout.addWidget(self.stopholdingbutton, 2, 2)
        """
        controlLayout.addWidget(gainContainer, 0, 0, 1, 2)
        controlLayout.addWidget(WavesettingsContainer, 1, 0, 1, 2)
        controlLayout.addWidget(Vm_measureContainer, 2, 0, 1, 2)
        controlLayout.addWidget(zapContainer, 3, 0, 1, 2)
        controlLayout.addWidget(HoldContainer, 4, 0, 1, 2)

        self.startButton = StylishQT.runButton("Start seal-test")
        self.startButton.clicked.connect(lambda: self.measure())
        # self.startButton.setFixedWidth(120)
        # self.startButton.clicked.connect(self.setRedlight)
        # self.startButton.clicked.connect(self.startUpdatingGUIThread)
        controlLayout.addWidget(self.startButton, 5, 0, 1, 1)

        self.stopButton = StylishQT.stop_deleteButton()
        self.stopButton.setEnabled(False)
        # self.stopButton.setFixedWidth(120)
        self.stopButton.clicked.connect(lambda: self.stopMeasurement())
        controlLayout.addWidget(self.stopButton, 5, 1, 1, 1)

        controlContainer.setLayout(controlLayout)

        # === Plots ===
        plotContainer = StylishQT.roundQGroupBox(title="Output")
        plotContainer.setFixedWidth(350)
        self.plotLayout = (
            QGridLayout()
        )  # We set the plotLayout as an attribute of the object (i.e. self.plotLayout instead of plotLayout)
        # This is to prevent the garbage collector of the C++ wrapper from deleting the layout and thus triggering errors.
        # Derived from: https://stackoverflow.com/questions/17914960/pyqt-runtimeerror-wrapped-c-c-object-has-been-deleted
        # and http://enki-editor.org/2014/08/23/Pyqt_mem_mgmt.html

        # self.outVolPlotWidget = SlidingWindow(200, title = "Voltage", unit = "V") #Should be bigger than the readvalue

        # Should be bigger than the readvalue
        self.outCurPlotWidget = SlidingWindow(200, title="Current", unit="A")

        self.display_tab_widget = QTabWidget()

        # self.plotLayout.addWidget(QLabel('Voltage (mV):'), 0, 0)
        self.display_tab_widget.addTab(self.outCurPlotWidget, "Current")
        # self.plotLayout.addWidget(self.outVolPlotWidget, 1, 0)
        # self.plotLayout.addWidget(QLabel('Current (pA):'), 0, 1)
        # self.display_tab_widget.addTab(self.outVolPlotWidget, "Voltage")
        # self.plotLayout.addWidget(self.outCurPlotWidget, 1, 1)

        valueContainer = QGroupBox("Resistance/Capacitance")
        self.valueLayout = QGridLayout()
        self.resistanceLabel = QLabel("Resistance: ")
        self.capacitanceLabel = QLabel("Capacitance: ")
        self.ratioLabel = QLabel("Ratio: ")
        self.valueLayout.addWidget(self.resistanceLabel, 0, 0)
        self.valueLayout.addWidget(self.capacitanceLabel, 0, 1)
        self.valueLayout.addWidget(self.ratioLabel, 0, 2)

        self.pipette_resistance = QLineEdit(self)
        self.pipette_resistance.setPlaceholderText("Pipette resistance")
        self.pipette_resistance.setFixedWidth(100)
        self.valueLayout.addWidget(self.pipette_resistance, 1, 0)

        self.savedataButton = QPushButton("Save figure")
        self.savedataButton.clicked.connect(lambda: self.savePatchfigure())
        self.valueLayout.addWidget(self.savedataButton, 1, 1)

        self.resetButton = QPushButton("Reset Iplot")
        self.resetButton.clicked.connect(lambda: self.ResetCurrentImgView())
        self.valueLayout.addWidget(self.resetButton, 1, 2)

        valueContainer.setLayout(self.valueLayout)
        self.plotLayout.addWidget(self.display_tab_widget, 0, 0, 1, 1)
        self.plotLayout.addWidget(valueContainer, 2, 0, 1, 1)

        plotContainer.setLayout(self.plotLayout)
        # === Adding to master ===
        master = QVBoxLayout()
        master.addWidget(controlContainer)
        master.addWidget(plotContainer)

        self.setLayout(master)

        # === Setting variables ===
        self.changeVolInGain(self.inVolGainList.currentText())
        self.changeVolOutGain(self.outVolGainList.currentText())
        self.changeCurOutGain(self.outCurGainList.currentText())
        self.changeProbeGain(self.probeGainList.currentText())

        self.inVolGainList.currentIndexChanged.connect(
            lambda: self.changeVolInGain(self.inVolGainList.currentText())
        )
        self.outVolGainList.currentIndexChanged.connect(
            lambda: self.changeVolOutGain(self.outVolGainList.currentText())
        )
        self.outCurGainList.currentIndexChanged.connect(
            lambda: self.changeCurOutGain(self.outCurGainList.currentText())
        )
        self.probeGainList.currentIndexChanged.connect(
            lambda: self.changeProbeGain(self.probeGainList.currentText())
        )

    def ResetCurrentImgView(self):
        """Closes the widget nicely, making sure to clear the graphics scene and release memory."""
        self.outCurPlotWidget.close()
        # self.outVolPlotWidget.close()

        # Replot the imageview
        self.outCurPlotWidget = SlidingWindow(200, title="Current", unit="A")

        self.display_tab_widget.addTab(self.outCurPlotWidget, "Current")
        # self.display_tab_widget.addTab(self.outVolPlotWidget, "Voltage")

    def measure(self):
        """Pop up window asking to check the gains.
        Returns
        True if the measurement can be done
        and
        False if not.
        """
        # Start the patchclamp measurement
        self.stopButton.setEnabled(True)
        self.startButton.setEnabled(False)

        self.diffvoltage = self.DiffVoltagebox.value() / 1000
        self.lowervoltage = self.LowerVoltagebox.value() / 1000
        self.sealTest.setWave(
            self.inVolGain, self.diffvoltage, self.lowervoltage
        )
        self.sealTest.start()
        self.is_sealtesting = True
        self.startUpdatingGUIThread()

    def measure_hold(self):
        """Pop up window asking to check the gains.
        Returns
        True if the measurement can be done
        and
        False if not.
        """
        self.holdTest.setWave(self.inVolGain, self.HoldingList.value())
        self.holdTest.start()
        self.is_sealtesting = True

    def measure_currentclamp(self):
        """Pop up window asking to check the gains.
        Returns
        True if the measurement can be done
        and
        False if not.
        """
        # Start the patchclamp measurement
        self.VmstartButton.setEnabled(False)
        self.VmstopButton.setEnabled(True)

        self.currentclamp_value = self.clampingcurrentbox.value()
        self.currentclampTest.setWave(
            self.inVolGain, self.probeGain, self.currentclamp_value
        )
        self.currentclampTest.start()
        self.is_sealtesting = True

    def hold(self):
        constant = self.HoldingList.value()
        self.executer = DAQmission()
        self.executer.sendSingleAnalog("patchAO", constant / 1000 * 10)
        logging.info(f"Holding vm at {constant} mV")

    def stophold(self):
        self.executer.sendSingleAnalog("patchAO", 0)
        logging.info("Stop holding")

    def btnstate(self):
        # source = self.sender()
        if self.holdingbutton.isChecked():
            self.hold()
            # self.setGreenlight()
        else:
            self.stophold()
            # self.setRedlight()

    def setRedlight(self):
        self.iconlabel.setPixmap(self.ICON_RED_LED)

    def setGreenlight(self):
        self.iconlabel.setPixmap(self.ICON_GREEN_LED)

    def startUpdatingGUIThread(self):
        time.sleep(0.3)
        StartGUIThread = threading.Thread(target=self.startUpdatingGUI)
        StartGUIThread.start()

    # else:
    # .disconnect()
    def handleMeasurement(self, voltOut, curOut):
        """Handle the measurement. Update the graph."""

        # Rescaling using gains
        self.voltOut = voltOut / self.outVolGain
        self.curOut = curOut / self.outCurGain / self.probeGain

    def startUpdatingGUI(self):
        while self.is_sealtesting is True:
            time.sleep(0.05)
            try:
                # self.outVolPlotWidget.append_(self.voltOut)
                self.outCurPlotWidget.append_(self.curOut)
                self.updateGraphs()
                self.updateLabels(self.curOut, self.voltOut)
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)

    def updateGraphs(self):
        """Update graphs."""
        self.outCurPlotWidget.updateWindow()
        # self.outVolPlotWidget.updateWindow()

    def updateLabels(self, curOut, voltOut):
        """Update the resistance and capacitance labels.
        http://scipy-lectures.org/intro/scipy/auto_examples/plot_curve_fit.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        """
        constants = MeasurementConstants()
        sampPerCyc = int(constants.patchSealSampRate / constants.patchSealFreq)

        try:
            curOutCyc = curOut.reshape(
                int(curOut.size / sampPerCyc), sampPerCyc
            )
            curData = np.mean(curOutCyc, axis=0)
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            curData = curOut

        voltData = voltOut
        try:
            # Computing resistance
            tres = np.mean(voltData)
            dV = np.mean(voltData[voltData > tres]) - np.mean(
                voltData[voltData < tres]
            )  # Computing the voltage difference

            start_selection_a = math.floor(0.15 * sampPerCyc)
            end_selection_a = math.floor(0.5 * sampPerCyc) - 2
            start_selection_b = math.floor(0.65 * sampPerCyc)
            end_selection_b = sampPerCyc - 2
            avg_a = np.mean(curData[start_selection_a:end_selection_a])
            avg_b = np.mean(curData[start_selection_b:end_selection_b])
            dIss = avg_a - avg_b  # Computing the current distance

            membraneResistance = dV / (dIss * 1000000)  # Ohms law (MegaOhm)
            self.resistanceLabel.setText(
                "Resistance:  %.4f M\u03a9" % membraneResistance
            )

            # The resistance of a typical patch of membrane, RM is 10000 Omega/{cm}^2
            self.estimated_size_resistance = 10000 / (
                membraneResistance * 1000000
            )
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.resistanceLabel.setText("Resistance:  %s" % "NaN")

        try:
            measured_vlotage = np.mean(voltData) * 1000
            self.membraneVoltLabel.setText("Vm:  %.2f mV" % measured_vlotage)
            self.membraneVoltLabel.setStyleSheet("color: red")
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.membraneVoltLabel.setText("Vm:  %s" % "NaN")

        try:
            # Computing capacitance
            points = 10 - 1
            maxCur = np.amax(curData)
            maxCurIndex = round(np.where(curData == maxCur)[0][0]) + 1
            last_index = maxCurIndex + points
            data_selection = curData[maxCurIndex:last_index]
            curFit = data_selection - 0.5 * dIss

            points_start = 3
            points_range = np.arange(points_start, points + points_start)
            timepoints = 1000 * points_range / constants.patchSealSampRate

            # Fitting the data to an exponential of the form y=a*exp(-b*x) where b = 1/tau and tau = RC
            # I(t)=I0*e^−t/τ, y=a*exp(-b*x), get log of both sides:log y = -bx + log a
            # Converting the exponential to a linear function and fitting it
            fit = np.polyfit(timepoints, curFit, 1)
            # Extracting data
            current = fit[0]
            resistance = dV * 1000 / current / 2  # Getting the resistance
            tau = -1 / fit[1]
            capacitance = 1000 * tau / resistance
            self.capacitanceLabel.setText("Capacitance:  %.4f" % capacitance)

            self.estimated_size_capacitance = capacitance * (10**-12) * (10**6)

            if (
                self.estimated_size_capacitance
                > self.estimated_size_resistance
            ):
                self.estimated_ratio = (
                    self.estimated_size_capacitance
                    / self.estimated_size_resistance
                )
            else:
                self.estimated_ratio = (
                    self.estimated_size_resistance
                    / self.estimated_size_capacitance
                )

            self.ratioLabel.setText(
                "Ratio:  %.4f" % self.estimated_ratio
            )  # http://www.cnbc.cmu.edu/~bard/passive2/node5.html

        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.capacitanceLabel.setText("Capacitance:  %s" % "NaN")
            self.ratioLabel.setText("Ratio:  %s" % "NaN")

        self.patch_parameters = "R_{}_C_{}".format(
            round(membraneResistance, 3), round(capacitance, 3)
        )

    def stopMeasurement(self):
        """Stop the seal test."""
        self.stopButton.setEnabled(False)
        self.startButton.setEnabled(True)
        self.sealTest.aboutToQuitHandler()
        self.is_sealtesting = False

    '''
    def closeEvent(self, event):
        """On closing the application we have to make sure that the measuremnt
        stops and the device gets freed."""
        self.stopMeasurement()
    '''

    def stop_hold_Measurement(self):
        """Stop the seal test."""
        self.holdTest.aboutToQuitHandler()
        self.is_sealtesting = False

    def close_hold_Event(self, event):
        """On closing the application we have to make sure that the measuremnt
        stops and the device gets freed."""
        self.stop_hold_Measurement()

    def stopMeasurement_currentclamp(self):
        """Stop the seal test."""
        self.VmstartButton.setEnabled(True)
        self.VmstopButton.setEnabled(False)

        self.currentclampTest.aboutToQuitHandler()
        self.is_sealtesting = False

    # Change gain
    def changeVolInGain(self, gain):
        if gain == "1":
            self.inVolGain = 1
        elif gain == "1/10":
            self.inVolGain = 0.1
        elif gain == "1/50":
            self.inVolGain = 1.0 / 50

    def changeVolOutGain(self, gain):
        self.outVolGain = float(gain)

    def changeCurOutGain(self, gain):
        self.outCurGain = float(gain)

    def changeProbeGain(self, gain):
        if gain == "100M\u03a9":
            self.probeGain = 100 * 10**6
        elif gain == "10G\u03a9":
            self.probeGain = 10 * 10**9

    def zap(self):
        self.zap_v = self.zapVbox.value()
        self.zap_time = self.zaptimebox.value()
        self.sealTest.aboutToQuitHandler()
        self.zapfunction.setWave(self.inVolGain, self.zap_v, self.zap_time)
        self.zapfunction.start()
        time.sleep(0.06)
        self.zapfunction.aboutToQuitHandler()
        """Start the patchclamp measurement"""
        self.diffvoltage = self.DiffVoltagebox.value() / 1000
        self.lowervoltage = self.LowerVoltagebox.value() / 1000
        self.sealTest.setWave(
            self.inVolGain, self.diffvoltage, self.lowervoltage
        )
        self.sealTest.start()

    def savePatchfigure(self):
        # create an exporter instance, as an argument give it
        # the item you wish to export
        exporter = pg.exporters.ImageExporter(
            self.outCurPlotWidget.getPlotItem()
        )

        # set export parameters if needed
        exporter.parameters()[
            "width"
        ] = 500  # (note this also affects height parameter)

        # save to file
        exporter.export(
            os.path.join(
                self.saving_dir,
                "SealTest_"
                + "Rpip_"
                + self.pipette_resistance.text()
                + "Mohm_"
                + self.patch_parameters
                + ".png",
            )
        )


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = PatchclampSealTestUI()
        mainwin.show()
        app.exec_()

    run_app()
