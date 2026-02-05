# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:04:00 2019

@author: Meng

    Inidival GUI for waveform generating and executing using NI-DAQ


"""

import logging
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from matplotlib import pyplot as plt
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)
from pyqtgraph import PlotDataItem

from .. import StylishQT
from ..ThorlabsFilterSlider.filterpyserial import ELL9Filter
from . import waveform_specification
from .DAQoperator import DAQmission
from .wavegenerator import (
    generate_AO,
    generate_AO_for640,
    generate_digital_waveform,
    generate_ramp,
    waveRecPic,
)

CAMERA_TRIGGER_INSERT_ARRAY = np.array(
    [False] * 25 + [True] * 25 + [False] * 25 + [True] * 25 + [False] * 15
)


class WaveformGenerator(QWidget):

    WaveformPackage = pyqtSignal(object)
    GalvoScanInfor = pyqtSignal(object)
    waveform_started = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            get_ipython = sys.modules["IPython"].get_ipython
        except KeyError:
            pass
        else:
            get_ipython().run_line_magic(
                "matplotlib", "qt"
            )  # before start, set spyder back to inline

        self.layout = QGridLayout(self)
        # Setting tabs
        self.tabs = StylishQT.roundQGroupBox("Waveforms")
        self.savedirectory = None

        # To solve camera losing first trigger issue, add one extra trigger
        # in the beginning.
        self.Adding_extra_camera_trigger_flag = False
        # In append mode, new added waveforms append behind
        self.Append_Mode = False
        # Reset channels by adding 0 at the start/end, includes extra trigger.
        self.Auto_padding_flag = False

        # These contour scanning signals will be set from main panel.
        self.handle_viewbox_coordinate_position_array_expanded = None
        self.time_per_contour = None
        self.handle_viewbox_coordinate_position_array_expanded_x = None
        self.handle_viewbox_coordinate_position_array_expanded_y = None
        self.Daq_sample_rate_pmt = None

        self.AnalogChannelList = [
            "galvos",
            "galvos_contour",
            "640AO",
            "532AO",
            "488AO",
            "patchAO",
            "fieldstimulator",
        ]
        self.DigitalChannelList = [
            "cameratrigger",
            "galvotrigger",
            "DMD_trigger",
            "blankingall",
            "640blanking",
            "532blanking",
            "488blanking",
            "LED",
            "Perfusion_8",
            "Perfusion_7",
            "Perfusion_6",
            "Perfusion_2",
            "2Pshutter",
            "532 servo",
            "640 servo",
            "488 servo",
        ]

        self.color_dictionary = {
            "galvos": [255, 255, 255],
            "galvos_X_contour": [255, 255, 255],
            "galvos_Y_contour": [255, 255, 255],
            "galvos_contour": [255, 255, 255],
            "640AO": [255, 0, 0],
            "488AO": [0, 0, 255],
            "532AO": [0, 255, 0],
            "patchAO": [100, 100, 0],
            "fieldstimulator": [255, 0, 255],
            "cameratrigger": [0, 255, 255],
            "galvotrigger": [100, 100, 200],
            "blankingall": [255, 229, 204],
            "640blanking": [255, 204, 255],
            "532blanking": [255, 255, 0],
            "488blanking": [255, 51, 153],
            "LED": [154, 205, 50],
            "Perfusion_8": [209, 104, 255],
            "Perfusion_7": [127, 255, 212],
            "Perfusion_6": [102, 40, 91],
            "Perfusion_2": [255, 215, 0],
            "2Pshutter": [229, 204, 255],
            "DMD_trigger": [255, 215, 0],
            "532 servo": [0, 200, 0],
            "640 servo": [200, 0, 0],
            "488 servo": [0, 0, 200],
        }

        self.PlotDataItem_dict = {}
        self.waveform_data_dict = {}

        self.servo_dict = {}

        self.setMinimumSize(1000, 650)
        self.setWindowTitle("Buon appetito!")

        self.Galvo_samples = None
        self.finalwave_640 = None
        self.finalwave_488 = None
        self.finalwave_532 = None
        self.finalwave_patch = None
        self.handle_viewbox_coordinate_position_array_expanded = None
        self.finalwave_cameratrigger = None
        self.final_galvotrigger = None
        self.finalwave_blankingall = None
        self.finalwave_640blanking = None
        self.finalwave_532blanking = None
        self.finalwave_488blanking = None
        self.finalwave_Perfusion_8 = None
        self.finalwave_Perfusion_7 = None
        self.finalwave_Perfusion_6 = None
        self.finalwave_Perfusion_2 = None
        self.finalwave_2Pshutter = None
        self.finalwave_DMD_trigger = None

        # self.current_Analog_channel.currentIndexChanged.connect(self.chosen_wave)
        self.wavetablayout = QGridLayout()

        self.wavetabs = QTabWidget()
        self.wavetab1 = QWidget()
        self.wavetab2 = QWidget()
        self.wavetab3 = QWidget()
        self.wavetab4 = QWidget()
        self.wavetab5 = QWidget()
        # Add tabs
        self.wavetabs.addTab(self.wavetab1, "Block")
        self.wavetabs.addTab(self.wavetab2, "Ramp")
        # self.wavetabs.addTab(self.wavetab3,"Import")
        self.wavetabs.addTab(self.wavetab4, "Galvo")
        self.wavetabs.addTab(self.wavetab5, "Photocycle")

        # self.wavetabs.setCurrentIndex(3)

        # Waveform General settings
        ReadContainer = QGroupBox("General settings")
        self.ReadLayout = QGridLayout()  # self.AnalogLayout manager

        ReferenceWaveform_container = StylishQT.roundQGroupBox(
            title="Reference waveform"
        )
        ReferenceWaveform_container_layout = QGridLayout()
        self.ReferenceWaveform_menu = QListWidget()
        self.ReferenceWaveform_menu.addItems(
            self.AnalogChannelList + self.DigitalChannelList
        )
        # self.transform_for_laser_menu.setFixedHeight(52)
        # self.transform_for_laser_menu.setFixedWidth(82)
        self.ReferenceWaveform_menu.setCurrentRow(0)

        ReferenceWaveform_container_layout.addWidget(
            self.ReferenceWaveform_menu, 0, 0
        )
        ReferenceWaveform_container.setLayout(
            ReferenceWaveform_container_layout
        )
        self.ReadLayout.addWidget(ReferenceWaveform_container, 0, 0, 3, 1)

        self.SamplingRateTextbox = QSpinBox(self)
        self.SamplingRateTextbox.setMinimum(0)
        self.SamplingRateTextbox.setMaximum(1000000)
        self.SamplingRateTextbox.setValue(50000)
        self.SamplingRateTextbox.setSingleStep(100000)
        self.ReadLayout.addWidget(self.SamplingRateTextbox, 0, 5)
        self.ReadLayout.addWidget(QLabel("Sampling rate:"), 0, 4)

        # ====================== Read-in channels =============================
        record_channel_container = StylishQT.roundQGroupBox(title="Recording")
        record_channel_container_layout = QGridLayout()

        self.ReadChanPMTTextbox = QCheckBox("   PMT   ")
        self.ReadChanPMTTextbox.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )
        record_channel_container_layout.addWidget(
            self.ReadChanPMTTextbox, 0, 3
        )

        # Without self. the QbuttonGroup won't work
        self.patchRecordingGroup = QButtonGroup()

        self.ReadChanVpTextbox = QCheckBox("Vm in I-clamp")
        self.ReadChanVpTextbox.setStyleSheet(
            'color:Indigo;font:bold "Times New Roman"'
        )
        self.ReadChanVpTextbox.setToolTip(
            "In current-clamp mode, record voltage from the amplifier."
        )
        record_channel_container_layout.addWidget(self.ReadChanVpTextbox, 2, 3)
        self.patchRecordingGroup.addButton(self.ReadChanVpTextbox)

        self.ReadChanIpTextbox = QCheckBox("Im in V-clamp")
        self.ReadChanIpTextbox.setStyleSheet(
            'color:DarkSlateGray	;font:bold "Times New Roman"'
        )
        self.ReadChanIpTextbox.setToolTip(
            "In voltage-clamp mode, record current from the amplifier."
        )
        record_channel_container_layout.addWidget(self.ReadChanIpTextbox, 1, 3)
        self.patchRecordingGroup.addButton(self.ReadChanIpTextbox)

        record_channel_container.setLayout(record_channel_container_layout)

        self.ReadLayout.addWidget(record_channel_container, 0, 2, 3, 1)

        self.ReadLayout.addWidget(QLabel("Master clock:"), 1, 4)

        self.clock_source = QComboBox()
        self.clock_source.addItems(["DAQ", "Camera"])
        self.ReadLayout.addWidget(self.clock_source, 1, 5)

        self.button_import_np_load = QPushButton("Load waveforms", self)
        self.ReadLayout.addWidget(self.button_import_np_load, 2, 5)
        self.button_import_np_load.clicked.connect(self.load_wave_np)

        self.saving_prefix = ""

        executionContainer = QGroupBox("Execution")
        executionContainerLayout = QGridLayout()  # self.AnalogLayout manager

        # Checkbox for saving waveforms
        self.checkbox_saveWaveforms = QCheckBox("Save waveforms")
        # self.checkbox_saveWaveforms.setChecked(True)
        self.checkbox_saveWaveforms.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )
        executionContainerLayout.addWidget(self.checkbox_saveWaveforms, 0, 0)

        # === Emission filter ===
        emission_channel_container = StylishQT.roundQGroupBox(
            title="Emission filter"
        )
        emission_channel_container_layout = QGridLayout()

        self.FilterButtongroup = QButtonGroup()

        self.ArchEmissionbox = QCheckBox("Arch")
        self.ArchEmissionbox.setStyleSheet(
            'color:red;font:bold "Times New Roman"'
        )
        emission_channel_container_layout.addWidget(self.ArchEmissionbox, 0, 0)
        self.FilterButtongroup.addButton(self.ArchEmissionbox)

        self.GFPEmissionbox = QCheckBox("GFP/Citrine")
        self.GFPEmissionbox.setStyleSheet(
            'color:green;font:bold "Times New Roman"'
        )
        emission_channel_container_layout.addWidget(self.GFPEmissionbox, 1, 0)
        self.FilterButtongroup.addButton(self.GFPEmissionbox)
        self.FilterButtongroup.setExclusive(True)

        emission_channel_container.setLayout(emission_channel_container_layout)

        executionContainerLayout.addWidget(
            emission_channel_container, 1, 0, 2, 1
        )

        executionContainerLayout.addWidget(QLabel("Progress:"), 0, 2)
        self.waveform_progressbar = QProgressBar(self)
        self.waveform_progressbar.setMaximumWidth(250)
        self.waveform_progressbar.setMinimumWidth(200)
        self.waveform_progressbar.setMaximum(100)
        self.waveform_progressbar.setStyleSheet(
            """
            QProgressBar {
                color: black;
                border: 2px solid grey;
                border-radius:8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #CD96CD;
                width: 10px;
                margin: 0.5px;
            }
            """
        )
        executionContainerLayout.addWidget(self.waveform_progressbar, 0, 3)

        self.button_all = StylishQT.generateButton()
        self.button_all.setFixedWidth(110)
        executionContainerLayout.addWidget(self.button_all, 0, 1)
        self.button_all.clicked.connect(self.organize_waveforms)

        self.button_execute = StylishQT.runButton("Execute")
        self.button_execute.setEnabled(False)
        self.button_execute.setFixedWidth(110)
        executionContainerLayout.addWidget(self.button_execute, 1, 1)

        self.button_execute.clicked.connect(self.execute_tread)
        self.button_execute.clicked.connect(self.startProgressBar)

        self.button_clear_canvas = StylishQT.cleanButton(label=" Canvas")
        executionContainerLayout.addWidget(self.button_clear_canvas, 2, 1)

        self.button_clear_canvas.clicked.connect(self.clear_canvas)

        executionContainer.setLayout(executionContainerLayout)

        ReadContainer.setLayout(self.ReadLayout)

        # === ANALOG ===
        AnalogContainer = QGroupBox("Analog signals")
        self.AnalogLayout = QGridLayout()  # self.AnalogLayout manager

        self.current_Analog_channel = QComboBox()
        self.current_Analog_channel.addItems(self.AnalogChannelList)
        self.AnalogLayout.addWidget(self.current_Analog_channel, 3, 0)
        self.current_Analog_channel.setCurrentIndex(2)

        self.add_waveform_button = StylishQT.addButton()
        self.add_waveform_button.setFixedHeight(32)
        self.AnalogLayout.addWidget(self.add_waveform_button, 3, 1)

        self.button_del_analog = StylishQT.stop_deleteButton()
        self.button_del_analog.setFixedHeight(32)
        self.AnalogLayout.addWidget(self.button_del_analog, 3, 2)

        self.switchAppendModeSwitch = StylishQT.MySwitch(
            "Append Mode", "spring green", "Non-append", "indian red", width=92
        )
        self.switchAppendModeSwitch.clicked.connect(
            lambda: self.setAppendModeFlag()
        )
        self.switchAppendModeSwitch.setToolTip(
            "In append mode, new waveforms will append at the end of the existing one."
        )

        self.AnalogLayout.addWidget(self.switchAppendModeSwitch, 3, 3)

        self.add_waveform_button.clicked.connect(self.add_waveform_analog)
        self.button_del_analog.clicked.connect(self.del_waveform_analog)

        # Tab for general block wave
        self.AnalogFreqTextbox = QLineEdit(self)
        self.wavetablayout.addWidget(self.AnalogFreqTextbox, 0, 1)
        self.wavetablayout.addWidget(QLabel("Frequency /s:"), 0, 0)

        self.AnalogOffsetTextbox = QLineEdit(self)
        self.AnalogOffsetTextbox.setPlaceholderText("0")
        self.wavetablayout.addWidget(self.AnalogOffsetTextbox, 1, 1)
        self.wavetablayout.addWidget(QLabel("Offset (ms):"), 1, 0)

        self.AnalogDurationTextbox = QLineEdit(self)
        self.wavetablayout.addWidget(self.AnalogDurationTextbox, 0, 3)
        self.wavetablayout.addWidget(QLabel("Duration (ms, 1 cycle):"), 0, 2)

        self.AnalogRepeatTextbox = QLineEdit(self)
        self.AnalogRepeatTextbox.setPlaceholderText("1")
        self.wavetablayout.addWidget(self.AnalogRepeatTextbox, 3, 3)
        self.wavetablayout.addWidget(QLabel("Number of cycles:"), 3, 2)

        self.wavetablayout.addWidget(QLabel("Duty cycle (%):"), 0, 4)
        self.AnalogDCTextbox = QDoubleSpinBox(self)
        self.AnalogDCTextbox.setMinimum(0)
        self.AnalogDCTextbox.setMaximum(100)
        self.AnalogDCTextbox.setValue(50)
        self.AnalogDCTextbox.setDecimals(2)
        self.AnalogDCTextbox.setSingleStep(5)
        self.wavetablayout.addWidget(self.AnalogDCTextbox, 0, 5)

        self.AnalogGapTextbox = QLineEdit(self)
        self.AnalogGapTextbox.setPlaceholderText("0")
        self.wavetablayout.addWidget(self.AnalogGapTextbox, 1, 5)
        self.wavetablayout.addWidget(
            QLabel("Gap between cycles (samples):"), 1, 4
        )

        self.wavetablayout.addWidget(QLabel("Starting amplitude (V):"), 2, 0)
        self.AnalogStartingAmpTextbox = QDoubleSpinBox(self)
        self.AnalogStartingAmpTextbox.setMinimum(-10)
        self.AnalogStartingAmpTextbox.setMaximum(10)
        self.AnalogStartingAmpTextbox.setValue(5)
        self.AnalogStartingAmpTextbox.setDecimals(5)
        self.AnalogStartingAmpTextbox.setSingleStep(0.5)
        self.wavetablayout.addWidget(self.AnalogStartingAmpTextbox, 2, 1)

        self.AnalogBaselineTextbox = QLineEdit(self)
        self.AnalogBaselineTextbox.setPlaceholderText("0")
        self.wavetablayout.addWidget(self.AnalogBaselineTextbox, 3, 1)
        self.wavetablayout.addWidget(QLabel("Baseline (V):"), 3, 0)

        self.wavetablayout.addWidget(QLabel("Change per step (V):"), 2, 2)
        self.AnalogStepTextbox = QDoubleSpinBox(self)
        self.AnalogStepTextbox.setMinimum(-10)
        self.AnalogStepTextbox.setMaximum(10)
        self.AnalogStepTextbox.setDecimals(4)
        self.AnalogStepTextbox.setValue(5)
        self.AnalogStepTextbox.setSingleStep(0.5)
        self.wavetablayout.addWidget(self.AnalogStepTextbox, 2, 3)

        self.wavetablayout.addWidget(QLabel("Steps in duration:"), 1, 2)
        self.AnalogCyclesTextbox = QSpinBox(self)
        self.AnalogCyclesTextbox.setMinimum(0)
        self.AnalogCyclesTextbox.setMaximum(100)
        self.AnalogCyclesTextbox.setValue(1)
        self.AnalogCyclesTextbox.setSingleStep(1)
        self.wavetablayout.addWidget(self.AnalogCyclesTextbox, 1, 3)

        self.wavetab1.setLayout(self.wavetablayout)

        # Tab for general Pramp wave
        self.wavetablayout_ramp = QGridLayout()
        self.AnalogFreqTextbox_ramp = QLineEdit(self)
        self.wavetablayout_ramp.addWidget(self.AnalogFreqTextbox_ramp, 0, 1)
        self.wavetablayout_ramp.addWidget(QLabel("Frequency in period:"), 0, 0)

        self.AnalogOffsetTextbox_ramp = QLineEdit(self)
        self.AnalogOffsetTextbox_ramp.setPlaceholderText("0")
        self.wavetablayout_ramp.addWidget(self.AnalogOffsetTextbox_ramp, 1, 1)
        self.wavetablayout_ramp.addWidget(QLabel("Offset (ms):"), 1, 0)

        self.AnalogDurationTextbox_ramp = QLineEdit(self)
        self.wavetablayout_ramp.addWidget(
            self.AnalogDurationTextbox_ramp, 0, 3
        )
        self.wavetablayout_ramp.addWidget(
            QLabel("Duration (ms, 1 cycle):"), 0, 2
        )

        self.AnalogDCTextbox_ramp = QLineEdit(self)
        self.AnalogDCTextbox_ramp.setPlaceholderText("0.5")
        self.wavetablayout_ramp.addWidget(self.AnalogDCTextbox_ramp, 0, 5)
        self.wavetablayout_ramp.addWidget(QLabel("Symmetry:"), 0, 4)

        self.AnalogRepeatTextbox_ramp = QLineEdit(self)
        self.AnalogRepeatTextbox_ramp.setPlaceholderText("1")
        self.wavetablayout_ramp.addWidget(self.AnalogRepeatTextbox_ramp, 1, 3)
        self.wavetablayout_ramp.addWidget(QLabel("Repeat:"), 1, 2)

        self.AnalogGapTextbox_ramp = QLineEdit(self)
        self.AnalogGapTextbox_ramp.setPlaceholderText("0")
        self.wavetablayout_ramp.addWidget(self.AnalogGapTextbox_ramp, 1, 5)
        self.wavetablayout_ramp.addWidget(
            QLabel("Gap between repeat (samples):"), 1, 4
        )

        self.wavetablayout_ramp.addWidget(QLabel("Height (V):"), 2, 0)
        self.AnalogStartingAmpTextbox_ramp = QDoubleSpinBox(self)
        self.AnalogStartingAmpTextbox_ramp.setMinimum(-10)
        self.AnalogStartingAmpTextbox_ramp.setMaximum(10)
        self.AnalogStartingAmpTextbox_ramp.setValue(2)
        self.AnalogStartingAmpTextbox_ramp.setSingleStep(0.5)
        self.wavetablayout_ramp.addWidget(
            self.AnalogStartingAmpTextbox_ramp, 2, 1
        )

        self.AnalogBaselineTextbox_ramp = QLineEdit(self)
        self.AnalogBaselineTextbox_ramp.setPlaceholderText("0")
        self.wavetablayout_ramp.addWidget(
            self.AnalogBaselineTextbox_ramp, 3, 1
        )
        self.wavetablayout_ramp.addWidget(QLabel("Baseline (V):"), 3, 0)

        self.wavetablayout_ramp.addWidget(QLabel("Step (V):"), 2, 2)
        self.AnalogStepTextbox_ramp = QDoubleSpinBox(self)
        self.AnalogStepTextbox_ramp.setMinimum(-10)
        self.AnalogStepTextbox_ramp.setMaximum(10)
        self.AnalogStepTextbox_ramp.setValue(1)
        self.AnalogStepTextbox_ramp.setSingleStep(0.5)
        self.wavetablayout_ramp.addWidget(self.AnalogStepTextbox_ramp, 2, 3)

        self.wavetablayout_ramp.addWidget(QLabel("Cycles:"), 3, 2)
        self.AnalogCyclesTextbox_ramp = QSpinBox(self)
        self.AnalogCyclesTextbox_ramp.setMinimum(0)
        self.AnalogCyclesTextbox_ramp.setMaximum(100)
        self.AnalogCyclesTextbox_ramp.setValue(1)
        self.AnalogCyclesTextbox_ramp.setSingleStep(1)
        self.wavetablayout_ramp.addWidget(self.AnalogCyclesTextbox_ramp, 3, 3)

        self.wavetab2.setLayout(self.wavetablayout_ramp)

        # === photocycle ===
        self.photocycletablayout = QGridLayout()

        # Tab for general block wave
        self.textbox_photocycleA = QLineEdit(self)
        self.photocycletablayout.addWidget(self.textbox_photocycleA, 0, 1)
        self.photocycletablayout.addWidget(QLabel("Frequency /s:"), 0, 0)

        self.textbox_photocycleB = QLineEdit(self)
        self.textbox_photocycleB.setPlaceholderText("100")
        self.photocycletablayout.addWidget(self.textbox_photocycleB, 1, 1)
        self.photocycletablayout.addWidget(QLabel("Offset (ms):"), 1, 0)

        self.textbox_photocycleC = QLineEdit(self)
        self.photocycletablayout.addWidget(self.textbox_photocycleC, 0, 3)
        self.photocycletablayout.addWidget(
            QLabel("Duration (ms, 1 cycle):"), 0, 2
        )

        self.textbox_photocycleD = QLineEdit(self)
        self.textbox_photocycleD.setPlaceholderText("10")
        self.photocycletablayout.addWidget(self.textbox_photocycleD, 1, 3)
        self.photocycletablayout.addWidget(QLabel("Repeat:"), 1, 2)

        self.photocycletablayout.addWidget(QLabel("DC (%):"), 0, 4)
        self.textbox_photocycleE = QComboBox()
        self.textbox_photocycleE.addItems(["50", "100", "0"])
        self.photocycletablayout.addWidget(self.textbox_photocycleE, 0, 5)

        self.textbox_photocycleF = QLineEdit(self)
        self.textbox_photocycleF.setPlaceholderText("100000")
        self.photocycletablayout.addWidget(self.textbox_photocycleF, 1, 5)
        self.photocycletablayout.addWidget(
            QLabel("Gap between repeat (samples):"), 1, 4
        )

        self.photocycletablayout.addWidget(
            QLabel("Starting amplitude (V):"), 2, 0
        )
        self.textbox_photocycleG = QDoubleSpinBox(self)
        self.textbox_photocycleG.setMinimum(-10)
        self.textbox_photocycleG.setMaximum(10)
        self.textbox_photocycleG.setDecimals(5)
        self.textbox_photocycleG.setValue(2)
        self.textbox_photocycleG.setSingleStep(0.5)
        self.photocycletablayout.addWidget(self.textbox_photocycleG, 2, 1)

        self.textbox_photocycleH = QLineEdit(self)
        self.textbox_photocycleH.setPlaceholderText("0")
        self.photocycletablayout.addWidget(self.textbox_photocycleH, 3, 1)
        self.photocycletablayout.addWidget(QLabel("Baseline (V):"), 3, 0)

        self.photocycletablayout.addWidget(QLabel("Step (V):"), 2, 2)
        self.textbox_photocycleI = QDoubleSpinBox(self)
        self.textbox_photocycleI.setMinimum(-10)
        self.textbox_photocycleI.setMaximum(10)
        self.textbox_photocycleI.setValue(0.33)
        self.textbox_photocycleI.setSingleStep(0.5)
        self.photocycletablayout.addWidget(self.textbox_photocycleI, 2, 3)

        self.photocycletablayout.addWidget(QLabel("Cycles:"), 3, 2)
        self.textbox_photocycleJ = QSpinBox(self)
        self.textbox_photocycleJ.setMinimum(0)
        self.textbox_photocycleJ.setMaximum(100)
        self.textbox_photocycleJ.setValue(1)
        self.textbox_photocycleJ.setSingleStep(1)
        self.photocycletablayout.addWidget(self.textbox_photocycleJ, 3, 3)

        self.photocycletablayout.addWidget(QLabel("start_point:"), 3, 4)
        self.textbox_photocycleK = QSpinBox(self)
        self.textbox_photocycleK.setMinimum(0)
        self.textbox_photocycleK.setMaximum(100)
        self.textbox_photocycleK.setValue(2)
        self.textbox_photocycleK.setSingleStep(1)
        self.photocycletablayout.addWidget(self.textbox_photocycleK, 3, 5)

        self.photocycletablayout.addWidget(QLabel("start_time:"), 3, 6)
        self.textbox_photocycleL = QDoubleSpinBox(self)
        self.textbox_photocycleL.setMinimum(0)
        self.textbox_photocycleL.setMaximum(100)
        self.textbox_photocycleL.setValue(0.5)
        self.textbox_photocycleL.setSingleStep(1)
        self.photocycletablayout.addWidget(self.textbox_photocycleL, 3, 7)

        self.photocycletablayout.addWidget(QLabel("control_amplitude:"), 2, 4)
        self.textbox_photocycleM = QDoubleSpinBox(self)
        self.textbox_photocycleM.setMinimum(0)
        self.textbox_photocycleM.setMaximum(100)
        self.textbox_photocycleM.setValue(0.33)
        self.textbox_photocycleM.setSingleStep(1)
        self.photocycletablayout.addWidget(self.textbox_photocycleM, 2, 5)

        self.wavetab5.setLayout(self.photocycletablayout)

        # Galvo scanning tab

        self.galvotablayout = QGridLayout()
        self.galvos_tabs = QTabWidget()
        self.normal_galvo_tab = QWidget()
        self.galvo_raster_tablayout = QGridLayout()
        self.contour_galvo_tab = QWidget()
        self.galvo_contour_tablayout = QGridLayout()
        # self.controlLayout.addWidget(QLabel("Galvo raster scanning : "), 1, 0)
        self.GalvoVoltXMinTextbox = QSpinBox(self)
        self.GalvoVoltXMinTextbox.setMinimum(-10)
        self.GalvoVoltXMinTextbox.setMaximum(10)
        self.GalvoVoltXMinTextbox.setValue(-5)
        self.GalvoVoltXMinTextbox.setSingleStep(1)
        self.galvo_raster_tablayout.addWidget(self.GalvoVoltXMinTextbox, 0, 1)
        self.galvo_raster_tablayout.addWidget(QLabel("voltXMin"), 0, 0)

        self.GalvoVoltXMaxTextbox = QSpinBox(self)
        self.GalvoVoltXMaxTextbox.setMinimum(-10)
        self.GalvoVoltXMaxTextbox.setMaximum(10)
        self.GalvoVoltXMaxTextbox.setValue(5)
        self.GalvoVoltXMaxTextbox.setSingleStep(1)
        self.galvo_raster_tablayout.addWidget(self.GalvoVoltXMaxTextbox, 1, 1)
        self.galvo_raster_tablayout.addWidget(QLabel("voltXMax"), 1, 0)

        self.GalvoVoltYMinTextbox = QSpinBox(self)
        self.GalvoVoltYMinTextbox.setMinimum(-10)
        self.GalvoVoltYMinTextbox.setMaximum(10)
        self.GalvoVoltYMinTextbox.setValue(-5)
        self.GalvoVoltYMinTextbox.setSingleStep(1)
        self.galvo_raster_tablayout.addWidget(self.GalvoVoltYMinTextbox, 0, 3)
        self.galvo_raster_tablayout.addWidget(QLabel("voltYMin"), 0, 2)

        self.GalvoVoltYMaxTextbox = QSpinBox(self)
        self.GalvoVoltYMaxTextbox.setMinimum(-10)
        self.GalvoVoltYMaxTextbox.setMaximum(10)
        self.GalvoVoltYMaxTextbox.setValue(5)
        self.GalvoVoltYMaxTextbox.setSingleStep(1)
        self.galvo_raster_tablayout.addWidget(self.GalvoVoltYMaxTextbox, 1, 3)
        self.galvo_raster_tablayout.addWidget(QLabel("voltYMax"), 1, 2)

        self.GalvoXpixelNumTextbox = QComboBox()
        self.GalvoXpixelNumTextbox.addItems(["500", "256"])
        self.galvo_raster_tablayout.addWidget(self.GalvoXpixelNumTextbox, 0, 5)
        self.galvo_raster_tablayout.addWidget(QLabel("X pixel number"), 0, 4)

        self.GalvoYpixelNumTextbox = QComboBox()
        self.GalvoYpixelNumTextbox.addItems(["500", "256"])
        self.galvo_raster_tablayout.addWidget(self.GalvoYpixelNumTextbox, 1, 5)
        self.galvo_raster_tablayout.addWidget(QLabel("Y pixel number"), 1, 4)

        self.GalvoOffsetTextbox = QLineEdit(self)
        self.GalvoOffsetTextbox.setPlaceholderText("0")
        self.galvo_raster_tablayout.addWidget(self.GalvoOffsetTextbox, 2, 1)
        self.galvo_raster_tablayout.addWidget(QLabel("Offset (ms):"), 2, 0)

        self.GalvoGapTextbox = QLineEdit(self)
        self.GalvoGapTextbox.setPlaceholderText("0")
        self.galvo_raster_tablayout.addWidget(self.GalvoGapTextbox, 2, 3)
        self.galvo_raster_tablayout.addWidget(
            QLabel("Gap between scans(ms):"), 2, 2
        )

        self.GalvoAvgNumTextbox = QSpinBox(self)
        self.GalvoAvgNumTextbox.setMinimum(1)
        self.GalvoAvgNumTextbox.setMaximum(20)
        self.GalvoAvgNumTextbox.setValue(2)
        self.GalvoAvgNumTextbox.setSingleStep(1)
        self.galvo_raster_tablayout.addWidget(self.GalvoAvgNumTextbox, 2, 5)
        self.galvo_raster_tablayout.addWidget(QLabel("average over:"), 2, 4)

        self.GalvoRepeatTextbox = QSpinBox(self)
        self.GalvoRepeatTextbox.setMinimum(1)
        self.GalvoRepeatTextbox.setMaximum(20)
        self.GalvoRepeatTextbox.setValue(1)
        self.GalvoRepeatTextbox.setSingleStep(1)
        self.galvo_raster_tablayout.addWidget(self.GalvoRepeatTextbox, 0, 7)
        self.galvo_raster_tablayout.addWidget(QLabel("Repeat:"), 0, 6)

        self.galvo_contour_label_1 = QLabel("Points in contour:")
        self.galvo_contour_tablayout.addWidget(
            self.galvo_contour_label_1, 0, 0
        )

        self.galvo_contour_label_2 = QLabel("Sampling rate: ")
        self.galvo_contour_tablayout.addWidget(
            self.galvo_contour_label_2, 0, 1
        )

        self.GalvoContourLastTextbox = QSpinBox(self)
        self.GalvoContourLastTextbox.setMinimum(000000)
        self.GalvoContourLastTextbox.setMaximum(20000000)
        self.GalvoContourLastTextbox.setValue(1000)
        self.GalvoContourLastTextbox.setSingleStep(500)
        self.galvo_contour_tablayout.addWidget(
            self.GalvoContourLastTextbox, 0, 3
        )
        self.galvo_contour_tablayout.addWidget(QLabel("Duration(ms):"), 0, 2)

        self.normal_galvo_tab.setLayout(self.galvo_raster_tablayout)
        self.contour_galvo_tab.setLayout(self.galvo_contour_tablayout)
        self.galvos_tabs.addTab(self.normal_galvo_tab, "Raster scanning")
        self.galvos_tabs.addTab(self.contour_galvo_tab, "Contour scanning")

        self.galvotablayout.addWidget(self.galvos_tabs, 0, 0)

        self.wavetab4.setLayout(self.galvotablayout)

        self.AnalogLayout.addWidget(self.wavetabs, 4, 0, 2, 6)

        AnalogContainer.setLayout(self.AnalogLayout)

        # === Digital ===
        DigitalContainer = QGroupBox("Digital signals")
        self.DigitalLayout = QGridLayout()  # self.AnalogLayout manager

        self.Digital_channel_combox = QComboBox()
        self.Digital_channel_combox.addItems(self.DigitalChannelList)
        self.DigitalLayout.addWidget(self.Digital_channel_combox, 0, 0)

        self.button3 = StylishQT.addButton()
        self.DigitalLayout.addWidget(self.button3, 0, 1)
        self.button3.clicked.connect(self.add_waveform_digital)

        self.button_del_digital = StylishQT.stop_deleteButton()
        self.button_del_digital.setFixedHeight(32)
        self.DigitalLayout.addWidget(self.button_del_digital, 0, 2)
        self.button_del_digital.clicked.connect(self.del_waveform_digital)

        self.switchAutoPadding = StylishQT.MySwitch(
            "Auto padding",
            "spring green",
            "Auto padding Off",
            "indian red",
            width=92,
        )
        self.switchAutoPadding.clicked.connect(self.setAutoPadding)
        self.switchAutoPadding.setToolTip(
            "Add one extra camera trigger at the start or not--"
            "Camera losing first frame"
        )

        self.DigitalLayout.addWidget(self.switchAutoPadding, 0, 3)

        self.switchExtraTrigger = StylishQT.MySwitch(
            "Extra camTrigger",
            "spring green",
            "E-cT Off",
            "indian red",
            width=92,
        )
        self.switchExtraTrigger.clicked.connect(self.setExtraTriggerFlag)
        self.switchExtraTrigger.setToolTip(
            "Add one extra camera trigger at the start or not--"
            "Camera losing first frame"
        )

        self.DigitalLayout.addWidget(self.switchExtraTrigger, 0, 4)

        # === Wave settings ===
        self.digitalwavetablayout = QGridLayout()

        self.digitalwavetabs = QTabWidget()
        self.digitalwavetab1 = QWidget()
        self.digitalwavetab2 = QWidget()
        self.digitalwavetab3 = QWidget()

        # Add tabs
        self.digitalwavetabs.addTab(self.digitalwavetab1, "Block")
        # self.digitalwavetabs.addTab(self.digitalwavetab2,"Ramp")
        # self.digitalwavetabs.addTab(self.digitalwavetab3,"Matlab")

        self.DigFreqTextbox = QLineEdit(self)
        self.digitalwavetablayout.addWidget(self.DigFreqTextbox, 0, 1)
        self.digitalwavetablayout.addWidget(QLabel("Frequency /s:"), 0, 0)

        self.DigOffsetTextbox = QLineEdit(self)
        self.DigOffsetTextbox.setPlaceholderText("0")
        self.digitalwavetablayout.addWidget(self.DigOffsetTextbox, 1, 1)
        self.digitalwavetablayout.addWidget(QLabel("Offset (ms):"), 1, 0)

        self.DigDurationTextbox = QLineEdit(self)
        self.digitalwavetablayout.addWidget(self.DigDurationTextbox, 0, 3)
        self.digitalwavetablayout.addWidget(QLabel("Duration (ms):"), 0, 2)

        self.DigRepeatTextbox = QLineEdit(self)
        self.DigRepeatTextbox.setPlaceholderText("1")
        self.digitalwavetablayout.addWidget(self.DigRepeatTextbox, 1, 3)
        self.digitalwavetablayout.addWidget(QLabel("Repeat:"), 1, 2)

        self.digitalwavetablayout.addWidget(QLabel("DC (%):"), 0, 4)
        self.digital_DC_spinbox = QDoubleSpinBox(self)
        self.digital_DC_spinbox.setMinimum(0)
        self.digital_DC_spinbox.setMaximum(100)
        self.digital_DC_spinbox.setValue(50)
        self.digital_DC_spinbox.setDecimals(2)
        self.digital_DC_spinbox.setSingleStep(5)
        self.digitalwavetablayout.addWidget(self.digital_DC_spinbox, 0, 5)

        self.DigGapTextbox = QLineEdit(self)
        self.DigGapTextbox.setPlaceholderText("0")
        self.digitalwavetablayout.addWidget(self.DigGapTextbox, 1, 5)
        self.digitalwavetablayout.addWidget(
            QLabel("Gap between repeat (samples):"), 1, 4
        )

        self.digitalwavetab1.setLayout(self.digitalwavetablayout)
        self.DigitalLayout.addWidget(self.digitalwavetabs, 2, 0, 3, 6)

        DigitalContainer.setLayout(self.DigitalLayout)

        # === Display win ===
        self.pw = pg.PlotWidget(title="Waveform plot")
        self.pw.setLabel("bottom", "Time", units="s")
        self.pw.setLabel("left", "Value", units="V")
        self.pw.addLine(x=0)
        self.pw.addLine(y=0)
        self.pw.setMinimumHeight(180)

        self.pw_PlotItem = self.pw.getPlotItem()
        self.pw_PlotItem.addLegend()
        self.pw_PlotItem.setDownsampling(auto=True, mode="mean")
        # === Data win ===
        self.pw_data = pg.PlotWidget(title="Data")
        self.pw_data.setLabel("bottom", "Time", units="s")
        self.pw_data.setMinimumHeight(180)
        # self.pw_data.setLabel('left', 'Value', units='V')
        # === Adding to master ===
        master_waveform = QGridLayout()
        master_waveform.addWidget(AnalogContainer, 1, 0, 1, 2)
        master_waveform.addWidget(DigitalContainer, 2, 0, 1, 2)
        master_waveform.addWidget(ReadContainer, 0, 0, 1, 1)
        master_waveform.addWidget(executionContainer, 0, 1, 1, 1)
        master_waveform.addWidget(self.pw, 3, 0, 1, 2)
        # master_waveform.addWidget(self.pw_data, 4, 0)
        self.tabs.setLayout(master_waveform)
        # self.setLayout(pmtmaster)
        self.layout.addWidget(self.tabs, 0, 0)
        self.setLayout(self.layout)

        # Automatically switch to galvo settings
        if self.current_Analog_channel.currentIndex() == 0:
            logging.info(1112132)
            self.wavetabs.setCurrentIndex(2)

    # %%
    # Functions for Waveform Tab

    def get_wave_file_np(self):
        self.wavenpfileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            "M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data",  # TODO hardcoded path
            "(*.npy)",
        )
        self.textbox_loadwave.setText(self.wavenpfileName)

    def load_wave_np(self):
        """
        Load the pre-configured waveforms saved in np format.

        Returns
        None.

        """
        # When loading waveform file, no padding needed.
        try:
            self.switchAutoPadding.setChecked(True)
            self.setAutoPadding()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)

        self.wavenpfileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            "",
            "(*.npy)",
        )

        try:
            temp_loaded_container = waveform_specification.load(
                self.wavenpfileName
            )

            self.uiDaq_sample_rate = waveform_specification.get_sample_rate(
                self.wavenpfileName
            )

            if self.uiDaq_sample_rate != int(self.SamplingRateTextbox.value()):
                logging.info("ERROR: Sampling rates is different!")

            self.PlotDataItem_dict = {}
            self.waveform_data_dict = {}

            for item in temp_loaded_container:
                channel_keyword = item["Specification"]
                self.waveform_data_dict[channel_keyword] = item["Waveform"]
                self.generate_graphy(
                    channel_keyword, self.waveform_data_dict[channel_keyword]
                )
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("File not valid.")

    # %%
    def setAppendModeFlag(self):
        # Add extra 4 samples of one camera trigger or not
        if self.switchAppendModeSwitch.isChecked():
            self.Append_Mode = True

        else:
            self.Append_Mode = False

    def add_waveform_analog(self):
        """
        Add analog waveforms.The waveform_data_dict dictionary will collect all
        the waveforms, with the key being the channel name.

        Returns
        None.

        """
        # make sure that the square wave tab is active now
        channel_keyword = self.current_Analog_channel.currentText()

        if self.wavetabs.currentIndex() != 2:

            if self.wavetabs.currentIndex() == 0:

                # === Square waves ===
                waveform_to_add = self.generate_analog(channel_keyword)

            elif self.wavetabs.currentIndex() == 1:
                # === Ramp waves ===
                waveform_to_add = self.generate_ramp(channel_keyword)

            if self.wavetabs.currentIndex() == 4:
                # === Photo cycle ===
                waveform_to_add = self.generate_photocycle(channel_keyword)

            if self.Append_Mode is False:
                self.waveform_data_dict[channel_keyword] = waveform_to_add
            else:
                # === In Append mode ===
                # If the waveform exists already
                if channel_keyword in self.waveform_data_dict.keys():
                    self.waveform_data_dict[channel_keyword] = np.append(
                        self.waveform_data_dict[channel_keyword],
                        waveform_to_add,
                    )
                # The first to append
                else:
                    self.waveform_data_dict[channel_keyword] = waveform_to_add

            self.generate_graphy(
                channel_keyword, self.waveform_data_dict[channel_keyword]
            )

        # === Galvo scanning ===
        elif self.wavetabs.currentIndex() == 2:

            if self.galvos_tabs.currentIndex() == 0:
                self.waveform_data_dict[channel_keyword] = (
                    self.generate_galvos()
                )
                self.generate_graphy(
                    channel_keyword,
                    self.waveform_data_dict[channel_keyword][1, :],
                )

            elif self.galvos_tabs.currentIndex() == 1:  # For contour
                self.waveform_data_dict["galvos_contour"] = (
                    self.generate_contour_for_waveform()
                )
                self.generate_graphy(
                    "galvos_contour",
                    self.waveform_data_dict["galvos_contour"][1, :],
                )

    def del_waveform_analog(self):

        channel_keyword = self.current_Analog_channel.currentText()

        if channel_keyword == "galvos_contour":
            try:
                # In case of just generated contour scanning signals
                self.pw_PlotItem.removeItem(
                    self.PlotDataItem_dict[channel_keyword]
                )

                del self.PlotDataItem_dict[channel_keyword]

            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                # In case of delete loaded contour scanning signals
                self.pw_PlotItem.removeItem(
                    self.PlotDataItem_dict["galvos_X_contour"]
                )
                self.pw_PlotItem.removeItem(
                    self.PlotDataItem_dict["galvos_Y_contour"]
                )

                del self.PlotDataItem_dict["galvos_X_contour"]
                del self.PlotDataItem_dict["galvos_Y_contour"]
        else:

            self.pw_PlotItem.removeItem(
                self.PlotDataItem_dict[channel_keyword]
            )

            del self.PlotDataItem_dict[channel_keyword]

        if "galvos" in channel_keyword:
            # As galvos key is changed, need to delete with adapted key name.
            galvo_key_to_delete = []
            for keys in self.waveform_data_dict:
                if "galvos" in keys:
                    galvo_key_to_delete.append(keys)
            for key in galvo_key_to_delete:
                del self.waveform_data_dict[key]
        else:
            del self.waveform_data_dict[channel_keyword]

        if channel_keyword == "fieldstimulator":
            waveform_to_add = self.generate_analog(
                channel_keyword
            )  # TODO: unused

    # %%

    def add_waveform_digital(self):
        """
        Add digital signals to the collection.

        Returns
        None.

        """
        channel_keyword = self.Digital_channel_combox.currentText()

        if channel_keyword != "galvotrigger":
            waveform_to_add = self.generate_digital(channel_keyword)
        else:
            waveform_to_add = self.generate_galvotrigger()

        if self.Append_Mode is False:
            self.waveform_data_dict[channel_keyword] = waveform_to_add
        else:
            # === In Append mode ===
            # If the waveform exists already
            if channel_keyword in self.waveform_data_dict.keys():
                self.waveform_data_dict[channel_keyword] = np.append(
                    self.waveform_data_dict[channel_keyword], waveform_to_add
                )
            # The first to append
            else:
                self.waveform_data_dict[channel_keyword] = waveform_to_add

        if channel_keyword == "cameratrigger":
            # For camera triggers,
            # set to zeros so that it does not block canvas.
            rectified_waveform = np.zeros(
                len(self.waveform_data_dict[channel_keyword]), dtype=bool
            )
        elif "servo" in channel_keyword:
            if channel_keyword in self.servo_dict and self.Append_Mode:
                servo_rectified_waveform = 5 * np.ones(
                    len(waveform_to_add), dtype=bool
                )
                servo_rectified_waveform[
                    : int(
                        self.uiDaq_sample_rate
                        * self.uiwaveoffset_digital_waveform
                        / 1000
                    )
                ] = 0
                self.servo_dict[channel_keyword]["rectified_waveform"] = (
                    np.append(
                        self.servo_dict[channel_keyword]["rectified_waveform"],
                        servo_rectified_waveform,
                    )
                )
            else:
                self.servo_dict[channel_keyword] = {
                    "offset": self.uiwaveoffset_digital_waveform,
                    "duration": self.uiwaveperiod_digital_waveform,
                }
                servo_rectified_waveform = 5 * np.ones(
                    len(waveform_to_add), dtype=bool
                )
                servo_rectified_waveform[
                    : int(
                        self.uiDaq_sample_rate
                        * self.uiwaveoffset_digital_waveform
                        / 1000
                    )
                ] = 0
                self.servo_dict[channel_keyword][
                    "rectified_waveform"
                ] = servo_rectified_waveform
            rectified_waveform = self.servo_dict[channel_keyword][
                "rectified_waveform"
            ]
        else:
            rectified_waveform = self.waveform_data_dict[channel_keyword]
        self.generate_graphy(channel_keyword, rectified_waveform)

    def del_waveform_digital(self):

        channel_keyword = self.Digital_channel_combox.currentText()

        self.pw_PlotItem.removeItem(self.PlotDataItem_dict[channel_keyword])

        del self.PlotDataItem_dict[channel_keyword]
        del self.waveform_data_dict[channel_keyword]
        del self.servo_dict[channel_keyword]

    def handle_stream_parameters(self, params):
        desired_fps = params["desired_fps"]
        total_time_s = params["total_time"]
        total_time_ms = total_time_s * 1000
        logging.info(
            f"Received stream parameters: desired_fps={desired_fps}, total_time={total_time_ms} ms."
        )

        self.Digital_channel_combox.setCurrentText("cameratrigger")
        self.DigFreqTextbox.setText(str(desired_fps))
        self.DigDurationTextbox.setText(str(total_time_ms))

    def setExtraTriggerFlag(self):
        # Add extra 4 samples of one camera trigger or not
        if self.switchExtraTrigger.isChecked():
            self.Adding_extra_camera_trigger_flag = True
            logging.info("Add one extra camera trigger.")
        else:
            self.Adding_extra_camera_trigger_flag = False
            logging.info("Don't add one extra camera trigger.")

    def setAutoPadding(self):
        # Add 0 at the start and the end of waveforms to reset channels.
        # For patchAO not necessarily 0.
        # Over command extra camera trigger.
        if self.switchAutoPadding.isChecked():
            self.Auto_padding_flag = True
            logging.info("Pad 0 to reset channels.")
        else:
            self.Auto_padding_flag = False
            logging.info("Don't pad 0 to reset channels.")

    # %%
    def generate_contour_for_waveform(self):
        """
        Variables related to contour scanning are set from main GUI,
        which comes from PMT widget.

        Returns
        TYPE
            DESCRIPTION.

        """
        self.total_contour_scanning_time = int(
            self.GalvoContourLastTextbox.value()
        )

        repeatnum_contour = int(
            self.total_contour_scanning_time / self.time_per_contour
        )
        repeated_contoursamples_1 = np.tile(
            self.handle_viewbox_coordinate_position_array_expanded_x,
            repeatnum_contour,
        )
        repeated_contoursamples_2 = np.tile(
            self.handle_viewbox_coordinate_position_array_expanded_y,
            repeatnum_contour,
        )

        # Append one extra values in the beginning like all others
        repeated_contoursamples_1 = np.insert(
            repeated_contoursamples_1, 0, repeated_contoursamples_1[0], axis=0
        )
        repeated_contoursamples_2 = np.insert(
            repeated_contoursamples_2, 0, repeated_contoursamples_2[0], axis=0
        )

        # Adding 0 to the end
        # repeated_contoursamples_1 = np.append(repeated_contoursamples_1,
        # 0)
        # repeated_contoursamples_2 = np.append(repeated_contoursamples_2,
        # 0)

        handle_viewbox_coordinate_position_array_expanded = np.vstack(
            (repeated_contoursamples_1, repeated_contoursamples_2)
        )

        return handle_viewbox_coordinate_position_array_expanded

    def generate_galvos(self):

        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())

        # Scanning settings
        Value_voltXMin = int(self.GalvoVoltXMinTextbox.value())
        Value_voltXMax = int(self.GalvoVoltXMaxTextbox.value())
        Value_voltYMin = int(self.GalvoVoltYMinTextbox.value())
        Value_voltYMax = int(self.GalvoVoltYMaxTextbox.value())
        Value_xPixels = int(self.GalvoXpixelNumTextbox.currentText())
        Value_yPixels = int(self.GalvoYpixelNumTextbox.currentText())
        self.averagenum = int(self.GalvoAvgNumTextbox.value())
        self.repeatnum = int(self.GalvoRepeatTextbox.value())
        if not self.GalvoOffsetTextbox.text():
            self.Galvo_samples_offset = 0
            self.offsetsamples_galvo = []

        else:
            self.Galvo_samples_offset = int(self.GalvoOffsetTextbox.text())

            # By default one 0 is added so that we have a rising edge at the
            # beginning.
            self.offsetsamples_number_galvo = int(
                (self.Galvo_samples_offset / 1000) * self.uiDaq_sample_rate
            )
            self.offsetsamples_galvo = np.zeros(
                self.offsetsamples_number_galvo
            )  # Be default offsetsamples_number is an integer.
        # Generate galvo samples
        self.samples_1, self.samples_2 = waveRecPic(
            sampleRate=self.uiDaq_sample_rate,
            imAngle=0,
            voltXMin=Value_voltXMin,
            voltXMax=Value_voltXMax,
            voltYMin=Value_voltYMin,
            voltYMax=Value_voltYMax,
            xPixels=Value_xPixels,
            yPixels=Value_yPixels,
            sawtooth=True,
        )
        # Totalscansamples = len(self.samples_1)*self.averagenum
        # Calculate number of samples to feed to scanner,
        # by default it's one frame
        self.ScanArrayXnum = int(
            len(self.samples_1) / Value_yPixels
        )  # number of samples of each individual line of x scanning
        if not self.GalvoGapTextbox.text():
            gap_sample = 0
            self.gapsamples_number_galvo = 0
        else:
            gap_sample = int(self.GalvoGapTextbox.text())

            self.gapsamples_number_galvo = int(
                (gap_sample / 1000) * self.uiDaq_sample_rate
            )

        self.repeated_samples_1 = np.tile(self.samples_1, self.averagenum)
        self.repeated_samples_2_yaxis = np.tile(
            self.samples_2, self.averagenum
        )

        # In index array, indexes where there's the first image are 1.
        self.PMT_data_index_array = np.ones(len(self.repeated_samples_1))
        # Adding gap between scans
        self.gap_samples_1 = self.repeated_samples_1[-1] * np.ones(
            self.gapsamples_number_galvo
        )
        self.repeated_samples_1 = np.append(
            self.repeated_samples_1, self.gap_samples_1
        )
        self.gap_samples_2 = self.repeated_samples_2_yaxis[-1] * np.ones(
            self.gapsamples_number_galvo
        )
        self.repeated_samples_2_yaxis = np.append(
            self.repeated_samples_2_yaxis, self.gap_samples_2
        )

        self.PMT_data_index_array = np.append(
            self.PMT_data_index_array, np.zeros(self.gapsamples_number_galvo)
        )

        self.repeated_samples_1 = np.tile(
            self.repeated_samples_1, self.repeatnum
        )
        self.repeated_samples_2_yaxis = np.tile(
            self.repeated_samples_2_yaxis, self.repeatnum
        )

        # self.PMT_data_index_array_repeated is created to help locate pmt data
        # at different pre-set average or repeat scanning scheme.
        for i in range(
            self.repeatnum
        ):  # Array value where sits the second PMT image will be 2, etc.
            if i == 0:
                self.PMT_data_index_array_repeated = self.PMT_data_index_array
            else:
                self.PMT_data_index_array_repeated = np.append(
                    self.PMT_data_index_array_repeated,
                    self.PMT_data_index_array * (i + 1),
                )

        self.repeated_samples_1 = np.append(
            self.offsetsamples_galvo, self.repeated_samples_1
        )
        self.repeated_samples_1 = np.append(
            self.repeated_samples_1, 0
        )  # Add 0 to clear up Daq
        self.repeated_samples_2_yaxis = np.append(
            self.offsetsamples_galvo, self.repeated_samples_2_yaxis
        )
        self.repeated_samples_2_yaxis = np.append(
            self.repeated_samples_2_yaxis, 0
        )

        self.PMT_data_index_array_repeated = np.append(
            self.offsetsamples_galvo, self.PMT_data_index_array_repeated
        )
        # self.PMT_data_index_array_repeated = np.append(
        # self.PMT_data_index_array_repeated, 0
        # )

        Galvo_samples = np.vstack(
            (self.repeated_samples_1, self.repeated_samples_2_yaxis)
        )

        return Galvo_samples

    def generate_galvotrigger(self):
        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())
        # Scanning settings
        Value_voltXMin = int(self.GalvoVoltXMinTextbox.value())
        Value_voltXMax = int(self.GalvoVoltXMaxTextbox.value())
        Value_voltYMin = int(self.GalvoVoltYMinTextbox.value())
        Value_voltYMax = int(self.GalvoVoltYMaxTextbox.value())
        Value_xPixels = int(self.GalvoXpixelNumTextbox.currentText())
        Value_yPixels = int(self.GalvoYpixelNumTextbox.currentText())
        self.averagenum = int(self.GalvoAvgNumTextbox.value())
        repeatnum = int(self.GalvoRepeatTextbox.value())
        if not self.GalvoOffsetTextbox.text():
            self.Galvo_samples_offset = 0
            self.offsetsamples_galvo = []

        else:
            self.Galvo_samples_offset = int(self.GalvoOffsetTextbox.text())

            # By default one 0 is added so that we have a rising edge at the
            # beginning.
            self.offsetsamples_number_galvo = int(
                (self.Galvo_samples_offset / 1000) * self.uiDaq_sample_rate
            )
            self.offsetsamples_galvo = np.zeros(
                self.offsetsamples_number_galvo
            )  # Be default offsetsamples_number is an integer.
        # Generate galvo samples
        self.samples_1, self.samples_2 = waveRecPic(
            sampleRate=self.uiDaq_sample_rate,
            imAngle=0,
            voltXMin=Value_voltXMin,
            voltXMax=Value_voltXMax,
            voltYMin=Value_voltYMin,
            voltYMax=Value_voltYMax,
            xPixels=Value_xPixels,
            yPixels=Value_yPixels,
            sawtooth=True,
        )
        self.ScanArrayXnum = int(
            len(self.samples_1) / Value_yPixels
        )  # number of samples of each individual line of x scanning
        if not self.GalvoGapTextbox.text():
            gap_sample = 0
            self.gapsamples_number_galvo = 0
        else:
            gap_sample = int(self.GalvoGapTextbox.text())

            self.gapsamples_number_galvo = int(
                (gap_sample / 1000) * self.uiDaq_sample_rate
            )
        # print(self.Digital_container_feeder[:, 0])

        self.repeated_samples_1 = np.tile(self.samples_1, self.averagenum)
        self.repeated_samples_2_yaxis = np.tile(
            self.samples_2, self.averagenum
        )

        # Adding gap between scans
        self.gap_samples_1 = self.repeated_samples_1[-1] * np.ones(
            self.gapsamples_number_galvo
        )
        self.repeated_samples_1 = np.append(
            self.repeated_samples_1, self.gap_samples_1
        )
        self.gap_samples_2 = self.repeated_samples_2_yaxis[-1] * np.ones(
            self.gapsamples_number_galvo
        )
        self.repeated_samples_2_yaxis = np.append(
            self.repeated_samples_2_yaxis, self.gap_samples_2
        )

        self.repeated_samples_1 = np.tile(self.repeated_samples_1, repeatnum)
        self.repeated_samples_2_yaxis = np.tile(
            self.repeated_samples_2_yaxis, repeatnum
        )

        self.repeated_samples_1 = np.append(
            self.offsetsamples_galvo, self.repeated_samples_1
        )
        self.repeated_samples_2_yaxis = np.append(
            self.offsetsamples_galvo, self.repeated_samples_2_yaxis
        )

        samplenumber_oneframe = len(self.samples_1)

        self.true_sample_num_singleperiod_galvotrigger = round(
            (20 / 1000) * self.uiDaq_sample_rate
        )  # Default the trigger lasts for 20 ms.
        self.false_sample_num_singleperiod_galvotrigger = (
            samplenumber_oneframe
            - self.true_sample_num_singleperiod_galvotrigger
        )

        self.true_sample_singleperiod_galvotrigger = np.ones(
            self.true_sample_num_singleperiod_galvotrigger, dtype=bool
        )
        self.true_sample_singleperiod_galvotrigger[0] = (
            False  # first one False to give a rise.
        )

        self.sample_singleperiod_galvotrigger = np.append(
            self.true_sample_singleperiod_galvotrigger,
            np.zeros(
                self.false_sample_num_singleperiod_galvotrigger, dtype=bool
            ),
        )

        self.sample_repeatedperiod_galvotrigger = np.tile(
            self.sample_singleperiod_galvotrigger, self.averagenum
        )

        self.gap_samples_galvotrigger = np.zeros(
            self.gapsamples_number_galvo, dtype=bool
        )
        self.gap_samples_galvotrigger = np.append(
            self.sample_repeatedperiod_galvotrigger,
            self.gap_samples_galvotrigger,
        )
        self.repeated_gap_samples_galvotrigger = np.tile(
            self.gap_samples_galvotrigger, repeatnum
        )
        self.offset_galvotrigger = np.array(
            self.offsetsamples_galvo, dtype=bool
        )

        final_galvotrigger = np.append(
            self.offset_galvotrigger, self.repeated_gap_samples_galvotrigger
        )

        # Adding a False in the end
        # final_galvotrigger = np.append(final_galvotrigger, False)

        return final_galvotrigger

    # %%
    def generate_analog(self, channel):
        """
        Generate analog signals.

        Parameters
        channel : TYPE
            DESCRIPTION.

        Returns
        finalwave : TYPE
            DESCRIPTION.

        """
        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())
        uiwavefrequency_2 = float(self.AnalogFreqTextbox.text())
        if not self.AnalogOffsetTextbox.text():
            uiwaveoffset_2 = 0
        else:
            uiwaveoffset_2 = float(self.AnalogOffsetTextbox.text())  # in ms
        uiwaveperiod_2 = int(self.AnalogDurationTextbox.text())
        uiwaveDC_2 = self.AnalogDCTextbox.value()
        if not self.AnalogRepeatTextbox.text():
            uiwaverepeat_2 = 1
        else:
            uiwaverepeat_2 = int(self.AnalogRepeatTextbox.text())
        if not self.AnalogGapTextbox.text():
            uiwavegap_2 = 0
        else:
            uiwavegap_2 = int(self.AnalogGapTextbox.text())
        uiwavestartamplitude_2 = float(self.AnalogStartingAmpTextbox.value())
        if not self.AnalogBaselineTextbox.text():
            uiwavebaseline_2 = 0
        else:
            uiwavebaseline_2 = float(self.AnalogBaselineTextbox.text())
        uiwavestep_2 = float(self.AnalogStepTextbox.value())
        uiwavecycles_2 = int(self.AnalogCyclesTextbox.value())

        s = generate_AO_for640(
            self.uiDaq_sample_rate,
            uiwavefrequency_2,
            uiwaveoffset_2,
            uiwaveperiod_2,
            uiwaveDC_2,
            uiwaverepeat_2,
            uiwavegap_2,
            uiwavestartamplitude_2,
            uiwavebaseline_2,
            uiwavestep_2,
            uiwavecycles_2,
        )
        finalwave = s.generate()

        return finalwave

    # %%
    # === for generating digital signals ===
    def generate_digital(self, channel):
        # self.uiwaveDC_digital_waveform = None

        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())
        try:
            self.uiwavefrequency_digital_waveform = float(
                self.DigFreqTextbox.text()
            )
        except Exception as exc:
            logging.info("IGNORE if using SERVO")
            logging.info("Error missing frequency", exc_info=exc)

        if not self.DigOffsetTextbox.text():
            self.uiwaveoffset_digital_waveform = 0
        else:
            self.uiwaveoffset_digital_waveform = float(
                self.DigOffsetTextbox.text()
            )
        self.uiwaveperiod_digital_waveform = int(
            self.DigDurationTextbox.text()
        )
        self.uiwaveDC_digital_waveform = int(self.digital_DC_spinbox.value())
        if not self.DigRepeatTextbox.text():
            self.uiwaverepeat_digital_waveform_number = 1
        else:
            self.uiwaverepeat_digital_waveform_number = int(
                self.DigRepeatTextbox.text()
            )
        if not self.DigGapTextbox.text():
            self.uiwavegap_digital_waveform = 0
        else:
            self.uiwavegap_digital_waveform = int(self.DigGapTextbox.text())

        # if int(self.textbox11A.currentText()) == 1:
        if "servo" in channel:
            logging.info("servo")
            digital_waveform = generate_digital_waveform(
                self.uiDaq_sample_rate,
                50,
                self.uiwaveoffset_digital_waveform,
                self.uiwaveperiod_digital_waveform,
                min(self.uiwaveDC_digital_waveform, 10),
                self.uiwaverepeat_digital_waveform_number,
                self.uiwavegap_digital_waveform,
                True,
            )
        else:
            logging.info("no servo")
            digital_waveform = generate_digital_waveform(
                self.uiDaq_sample_rate,
                self.uiwavefrequency_digital_waveform,
                self.uiwaveoffset_digital_waveform,
                self.uiwaveperiod_digital_waveform,
                self.uiwaveDC_digital_waveform,
                self.uiwaverepeat_digital_waveform_number,
                self.uiwavegap_digital_waveform,
                False,
            )

        return digital_waveform.generate()

    # %%
    # === for generating ramp voltage signals ===
    def generate_ramp(self, channel):
        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())
        self.uiwavefrequency_ramp = float(self.AnalogFreqTextbox_ramp.text())
        if not self.AnalogOffsetTextbox_ramp.text():
            self.uiwaveoffset_ramp = 0
        else:
            self.uiwaveoffset_ramp = int(
                self.AnalogOffsetTextbox_ramp.text()
            )  # in ms
        self.uiwaveperiod_ramp = int(self.AnalogDurationTextbox_ramp.text())
        if not self.AnalogDCTextbox_ramp.text():
            self.uiwavesymmetry_ramp = 0.5
        else:
            self.uiwavesymmetry_ramp = float(self.AnalogDCTextbox_ramp.text())
        if not self.AnalogRepeatTextbox_ramp.text():
            self.uiwaverepeat_ramp = 1
        else:
            self.uiwaverepeat_ramp = int(self.AnalogRepeatTextbox_ramp.text())
        if not self.AnalogGapTextbox_ramp.text():
            self.uiwavegap_ramp = 0
        else:
            self.uiwavegap_ramp = int(self.AnalogGapTextbox_ramp.text())
        self.uiwavestartamplitude_ramp = float(
            self.AnalogStartingAmpTextbox_ramp.value()
        )
        if not self.AnalogBaselineTextbox_ramp.text():
            self.uiwavebaseline_ramp = 0
        else:
            self.uiwavebaseline_ramp = float(
                self.AnalogBaselineTextbox_ramp.text()
            )
        self.uiwavestep_ramp = float(self.AnalogStepTextbox_ramp.value())
        self.uiwavecycles_ramp = int(self.AnalogCyclesTextbox_ramp.value())

        ramp_instance = generate_ramp(
            self.uiDaq_sample_rate,
            self.uiwavefrequency_ramp,
            self.uiwaveoffset_ramp,
            self.uiwaveperiod_ramp,
            self.uiwavesymmetry_ramp,
            self.uiwaverepeat_ramp,
            self.uiwavegap_ramp,
            self.uiwavestartamplitude_ramp,
            self.uiwavebaseline_ramp,
            self.uiwavestep_ramp,
            self.uiwavecycles_ramp,
        )

        return ramp_instance.generate()

    # %%
    def generate_photocycle(self, channel):

        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())
        self.uiwavefrequency_photocycle_488 = float(
            self.textbox_photocycleA.text()
        )
        if not self.textbox_photocycleB.text():
            self.uiwavefrequency_offset_photocycle_488 = 100
        else:
            self.uiwavefrequency_offset_photocycle_488 = int(
                self.textbox_photocycleB.text()
            )
        self.uiwaveperiod_photocycle_488 = int(self.textbox_photocycleC.text())
        self.uiwaveDC_photocycle_488 = int(
            self.textbox_photocycleE.currentText()
        )
        if not self.textbox_photocycleD.text():
            self.uiwaverepeat_photocycle_488 = 10
        else:
            self.uiwaverepeat_photocycle_488 = int(
                self.textbox_photocycleD.text()
            )
        if not self.textbox_photocycleF.text():
            self.uiwavegap_photocycle_488 = 100000
        else:
            self.uiwavegap_photocycle_488 = int(
                self.textbox_photocycleF.text()
            )
        self.uiwavestartamplitude_photocycle_488 = float(
            self.textbox_photocycleG.value()
        )
        if not self.textbox_photocycleH.text():
            self.uiwavebaseline_photocycle_488 = 0
        else:
            self.uiwavebaseline_photocycle_488 = float(
                self.textbox_photocycleH.text()
            )
        self.uiwavestep_photocycle_488 = float(
            self.textbox_photocycleI.value()
        )
        self.uiwavecycles_photocycle_488 = float(
            self.textbox_photocycleJ.value()
        )
        self.uiwavestart_time_photocycle_488 = float(
            self.textbox_photocycleL.value()
        )

        self.uiwavecontrol_amplitude_photocycle_488 = float(
            self.textbox_photocycleM.value()
        )

        s = generate_AO(
            self.uiDaq_sample_rate,
            self.uiwavefrequency_photocycle_488,
            self.uiwavefrequency_offset_photocycle_488,
            self.uiwaveperiod_photocycle_488,
            self.uiwaveDC_photocycle_488,
            self.uiwaverepeat_photocycle_488,
            self.uiwavegap_photocycle_488,
            self.uiwavestartamplitude_photocycle_488,
            self.uiwavebaseline_photocycle_488,
            self.uiwavestep_photocycle_488,
            self.uiwavecycles_photocycle_488,
            self.uiwavestart_time_photocycle_488,
            self.uiwavecontrol_amplitude_photocycle_488,
        )
        finalwave = s.generate()

        return finalwave

    # %%
    def generate_graphy(self, channel, waveform):

        self.uiDaq_sample_rate = int(self.SamplingRateTextbox.value())
        if waveform.dtype == "bool":
            waveform = waveform.astype(int)

        x_label = np.arange(len(waveform)) / self.uiDaq_sample_rate
        if "servo" in channel:
            # if showing servo -> zero for offset
            current_PlotDataItem = PlotDataItem(
                x_label, waveform, name=channel
            )

        else:
            current_PlotDataItem = PlotDataItem(
                x_label, waveform, name=channel
            )
        current_PlotDataItem.setPen(self.color_dictionary[channel])

        if self.Append_Mode is True:
            try:
                self.pw_PlotItem.removeItem(self.PlotDataItem_dict[channel])
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
        self.pw_PlotItem.addItem(current_PlotDataItem)

        self.PlotDataItem_dict[channel] = current_PlotDataItem

    # %%

    def clear_canvas(self):
        # Back to initial state
        self.pw.clear()
        self.PlotDataItem_dict = {}
        self.waveform_data_dict = {}
        self.servo_dict = {}

    def organize_waveforms(self):
        """
        Each waveforms are first placed into a structure array with data tpye:
        np.dtype([('Waveform', float, (length_of_sig,)), ('Specification', 'U20')])
        and then append to a list and saved as np file.

        It's the last step before executing waveforms.

        To load the waveforms from saved np file:
            Trace = np.load(file path)[index of channel]("Waveform")
            Channel name = np.load(file path)[index of channel]("Specification")

        Returns
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        # === Find the reference waveform length. ===
        ReferenceWaveform_menu_text = (
            self.ReferenceWaveform_menu.selectedItems()[0].text()
        )

        try:
            if "galvos_X_contour" in self.waveform_data_dict.keys():
                # In case of loading contour waveforms when keys are not
                # "galvos_contour"
                reference_wave = self.waveform_data_dict["galvos_X_contour"]
            else:
                reference_wave = self.waveform_data_dict[
                    ReferenceWaveform_menu_text
                ]
        except KeyError:
            QMessageBox.warning(
                self,
                "Oops",
                "Please select the right reference waveform!",
                QMessageBox.Ok,
            )

        # Adding 4 values at the front, same for all waveforms except the
        # Camera trigger, which adds one extra trigger to solve the missing
        # trigger in the beginning issue.
        if self.Auto_padding_flag is True:
            logging.info("Auto-padding to reset channels.")

            self.padding_number = 115

            if self.Adding_extra_camera_trigger_flag is True:
                for waveform_key in self.waveform_data_dict:
                    if waveform_key == "cameratrigger":
                        # In case of cameratrigger, add a trigger composed of
                        # 4 values
                        insert_array = CAMERA_TRIGGER_INSERT_ARRAY

                        # Add False in the end
                        self.waveform_data_dict[waveform_key] = np.append(
                            self.waveform_data_dict[waveform_key], False
                        )
                    elif waveform_key in self.AnalogChannelList:
                        # === Padding at the end ===
                        if waveform_key != "patchAO":
                            # For analog channels,
                            insert_array = np.zeros(self.padding_number)

                            # Add 0 in the end
                            self.waveform_data_dict[waveform_key] = np.append(
                                self.waveform_data_dict[waveform_key], 0
                            )
                        else:
                            if np.amax(
                                self.waveform_data_dict[waveform_key]
                            ) == np.amin(
                                self.waveform_data_dict[waveform_key]
                            ):
                                # In case of holding potential
                                # For patch channels, add 4 same float values
                                insert_array = (
                                    np.ones([self.padding_number])
                                    * self.waveform_data_dict[waveform_key][0]
                                )
                            else:
                                # In case of step waves,
                                # append baseline values
                                insert_array = np.ones(
                                    [self.padding_number]
                                ) * np.amin(
                                    self.waveform_data_dict[waveform_key]
                                )

                            # In case of patch clamp, append same last value
                            self.waveform_data_dict[waveform_key] = np.append(
                                self.waveform_data_dict[waveform_key],
                                self.waveform_data_dict[waveform_key][-1],
                            )
                    else:
                        # For digital boolean signals
                        insert_array = np.zeros(
                            [self.padding_number], dtype=bool
                        )

                        # Add False in the end
                        self.waveform_data_dict[waveform_key] = np.append(
                            self.waveform_data_dict[waveform_key], False
                        )

                    # Insert the appendix
                    self.waveform_data_dict[waveform_key] = np.insert(
                        self.waveform_data_dict[waveform_key], 0, insert_array
                    )
            else:
                for waveform_key in self.waveform_data_dict:
                    if waveform_key in self.AnalogChannelList:
                        # === Padding at the end ===
                        if waveform_key != "patchAO":
                            # For analog channels,
                            insert_array = np.zeros(self.padding_number)
                            # Add 0 in the end
                            self.waveform_data_dict[waveform_key] = np.append(
                                self.waveform_data_dict[waveform_key], 0
                            )
                        else:
                            # For patch channels, add 4 same float values
                            insert_array = (
                                np.ones([self.padding_number])
                                * self.waveform_data_dict[waveform_key][0]
                            )
                            # In case of patch clamp, append same last value
                            self.waveform_data_dict[waveform_key] = np.append(
                                self.waveform_data_dict[waveform_key],
                                self.waveform_data_dict[waveform_key][-1],
                            )
                    else:
                        insert_array = np.zeros(
                            self.padding_number, dtype=bool
                        )
                        # Add False in the end
                        self.waveform_data_dict[waveform_key] = np.append(
                            self.waveform_data_dict[waveform_key], False
                        )

                    # Insert the appendix
                    self.waveform_data_dict[waveform_key] = np.insert(
                        self.waveform_data_dict[waveform_key], 0, insert_array
                    )
        else:
            logging.info("No Auto-padding to reset channels.")

        if ReferenceWaveform_menu_text == "galvos":
            # in case of using galvos as reference wave
            self.reference_length = len(reference_wave[0, :])

        elif ReferenceWaveform_menu_text == "galvos_contour":
            if "galvos_X_contour" in self.waveform_data_dict.keys():
                # In case of loading contour waveforms when keys are not
                # "galvos_contour"
                self.reference_length = len(reference_wave)
            else:
                self.reference_length = len(reference_wave[0, :])
        else:
            self.reference_length = len(reference_wave)

        if self.Auto_padding_flag is True:
            if self.Adding_extra_camera_trigger_flag is True:
                # In case of adding extra camera trigger, 4 values are added to
                # all channels at the start
                self.reference_length += len(insert_array) + 1
            else:
                # No extra camera trigger, 1 each extra in the beginning and at
                # the end
                self.reference_length += 2
            logging.info(f"reference_length: {self.reference_length}")
        else:
            # Without auto padding, reference length is the same as original
            # waveform.
            logging.info(f"reference_length: {self.reference_length}")

        # === Get all waveforms the same length. ===

        for waveform_key in self.waveform_data_dict:
            if self.waveform_data_dict[waveform_key].ndim == 1:
                # Cut or append 0 to the data for non-reference waveforms.
                len_waveform_data = len(self.waveform_data_dict[waveform_key])
                if len_waveform_data >= self.reference_length:
                    self.waveform_data_dict[waveform_key] = (
                        self.waveform_data_dict[waveform_key][
                            0 : self.reference_length
                        ]
                    )
                else:
                    if (
                        self.waveform_data_dict[waveform_key].dtype
                        == "float64"
                    ):
                        append_waveforms = np.zeros(
                            self.reference_length - len_waveform_data
                        )
                    else:
                        append_waveforms = np.zeros(
                            self.reference_length - len_waveform_data,
                            dtype=bool,
                        )

                    self.waveform_data_dict[waveform_key] = np.append(
                        self.waveform_data_dict[waveform_key], append_waveforms
                    )
            else:  # In case of galvos which has dimension 2.
                # Cut or append 0 to the data for non-reference waveforms.
                if (
                    len(self.waveform_data_dict[waveform_key][0, :])
                    >= self.reference_length
                ):
                    self.waveform_data_dict[waveform_key][0, :] = (
                        self.waveform_data_dict[waveform_key][0, :][
                            0 : self.reference_length
                        ]
                    )
                    self.waveform_data_dict[waveform_key][1, :] = (
                        self.waveform_data_dict[waveform_key][1, :][
                            0 : self.reference_length
                        ]
                    )

                else:
                    append_waveforms = np.zeros(
                        self.reference_length
                        - len(self.waveform_data_dict[waveform_key][0, :])
                    )
                    self.waveform_data_dict[waveform_key] = np.stack(
                        (
                            np.append(
                                self.waveform_data_dict[waveform_key][0, :],
                                append_waveforms,
                            ),
                            np.append(
                                self.waveform_data_dict[waveform_key][1, :],
                                append_waveforms,
                            ),
                        )
                    )

                    # Reset the PlotDataItem
                    # self.PlotDataItem_dict[waveform_key].setData(
                    # x_label,
                    # self.waveform_data_dict[waveform_key][1, :],
                    # name=waveform_key,
                    # )

        # === Set galvos sampele stack apart ===
        # Keys for contour scanning in waveform_data_dict change to
        # "galvos_X_contour" and "galvos_Y_contour"
        if (
            "galvos" in self.waveform_data_dict
            and "galvos_contour" not in self.waveform_data_dict
        ):
            self.waveform_data_dict[
                "galvosx"
                + "avgnum_"
                + str(int(self.GalvoAvgNumTextbox.value()))
            ] = self.waveform_data_dict["galvos"][0, :]
            self.waveform_data_dict[
                "galvosy"
                + "ypixels_"
                + str(int(self.GalvoYpixelNumTextbox.currentText()))
            ] = self.waveform_data_dict["galvos"][1, :]
            del self.waveform_data_dict["galvos"]

        if "galvos_contour" in self.waveform_data_dict:
            self.waveform_data_dict["galvos_X" + "_contour"] = (
                self.waveform_data_dict["galvos_contour"][0, :]
            )
            self.waveform_data_dict["galvos_Y" + "_contour"] = (
                self.waveform_data_dict["galvos_contour"][1, :]
            )
            del self.waveform_data_dict["galvos_contour"]

        # Structured array to contain
        # https://stackoverflow.com/questions/39622533/numpy-array-as-datatype-in-a-structured-array

        dataType_analog = waveform_specification.make_dtype(
            self.reference_length, float
        )
        dataType_digital = waveform_specification.make_dtype(
            self.reference_length, bool
        )

        # === Reset the PlotDataItem ===
        x_label = np.arange(self.reference_length) / self.uiDaq_sample_rate

        for waveform_key in self.waveform_data_dict:
            if self.waveform_data_dict[waveform_key].dtype == "float64":
                # In case of galvos re-drawing
                if "galvos_contour" in waveform_key:
                    self.PlotDataItem_dict["galvos_contour"].setData(
                        x_label,
                        self.waveform_data_dict[waveform_key],
                        name=waveform_key,
                    )
                elif "galvosx" in waveform_key or "galvosy" in waveform_key:
                    self.PlotDataItem_dict["galvos"].setData(
                        x_label,
                        self.waveform_data_dict[waveform_key],
                        name=waveform_key,
                    )
                elif "galvos_X" in waveform_key or "galvos_Y" in waveform_key:
                    self.PlotDataItem_dict["galvos_contour"].setData(
                        x_label,
                        self.waveform_data_dict[waveform_key],
                        name=waveform_key,
                    )
                else:
                    self.PlotDataItem_dict[waveform_key].setData(
                        x_label,
                        self.waveform_data_dict[waveform_key],
                        name=waveform_key,
                    )

            # THIS PIECE OF THE CODE IS FOR FORMATTING THE WAVEFORM DISPLAY, BUT IT DOESN'T WORK. IT DOES NOT SEEM ESSENTIAL SO I COMMENTED IT OUT - MARCO POST 19-03-2025
            """
            else:
                if 'servo' in waveform_key:
                    # create and pad plot
                    if hasattr(self, 'padding_number'):
                        temp_array = np.concatenate((np.zeros(self.padding_number), self.servo_dict[waveform_key]['rectified_waveform']))
                    else:
                            temp_array = self.servo_dict[waveform_key]['rectified_waveform']
                    if len(temp_array) < len(self.waveform_data_dict[waveform_key]):
                        temp_array = np.concatenate((temp_array, np.zeros(len(self.waveform_data_dict[waveform_key]) - len(temp_array))))
                    elif len(temp_array) > len(self.waveform_data_dict[waveform_key]):
                        temp_array = temp_array[:len(self.waveform_data_dict[waveform_key])]

                    self.PlotDataItem_dict[waveform_key].setData(
                        x_label,
                        temp_array.astype(int),
                        name=waveform_key,
                    )
                elif waveform_key != "cameratrigger": # and not 'servo' in waveform_key:
                    # In case of digital boolean signals, convert to int before
                    # plotting.
                    self.PlotDataItem_dict[waveform_key].setData(
                        x_label,
                        self.waveform_data_dict[waveform_key].astype(int),
                        name=waveform_key,
                    )
                else:
                    # For camera triggers, set to zeros so that it does not
                    # block canvas.
                    rectified_waveform = np.zeros(
                        len(self.waveform_data_dict[waveform_key]), dtype=bool
                    )

                    self.PlotDataItem_dict[waveform_key].setData(
                        x_label,
                        rectified_waveform.astype(int),
                        name=waveform_key,
                    )
              """

        # === Making containers ===
        digital_line_num = 0
        for waveform_key in self.waveform_data_dict:
            if waveform_key in self.DigitalChannelList:
                digital_line_num += 1
        analog_line_num = (
            len(self.waveform_data_dict.keys()) - digital_line_num
        )

        self.analog_array = np.zeros(analog_line_num, dtype=dataType_analog)
        self.digital_array = np.zeros(digital_line_num, dtype=dataType_digital)

        digital_line_num = 0
        analog_line_num = 0
        for waveform_key in self.waveform_data_dict:

            if (
                waveform_key in self.AnalogChannelList
                or "galvos" in waveform_key
            ):
                logging.info(len(self.waveform_data_dict[waveform_key]))
                self.analog_array[analog_line_num] = np.array(
                    [(self.waveform_data_dict[waveform_key], waveform_key)],
                    dtype=dataType_analog,
                )
                analog_line_num += 1

            elif waveform_key in self.DigitalChannelList:

                self.digital_array[digital_line_num] = np.array(
                    [(self.waveform_data_dict[waveform_key], waveform_key)],
                    dtype=dataType_digital,
                )
                digital_line_num += 1
        logging.info(
            "Writing channels: {}".format(self.waveform_data_dict.keys())
        )

        # === Saving configed waveforms ===
        if self.checkbox_saveWaveforms.isChecked():

            ciao = []  # Variable name 'ciao' was defined by Nicolo Ceffa.

            for i in range(len(self.analog_array["Specification"])):
                ciao.append(self.analog_array[i])

            for i in range(len(self.digital_array["Specification"])):
                ciao.append(self.digital_array[i])

            filename = waveform_specification.create_filename(
                self.saving_prefix, self.SamplingRateTextbox.value()
            )
            np.save(
                os.path.join(self.savedirectory, filename),
                ciao,
            )

            self.save_plot_figure()

        self.readinchan = []

        if self.ReadChanPMTTextbox.isChecked():
            self.readinchan.append("PMT")
        if self.ReadChanVpTextbox.isChecked():
            self.readinchan.append("Vp")
        if self.ReadChanIpTextbox.isChecked():
            self.readinchan.append("Ip")

        logging.info("Recording channels: {}".format(self.readinchan))

        self.GeneratedWaveformPackage = (
            int(self.SamplingRateTextbox.value()),
            self.analog_array,
            self.digital_array,
            self.readinchan,
        )
        self.WaveformPackage.emit(self.GeneratedWaveformPackage)

        try:
            self.GalvoScanInforPackage = (
                self.readinchan,
                self.repeatnum,
                self.PMT_data_index_array_repeated,
                self.averagenum,
                len(self.samples_1),
                self.ScanArrayXnum,
            )  # Emit a tuple
            self.GalvoScanInfor.emit(self.GalvoScanInforPackage)
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.GalvoScanInfor.emit("NoGalvo")  # Emit a string

        self.button_execute.setEnabled(True)

        return self.analog_array, self.digital_array, self.readinchan

    def save_plot_figure(self):
        """
        # create an exporter instance, as an argument give it
        # the item you wish to export
        """
        exporter = pg.exporters.ImageExporter(self.pw.getPlotItem())

        # set export parameters if needed
        exporter.parameters()[
            "width"
        ] = 1000  # (note this also affects height parameter)

        # save to file
        filename = waveform_specification.create_filename(
            self.saving_prefix, self.SamplingRateTextbox.value()
        )
        filename += ".png"
        exporter.export(os.path.join(self.savedirectory, filename))

    def execute_tread(self):
        """
        Tread to move filter in advance.

        Returns
        None.

        """
        if self.FilterButtongroup.checkedId() == -1:
            # No emission filter configured.
            pass
        elif self.FilterButtongroup.checkedId() == -2:
            # Arch filter selected.
            move_emission_filter_thread = threading.Thread(
                target=self.filter_move_towards("COM15", 0)
            )
            move_emission_filter_thread.start()
            move_emission_filter_thread.join()
            logging.info("Emission filter moved to Arch.")
            time.sleep(0.7)
        elif self.FilterButtongroup.checkedId() == -3:
            # Arch filter selected.
            move_emission_filter_thread = threading.Thread(
                target=self.filter_move_towards("COM15", 1)
            )
            move_emission_filter_thread.start()
            move_emission_filter_thread.join()
            logging.info("Emission filter moved to GFP/Citrine.")
            time.sleep(0.7)

        run_DAQ_Waveforms_thread = threading.Thread(
            target=self.run_DAQ_Waveforms, daemon=False
        )
        run_DAQ_Waveforms_thread.start()
        self.waveform_started.emit()

    def filter_move_towards(self, COMport, pos):
        # Filter move command.
        ELL9Filter_ins = ELL9Filter(COMport)
        ELL9Filter_ins.moveToPosition(pos)

    def run_DAQ_Waveforms(self):
        # Execute the runWaveforms function from NIdaq
        self.adcollector = DAQmission()
        self.adcollector.runWaveforms(
            clock_source=self.clock_source.currentText(),
            sampling_rate=self.uiDaq_sample_rate,
            analog_signals=self.analog_array,
            digital_signals=self.digital_array,
            readin_channels=self.readinchan,
        )
        self.adcollector.save_as_binary(self.savedirectory)

        # self.button_execute.setEnabled(False)

    def load_waveforms(self, WaveformTuple):
        self.WaveformSamplingRate = WaveformTuple[0]
        self.WaveformAnalogContainer = WaveformTuple[1]
        self.WaveformDigitalContainer = WaveformTuple[2]
        self.WaveformRecordingChannContainer = WaveformTuple[3]

        return (
            self.WaveformSamplingRate,
            self.WaveformAnalogContainer,
            self.WaveformDigitalContainer,
            self.WaveformRecordingChannContainer,
        )

    def recive_data(self, data_waveformreceived):
        """
        Display the recorded signals after execution.

        Parameters
        data_waveformreceived : TYPE
            DESCRIPTION.

        Returns
        None.

        """
        self.adcollector.save_as_binary(self.savedirectory)
        self.channel_number = len(data_waveformreceived)
        if self.channel_number == 1:
            if "Vp" in self.readinchan:
                self.data_collected_0 = data_waveformreceived[0]

                self.PlotDataItem_patch_voltage = PlotDataItem(
                    self.xlabelhere_all, self.data_collected_0
                )
                # use the same color as before, taking advantages of employing
                # same keys in dictionary
                self.PlotDataItem_patch_voltage.setPen("w")
                self.pw_data.addItem(self.PlotDataItem_patch_voltage)

                self.textitem_patch_voltage = pg.TextItem(
                    ("Vp"), color=("w"), anchor=(1, 1)
                )
                self.textitem_patch_voltage.setPos(0, 1)
                self.pw_data.addItem(self.textitem_patch_voltage)
            elif "Ip" in self.readinchan:
                self.data_collected_0 = data_waveformreceived[0]

                self.PlotDataItem_patch_current = PlotDataItem(
                    self.xlabelhere_all, self.data_collected_0
                )
                # use the same color as before, taking advantages of employing
                # same keys in dictionary
                self.PlotDataItem_patch_current.setPen("c")
                self.pw_data.addItem(self.PlotDataItem_patch_current)

                self.textitem_patch_current = pg.TextItem(
                    ("Ip"), color=("w"), anchor=(1, 1)
                )
                self.textitem_patch_current.setPos(0, 1)
                self.pw_data.addItem(self.textitem_patch_current)
            elif (
                "PMT" in self.readinchan
            ):  # repeatnum, PMT_data_index_array, averagenum, ScanArrayXnum
                self.data_collected_0 = data_waveformreceived[0] * -1
                self.data_collected_0 = self.data_collected_0[
                    0 : len(self.data_collected_0) - 1
                ]

                # pmt data could come from raster scanning mode or from contour
                # scanning mode.
                try:
                    for i in range(self.repeatnum):
                        self.PMT_image_reconstructed_array = (
                            self.data_collected_0[
                                np.where(
                                    self.PMT_data_index_array_repeated == i + 1
                                )
                            ]
                        )
                        Dataholder_average = np.mean(
                            self.PMT_image_reconstructed_array.reshape(
                                self.averagenum, -1
                            ),
                            axis=0,
                        )
                        Value_yPixels = int(
                            len(self.samples_1) / self.ScanArrayXnum
                        )
                        self.PMT_image_reconstructed = np.reshape(
                            Dataholder_average,
                            (Value_yPixels, self.ScanArrayXnum),
                        )

                        # Stack the arrays into a 3d array
                        if i == 0:
                            self.PMT_image_reconstructed_stack = (
                                self.PMT_image_reconstructed
                            )
                        else:
                            self.PMT_image_reconstructed_stack = (
                                np.concatenate(
                                    (
                                        self.PMT_image_reconstructed_stack,
                                        self.PMT_image_reconstructed,
                                    ),
                                    axis=0,
                                )
                            )

                        # generate an image object
                        Localimg = Image.fromarray(
                            self.PMT_image_reconstructed
                        )
                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )
                        filename = (
                            f"{timestamp}_PMT_{self.saving_prefix}_{i}.tif"
                        )
                        # save as tif
                        Localimg.save(
                            os.path.join(self.savedirectory, filename)
                        )

                        plt.figure()
                        plt.imshow(
                            self.PMT_image_reconstructed, cmap=plt.cm.gray
                        )
                        plt.show()
                except Exception as exc:
                    logging.critical("caught exception", exc_info=exc)
                    np.save(
                        os.path.join(
                            self.savedirectory,
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            + "_PMT_"
                            + self.saving_prefix
                            + "_"
                            + "flatten",
                        ),
                        self.data_collected_0,
                    )

        elif self.channel_number == 2:
            if "PMT" not in self.readinchan:
                self.data_collected_0 = data_waveformreceived[0]

                self.PlotDataItem_patch_voltage = PlotDataItem(
                    self.xlabelhere_all, self.data_collected_0
                )
                # use the same color as before, taking advantage of employing
                # same keys in dictionary
                self.PlotDataItem_patch_voltage.setPen("w")
                self.pw_data.addItem(self.PlotDataItem_patch_voltage)

                self.textitem_patch_voltage = pg.TextItem(
                    ("Vp"), color=("w"), anchor=(1, 1)
                )
                self.textitem_patch_voltage.setPos(0, 1)
                self.pw_data.addItem(self.textitem_patch_voltage)

                self.data_collected_1 = data_waveformreceived[1]

                self.PlotDataItem_patch_current = PlotDataItem(
                    self.xlabelhere_all, self.data_collected_1
                )
                # use the same color as before, taking advantage of employing
                # same keys in dictionary
                self.PlotDataItem_patch_current.setPen("c")
                self.pw_data.addItem(self.PlotDataItem_patch_current)

                self.textitem_patch_current = pg.TextItem(
                    ("Ip"), color=("w"), anchor=(1, 1)
                )
                self.textitem_patch_current.setPos(0, 1)
                self.pw_data.addItem(self.textitem_patch_current)
            elif "PMT" in self.readinchan:
                self.data_collected_0 = data_waveformreceived[0] * -1
                self.data_collected_0 = self.data_collected_0[
                    0 : len(self.data_collected_0) - 1
                ]

                try:
                    for i in range(self.repeatnum):
                        self.PMT_image_reconstructed_array = (
                            self.data_collected_0[
                                np.where(
                                    self.PMT_data_index_array_repeated == i + 1
                                )
                            ]
                        )
                        Dataholder_average = np.mean(
                            self.PMT_image_reconstructed_array.reshape(
                                self.averagenum, -1
                            ),
                            axis=0,
                        )
                        Value_yPixels = int(
                            len(self.samples_1) / self.ScanArrayXnum
                        )
                        self.PMT_image_reconstructed = np.reshape(
                            Dataholder_average,
                            (Value_yPixels, self.ScanArrayXnum),
                        )

                        # Stack the arrays into a 3d array
                        if i == 0:
                            self.PMT_image_reconstructed_stack = (
                                self.PMT_image_reconstructed
                            )
                        else:
                            self.PMT_image_reconstructed_stack = (
                                np.concatenate(
                                    (
                                        self.PMT_image_reconstructed_stack,
                                        self.PMT_image_reconstructed,
                                    ),
                                    axis=0,
                                )
                            )

                        # generate an image object
                        Localimg = Image.fromarray(
                            self.PMT_image_reconstructed
                        )
                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )
                        filename = (
                            f"{timestamp}_PMT_{self.saving_prefix}_{i}.tif"
                        )
                        # save as tif
                        Localimg.save(
                            os.path.join(self.savedirectory, filename)
                        )

                        plt.figure()
                        plt.imshow(
                            self.PMT_image_reconstructed, cmap=plt.cm.gray
                        )
                        plt.show()
                except Exception as exc:
                    logging.critical("caught exception", exc_info=exc)
                    np.save(
                        os.path.join(
                            self.savedirectory,
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            + "_PMT_"
                            + self.saving_prefix
                            + "_"
                            + "contourscanning",
                        ),
                        self.data_collected_0,
                    )

                if "Vp" in self.readinchan:
                    self.data_collected_1 = data_waveformreceived[1]

                    self.PlotDataItem_patch_voltage = PlotDataItem(
                        self.xlabelhere_all, self.data_collected_1
                    )
                    # use the same color as before, taking advantage of
                    # employing same keys in dictionary
                    self.PlotDataItem_patch_voltage.setPen("w")
                    self.pw_data.addItem(self.PlotDataItem_patch_voltage)

                    self.textitem_patch_voltage = pg.TextItem(
                        ("Vp"), color=("w"), anchor=(1, 1)
                    )
                    self.textitem_patch_voltage.setPos(0, 1)
                    self.pw_data.addItem(self.textitem_patch_voltage)
                elif "Ip" in self.readinchan:
                    self.data_collected_1 = data_waveformreceived[1]

                    self.PlotDataItem_patch_current = PlotDataItem(
                        self.xlabelhere_all, self.data_collected_1
                    )
                    # use the same color as before, taking advantage of
                    # employing same keys in dictionary
                    self.PlotDataItem_patch_current.setPen("c")
                    self.pw_data.addItem(self.PlotDataItem_patch_current)

                    self.textitem_patch_current = pg.TextItem(
                        ("Ip"), color=("w"), anchor=(1, 1)
                    )
                    self.textitem_patch_current.setPos(0, 1)
                    self.pw_data.addItem(self.textitem_patch_current)

    def startProgressBar(self):
        self.DaqProgressBar_thread = DaqProgressBar()
        self.TotalTimeProgressBar = round(
            (self.reference_length) / int(self.SamplingRateTextbox.value()), 6
        )
        self.DaqProgressBar_thread.setlength(self.TotalTimeProgressBar)
        self.DaqProgressBar_thread.change_value.connect(self.setProgressVal)
        self.DaqProgressBar_thread.start()

    def setProgressVal(self, val):
        self.waveform_progressbar.setValue(val)

    def set_waveform_prefix(self):
        self.saving_prefix = str(self.prefixtextbox.text())

    def _open_file_dialog(self):
        self.saving_prefix = str(self.prefixtextbox.text())
        self.savedirectory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.savedirectorytextbox.setText(self.savedirectory)

    def stopMeasurement_daqer(self):
        """Stop"""
        self.adcollector.aboutToQuitHandler()

    def closeEvent(self, event):
        QtWidgets.QApplication.quit()
        event.accept()


class DaqProgressBar(QThread):
    # Create a counter thread
    change_value = pyqtSignal(int)

    def setlength(self, TotalTimeProgressBar):
        self.time_to_sleep_along_one_percent = round(
            TotalTimeProgressBar / 100, 6
        )

    def run(self):
        cnt = 0
        while cnt < 100:
            cnt += 1
            time.sleep(self.time_to_sleep_along_one_percent)
            self.change_value.emit(cnt)


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = WaveformGenerator()
        mainwin.show()
        app.exec_()

    run_app()
