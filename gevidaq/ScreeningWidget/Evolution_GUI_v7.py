# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:40:26 2019

@author: Meng
"""

import copy
import datetime
import logging
import os
import sys
import threading

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QTextCursor

from .. import NIDAQ, Icons, StylishQT
from ..ImageAnalysis import EvolutionAnalysisWidget
from .EvolutionScanningThread import ScanningExecutionThread


class Mainbody(QtWidgets.QWidget):
    # waveforms_generated = pyqtSignal(object, object, list, int)
    # %%
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        with Icons.Path("screening.png") as path:
            self.setWindowIcon(QIcon(path))
        self.setWindowTitle("Gorgonzola")
        self.layout = QtWidgets.QGridLayout(self)

        self.RoundQueueDict = {}
        self.WaveformQueueDict = {}
        self.CamOperationDict = {}
        self.PhotocycleDict = {}
        self.RoundQueueDict["InsightEvents"] = []
        self.RoundQueueDict["FilterEvents"] = []

        self.RoundCoordsDict = {}
        self.WaveformQueueDict_GalvoInfor = {}
        self.GeneralSettingDict = {}
        self.FocusCorrectionMatrixDict = {}
        self.FocusStackInfoDict = {}
        self.popnexttopimgcounter = 0

        self.quick_start_location = [
            r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\2020-8-26 Screening Lenti Archon\2020-08-26_11-28-50_Pipeline.npy"  # TODO hardcoded path
        ]

        self.Tag_round_infor = []
        self.Lib_round_infor = []

        # === GUI for Quick start ===
        self.Quick_startContainer = StylishQT.roundQGroupBox("Quick start")
        self.Quick_startContainerLayout = QtWidgets.QGridLayout()

        self.prefixtextbox = QtWidgets.QLineEdit(self)
        self.prefixtextbox.setPlaceholderText("Folder prefix")
        self.prefixtextbox.setFixedWidth(80)
        self.prefixtextbox.returnPressed.connect(self.set_prefix)
        self.Quick_startContainerLayout.addWidget(self.prefixtextbox, 0, 0)

        self.OpenSettingWidgetButton = QtWidgets.QPushButton(
            "New pipeline", self
        )
        # self.OpenSettingWidgetButton.setCheckable(True)
        self.Quick_startContainerLayout.addWidget(
            self.OpenSettingWidgetButton, 0, 1
        )
        self.OpenSettingWidgetButton.clicked.connect(
            self.showPipelineConfigWidget
        )

        self.QuickStartButton_1 = QtWidgets.QPushButton("Config 1", self)
        self.Quick_startContainerLayout.addWidget(
            self.QuickStartButton_1, 1, 0
        )
        self.QuickStartButton_1.clicked.connect(lambda: self.quick_start(0))

        self.QuickStartButton_2 = QtWidgets.QPushButton("Config 2", self)
        self.Quick_startContainerLayout.addWidget(
            self.QuickStartButton_2, 2, 0
        )

        self.openScreenAnalysisMLWidgetButton = QtWidgets.QPushButton(
            "Screen Analysis ML", self
        )
        self.Quick_startContainerLayout.addWidget(
            self.openScreenAnalysisMLWidgetButton, 3, 0
        )
        self.openScreenAnalysisMLWidgetButton.clicked.connect(
            self.openScreenAnalysisMLWidget
        )

        self.Quick_startContainer.setFixedWidth(380)
        # self.Quick_startContainer.setFixedHeight(300)
        self.Quick_startContainer.setLayout(self.Quick_startContainerLayout)

        # === GUI for GeneralSettings ===
        self.GeneralSettingContainer = StylishQT.roundQGroupBox("Execution")
        self.GeneralSettingContainerLayout = QtWidgets.QGridLayout()

        self.saving_prefix = ""
        self.savedirectorytextbox = QtWidgets.QLineEdit(self)
        self.savedirectorytextbox.setPlaceholderText("Saving directory")
        self.savedirectorytextbox.setFixedWidth(300)
        self.savedirectorytextbox.returnPressed.connect(
            self.update_saving_directory
        )
        self.GeneralSettingContainerLayout.addWidget(
            self.savedirectorytextbox, 0, 1, 1, 2
        )

        self.prefixtextbox = QtWidgets.QLineEdit(self)
        self.prefixtextbox.setPlaceholderText("Folder prefix")
        self.prefixtextbox.setFixedWidth(80)
        self.prefixtextbox.returnPressed.connect(self.set_prefix)
        self.GeneralSettingContainerLayout.addWidget(self.prefixtextbox, 0, 3)

        self.toolButtonOpenDialog = QtWidgets.QPushButton()
        with Icons.Path("Browse.png") as path:
            self.toolButtonOpenDialog.setIcon(QIcon(path))
        self.toolButtonOpenDialog.clicked.connect(self._open_file_dialog)

        self.GeneralSettingContainerLayout.addWidget(
            self.toolButtonOpenDialog, 0, 0
        )

        ButtonConfigurePipeline = StylishQT.generateButton()
        ButtonConfigurePipeline.clicked.connect(self.ConfigGeneralSettings)
        # ButtonConfigurePipeline.clicked.connect(self.GenerateFocusCorrectionMatrix)

        ButtonExePipeline = StylishQT.runButton()
        ButtonExePipeline.clicked.connect(self.ExecutePipeline)

        ButtonSavePipeline = StylishQT.saveButton()
        ButtonSavePipeline.clicked.connect(self.Savepipeline)

        self.ImportPipelineButton = StylishQT.loadButton()
        self.GeneralSettingContainerLayout.addWidget(
            self.ImportPipelineButton, 0, 5
        )
        self.ImportPipelineButton.clicked.connect(self.GetPipelineNPFile)

        self.GeneralSettingContainerLayout.addWidget(
            ButtonConfigurePipeline, 0, 6
        )
        self.GeneralSettingContainerLayout.addWidget(ButtonExePipeline, 0, 8)
        self.GeneralSettingContainerLayout.addWidget(ButtonSavePipeline, 0, 7)

        self.Analyse_roundCheckbox = QtWidgets.QCheckBox("Analyse")
        self.Analyse_roundCheckbox.setStyleSheet(
            'color:navy;font:bold "Times New Roman"'
        )
        self.Analyse_roundCheckbox.setChecked(True)
        self.Analyse_roundCheckbox.setToolTip(
            "Start analysis as configured in screening analysis GUI right after screening."
        )
        self.GeneralSettingContainerLayout.addWidget(
            self.Analyse_roundCheckbox, 0, 4
        )

        self.GeneralSettingContainer.setLayout(
            self.GeneralSettingContainerLayout
        )

        # === GUI for Billboard display ===
        self.ImageDisplayContainer = QtWidgets.QGroupBox()
        self.ImageDisplayContainerLayout = QtWidgets.QGridLayout()

        # AOTFWidgetInstance = NIDAQ.AOTFWidget.AOTFWidgetUI()
        # self.ImageDisplayContainerLayout.addWidget(AOTFWidgetInstance, 0, 0, 1, 1)

        # FilterSliderWidgetInstance = ThorlabsFilterSlider.FilterSliderWidget.FilterSliderWidgetUI()
        # self.ImageDisplayContainerLayout.addWidget(FilterSliderWidgetInstance, 1, 0, 1, 1)

        self.ConsoleTextDisplay = QtWidgets.QTextEdit()
        self.ConsoleTextDisplay.setFontItalic(True)
        self.ConsoleTextDisplay.setPlaceholderText(
            "Notice board from console."
        )
        # self.ConsoleTextDisplay.setMaximumHeight(200)
        # self.ConsoleTextDisplay.setFixedWidth(200)
        self.ImageDisplayContainerLayout.addWidget(
            self.ConsoleTextDisplay, 0, 1, 2, 1
        )

        # self.ImageDisplayContainerLayout.addWidget(ToolWidgetsContainer, 2, 0, 1, 1)

        self.ImageDisplayContainer.setLayout(self.ImageDisplayContainerLayout)
        # self.ImageDisplayContainer.setMinimumHeight(400)
        # self.ImageDisplayContainer.setMinimumWidth(100)

        # === Pipeline configure widget ===
        self.PipelineConfigureWidget = QtWidgets.QWidget()
        self.PipelineConfigureWidget.layout = QtWidgets.QGridLayout()

        # === GUI for PiplineContainer ===

        self.PipelineContainer = StylishQT.roundQGroupBox("Pipeline settings")
        self.PipelineContainerLayout = QtWidgets.QGridLayout()

        self.RoundOrderBox = QtWidgets.QSpinBox(self)
        self.RoundOrderBox.setMinimum(1)
        self.RoundOrderBox.setMaximum(1000)
        self.RoundOrderBox.setValue(1)
        self.RoundOrderBox.setSingleStep(1)
        self.RoundOrderBox.setMaximumWidth(30)
        self.PipelineContainerLayout.addWidget(self.RoundOrderBox, 1, 1)
        self.PipelineContainerLayout.addWidget(
            QtWidgets.QLabel("Round sequence:"), 1, 0
        )

        ButtonAddRound = StylishQT.addButton()
        ButtonDeleteRound = StylishQT.stop_deleteButton()

        self.PipelineContainerLayout.addWidget(ButtonAddRound, 1, 2)
        ButtonAddRound.clicked.connect(self.AddFreshRound)
        ButtonAddRound.clicked.connect(self.GenerateScanCoords)

        self.PipelineContainerLayout.addWidget(ButtonDeleteRound, 1, 3)
        ButtonDeleteRound.clicked.connect(self.DeleteFreshRound)

        ButtonClearRound = StylishQT.cleanButton()
        self.PipelineContainerLayout.addWidget(ButtonClearRound, 1, 4)
        ButtonClearRound.clicked.connect(self.ClearRoundQueue)

        self.ScanRepeatTextbox = QtWidgets.QSpinBox(self)
        self.ScanRepeatTextbox.setMinimum(1)
        self.ScanRepeatTextbox.setValue(1)
        self.ScanRepeatTextbox.setMaximum(100000)
        self.ScanRepeatTextbox.setSingleStep(1)
        self.PipelineContainerLayout.addWidget(self.ScanRepeatTextbox, 0, 1)
        self.PipelineContainerLayout.addWidget(
            QtWidgets.QLabel("Meshgrid:"), 0, 0
        )

        self.OpenTwoPLaserShutterCheckbox = QtWidgets.QCheckBox(
            "Open 2P shutter first"
        )
        self.OpenTwoPLaserShutterCheckbox.setStyleSheet(
            'color:navy;font:bold "Times New Roman"'
        )
        self.OpenTwoPLaserShutterCheckbox.setChecked(True)
        self.PipelineContainerLayout.addWidget(
            self.OpenTwoPLaserShutterCheckbox, 0, 2
        )

        # === GUI for StageScanContainer ===
        ScanContainer = QtWidgets.QWidget()
        ScanSettingLayout = QtWidgets.QGridLayout()  # Layout manager
        ScanContainer.layout = ScanSettingLayout

        self.ScanStepsNumTextbox = QtWidgets.QSpinBox(self)
        self.ScanStepsNumTextbox.setMinimum(1)
        self.ScanStepsNumTextbox.setMaximum(100000)
        self.ScanStepsNumTextbox.setValue(6)
        self.ScanStepsNumTextbox.setSingleStep(2)
        ScanSettingLayout.addWidget(self.ScanStepsNumTextbox, 0, 1)
        ScanSettingLayout.addWidget(
            QtWidgets.QLabel("Stage scanning step number:"), 0, 0
        )

        self.ScanstepTextbox = QtWidgets.QSpinBox(self)
        self.ScanstepTextbox.setMaximum(20000)
        self.ScanstepTextbox.setValue(1568)
        self.ScanstepTextbox.setSingleStep(500)
        ScanSettingLayout.addWidget(self.ScanstepTextbox, 1, 1)
        ScanSettingLayout.addWidget(
            QtWidgets.QLabel("Stage scanning step size:"), 1, 0
        )

        self.AutoFocusGapTextbox = QtWidgets.QSpinBox(self)
        self.AutoFocusGapTextbox.setMinimum(0)
        self.AutoFocusGapTextbox.setMaximum(100000)
        self.AutoFocusGapTextbox.setValue(0)
        self.AutoFocusGapTextbox.setSingleStep(2)
        self.AutoFocusGapTextbox.setToolTip(
            "For example if =2 then there's 1 coordinate between AF. \nIf =0, all AF settings are omitted. \nIn pure-AF mode, put the same as the first AF round."
        )
        # if value = 0, then no auto-focus.
        ScanSettingLayout.addWidget(self.AutoFocusGapTextbox, 0, 5)
        Auto_focus_grid_label = QtWidgets.QLabel("Auto focus grid steps:")
        Auto_focus_grid_label.setToolTip(
            "For example if =2 then there's 1 coordinate between AF. \nIf =0, all AF settings are omitted. \nIn pure-AF mode, put the same as the first AF round."
        )
        ScanSettingLayout.addWidget(Auto_focus_grid_label, 0, 4)

        self.AF_roundCheckbox = QtWidgets.QCheckBox("Auto-focus round")
        self.AF_roundCheckbox.setStyleSheet(
            'color:navy;font:bold "Times New Roman"'
        )
        self.AF_roundCheckbox.setChecked(False)
        self.AF_roundCheckbox.setToolTip(
            "No waveform configuration needed for AF round."
        )
        ScanSettingLayout.addWidget(self.AF_roundCheckbox, 0, 6)

        self.FocusStackNumTextbox = QtWidgets.QSpinBox(self)
        self.FocusStackNumTextbox.setMinimum(1)
        self.FocusStackNumTextbox.setMaximum(20000)
        self.FocusStackNumTextbox.setValue(1)
        self.FocusStackNumTextbox.setSingleStep(1)
        ScanSettingLayout.addWidget(self.FocusStackNumTextbox, 1, 5)
        ScanSettingLayout.addWidget(
            QtWidgets.QLabel("Focus stack number:"), 1, 4
        )

        self.FocusStackStepTextbox = QtWidgets.QDoubleSpinBox(self)
        self.FocusStackStepTextbox.setMinimum(0)
        self.FocusStackStepTextbox.setMaximum(10000)
        self.FocusStackStepTextbox.setDecimals(6)
        self.FocusStackStepTextbox.setValue(0.002)
        self.FocusStackStepTextbox.setSingleStep(0.001)
        ScanSettingLayout.addWidget(self.FocusStackStepTextbox, 1, 7)
        ScanSettingLayout.addWidget(
            QtWidgets.QLabel("Focus stack step size(mm):"), 1, 6
        )

        ScanContainer.setLayout(ScanSettingLayout)

        # === GUI for Laser/filter ===
        TwoPLaserContainer = QtWidgets.QGroupBox()
        TwoPLaserSettingLayout = QtWidgets.QGridLayout()  # Layout manager

        self.TwoPLaserFilterCheckbox = QtWidgets.QCheckBox(
            "Insight/Filter event"
        )
        self.TwoPLaserFilterCheckbox.setStyleSheet(
            'color:blue;font:bold "Times New Roman"'
        )
        TwoPLaserSettingLayout.addWidget(self.TwoPLaserFilterCheckbox, 0, 0)

        TwoPLaserSettingLayout.addWidget(
            QtWidgets.QLabel("2-P wavelength:"), 0, 1
        )

        self.TwoPLaserWavelengthbox = QtWidgets.QSpinBox(self)
        self.TwoPLaserWavelengthbox.setMinimum(680)
        self.TwoPLaserWavelengthbox.setMaximum(1300)
        self.TwoPLaserWavelengthbox.setSingleStep(100)
        self.TwoPLaserWavelengthbox.setValue(900)
        TwoPLaserSettingLayout.addWidget(self.TwoPLaserWavelengthbox, 0, 2)

        self.TwoPLaserShutterCombox = QtWidgets.QComboBox()
        self.TwoPLaserShutterCombox.addItems(
            ["No shutter event", "Open", "Close"]
        )
        TwoPLaserSettingLayout.addWidget(self.TwoPLaserShutterCombox, 0, 3)

        # === filter ===
        self.FilterCheckbox = QtWidgets.QCheckBox("Filter event only")
        self.FilterCheckbox.setStyleSheet(
            'color:blue;font:bold "Times New Roman"'
        )
        TwoPLaserSettingLayout.addWidget(self.FilterCheckbox, 1, 0)

        NDfilterlabel = QtWidgets.QLabel("ND filter:")
        TwoPLaserSettingLayout.addWidget(NDfilterlabel, 1, 1)
        # NDfilterlabel.setAlignment(Qt.AlignRight)
        self.NDfilterCombox = QtWidgets.QComboBox()
        self.NDfilterCombox.addItems(
            ["2", "0.3", "0.5", "1", "1.1", "1.3", "1.5", "2.3", "2.5", "3"]
        )
        TwoPLaserSettingLayout.addWidget(self.NDfilterCombox, 1, 2)

        Emifilterlabel = QtWidgets.QLabel("Emission filter:")
        TwoPLaserSettingLayout.addWidget(Emifilterlabel, 1, 3)
        Emifilterlabel.setAlignment(Qt.AlignRight)
        self.EmisfilterCombox = QtWidgets.QComboBox()
        self.EmisfilterCombox.addItems(["eGFP", "Arch", "Citrine"])
        TwoPLaserSettingLayout.addWidget(self.EmisfilterCombox, 1, 4)

        ButtonDelEvent = QtWidgets.QPushButton("Delete event", self)
        TwoPLaserSettingLayout.addWidget(ButtonDelEvent, 1, 5)
        ButtonDelEvent.clicked.connect(self.DelFilterEvent)
        ButtonDelEvent.clicked.connect(self.DelInsightEvent)

        TwoPLaserContainer.setLayout(TwoPLaserSettingLayout)

        # === GUI for StageScanContainer ===
        CamAFsettingsContainer = QtWidgets.QWidget()
        CamAFsettingsContainerLayout = (
            QtWidgets.QGridLayout()
        )  # Layout manager
        CamAFsettingsContainer.layout = CamAFsettingsContainerLayout

        self.AFmethodCombox = QtWidgets.QComboBox()
        self.AFmethodCombox.addItems(["PMT auto-focus", "Camera auto-focus"])
        ScanSettingLayout.addWidget(self.AFmethodCombox, 0, 7)
        # CamAFsettingsContainerLayout.addWidget(self.AFmethodCombox, 0, 0)
        # self.AFmethodCombox.currentIndexChanged().connect(lambda:DelInsightEvent)
        self.AFmethodCombox.currentIndexChanged.connect(
            lambda: self.AFsettings_container_stack.setCurrentIndex(
                self.AFmethodCombox.currentIndex()
            )
        )

        self.AFsettings_container_stack = QtWidgets.QStackedWidget()

        # === PMT auto focus ===
        self.PMT_autofocus_setting_group = StylishQT.roundQGroupBox(
            "PMT auto focus settings"
        )
        PMT_autofocus_setting_group_layout = QtWidgets.QGridLayout()

        self.PMT_AF_init_step_sizeBox = QtWidgets.QDoubleSpinBox(self)
        self.PMT_AF_init_step_sizeBox.setDecimals(3)
        self.PMT_AF_init_step_sizeBox.setMinimum(0)
        self.PMT_AF_init_step_sizeBox.setMaximum(10)
        self.PMT_AF_init_step_sizeBox.setValue(0.010)
        self.PMT_AF_init_step_sizeBox.setSingleStep(0.001)
        PMT_autofocus_setting_group_layout.addWidget(
            self.PMT_AF_init_step_sizeBox, 0, 1
        )
        PMT_autofocus_setting_group_layout.addWidget(
            QtWidgets.QLabel("Init. searching range(mm):"), 0, 0
        )

        self.PMT_AF_step_numBox = QtWidgets.QSpinBox(self)
        self.PMT_AF_step_numBox.setMinimum(1)
        self.PMT_AF_step_numBox.setMaximum(1000)
        self.PMT_AF_step_numBox.setValue(5)
        self.PMT_AF_step_numBox.setSingleStep(1)
        PMT_autofocus_setting_group_layout.addWidget(
            self.PMT_AF_step_numBox, 0, 3
        )
        PMT_autofocus_setting_group_layout.addWidget(
            QtWidgets.QLabel("Searching increment number:"), 0, 2
        )

        self.PMT_AF_scan_voltBox = QtWidgets.QSpinBox(self)
        self.PMT_AF_scan_voltBox.setMinimum(1)
        self.PMT_AF_scan_voltBox.setMaximum(8)
        self.PMT_AF_scan_voltBox.setValue(5)
        self.PMT_AF_scan_voltBox.setSingleStep(1)
        PMT_autofocus_setting_group_layout.addWidget(
            self.PMT_AF_scan_voltBox, 1, 1
        )
        PMT_autofocus_setting_group_layout.addWidget(
            QtWidgets.QLabel("Scanning voltage:"), 1, 0
        )

        self.PMT_autofocus_setting_group.setLayout(
            PMT_autofocus_setting_group_layout
        )

        self.AFsettings_container_stack.addWidget(
            self.PMT_autofocus_setting_group
        )

        # === Camera auto focus ===
        self.Cam_autofocus_setting_group = StylishQT.roundQGroupBox(
            "Camera auto focus settings"
        )
        Cam_autofocus_setting_group_layout = QtWidgets.QGridLayout()

        self.Cam_AF_init_step_sizeBox = QtWidgets.QDoubleSpinBox(self)
        self.Cam_AF_init_step_sizeBox.setDecimals(3)
        self.Cam_AF_init_step_sizeBox.setMinimum(0)
        self.Cam_AF_init_step_sizeBox.setMaximum(10)
        self.Cam_AF_init_step_sizeBox.setValue(0.025)
        self.Cam_AF_init_step_sizeBox.setSingleStep(0.001)
        Cam_autofocus_setting_group_layout.addWidget(
            self.Cam_AF_init_step_sizeBox, 0, 1
        )
        Cam_autofocus_setting_group_layout.addWidget(
            QtWidgets.QLabel("Init. searching range(mm):"), 0, 0
        )

        self.Cam_AF_step_numBox = QtWidgets.QSpinBox(self)
        self.Cam_AF_step_numBox.setMinimum(1)
        self.Cam_AF_step_numBox.setMaximum(1000)
        self.Cam_AF_step_numBox.setValue(10)
        self.Cam_AF_step_numBox.setSingleStep(1)
        Cam_autofocus_setting_group_layout.addWidget(
            self.Cam_AF_step_numBox, 0, 3
        )
        Cam_autofocus_setting_group_layout.addWidget(
            QtWidgets.QLabel("Searching increment number:"), 0, 2
        )

        self.Cam_AF_ExposureBox = QtWidgets.QDoubleSpinBox(self)
        self.Cam_AF_ExposureBox.setDecimals(5)
        self.Cam_AF_ExposureBox.setMinimum(0)
        self.Cam_AF_ExposureBox.setMaximum(100)
        self.Cam_AF_ExposureBox.setValue(0.003)
        self.Cam_AF_ExposureBox.setSingleStep(0.001)
        Cam_autofocus_setting_group_layout.addWidget(
            self.Cam_AF_ExposureBox, 1, 1
        )
        Cam_autofocus_setting_group_layout.addWidget(
            QtWidgets.QLabel("Exposure time(s):"), 1, 0
        )

        self.Cam_AF_AOTF_valueBox = QtWidgets.QDoubleSpinBox(self)
        self.Cam_AF_AOTF_valueBox.setDecimals(3)
        self.Cam_AF_AOTF_valueBox.setMinimum(0)
        self.Cam_AF_AOTF_valueBox.setMaximum(5)
        self.Cam_AF_AOTF_valueBox.setValue(3)
        self.Cam_AF_AOTF_valueBox.setSingleStep(0.5)
        Cam_autofocus_setting_group_layout.addWidget(
            self.Cam_AF_AOTF_valueBox, 1, 3
        )

        self.Cam_AF_AOTF_settingBox = QtWidgets.QComboBox()
        self.Cam_AF_AOTF_settingBox.addItems(["488AO", "532AO", "640AO"])
        Cam_autofocus_setting_group_layout.addWidget(
            self.Cam_AF_AOTF_settingBox, 1, 2
        )

        self.Cam_autofocus_setting_group.setLayout(
            Cam_autofocus_setting_group_layout
        )

        self.AFsettings_container_stack.addWidget(
            self.Cam_autofocus_setting_group
        )

        CamAFsettingsContainerLayout.addWidget(
            self.AFsettings_container_stack, 0, 1
        )

        CamAFsettingsContainer.setLayout(CamAFsettingsContainerLayout)
        self.RoundGeneralSettingTabs = QtWidgets.QTabWidget()
        self.RoundGeneralSettingTabs.addTab(ScanContainer, "Scanning settings")
        self.RoundGeneralSettingTabs.addTab(
            TwoPLaserContainer, "Pulse laser/Filter settings"
        )
        self.RoundGeneralSettingTabs.addTab(
            CamAFsettingsContainer, "Auto-focus settings"
        )

        self.PipelineContainerLayout.addWidget(
            self.RoundGeneralSettingTabs, 3, 0, 1, 10
        )

        self.WaveformOrderBox = QtWidgets.QSpinBox(self)
        self.WaveformOrderBox.setMinimum(1)
        self.WaveformOrderBox.setMaximum(1000)
        self.WaveformOrderBox.setValue(1)
        self.WaveformOrderBox.setSingleStep(1)
        self.WaveformOrderBox.setMaximumWidth(30)
        self.PipelineContainerLayout.addWidget(self.WaveformOrderBox, 4, 1)
        self.PipelineContainerLayout.addWidget(
            QtWidgets.QLabel("Waveform/Camera sequence:"), 4, 0
        )

        ButtonAddWaveform = StylishQT.addButton()
        ButtonDeleteWaveform = StylishQT.stop_deleteButton()

        ButtonClearWaveform = StylishQT.cleanButton()

        self.PipelineContainerLayout.addWidget(ButtonAddWaveform, 4, 2)
        self.PipelineContainerLayout.addWidget(ButtonDeleteWaveform, 4, 3)
        self.PipelineContainerLayout.addWidget(ButtonClearWaveform, 4, 4)

        ButtonAddWaveform.clicked.connect(self.AddFreshWaveform)
        ButtonAddWaveform.clicked.connect(self.AddCameraOperation)
        ButtonAddWaveform.clicked.connect(self.AddPhotocycleOperation)

        ButtonDeleteWaveform.clicked.connect(self.DeleteFreshWaveform)
        ButtonDeleteWaveform.clicked.connect(self.DeleteCameraOperation)
        ButtonDeleteWaveform.clicked.connect(self.DeletePhotocycleOperation)

        ButtonClearWaveform.clicked.connect(self.ClearWaveformQueue)
        ButtonClearWaveform.clicked.connect(self.CleanCameraOperation)
        ButtonClearWaveform.clicked.connect(self.CleanPhotocycleOperation)
        self.EachCoordDwellSettingTabs = QtWidgets.QTabWidget()

        # Waveforms tab settings
        waveformTab = QtWidgets.QWidget()
        waveformTabLayout = QtWidgets.QGridLayout()

        self.Waveformer_widget_instance = (
            NIDAQ.WaveformWidget.WaveformGenerator()
        )
        self.Waveformer_widget_instance.checkbox_saveWaveforms.setEnabled(
            False
        )
        self.Waveformer_widget_instance.WaveformPackage.connect(
            self.UpdateWaveformerSignal
        )
        self.Waveformer_widget_instance.GalvoScanInfor.connect(
            self.UpdateWaveformerGalvoInfor
        )

        waveformTabLayout.addWidget(
            self.Waveformer_widget_instance, 2, 0, 2, 9
        )
        waveformTab.setLayout(waveformTabLayout)

        # Camera tab settings
        CameraDwellTab = QtWidgets.QWidget()
        CameraDwellTabLayout = QtWidgets.QGridLayout()

        self.photocycleChecbox = QtWidgets.QCheckBox("Photo cycle")
        self.photocycleChecbox.setStyleSheet(
            'color:Indigo;font:bold "Times New Roman"'
        )
        CameraDwellTabLayout.addWidget(self.photocycleChecbox, 0, 0)

        self.CamTriggerSettingBox = QtWidgets.QComboBox()
        self.CamTriggerSettingBox.addItems(["EXTERNAL", "INTERNAL"])

        self.CamTriggerActive_SettingBox = QtWidgets.QComboBox()
        self.CamTriggerActive_SettingBox.addItems(
            ["EDGE", "LEVEL", "SYNCREADOUT"]
        )

        CameraDwellTabLayout.addWidget(QtWidgets.QLabel("Trigger:"), 2, 0)
        CameraDwellTabLayout.addWidget(self.CamTriggerSettingBox, 2, 1)
        CameraDwellTabLayout.addWidget(self.CamTriggerActive_SettingBox, 2, 2)

        self.StreamBufferTotalFrames_spinbox = QtWidgets.QSpinBox()
        self.StreamBufferTotalFrames_spinbox.setMaximum(120000)
        self.StreamBufferTotalFrames_spinbox.setValue(0)
        CameraDwellTabLayout.addWidget(
            self.StreamBufferTotalFrames_spinbox, 2, 4
        )
        CameraDwellTabLayout.addWidget(QtWidgets.QLabel("Buffers:"), 2, 3)

        self.CamExposureBox = QtWidgets.QDoubleSpinBox(self)
        self.CamExposureBox.setDecimals(6)
        self.CamExposureBox.setMinimum(0)
        self.CamExposureBox.setMaximum(100)
        self.CamExposureBox.setValue(0.001501)
        self.CamExposureBox.setSingleStep(0.001)
        CameraDwellTabLayout.addWidget(self.CamExposureBox, 2, 6)
        CameraDwellTabLayout.addWidget(
            QtWidgets.QLabel("Exposure time(s):"), 2, 5
        )

        # === Camera ROI settings ===
        CameraROIPosContainer = QtWidgets.QGroupBox("ROI position")
        CameraROIPosContainer.setStyleSheet(
            "QGroupBox { background-color:#F5F5F5;}"
        )
        CameraROIPosLayout = QtWidgets.QGridLayout()

        OffsetLabel = QtWidgets.QLabel("Offset")
        OffsetLabel.setFixedHeight(30)
        ROISizeLabel = QtWidgets.QLabel("Size")
        ROISizeLabel.setFixedHeight(30)

        CameraROIPosLayout.addWidget(OffsetLabel, 2, 1)
        CameraROIPosLayout.addWidget(ROISizeLabel, 2, 2)

        self.ROI_hpos_spinbox = QtWidgets.QSpinBox()
        self.ROI_hpos_spinbox.setMaximum(2048)
        self.ROI_hpos_spinbox.setValue(0)

        CameraROIPosLayout.addWidget(self.ROI_hpos_spinbox, 3, 1)

        self.ROI_vpos_spinbox = QtWidgets.QSpinBox()
        self.ROI_vpos_spinbox.setMaximum(2048)
        self.ROI_vpos_spinbox.setValue(0)

        CameraROIPosLayout.addWidget(self.ROI_vpos_spinbox, 4, 1)

        self.ROI_hsize_spinbox = QtWidgets.QSpinBox()
        self.ROI_hsize_spinbox.setMaximum(2048)
        self.ROI_hsize_spinbox.setValue(2048)

        CameraROIPosLayout.addWidget(self.ROI_hsize_spinbox, 3, 2)

        self.ROI_vsize_spinbox = QtWidgets.QSpinBox()
        self.ROI_vsize_spinbox.setMaximum(2048)
        self.ROI_vsize_spinbox.setValue(2048)

        CameraROIPosLayout.addWidget(self.ROI_vsize_spinbox, 4, 2)

        CameraROIPosLayout.addWidget(QtWidgets.QLabel("Horizontal"), 3, 0)
        CameraROIPosLayout.addWidget(QtWidgets.QLabel("Vertical"), 4, 0)

        CameraROIPosContainer.setLayout(CameraROIPosLayout)
        CameraROIPosContainer.setFixedHeight(105)

        CameraDwellTabLayout.addWidget(CameraROIPosContainer, 3, 1, 3, 3)

        CameraDwellTab.setLayout(CameraDwellTabLayout)

        self.EachCoordDwellSettingTabs.addTab(
            waveformTab, "Waveforms settings"
        )
        self.EachCoordDwellSettingTabs.addTab(
            CameraDwellTab, "Camera operations"
        )

        self.PipelineContainerLayout.addWidget(
            self.EachCoordDwellSettingTabs, 5, 0, 4, 10
        )

        self.PipelineContainer.setLayout(self.PipelineContainerLayout)

        self.PipelineConfigureWidget.layout.addWidget(
            self.GeneralSettingContainer, 0, 0
        )
        self.PipelineConfigureWidget.layout.addWidget(
            self.PipelineContainer, 1, 0
        )
        self.PipelineConfigureWidget.setLayout(
            self.PipelineConfigureWidget.layout
        )

        # === GUI for Stack widget ===
        startupWidget = QtWidgets.QWidget()

        self.settingStackedWidget = QtWidgets.QStackedWidget()
        self.settingStackedWidget.addWidget(startupWidget)
        self.settingStackedWidget.addWidget(self.PipelineConfigureWidget)
        self.settingStackedWidget.setCurrentIndex(0)
        # self.setFixedWidth(400)
        # self.setFixedHeight(300)

        self.layout.addWidget(self.Quick_startContainer, 1, 0, 1, 2)

        self.setLayout(self.layout)

        self.showPipelineConfigWidget()

        self.ScreenAnalysisMLWindow = EvolutionAnalysisWidget.MainGUI()
        self.ScreenAnalysisMLWindow.show()

    def showPipelineConfigWidget(self):
        self.layout.addWidget(self.ImageDisplayContainer, 1, 2, 1, 2)
        self.layout.addWidget(self.settingStackedWidget, 2, 0, 1, 4)

        self.settingStackedWidget.setCurrentIndex(1)

    # %%
    """
    # FUNCTIONS FOR EXECUTION

    === Screening routine configuration Structure ===

    ====RoundQueueDict====                              Dictionary=============

      -- key: RoundPackage_{}                           List of operations at each coordinate. {} stands for round sequence number.
            |__ WaveformQueueDict                       Dictionary
                key: WaveformPackage_{}                 Waveforms tuple signal from Waveformer. At each coordinate.

            |__ CamOperationDict                        Dictionary
                key: CameraPackage_{}                   Camera operations at each coordinate. {} stands for waveform/camera sequence number.

            |__ PhotocycleDict                          Dictionary
                key: PhotocyclePackage_{}               Photocycle experiment information. {} stands for waveform/camera sequence number.

      -- key: GalvoInforPackage_{}
            |__ WaveformQueueDict_GalvoInfor            Dictionary
                key: GalvoInfor_{}                      Galvo scanning configuration signal from Waveformer. At each coordinate.

      -- key: FilterEvents
            |__ List of filter operation strings, in the round.

      -- key: InsightEvents
            |__ List of insight laser operation strings, in the round.

    ====RoundCoordsDict====                             Dictionary=============

      -- key: CoordsPackage_{}
            |__ np.array of scanning coordinates.

    ====GeneralSettingDict====                          Dictionary=============

      -- key: 'savedirectory'                           screening data saving directory.

      -- key: 'FocusCorrectionMatrixDict'               Dictionary
                  |__ key: RoundPackage_{}
                      or RoundPackage_{}_Grid_{}        np.array of pre-calibrated focus positions.

      -- key: 'FocusStackInfoDict'                      Dictionary
                  |__ key: RoundPackage_{}              String specifies 'NumberOfFocus{}WithIncrementBeing{}'.

      -- key: 'Meshgrid'                                int if scanning grid number. meshrepeat

      -- key: 'Scanning step'                           Scanning stage step. self.step

      -- key: 'StartUpEvents'                           List of strings, like Shutter_Open
    """
    # === Waveform package functions at each coordinate ===
    """
    Every time when the 'configure' button in waveformer widget is hit, the 'WaveformPackage' and 'GalvoInfor' signals are sent here.
    """

    def UpdateWaveformerSignal(self, WaveformPackage):
        """
        Capture the newest generated waveform tuple signal from Waveformer, which contains 4 parts in tuple:
        (sampling rate, analogcontainer_array, digitalcontainer_array, recording channel list)
        """
        self.FreshWaveformPackage = WaveformPackage

    def UpdateWaveformerGalvoInfor(self, GalvoInfor):
        self.FreshWaveformGalvoInfor = GalvoInfor

    def AddFreshWaveform(self):  # Add waveform package for single round.
        CurrentWaveformPackageSequence = self.WaveformOrderBox.value()
        try:
            self.WaveformQueueDict[
                "WaveformPackage_{}".format(CurrentWaveformPackageSequence)
            ] = self.FreshWaveformPackage
        except AttributeError:
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                "Click configure waveform first!",
                QtWidgets.QMessageBox.Ok,
            )

        self.WaveformQueueDict_GalvoInfor[
            "GalvoInfor_{}".format(CurrentWaveformPackageSequence)
        ] = self.FreshWaveformGalvoInfor
        self.normalOutputWritten(
            "Waveform{} added.\n".format(CurrentWaveformPackageSequence)
        )
        logging.info("Waveform added.")

    def DeleteFreshWaveform(
        self,
    ):  # Empty the waveform container to avoid crosstalk between rounds.
        CurrentWaveformPackageSequence = self.WaveformOrderBox.value()
        del self.WaveformQueueDict[
            "WaveformPackage_{}".format(CurrentWaveformPackageSequence)
        ]

        del self.WaveformQueueDict_GalvoInfor[
            "GalvoInfor_{}".format(CurrentWaveformPackageSequence)
        ]

    def ClearWaveformQueue(self):
        self.WaveformQueueDict = {}
        self.WaveformQueueDict_GalvoInfor = {}

    # === Camera operation at each coordinate ===
    def AddCameraOperation(self):
        CurrentCamPackageSequence = self.WaveformOrderBox.value()

        if self.StreamBufferTotalFrames_spinbox.value() != 0:
            CameraOperation = {
                "Settings": [
                    "trigger_source",
                    self.CamTriggerSettingBox.currentText(),
                    "exposure_time",
                    self.CamExposureBox.value(),
                    "trigger_active",
                    self.CamTriggerActive_SettingBox.currentText(),
                    "subarray_hsize",
                    self.ROI_hsize_spinbox.value(),
                    "subarray_vsize",
                    self.ROI_vsize_spinbox.value(),
                    "subarray_hpos",
                    self.ROI_hpos_spinbox.value(),
                    "subarray_vpos",
                    self.ROI_vpos_spinbox.value(),
                ],
                "Buffer_number": self.StreamBufferTotalFrames_spinbox.value(),
            }

            self.CamOperationDict[
                "CameraPackage_{}".format(CurrentCamPackageSequence)
            ] = CameraOperation
        else:
            self.CamOperationDict[
                "CameraPackage_{}".format(CurrentCamPackageSequence)
            ] = {}

    def DeleteCameraOperation(
        self,
    ):  # Empty the waveform container to avoid crosstalk between rounds.
        CurrentCamPackageSequence = self.WaveformOrderBox.value()
        del self.CamOperationDict[
            "CameraPackage_{}".format(CurrentCamPackageSequence)
        ]

    def CleanCameraOperation(self):
        self.CamOperationDict = {}

    # === Photocycle operation at each coordinate ===
    def AddPhotocycleOperation(self):
        CurrentPhotocycleSequence = self.WaveformOrderBox.value()

        if self.photocycleChecbox.isChecked():
            PhotocycleOperation = [True]
            self.PhotocycleDict[
                "PhotocyclePackage_{}".format(CurrentPhotocycleSequence)
            ] = PhotocycleOperation
        else:
            self.PhotocycleDict[
                "PhotocyclePackage_{}".format(CurrentPhotocycleSequence)
            ] = {}

    def DeletePhotocycleOperation(self):
        CurrentPhotocycleSequence = self.WaveformOrderBox.value()
        del self.PhotocycleDict[
            "PhotocyclePackage_{}".format(CurrentPhotocycleSequence)
        ]

    def CleanPhotocycleOperation(self):
        self.PhotocycleDict = {}

    # === Settings at each round ===
    def AddFreshRound(self):
        CurrentRoundSequence = self.RoundOrderBox.value()

        WaveformQueueDict = copy.deepcopy(
            self.WaveformQueueDict
        )  # Here we make the self.WaveformQueueDict private so that other rounds won't refer to the same variable.
        WaveformQueueDict_GalvoInfor = copy.deepcopy(
            self.WaveformQueueDict_GalvoInfor
        )
        CamOperationDict = copy.deepcopy(self.CamOperationDict)
        PhotocycleDict = copy.deepcopy(self.PhotocycleDict)

        self.RoundQueueDict["RoundPackage_{}".format(CurrentRoundSequence)] = [
            WaveformQueueDict,
            CamOperationDict,
            PhotocycleDict,
        ]
        self.RoundQueueDict[
            "GalvoInforPackage_{}".format(CurrentRoundSequence)
        ] = WaveformQueueDict_GalvoInfor  # Information we need to restore pmt scanning images.

        # Configure information for Z-stack
        ZstackNumber = self.FocusStackNumTextbox.value()
        ZstackStep = self.FocusStackStepTextbox.value()

        self.FocusStackInfoDict[
            "RoundPackage_{}".format(CurrentRoundSequence)
        ] = "NumberOfFocus{}WithIncrementBeing{}".format(
            ZstackNumber, ZstackStep
        )

        self.AddFilterEvent()

        self.AddInsightEvent()

        self.normalOutputWritten(
            "Round{} added.\n".format(CurrentRoundSequence)
        )
        logging.info("Round added.")

    # === Configure filter event ===
    def AddFilterEvent(self):
        CurrentRoundSequence = self.RoundOrderBox.value()

        if (
            self.FilterCheckbox.isChecked()
            or self.TwoPLaserFilterCheckbox.isChecked()
        ):
            self.RoundQueueDict["FilterEvents"].append(
                "Round_{}_ND_ToPos_{}".format(
                    CurrentRoundSequence, self.NDfilterCombox.currentText()
                )
            )
            self.RoundQueueDict["FilterEvents"].append(
                "Round_{}_EM_ToPos_{}".format(
                    CurrentRoundSequence, self.EmisfilterCombox.currentText()
                )
            )
            logging.info(self.RoundQueueDict["FilterEvents"])
            self.normalOutputWritten(
                "FilterEvents"
                + str(self.RoundQueueDict["FilterEvents"])
                + "\n"
            )

    def DelFilterEvent(self):
        CurrentRoundSequence = self.RoundOrderBox.value()

        if (
            "Round_{}_ND_ToPos_{}".format(
                CurrentRoundSequence, self.NDfilterCombox.currentText()
            )
            in self.RoundQueueDict["FilterEvents"]
        ):
            self.RoundQueueDict["FilterEvents"].remove(
                "Round_{}_ND_ToPos_{}".format(
                    CurrentRoundSequence, self.NDfilterCombox.currentText()
                )
            )
            self.RoundQueueDict["FilterEvents"].remove(
                "Round_{}_EM_ToPos_{}".format(
                    CurrentRoundSequence, self.EmisfilterCombox.currentText()
                )
            )
        logging.info(self.RoundQueueDict["FilterEvents"])
        self.normalOutputWritten(
            str(self.RoundQueueDict["FilterEvents"]) + "\n"
        )

    # === Configure insight event ===
    def AddInsightEvent(self):
        CurrentRoundSequence = self.RoundOrderBox.value()

        if self.TwoPLaserFilterCheckbox.isChecked():
            self.RoundQueueDict["InsightEvents"].append(
                "Round_{}_WavelengthTo_{}".format(
                    CurrentRoundSequence, self.TwoPLaserWavelengthbox.value()
                )
            )

            if self.TwoPLaserShutterCombox.currentText() != "No shutter event":
                self.RoundQueueDict["InsightEvents"].append(
                    "Round_{}_Shutter_{}".format(
                        CurrentRoundSequence,
                        self.TwoPLaserShutterCombox.currentText(),
                    )
                )

        logging.info(self.RoundQueueDict["InsightEvents"])
        self.normalOutputWritten(
            "InsightEvents" + str(self.RoundQueueDict["InsightEvents"]) + "\n"
        )

    def DelInsightEvent(self):
        CurrentRoundSequence = self.RoundOrderBox.value()

        if self.TwoPLaserFilterCheckbox.isChecked():
            self.RoundQueueDict["InsightEvents"].remove(
                "Round_{}_WavelengthTo_{}".format(
                    CurrentRoundSequence, self.TwoPLaserWavelengthbox.value()
                )
            )

            if self.TwoPLaserShutterCombox.currentText() != "No shutter event":
                self.RoundQueueDict["InsightEvents"].remove(
                    "Round_{}_Shutter_{}".format(
                        CurrentRoundSequence,
                        self.TwoPLaserShutterCombox.currentText(),
                    )
                )

        logging.info(self.RoundQueueDict["InsightEvents"])
        self.normalOutputWritten(
            str(self.RoundQueueDict["InsightEvents"]) + "\n"
        )

    # === Generate Scan Coords ===
    def GenerateScanCoords(self):
        """
        Generate Scan coordinates structured array with auto-focus fields.

        Returns
        None.

        """
        CurrentRoundSequence = self.RoundOrderBox.value()
        # settings for scanning index
        step = self.ScanstepTextbox.value()

        row_start = 0
        row_end = (self.ScanStepsNumTextbox.value() - 1) * step

        column_start = 0
        column_end = (self.ScanStepsNumTextbox.value() - 1) * step

        # Generate structured array containing scanning coordinates' information.
        AutoFocusGrid_steps = self.AutoFocusGapTextbox.value()
        AutoFocusCoordGap = AutoFocusGrid_steps * step

        # Number of coordinates per row
        Coords_number_per_row = (row_end - row_start) / step + 1

        # Generate the
        if AutoFocusGrid_steps != 0:
            AutoFocusGridNum = int(Coords_number_per_row / AutoFocusGrid_steps)
            AutoFocusGridOffsetList = []
            # Auto focus grid coordinates offset list
            for row_offset in range(AutoFocusGridNum):
                for col_offset in range(AutoFocusGridNum):
                    AutoFocusGridOffsetList.append(
                        [
                            row_offset * step * AutoFocusGrid_steps,
                            col_offset * step * AutoFocusGrid_steps,
                        ]
                    )
        else:
            AutoFocusGridOffsetList = [[0, 0]]

        # Data type of structured array.
        Coords_array_dtype = np.dtype(
            [
                ("row", "i4"),
                ("col", "i4"),
                ("auto_focus_flag", "U10"),
                ("focus_position", "f4"),
            ]
        )

        Coords_array = np.array([], dtype=Coords_array_dtype)

        if not self.AF_roundCheckbox.isChecked():
            # If not auto-focus round
            for AutoFocusGridOffset in AutoFocusGridOffsetList:
                # In each small auto-focus grid
                AutoFocusOffset_row = AutoFocusGridOffset[0]
                AutoFocusOffset_col = AutoFocusGridOffset[1]

                if AutoFocusCoordGap != 0:
                    # If auto-focus involved:
                    for row_pos in range(row_start, AutoFocusCoordGap, step):
                        for col_pos in range(
                            column_start, AutoFocusCoordGap, step
                        ):
                            # At each left-top corner of the coordinates grid, place the
                            # flag for auto focus
                            if (
                                col_pos == 0
                                and row_pos == 0
                                and AutoFocusGrid_steps != 0
                            ):
                                current_coord_array = np.array(
                                    [
                                        (
                                            row_pos + AutoFocusOffset_row,
                                            col_pos + AutoFocusOffset_col,
                                            "yes",
                                            -1,
                                        )
                                    ],
                                    dtype=Coords_array_dtype,
                                )
                            else:
                                current_coord_array = np.array(
                                    [
                                        (
                                            row_pos + AutoFocusOffset_row,
                                            col_pos + AutoFocusOffset_col,
                                            "no",
                                            -1,
                                        )
                                    ],
                                    dtype=Coords_array_dtype,
                                )

                            Coords_array = np.append(
                                Coords_array, current_coord_array
                            )

                else:
                    # If no auto-focus, put 'no' in auto_focus_flag field.
                    for row_pos in range(row_start, row_end + step, step):
                        for col_pos in range(
                            column_start, column_end + step, step
                        ):
                            current_coord_array = np.array(
                                [(row_pos, col_pos, "no", -1)],
                                dtype=Coords_array_dtype,
                            )

                            Coords_array = np.append(
                                Coords_array, current_coord_array
                            )

        else:
            # In case of pure auto-focus round, generate coordinates at only
            # calibration positions, and put flag 'pure AF'.
            for row_pos in range(row_start, row_end, AutoFocusCoordGap):
                for col_pos in range(
                    column_start, column_end, AutoFocusCoordGap
                ):
                    current_coord_array = np.array(
                        [(row_pos, col_pos, "pure AF", -1)],
                        dtype=Coords_array_dtype,
                    )

                    Coords_array = np.append(Coords_array, current_coord_array)
        logging.info(Coords_array)
        self.RoundCoordsDict[
            "CoordsPackage_{}".format(CurrentRoundSequence)
        ] = Coords_array

    def DeleteFreshRound(self):
        CurrentRoundSequence = self.RoundOrderBox.value()
        del self.RoundQueueDict["RoundPackage_{}".format(CurrentRoundSequence)]
        del self.RoundCoordsDict[
            "CoordsPackage_{}".format(CurrentRoundSequence)
        ]
        del self.RoundQueueDict[
            "GalvoInforPackage_{}".format(CurrentRoundSequence)
        ]
        logging.info(self.RoundQueueDict.keys())

    def ClearRoundQueue(self):
        self.WaveformQueueDict = {}
        self.CamOperationDict = {}
        self.PhotocycleDict = {}
        self.RoundQueueDict = {}
        self.RoundQueueDict["InsightEvents"] = []
        self.RoundQueueDict["FilterEvents"] = []
        self.RoundCoordsDict = {}
        self.WaveformQueueDict_GalvoInfor = {}
        self.GeneralSettingDict = {}
        self.FocusStackInfoDict = {}

        self.normalOutputWritten("Rounds cleared.\n")
        logging.info("Rounds cleared.")

    # %%
    """
    # Configure general settings, get ready for execution
    """

    def ConfigGeneralSettings(self):
        savedirectory = self.savedirectory
        meshrepeat = self.ScanRepeatTextbox.value()
        StageGridOffset = (
            self.ScanStepsNumTextbox.value() * self.ScanstepTextbox.value()
        )

        StartUpEvents = []
        if self.OpenTwoPLaserShutterCheckbox.isChecked():
            StartUpEvents.append("Shutter_Open")

        if self.AFmethodCombox.currentIndex() == 0:
            # If use PMT auto-focus
            AutoFocusConfig = {
                "source_of_image": "PMT",
                "init_search_range": self.PMT_AF_init_step_sizeBox.value(),
                "total_step_number": self.PMT_AF_step_numBox.value(),
                "imaging_conditions": {
                    "edge_volt": self.PMT_AF_scan_voltBox.value()
                },
            }
        elif self.AFmethodCombox.currentIndex() == 1:
            # If use camera auto-focus
            AutoFocusConfig = {
                "source_of_image": "Camera",
                "init_search_range": self.Cam_AF_init_step_sizeBox.value(),
                "total_step_number": self.Cam_AF_step_numBox.value(),
                "imaging_conditions": {
                    self.Cam_AF_AOTF_settingBox.currentText(): self.Cam_AF_AOTF_valueBox.value(),
                    "exposure_time": self.Cam_AF_ExposureBox.value(),
                },
            }

        self.normalOutputWritten("Auto-focus :{}".format(AutoFocusConfig))

        # Interpolate in between the focus correction positions
        FocusCorrectionMatrixDict = {}  # self.upsize_focus_matrix()

        generalnamelist = [
            "savedirectory",
            "FocusCorrectionMatrixDict",
            "FocusStackInfoDict",
            "StageGridOffset",
            "Meshgrid",
            "StartUpEvents",
            "AutoFocusConfig",
        ]

        generallist = [
            savedirectory,
            FocusCorrectionMatrixDict,
            self.FocusStackInfoDict,
            StageGridOffset,
            meshrepeat,
            StartUpEvents,
            AutoFocusConfig,
        ]

        for item in range(len(generallist)):
            self.GeneralSettingDict[generalnamelist[item]] = generallist[item]
        # print(self.GeneralSettingDict['FocusStackInfoDict'])
        self.normalOutputWritten("Rounds configured.\n")

        self.show_pipline_infor()

    def auto_saving_directory(self):
        self.savedirectory = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\Evolution screening\{}_{}".format(  # TODO hardcoded path
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            str(self.prefixtextbox.text()),
        )

        self.GeneralSettingDict["savedirectory"] = self.savedirectory

        os.mkdir(self.savedirectory)  # Create the folder

    def _open_file_dialog(self):
        self.savedirectory = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                directory="M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data"  # TODO hardcoded path
            )
        )
        self.savedirectorytextbox.setText(self.savedirectory)
        try:
            self.GeneralSettingDict["savedirectory"] = self.savedirectory
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
        self.set_prefix()

    def update_saving_directory(self):
        self.savedirectory = str(self.savedirectorytextbox.text())

    def set_prefix(self):
        self.saving_prefix = str(self.prefixtextbox.text())

    # === GenerateFocusCorrectionMatrix ===
    def CaptureFocusCorrectionMatrix(self, CorrectionFomula):
        self.CorrectionFomula = CorrectionFomula

    def CaptureFocusDuplicateMethodMatrix(
        self, CorrectionDictForDuplicateMethod
    ):
        self.FocusDuplicateMethodInfor = CorrectionDictForDuplicateMethod

    def ExecutePipeline(self):
        self.Savepipeline()

        try:
            get_ipython = sys.modules["IPython"].get_ipython
        except KeyError:
            pass
        else:
            get_ipython().run_line_magic(
                "matplotlib", "inline"
            )  # before start, set spyder back to inline

        self.ExecuteThreadInstance = ScanningExecutionThread(
            self.RoundQueueDict, self.RoundCoordsDict, self.GeneralSettingDict
        )
        self.ExecuteThreadInstance.start()

        self.ExecuteThreadInstance.finished.connect(
            lambda: self.run_in_thread(self.start_analysis)
        )

    def Savepipeline(self):
        SavepipelineInstance = []
        SavepipelineInstance.extend(
            [
                self.RoundQueueDict,
                self.RoundCoordsDict,
                self.GeneralSettingDict,
            ]
        )

        np.save(
            os.path.join(
                self.savedirectory,
                self.saving_prefix
                + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + "_Pipeline",
            ),
            SavepipelineInstance,
        )

    # === Auto analysis ===
    def SetAnalysisRound(self):
        """
        Sepcify the round numbers and store the information in list.

        Returns
        None.

        """
        if self.FilepathSwitchBox.currentText() == "Tag":
            self.Tag_round_infor.append(self.AnalysisRoundBox.value())
        elif self.FilepathSwitchBox.currentText() == "Lib":
            self.Lib_round_infor.append(self.AnalysisRoundBox.value())

        self.normalOutputWritten(
            "Tag_round_infor: {}\nLib_round_infor: {}\n".format(
                str(self.Tag_round_infor), str(self.Lib_round_infor)
            )
        )

    def start_analysis(self):
        """
        Start the screening analysis by calling the EvolutionAnalysisWidget.

        Returns
        None.

        """
        if self.Analyse_roundCheckbox.isChecked():
            # By default all data is stored in the same folder.
            self.ScreenAnalysisMLWindow.Tag_folder = self.savedirectory
            self.ScreenAnalysisMLWindow.Lib_folder = self.savedirectory

            # self.ScreenAnalysisMLWindow.Tag_round_infor = self.Tag_round_infor
            # self.ScreenAnalysisMLWindow.Lib_round_infor = self.Lib_round_infor

            self.ScreenAnalysisMLWindow.ScreeningAnalysis()

    def run_in_thread(self, fn, *args, **kwargs):
        """
        Send target function to thread.
        Usage: lambda: self.run_in_thread(self.fn)

        Parameters
        fn : function
            Target function to put in thread.

        Returns
        thread : TYPE
            Threading handle.

        """
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()

        return thread

    # %%
    """
    # For save and load file.
    """

    def GetPipelineNPFile(self):
        self.pipelinenpfileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            "M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data",  # TODO hardcoded path
            "(*.npy)",
        )

        self.LoadPipelineFile()

    def LoadPipelineFile(self):
        temp_loaded_container = np.load(
            self.pipelinenpfileName, allow_pickle=True
        )
        self.RoundQueueDict = temp_loaded_container[0]
        self.RoundCoordsDict = temp_loaded_container[1]
        self.GeneralSettingDict = temp_loaded_container[2]

        # Interpolate in between the focus correction positions
        FocusCorrectionMatrixDict = {}  # self.upsize_focus_matrix()

        # Refresh the focus correction
        self.GeneralSettingDict["FocusCorrectionMatrixDict"] = (
            FocusCorrectionMatrixDict
        )

        # if saving directory is re-configured, refresh it, otherwise keep as it is.
        self.auto_saving_directory()

        self.normalOutputWritten("Pipeline loaded.\n")
        logging.info("Pipeline loaded.")

        self.show_pipline_infor()

    def show_pipline_infor(self):
        """
        Show general information of the pipeline.

        Returns
        None.

        """
        self.normalOutputWritten("--------Pipeline general info--------\n")
        for eachround in range(int(len(self.RoundQueueDict) / 2 - 1)):
            # show waveform settings
            waveformPackage = self.RoundQueueDict[
                "RoundPackage_" + str(eachround + 1)
            ][0]
            camOperationPackage = self.RoundQueueDict[
                "RoundPackage_" + str(eachround + 1)
            ][1]
            waveform_sequence = 1

            for eachwaveform in waveformPackage:
                try:
                    if len(waveformPackage[eachwaveform][3]) != 0:
                        self.normalOutputWritten(
                            "Round {}, sequence {}, recording channels:{}.\n".format(
                                eachround + 1,
                                waveform_sequence,
                                waveformPackage[eachwaveform][3],
                            )
                        )
                        logging.info(
                            "Round {}, recording channels:{}.".format(
                                eachround + 1, waveformPackage[eachwaveform][3]
                            )
                        )  # [1]['Specification']
                # else:
                # self.normalOutputWritten('Round {} No recording channel.\n'.format(eachround+1))
                except Exception as exc:
                    logging.critical("caught exception", exc_info=exc)
                    self.normalOutputWritten("No recording channel.\n")
                    logging.info("No recording channel.")
                try:
                    self.normalOutputWritten(
                        "Round {}, Analog signals:{}.\n".format(
                            eachround + 1,
                            waveformPackage[eachwaveform][1]["Specification"],
                        )
                    )
                    logging.info(
                        "Round {}, Analog signals:{}.".format(
                            eachround + 1,
                            waveformPackage[eachwaveform][1]["Specification"],
                        )
                    )  #
                except Exception as exc:
                    logging.critical("caught exception", exc_info=exc)
                    self.normalOutputWritten("No Analog signals.\n")
                    logging.info("No Analog signals.")
                try:
                    if (
                        len(waveformPackage[eachwaveform][2]["Specification"])
                        != 0
                    ):
                        self.normalOutputWritten(
                            "Round {}, Digital signals:{}.\n".format(
                                eachround + 1,
                                waveformPackage[eachwaveform][2][
                                    "Specification"
                                ],
                            )
                        )
                        self.normalOutputWritten(
                            "Lasting time:{} s.\n".format(
                                len(
                                    waveformPackage[eachwaveform][2][
                                        "Waveform"
                                    ][0]
                                )
                                / waveformPackage[eachwaveform][0]
                            )
                        )

                        logging.info(
                            "Lasting time:{} s.\n".format(
                                len(
                                    waveformPackage[eachwaveform][2][
                                        "Waveform"
                                    ][0]
                                )
                                / waveformPackage[eachwaveform][0]
                            )
                        )
                        logging.info(
                            "Round {}, Digital signals:{}.".format(
                                eachround + 1,
                                waveformPackage[eachwaveform][2][
                                    "Specification"
                                ],
                            )
                        )  #
                # else:
                # self.normalOutputWritten('Round {} No Digital signals.\n'.format(eachround+1))
                except Exception as exc:
                    logging.critical("caught exception", exc_info=exc)
                    self.normalOutputWritten("No Digital signals.\n")
                    logging.info("No Digital signals.")
                waveform_sequence += 1
                self.normalOutputWritten("\n")

            for eachcamoperation in camOperationPackage:
                # Show camera operations

                try:
                    if len(camOperationPackage[eachcamoperation]) != 0:
                        self.normalOutputWritten(
                            "Round {}, cam Buffer_number:{}.\n".format(
                                eachround + 1,
                                camOperationPackage[eachcamoperation][
                                    "Buffer_number"
                                ],
                            )
                        )
                        logging.info(
                            "Round {}, cam Buffer_number:{}.\n".format(
                                eachround + 1,
                                camOperationPackage[eachcamoperation][
                                    "Buffer_number"
                                ],
                            )
                        )  #
                # else:
                # self.normalOutputWritten('Round {} No Digital signals.\n'.format(eachround+1))
                except Exception as exc:
                    logging.critical("caught exception", exc_info=exc)
                    self.normalOutputWritten("No camera operations.\n")
                    logging.info("No camera operations.")

            self.normalOutputWritten("-----------end of round-----------\n")
        self.normalOutputWritten("----------------------------------------\n")

    # %%
    """
    # FUNCTIONS FOR QUICK START
    """

    def quick_start(self, config_number):
        # Load pre-saved pipeline
        self.pipelinenpfileName = self.quick_start_location[config_number]
        temp_loaded_container = np.load(
            self.pipelinenpfileName, allow_pickle=True
        )
        self.RoundQueueDict = temp_loaded_container[0]
        self.RoundCoordsDict = temp_loaded_container[1]
        self.GeneralSettingDict = temp_loaded_container[2]

        self.auto_saving_directory()

        self.normalOutputWritten("Pipeline loaded.\n")
        logging.info("Pipeline loaded.")

        self.show_pipline_infor()

        # Execute
        self.ExecutePipeline()

    # === functions for console display ===
    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.ConsoleTextDisplay.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.ConsoleTextDisplay.setTextCursor(cursor)
        self.ConsoleTextDisplay.ensureCursorVisible()

    def openScreenAnalysisMLWidget(self):
        self.ScreenAnalysisMLWindow.deleteLater()
        self.ScreenAnalysisMLWindow = EvolutionAnalysisWidget.MainGUI()
        self.ScreenAnalysisMLWindow.show()

    # %%


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        QtWidgets.QApplication.setStyle(
            QtWidgets.QStyleFactory.create("Fusion")
        )
        mainwin = Mainbody()
        mainwin.show()
        app.exec_()

    run_app()
