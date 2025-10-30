# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:47:31 2020

@author: xinmeng
"""

import ctypes
import ctypes.util
import importlib.resources
import logging
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import tifffile as skimtiff
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QMovie,
    QPen,
)
from PyQt5.QtWidgets import (
    QAction,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyleFactory,
    QTabWidget,
    QToolButton,
    QWidget,
)
from skimage.measure import block_reduce

from .. import Icons, StylishQT
from ..GalvoWidget import PMTWidget
from ..HamamatsuCam import HamamatsuDCAM
from ..NIDAQ import WaveformWidget
from ..PythonScriptsNike.camera_pmt_mapping import CameraPmtMapping
from ..PythonScriptsNike.camera_pmt_registration import CameraPmtRegistration
from ..PythonScriptsNike.camera_pmt_registration_points import (
    CameraPmtRegistrationPoints,
)
from ..PythonScriptsNike.image_analyzer import ImageAnalyzer

"""
Some general settings for pyqtgraph, these only have to do with appearance
except for row-major, which inverts the image and puts mirrors some axes.
"""

pg.setConfigOptions(imageAxisOrder="row-major")
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")
pg.setConfigOption("useOpenGL", True)
pg.setConfigOption("leftButtonPan", False)


class CameraUI(QMainWindow):

    output_signal_SnapImg = pyqtSignal(np.ndarray)
    output_signal_LiveImg = pyqtSignal(np.ndarray)
    output_signal_camera_handle = pyqtSignal(object)
    output_signal_camera_pmt_contour = pyqtSignal(object)
    stream_parameters = pyqtSignal(object)

    def __init__(self, pmt_widget_ui, waveform_widget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pmt_widget_ui = pmt_widget_ui
        pmt_widget_ui.GalvoCoordinatesCommand.connect(
            self.handle_galvo_coordinates
        )
        self.waveform_widget = waveform_widget
        self.stream_parameters.connect(
            self.waveform_widget.handle_stream_parameters
        )

        self.cameraIsLive = False
        self.cameraIsStreaming = False
        self.Live_item_autolevel = True
        self.ShowROIImgSwitch = False
        self.ROIselector_ispresented = False
        self.Live_sleeptime = 0.06666  # default camera live fps
        self.minimum_live_update_interval = 0.04
        self.default_folder = "M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data"  # TODO hardcoded path

        # === GUI ===
        self.setWindowTitle("Hamamatsu Orca Flash")
        self.setFont(QFont("Arial"))
        self.setMinimumSize(1280, 1000)
        self.layout = QGridLayout()
        # === Create menu bar and add action ===
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("&Camera")

        with Icons.Path("on.png") as path:
            ActConnectCamera = QAction(QIcon(path), "Connect camera", self)
        ActConnectCamera.setShortcut("Ctrl+c")
        ActConnectCamera.setStatusTip("Connect camera")
        ActConnectCamera.triggered.connect(self.ConnectCamera)

        with Icons.Path("off.png") as path:
            ActDisconnectCamera = QAction(
                QIcon(path), "Disconnect camera", self
            )
        ActDisconnectCamera.setShortcut("Ctrl+d")
        ActDisconnectCamera.triggered.connect(self.DisconnectCamera)

        ActListCameraProperties = QAction("List properties", self)
        ActListCameraProperties.setShortcut("Ctrl+l")
        ActListCameraProperties.triggered.connect(self.ListCameraProperties)

        fileMenu.addAction(ActConnectCamera)
        fileMenu.addAction(ActDisconnectCamera)
        fileMenu.addAction(ActListCameraProperties)

        MainWinCentralWidget = QWidget()
        main_layout = QGridLayout()  # Create the layout
        main_layout.setSpacing(10)
        MainWinCentralWidget.setLayout(
            main_layout
        )  # Set the layout to MainWinCentralWidget

        """
        # Camera settings container.
        """
        CameraSettingContainer = StylishQT.roundQGroupBox(
            title="General settings"
        )
        CameraSettingContainer.setMaximumHeight(350)
        CameraSettingContainer.setMaximumWidth(332)
        CameraSettingLayout = QGridLayout()

        CameraSettingContainer.setStyleSheet(
            "QGroupBox {"
            "    font: bold;"
            "    border: 1px solid silver;"
            "    border-radius: 6px;"
            "    margin-top: 10px;"
            "    padding: 0px;"  # Remove padding
            "    color: Navy;"
            "    font-size: 12px;"
            "}"
            "QGroupBox::title {"
            "    subcontrol-origin: margin;"
            "    left: 7px;"
            "    padding: 5px 5px 5px 5px;"
            "}"
        )

        self.CamStatusLabel = QLabel("Camera not connected.")
        self.CamStatusLabel.setStyleSheet(
            "QLabel { background-color : azure; color : blue; }"
        )
        self.CamStatusLabel.setFixedHeight(30)
        self.CamStatusLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        CameraSettingLayout.addWidget(self.CamStatusLabel, 0, 0, 1, 1)

        self.cam_connect_button = StylishQT.connectButton()
        self.cam_connect_button.setFixedWidth(70)
        CameraSettingLayout.addWidget(self.cam_connect_button, 0, 1)
        self.cam_connect_button.clicked.connect(
            lambda: self.run_in_thread(self.cam_connect_switch)
        )

        CameraSettingTab = QTabWidget()
        CameraSettingTab.layout = QGridLayout()

        """
        === Camera tab ===
        """
        CameraSettingTab_1 = QWidget()
        CameraSettingTab_1.layout = QGridLayout()

        Label_readoutspeed = QLabel("Readout speed")
        Label_readoutspeed.setToolTip(
            "The standard scan readout speed can achieve a frame rate of 100 "
            "fps for full resolution with low noise (1.0 electrons (median), "
            "1.6 electrons (r.m.s.)), and the slow scan readout speed can "
            "achieve even lower noise (0.8 electrons (median), 1.4 electrons "
            "(r.m.s.)) with a frame rate of 30 fps for full resolution."
        )
        CameraSettingTab_1.layout.addWidget(Label_readoutspeed, 2, 0)
        self.ReadoutSpeedSwitchButton = StylishQT.MySwitch(
            "Normal", "yellow", "Fast", "cyan", width=42
        )
        self.ReadoutSpeedSwitchButton.clicked.connect(
            self.ReadoutSpeedSwitchEvent
        )
        CameraSettingTab_1.layout.addWidget(
            self.ReadoutSpeedSwitchButton, 2, 1, 1, 2
        )

        self.DefectCorrectionButton = QPushButton("Pixel corr.")
        self.DefectCorrectionButton.setCheckable(True)
        self.DefectCorrectionButton.setChecked(True)
        self.DefectCorrectionButton.clicked.connect(
            self.DefectCorrectionSwitchEvent
        )
        self.DefectCorrectionButton.setToolTip(
            "There are a few pixels in CMOS image sensor that have slightly "
            "higher readout noise performance compared to surrounding pixels. "
            "And the extended exposures may cause a few white spots which is "
            "caused by failure in part of the silicon wafer in CMOS image "
            "sensor. The camera has real-time variant pixel correction "
            "features to improve image quality."
        )
        CameraSettingTab_1.layout.addWidget(self.DefectCorrectionButton, 2, 3)

        CameraImageFormatContainer = QGroupBox("Image format")
        CameraImageFormatContainer.setStyleSheet(
            "QGroupBox { background-color:#F5F5F5;}"
        )
        CameraImageFormatLayout = QGridLayout()

        self.BinningButtongroup = QButtonGroup(self)
        self.BinningButton_1 = QPushButton("1x1")
        self.BinningButton_1.setCheckable(True)
        self.BinningButton_1.setChecked(True)
        self.BinningButtongroup.addButton(self.BinningButton_1, 1)
        self.BinningButton_2 = QPushButton("2x2")
        self.BinningButton_2.setCheckable(True)
        self.BinningButtongroup.addButton(self.BinningButton_2, 2)
        self.BinningButton_4 = QPushButton("4x4")
        self.BinningButton_4.setCheckable(True)
        self.BinningButtongroup.addButton(self.BinningButton_4, 3)
        self.BinningButtongroup.setExclusive(True)
        self.BinningButtongroup.buttonClicked[int].connect(self.SetBinning)

        Label_binning = QLabel("Binning:")
        Label_binning.setToolTip(
            "Binning readout is a method for achieving high sensitivity in "
            "exchange for losing resolution."
        )
        CameraImageFormatLayout.addWidget(Label_binning, 0, 0)
        CameraImageFormatLayout.addWidget(self.BinningButton_1, 0, 1)
        CameraImageFormatLayout.addWidget(self.BinningButton_2, 0, 2)
        CameraImageFormatLayout.addWidget(self.BinningButton_4, 0, 3)

        self.PixelTypeButtongroup = QButtonGroup(self)
        self.PixelTypeButton_1 = QPushButton("8")
        self.PixelTypeButton_1.setCheckable(True)
        self.PixelTypeButtongroup.addButton(self.PixelTypeButton_1, 1)
        self.PixelTypeButton_2 = QPushButton("12")
        self.PixelTypeButton_2.setCheckable(True)
        self.PixelTypeButtongroup.addButton(self.PixelTypeButton_2, 2)
        self.PixelTypeButton_3 = QPushButton("16")
        self.PixelTypeButton_3.setCheckable(True)
        self.PixelTypeButton_3.setChecked(True)
        self.PixelTypeButtongroup.addButton(self.PixelTypeButton_3, 3)
        self.PixelTypeButtongroup.setExclusive(True)
        self.PixelTypeButtongroup.buttonClicked[int].connect(self.SetPixelType)

        CameraImageFormatLayout.addWidget(QLabel("Pixel bit depth:"), 1, 0)
        CameraImageFormatLayout.addWidget(self.PixelTypeButton_1, 1, 1)
        CameraImageFormatLayout.addWidget(self.PixelTypeButton_2, 1, 2)
        CameraImageFormatLayout.addWidget(self.PixelTypeButton_3, 1, 3)

        CameraImageFormatContainer.setLayout(CameraImageFormatLayout)
        CameraImageFormatContainer.setFixedHeight(100)
        CameraSettingTab_1.layout.addWidget(
            CameraImageFormatContainer, 0, 0, 1, 4
        )

        self.CamExposureBox = QDoubleSpinBox(self)
        self.CamExposureBox.setDecimals(5)
        self.CamExposureBox.setMinimum(0)
        self.CamExposureBox.setMaximum(10)
        self.CamExposureBox.setSingleStep(0.001)

        CamExposureButton = QPushButton("Set exposure")
        CamExposureButton.clicked.connect(self.SetExposureTimeFromCamera)
        self.allowUserInputForExposure = True

        CameraSettingTab_1.layout.addWidget(
            QLabel("Exposure time:"), 4, 0, 1, 1
        )
        CameraSettingTab_1.layout.addWidget(self.CamExposureBox, 4, 1, 1, 1)
        CameraSettingTab_1.layout.addWidget(CamExposureButton, 4, 2, 1, 1)
        self.CamExposureBox.setToolTip(
            "The exposure time setting can be done by the units of seconds."
        )

        self.CamExposureBox.setKeyboardTracking(False)

        CameraSettingTab_1.setLayout(CameraSettingTab_1.layout)

        """
        === ROI tab ===
        """
        CameraSettingTab_2 = QWidget()
        CameraSettingTab_2.layout = QGridLayout()

        label_sub_array_mode_switch = QLabel("Readout region:")
        label_sub_array_mode_switch.setToolTip(
            "Sub-array readout is a procedure only a region of interest is "
            "scanned; while full size images whole frame."
        )
        CameraSettingTab_2.layout.addWidget(label_sub_array_mode_switch, 0, 0)
        self.SubArrayModeSwitchButton = StylishQT.MySwitch(
            "Sub Array Mode",
            "lemon chiffon",
            "Full Image Size",
            "lavender",
            width=100,
        )
        self.SubArrayModeSwitchButton.setChecked(False)
        self.SubArrayModeSwitchButton.clicked.connect(
            self.SubArrayModeSwitchEvent
        )
        CameraSettingTab_2.layout.addWidget(
            self.SubArrayModeSwitchButton, 0, 1, 1, 3
        )

        # Adapted from Douwe's ROI part.
        self.center_roiButton = QPushButton()
        self.center_roiButton.setText("Symmetric to Center Line")
        self.center_roiButton.setToolTip(
            "In normal configuration, place ROI symmetric to chip center line "
            "to achieve fastest frame rate."
        )
        self.center_roiButton.clicked.connect(lambda: self.set_roi_flag())
        """
        set_roi_flag checks whether the centering button is pushed and
        acts accordingly.
        """
        self.center_roiButton.setCheckable(True)
        CameraSettingTab_2.layout.addWidget(self.center_roiButton, 1, 1, 1, 3)
        """
        The ROI needs to be centered to maximise the framerate of the hamamatsu
        CMOS. When not centered it will count the outermost vertical pixel and
        treats it as the size of the ROI. See the camera manual for a more
        detailed explanation.
        """

        self.ShowROISelectorButton = QPushButton()
        self.ShowROISelectorButton.setText("Show ROI Selector")
        self.ShowROISelectorButton.clicked.connect(self.ShowROISelector)
        self.ShowROISelectorButton.setCheckable(True)
        CameraSettingTab_2.layout.addWidget(
            self.ShowROISelectorButton, 2, 1, 1, 2
        )

        self.ShowROIImgButton = QPushButton()
        self.ShowROIImgButton.setText("Check ROI (R)")
        self.ShowROIImgButton.setToolTip("Short key: R ")
        self.ShowROIImgButton.clicked.connect(self.SetShowROIImgSwitch)
        self.ShowROIImgButton.setShortcut("r")
        self.ShowROIImgButton.setCheckable(True)
        self.ShowROIImgButton.setEnabled(False)
        CameraSettingTab_2.layout.addWidget(self.ShowROIImgButton, 2, 3, 1, 1)

        CameraROIPosContainer = QGroupBox("ROI position")
        CameraROIPosContainer.setStyleSheet(
            "QGroupBox { background-color:#F5F5F5;}"
        )
        CameraROIPosLayout = QGridLayout()

        OffsetLabel = QLabel("Offset")
        OffsetLabel.setFixedHeight(20)
        ROISizeLabel = QLabel("Size")
        ROISizeLabel.setFixedHeight(20)

        CameraROIPosLayout.addWidget(OffsetLabel, 0, 1)
        CameraROIPosLayout.addWidget(ROISizeLabel, 0, 2)

        self.ROI_hpos_spinbox = QSpinBox()
        self.ROI_hpos_spinbox.setMaximum(2048)
        self.ROI_hpos_spinbox.setValue(0)

        CameraROIPosLayout.addWidget(self.ROI_hpos_spinbox, 1, 1)

        self.ROI_vpos_spinbox = QSpinBox()
        self.ROI_vpos_spinbox.setMaximum(2048)
        self.ROI_vpos_spinbox.setValue(0)

        CameraROIPosLayout.addWidget(self.ROI_vpos_spinbox, 2, 1)

        self.ROI_hsize_spinbox = QSpinBox()
        self.ROI_hsize_spinbox.setMaximum(2048)
        self.ROI_hsize_spinbox.setValue(2048)

        CameraROIPosLayout.addWidget(self.ROI_hsize_spinbox, 1, 2)

        self.ROI_vsize_spinbox = QSpinBox()
        self.ROI_vsize_spinbox.setMaximum(2048)
        self.ROI_vsize_spinbox.setValue(2048)

        CameraROIPosLayout.addWidget(self.ROI_vsize_spinbox, 2, 2)

        CameraROIPosLayout.addWidget(QLabel("Horizontal"), 1, 0)
        CameraROIPosLayout.addWidget(QLabel("Vertical"), 2, 0)

        CameraROIPosContainer.setLayout(CameraROIPosLayout)
        CameraROIPosContainer.setFixedHeight(100)
        CameraSettingTab_2.layout.addWidget(CameraROIPosContainer, 3, 0, 1, 4)

        self.ApplyROIButton = QPushButton()
        self.ApplyROIButton.setText("Apply ROI")
        self.ApplyROIButton.clicked.connect(self.SetROI)
        CameraSettingTab_2.layout.addWidget(self.ApplyROIButton, 4, 0, 1, 2)

        self.ClearROIButton = QPushButton()
        self.ClearROIButton.setText("Clear ROI")

        CameraSettingTab_2.layout.addWidget(self.ClearROIButton, 4, 2, 1, 2)
        CameraSettingTab_2.setLayout(CameraSettingTab_2.layout)

        # === Timing tab ===
        CameraSettingTab_3 = QWidget()
        CameraSettingTab_3.layout = QGridLayout()

        self.TriggerButtongroup = QButtonGroup(self)
        self.TriggerButton_1 = QPushButton("Intern")
        self.TriggerButton_1.setCheckable(True)
        self.TriggerButtongroup.addButton(self.TriggerButton_1, 1)
        self.TriggerButton_1.clicked.connect(
            lambda: self.TimingstackedWidget.setCurrentIndex(0)
        )

        self.TriggerButton_2 = QPushButton("Extern")
        self.TriggerButton_2.setCheckable(True)
        self.TriggerButtongroup.addButton(self.TriggerButton_2, 2)
        self.TriggerButton_2.clicked.connect(
            lambda: self.TimingstackedWidget.setCurrentIndex(1)
        )

        self.TriggerButton_3 = QPushButton("MasterPulse")
        self.TriggerButton_3.setCheckable(True)
        self.TriggerButtongroup.addButton(self.TriggerButton_3, 3)
        self.TriggerButton_3.clicked.connect(
            lambda: self.TimingstackedWidget.setCurrentIndex(2)
        )
        self.TriggerButtongroup.setExclusive(True)

        self.TriggerButtongroup.buttonClicked[int].connect(
            self.SetTimingTrigger
        )

        CameraSettingTab_3.layout.addWidget(
            QLabel("Acquisition Control:"), 0, 0, 1, 2
        )
        CameraSettingTab_3.layout.addWidget(self.TriggerButton_1, 1, 1)
        CameraSettingTab_3.layout.addWidget(self.TriggerButton_2, 1, 2)
        CameraSettingTab_3.layout.addWidget(self.TriggerButton_3, 1, 3)

        InternTriggerWidget = QWidget()
        ExternTriggerWidget = QWidget()
        MasterPulseWidget = QWidget()

        self.TimingstackedWidget = QStackedWidget()
        self.TimingstackedWidget.addWidget(InternTriggerWidget)
        self.TimingstackedWidget.addWidget(ExternTriggerWidget)
        self.TimingstackedWidget.addWidget(MasterPulseWidget)
        self.TimingstackedWidget.setCurrentIndex(0)

        # === ExternTrigger ===
        ExternTriggerWidget.layout = QGridLayout()
        ExternTriggerWidget.layout.addWidget(QLabel("Trigger Signal:"), 0, 0)
        self.ExternTriggerSignalComboBox = QComboBox()
        self.ExternTriggerSignalComboBox.addItems(
            ["EDGE", "LEVEL", "SYNCREADOUT"]
        )
        self.ExternTriggerSignalComboBox.setCurrentIndex(2)
        self.ExternTriggerSignalComboBox.activated.connect(
            self.SetTriggerActive
        )

        ExternTriggerWidget.layout.addWidget(
            self.ExternTriggerSignalComboBox, 0, 1
        )
        ExternTriggerWidget.setLayout(ExternTriggerWidget.layout)

        CameraSettingTab_3.layout.addWidget(
            self.TimingstackedWidget, 2, 0, 4, 4
        )
        CameraSettingTab_3.setLayout(CameraSettingTab_3.layout)

        # === Registration tab ===
        CameraSettingTab_4 = QWidget()
        CameraSettingTab_4.layout = QGridLayout()

        self.camera_pmt_registration = CameraPmtRegistration()

        registration_point_pmt_label = QLabel("Registration Point PMT")
        registration_point_pmt_label.setStyleSheet("font-weight: bold;")
        registration_x_coord_pmt_label = QLabel("X-coordinate: ")
        self.registration_x_coord_pmt_value = QLineEdit()
        self.registration_x_coord_pmt_value.setEnabled(False)
        registration_y_coord_pmt_label = QLabel("Y-coordinate: ")
        self.registration_y_coord_pmt_value = QLineEdit()
        self.registration_y_coord_pmt_value.setEnabled(False)

        CameraSettingTab_4.layout.addWidget(
            registration_point_pmt_label, 0, 0, 1, 2
        )
        CameraSettingTab_4.layout.addWidget(
            registration_x_coord_pmt_label, 1, 0
        )
        CameraSettingTab_4.layout.addWidget(
            self.registration_x_coord_pmt_value, 1, 1
        )
        CameraSettingTab_4.layout.addWidget(
            registration_y_coord_pmt_label, 1, 2
        )
        CameraSettingTab_4.layout.addWidget(
            self.registration_y_coord_pmt_value, 1, 3
        )

        registration_point_camera_label = QLabel("Registration Point Camera")
        registration_point_camera_label.setStyleSheet("font-weight: bold;")
        registration_x_coord_camera_label = QLabel("X-coordinate:")
        self.registration_x_coord_camera_value = QLineEdit()
        self.registration_x_coord_camera_value.setEnabled(False)
        registration_y_coord_camera_label = QLabel("Y-coordinate: ")
        self.registration_y_coord_camera_value = QLineEdit()
        self.registration_y_coord_camera_value.setEnabled(False)

        CameraSettingTab_4.layout.addWidget(
            registration_point_camera_label, 2, 0, 1, 2
        )
        CameraSettingTab_4.layout.addWidget(
            registration_x_coord_camera_label, 3, 0
        )
        CameraSettingTab_4.layout.addWidget(
            self.registration_x_coord_camera_value, 3, 1
        )
        CameraSettingTab_4.layout.addWidget(
            registration_y_coord_camera_label, 3, 2
        )
        CameraSettingTab_4.layout.addWidget(
            self.registration_y_coord_camera_value, 3, 3
        )

        amplitude_label = QLabel("Amplitude: ")
        self.amplitude_value = QLineEdit()
        self.amplitude_value.setEnabled(False)
        offset_label = QLabel("Offset: ")
        self.camera_offset_value = QLineEdit()
        self.camera_offset_value.setEnabled(False)

        CameraSettingTab_4.layout.addWidget(amplitude_label, 4, 0)
        CameraSettingTab_4.layout.addWidget(self.amplitude_value, 4, 1)
        CameraSettingTab_4.layout.addWidget(offset_label, 4, 2)
        CameraSettingTab_4.layout.addWidget(self.camera_offset_value, 4, 3)

        sigma_x_label = QLabel("Sigma x: ")
        self.sigma_x_value = QLineEdit()
        self.sigma_x_value.setEnabled(False)
        sigma_y_label = QLabel("Sigma y: ")
        self.sigma_y_value = QLineEdit()
        self.sigma_y_value.setEnabled(False)

        CameraSettingTab_4.layout.addWidget(sigma_x_label, 5, 0)
        CameraSettingTab_4.layout.addWidget(self.sigma_x_value, 5, 1)
        CameraSettingTab_4.layout.addWidget(sigma_y_label, 5, 2)
        CameraSettingTab_4.layout.addWidget(self.sigma_y_value, 5, 3)

        self.find_camera_registration_point = StylishQT.GeneralFancyButton(
            label="Find camera point"
        )
        self.find_camera_registration_point.setEnabled(False)
        self.find_camera_registration_point.clicked.connect(
            self.fit_gaussian_over_camera_img
        )
        CameraSettingTab_4.layout.addWidget(
            self.find_camera_registration_point, 6, 1, 1, 2
        )

        CameraSettingTab_4.setLayout(CameraSettingTab_4.layout)

        CameraSettingTab.addTab(CameraSettingTab_1, "Camera")
        CameraSettingTab.addTab(CameraSettingTab_2, "ROI")
        CameraSettingTab.addTab(CameraSettingTab_3, "Timing")
        CameraSettingTab.addTab(CameraSettingTab_4, "2P-registration")

        CameraSettingTab.setStyleSheet(
            "QTabBar { width: 200px; font-size: 8pt; font: bold;}"
        )
        CameraSettingLayout.addWidget(CameraSettingTab, 1, 0, 1, 2)
        CameraSettingContainer.setLayout(CameraSettingLayout)

        """
        # Camera tiff file import container.
        # Added by Nike Celie 8-1-2025
        """

        # === TIFF File Import Container ===
        tiffImportContainer = StylishQT.roundQGroupBox("Import TIFF files")
        tiffImportContainer.setMaximumHeight(70)
        tiffImportContainer.setMaximumWidth(325)
        tiffImportLayout = QGridLayout()

        # Add "Browse" button for selecting the file
        self.browseTiffFileButton = QPushButton("Browse TIFF-files")
        self.browseTiffFileButton.setIcon(QIcon("./Icons/Browse.png"))
        self.browseTiffFileButton.clicked.connect(self.browseTiffFiles)
        tiffImportLayout.addWidget(self.browseTiffFileButton, 0, 0)

        # Add text box for displaying the selected TIFF file path
        self.tiffDirectoryTextbox = QLineEdit(self)
        self.tiffDirectoryTextbox.setPlaceholderText("Tiff file path")
        tiffImportLayout.addWidget(self.tiffDirectoryTextbox, 0, 1)

        # Add the "Clear" button next to the file path textbox
        self.clearImageButton = QPushButton("Clear Image")
        self.clearImageButton.clicked.connect(self.clearImage)
        self.clearImageButton.setEnabled(False)
        tiffImportLayout.addWidget(self.clearImageButton, 0, 2)

        # Set the layout for the container
        tiffImportContainer.setLayout(tiffImportLayout)

        """
        # Camera image inspection
        # Added by Nike Celie 8-1-2025
        """

        # Create an instance of the ImageAnalyzer class
        self.image_analyzer = ImageAnalyzer()
        self.registrationPoints = CameraPmtRegistrationPoints()

        # Connect the output_signal_camera_pmt_contour signal to the handleoutput_signal_camera_pmt_contour slot
        self.output_signal_camera_pmt_contour.connect(
            self.pmt_widget_ui.handle_received_camera_contour
        )

        # Camera Image Inspection Container
        CameraImageInspectionContainer = StylishQT.roundQGroupBox(
            "Contour generation"
        )
        CameraImageInspectionContainer.setMaximumHeight(150)
        CameraImageInspectionContainer.setFixedWidth(325)

        # Create the layout for the container
        self.CameraImageInspectionLayout = QGridLayout()

        # Create three QLabel widgets for displaying values (e.g., pixel coordinates and intensity)
        self.x_label = QLabel("X-coordinate: _")
        self.y_label = QLabel("Y-coordinate: _")
        self.intensity_label = QLabel("Intensity: _")

        # Add the labels to the layout
        self.CameraImageInspectionLayout.addWidget(self.x_label, 0, 0)
        self.CameraImageInspectionLayout.addWidget(self.y_label, 0, 1)
        self.CameraImageInspectionLayout.addWidget(self.intensity_label, 0, 2)

        # Create widgets for contour generation
        self.contourGenerationButton = QPushButton("Generate contour")
        self.contourGenerationButton.clicked.connect(self.generateContour)
        self.contourGenerationButton.setEnabled(False)
        self.contourIntensityInput = QLineEdit(self)
        self.contourIntensityInput.textChanged.connect(
            self.checkContourIntensityInput
        )
        self.contourIntensityInput.setPlaceholderText(
            "Define intensity threshold"
        )
        self.removeGeneratedContourButton = QPushButton("Remove contour")
        self.removeGeneratedContourButton.clicked.connect(
            self.removeGeneratedContour
        )
        self.removeGeneratedContourButton.setVisible(False)

        # Create sliders for adjusting the generated contour
        self.contourSizeSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.contourSizeSlider.setMinimum(int(0.5 * 100))
        self.contourSizeSlider.setMaximum(int(1.5 * 100))
        self.contourSizeSlider.setSingleStep(int(0.05 * 100))
        self.contourSizeSlider.setValue(int(1.0 * 100))
        self.contourSizeSlider.valueChanged.connect(
            self.updateGeneratedContour
        )
        self.contourSizeSlider.setVisible(False)

        self.contourSizeLabel = QLabel()
        self.contourSizeLabel.setVisible(False)

        self.contourSmoothnessSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.contourSmoothnessSlider.setMinimum(1)
        self.contourSmoothnessSlider.setMaximum(25)
        self.contourSmoothnessSlider.setSingleStep(1)
        self.contourSmoothnessSlider.setValue(1)
        self.contourSmoothnessSlider.valueChanged.connect(
            self.updateGeneratedContour
        )
        self.contourSmoothnessSlider.setVisible(False)

        self.contourSmoothnessLabel = QLabel()
        self.contourSmoothnessLabel.setVisible(False)

        self.contourIntensitySlider = QtWidgets.QSlider(Qt.Horizontal)
        self.contourIntensitySlider.setSingleStep(50)
        self.contourIntensitySlider.valueChanged.connect(
            self.updateGeneratedContour
        )
        self.contourIntensitySlider.setVisible(False)

        self.contourIntensityLabel = QLabel()
        self.contourIntensityLabel.setVisible(False)

        # Create widgets for contour creation
        self.contourCreationProgress = QLabel()
        self.contourCreationProgress.setVisible(False)

        self.saveContourButton = QPushButton("Save contour")
        self.saveContourButton.clicked.connect(self.saveCustomContour)
        self.saveContourButton.setVisible(False)

        # Add the widgets to the layout
        self.CameraImageInspectionLayout.addWidget(
            self.contourGenerationButton, 1, 0
        )
        self.CameraImageInspectionLayout.addWidget(
            self.contourIntensityInput, 1, 1
        )
        self.CameraImageInspectionLayout.addWidget(
            self.removeGeneratedContourButton, 1, 2
        )
        self.CameraImageInspectionLayout.addWidget(
            self.contourIntensityLabel, 2, 0
        )
        self.CameraImageInspectionLayout.addWidget(
            self.contourSmoothnessLabel, 2, 1
        )
        self.CameraImageInspectionLayout.addWidget(self.contourSizeLabel, 2, 2)
        self.CameraImageInspectionLayout.addWidget(
            self.contourIntensitySlider, 3, 0
        )
        self.CameraImageInspectionLayout.addWidget(
            self.contourSmoothnessSlider, 3, 1
        )
        self.CameraImageInspectionLayout.addWidget(
            self.contourSizeSlider, 3, 2
        )
        self.CameraImageInspectionLayout.addWidget(
            self.contourCreationProgress, 4, 0
        )
        self.CameraImageInspectionLayout.addWidget(
            self.saveContourButton, 4, 1
        )

        # Uncomment for user input contour, not fully debugged tho

        # self.contourCreationButton = QPushButton("Create contour")
        # self.contourCreationButton.clicked.connect(self.createCustomContour)
        # self.contourCreationButton.setEnabled(False)

        # self.contourSizeInput = QLineEdit(self)
        # self.contourSizeInput.textChanged.connect(self.checkContourSizeInput)
        # self.contourSizeInput.setPlaceholderText("Define int # of points")

        self.createContour = False
        # self.CameraImageInspectionLayout.addWidget(self.contourCreationButton, 5, 0)
        # self.CameraImageInspectionLayout.addWidget(self.contourSizeInput, 5, 1)

        # Set the layout for the container
        CameraImageInspectionContainer.setLayout(
            self.CameraImageInspectionLayout
        )

        # Store the connection for later disconnection
        self.mouse_click_connection = None

        """
        # Camera acquisition container.
        """

        CameraAcquisitionContainer = StylishQT.roundQGroupBox("Acquisition")
        CameraAcquisitionContainer.setMaximumHeight(438)
        CameraAcquisitionContainer.setMaximumWidth(325)
        CameraAcquisitionLayout = QGridLayout()

        CamSpecContainer = QGroupBox()
        CamSpecContainer.setStyleSheet(
            "QGroupBox {"
            "    font: bold;"
            "    border: 1px solid silver;"
            "    border-radius: 6px;"
            "    margin-top: 6px;"
            "    color: olive; background-color: azure;"
            "}"
            "QGroupBox::title {"
            "    subcontrol-origin: margin;"
            "    left: 7px;"
            "    padding: 0px 5px 0px 5px;"
            "}"
        )

        # Create a QToolButton for toggling visibility
        toggle_button = QToolButton()
        toggle_button.setText("Hamamatsu specs")
        toggle_button.setCheckable(True)
        toggle_button.setChecked(True)
        toggle_button.setStyleSheet("QToolButton { font: bold; }")

        # Create a layout for the QGroupBox title and button
        title_layout = QtWidgets.QHBoxLayout()
        title_layout.addWidget(toggle_button)
        title_layout.addStretch(1)
        CamSpecContainer.setLayout(title_layout)

        # Create the layout for the Camera specs
        CamSpecLayout = QGridLayout()

        self.CamFPSLabel = QLabel("Internal frame rate:     ")
        self.CamFPSLabel.setStyleSheet(
            "QLabel { background-color : azure; color : teal; }"
        )
        CamSpecLayout.addWidget(self.CamFPSLabel, 0, 0, 1, 1)

        self.CamExposureTimeLabel = QLabel("Exposure time:     ")
        self.CamExposureTimeLabel.setStyleSheet(
            "QLabel { background-color : azure; color : teal; }"
        )
        CamSpecLayout.addWidget(self.CamExposureTimeLabel, 1, 0, 1, 1)

        self.CamReadoutTimeLabel = QLabel("Readout speed:     ")
        self.CamReadoutTimeLabel.setStyleSheet(
            "QLabel { background-color : azure; color : teal; }"
        )
        CamSpecLayout.addWidget(self.CamReadoutTimeLabel, 2, 0, 1, 1)

        # Add the Camera specs layout to a widget
        CamSpecWidget = QWidget()
        CamSpecWidget.setLayout(CamSpecLayout)

        # Add the Camera specs widget to the QGroupBox
        title_layout.addWidget(CamSpecWidget)

        # Connect the toggle button to a slot to toggle visibility
        toggle_button.clicked.connect(
            lambda: CamSpecWidget.setVisible(toggle_button.isChecked())
        )

        # Add the QGroupBox to the CameraAcquisitionLayout
        CameraAcquisitionLayout.addWidget(CamSpecContainer, 0, 0)

        # === Saving directory ===
        dir_container = StylishQT.roundQGroupBox()
        dir_container_layout = QGridLayout()

        self.BrowseStreamFileButton = QPushButton()
        with Icons.Path("Browse.png") as path:
            self.BrowseStreamFileButton.setIcon(QIcon(path))
        self.BrowseStreamFileButton.clicked.connect(
            lambda: self.SetSavingDirectory()
        )
        dir_container_layout.addWidget(self.BrowseStreamFileButton, 0, 0)

        self.CamSaving_directory_textbox = QLineEdit(self)
        self.CamSaving_directory_textbox.setPlaceholderText("Saving folder")
        dir_container_layout.addWidget(self.CamSaving_directory_textbox, 0, 1)

        self.CamSaving_filename_textbox = QLineEdit(self)
        self.CamSaving_filename_textbox.setPlaceholderText("Tiff file name")
        dir_container_layout.addWidget(self.CamSaving_filename_textbox, 0, 2)

        dir_container.setLayout(dir_container_layout)

        CameraAcquisitionLayout.addWidget(dir_container, 1, 0)
        self.AcquisitionROIstackedWidget = QStackedWidget()

        # === AcquisitionTabs ===
        CameraAcquisitionTab = QTabWidget()
        CameraAcquisitionTab.layout = QGridLayout()
        """
        === Live tab ===
        """
        CameraAcquisitionTab_1 = QWidget()
        CameraAcquisitionTab_1.layout = QGridLayout()

        CamLiveActionContainer = QGroupBox()
        # CamLiveActionContainer.setFixedHeight(110)
        CamLiveActionContainer.setStyleSheet(
            "QGroupBox { background-color:#F5F5F5;}"
        )
        CamLiveActionLayout = QGridLayout()

        # self.LiveSwitchLabel = QLabel("Live switch:")
        # self.LiveSwitchLabel.setStyleSheet(
        # "QLabel { color : navy; font-size: 10pt; }")
        # self.LiveSwitchLabel.setFixedHeight(45)
        # self.LiveSwitchLabel.setAlignment(Qt.AlignCenter)
        # CamLiveActionLayout.addWidget(self.LiveSwitchLabel, 0, 0)
        self.LiveButton = StylishQT.MySwitch(
            "Stop live",
            "indian red",
            "Start live",
            "spring green",
            width=85,
            font_size=10,
        )
        self.LiveButton.clicked.connect(self.LiveSwitchEvent)
        CamLiveActionLayout.addWidget(self.LiveButton, 0, 1, 1, 2)

        SnapImgButton = StylishQT.FancyPushButton(
            23, 32, color1=(255, 204, 229), color2=(153, 153, 255)
        )
        with Icons.Path("snap.png") as path:
            SnapImgButton.setIcon(QIcon(path))
        SnapImgButton.clicked.connect(self.SnapImg)
        SnapImgButton.setToolTip("Snap an image.")
        CamLiveActionLayout.addWidget(SnapImgButton, 1, 1, 1, 1)

        SaveLiveImgButton = StylishQT.saveButton()
        SaveLiveImgButton.setToolTip("Save live image directly.")
        SaveLiveImgButton.clicked.connect(
            lambda: self.run_in_thread(self.SaveLiveImg())
        )
        CamLiveActionLayout.addWidget(SaveLiveImgButton, 1, 2, 1, 1)

        CamLiveActionContainer.setLayout(CamLiveActionLayout)
        CameraAcquisitionTab_1.layout.addWidget(
            CamLiveActionContainer, 0, 1, 4, 4
        )

        self.LiveImgViewResetButton = QPushButton()
        self.LiveImgViewResetButton.setText("Reset ImageView")
        self.LiveImgViewResetButton.setToolTip(
            "Restart the view window in case it gets stuck."
        )
        self.LiveImgViewResetButton.clicked.connect(self.ResetLiveImgView)
        CameraAcquisitionTab_1.layout.addWidget(
            self.LiveImgViewResetButton, 0, 0, 1, 1
        )

        self.LiveAutoLevelSwitchButton = QPushButton()
        self.LiveAutoLevelSwitchButton.setText("Auto Level(A)")
        self.LiveAutoLevelSwitchButton.setToolTip(
            "Automatically adjust the contrast of image."
        )
        self.LiveAutoLevelSwitchButton.setShortcut("a")
        self.LiveAutoLevelSwitchButton.clicked.connect(
            self.AutoLevelSwitchEvent
        )
        self.LiveAutoLevelSwitchButton.setCheckable(True)
        self.LiveAutoLevelSwitchButton.setChecked(True)
        CameraAcquisitionTab_1.layout.addWidget(
            self.LiveAutoLevelSwitchButton, 1, 0, 1, 1
        )

        CameraAcquisitionTab_1.setLayout(CameraAcquisitionTab_1.layout)

        """
        === Stream tab ===
        """
        CameraAcquisitionTab_2 = QWidget()
        CameraAcquisitionTab_2.layout = QGridLayout()

        self.CamStreamActionContainer = QGroupBox()
        CamStreamActionLayout = QGridLayout()

        self.StreamStopSignalComBox = QComboBox()
        self.StreamStopSignalComBox.addItems(
            ["Stop signal: Time", "Stop signal: Frames"]
        )
        self.StreamStopSignalComBox.setToolTip(
            "End acquisition after getting certain number of frames or "
            "pre-set time is past."
        )
        CamStreamActionLayout.addWidget(self.StreamStopSignalComBox, 1, 0)

        desired_fps_label = QLabel("Desired FPS")
        desired_fps_label.setToolTip("Estimated frame rate of video.")
        TotalTimeLabel = QLabel("Total time")
        TotalTimeLabel.setToolTip("Length of the video.")

        CamStreamActionLayout.addWidget(desired_fps_label, 0, 1)
        CamStreamActionLayout.addWidget(TotalTimeLabel, 0, 2)

        self.desired_fps_spinbox = QSpinBox()
        self.desired_fps_spinbox.setMaximum(4048)
        self.desired_fps_spinbox.setValue(500)
        self.desired_fps_spinbox.setKeyboardTracking(False)
        CamStreamActionLayout.addWidget(self.desired_fps_spinbox, 1, 1)
        self.desired_fps_spinbox.valueChanged.connect(self.UpdateBufferNumber)

        self.StreamTotalTime_spinbox = QSpinBox()
        self.StreamTotalTime_spinbox.setMaximum(1200)
        self.StreamTotalTime_spinbox.setValue(0)
        CamStreamActionLayout.addWidget(self.StreamTotalTime_spinbox, 1, 2)
        self.StreamTotalTime_spinbox.valueChanged.connect(
            self.UpdateBufferNumber
        )

        self.StreamBufferTotalFrames_spinbox = QSpinBox()
        self.StreamBufferTotalFrames_spinbox.setMaximum(120000)
        self.StreamBufferTotalFrames_spinbox.setValue(0)
        CamStreamActionLayout.addWidget(
            self.StreamBufferTotalFrames_spinbox, 2, 2
        )
        Label_buffernumber = QLabel("Buffer size:")
        Label_buffernumber.setToolTip(
            "Amount of frames to allocate in buffer to store the video."
        )
        CamStreamActionLayout.addWidget(Label_buffernumber, 2, 1)

        self.StreamMemMethodComBox = QComboBox()
        self.StreamMemMethodComBox.addItems(
            ["Stream to Hard disk", "Stream to RAM"]
        )
        CamStreamActionLayout.addWidget(self.StreamMemMethodComBox, 2, 0)

        ApplyStreamSettingButton = StylishQT.FancyPushButton(50, 22)
        ApplyStreamSettingButton.setText("Apply")
        ApplyStreamSettingButton.clicked.connect(self.SetStreamSpecs)
        CameraAcquisitionTab_2.layout.addWidget(ApplyStreamSettingButton, 5, 2)

        self.startOrStopStreamButton = QPushButton()
        self.startOrStopStreamButton.setToolTip("Stream")
        with Icons.Path("StartStreaming.png") as path:
            self.startOrStopStreamButton.setIcon(QIcon(path))
        self.startOrStopStreamButton.setCheckable(True)
        self.startOrStopStreamButton.setEnabled(False)
        self.startOrStopStreamButton.clicked.connect(self.StreamingSwitchEvent)
        CameraAcquisitionTab_2.layout.addWidget(
            self.startOrStopStreamButton, 5, 3
        )

        """
        === Acquisition status ===
        """

        self.CamStreamIsFree = QLabel("No Stream Activity")
        self.CamStreamIsFree.setStyleSheet(
            "QLabel { background-color : azure; color : teal; font: bold;}"
        )
        self.CamStreamIsFree.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        CamStreamBusyWidget = QWidget()
        CamStreamSavingWidget = QWidget()

        self.StreamStatusStackedWidget = QStackedWidget()
        self.StreamStatusStackedWidget.setFixedHeight(50)
        self.StreamStatusStackedWidget.setStyleSheet(
            "QStackedWidget { background-color : #F0F8FF;}"
        )

        self.StreamStatusStackedWidget.addWidget(self.CamStreamIsFree)
        self.StreamStatusStackedWidget.addWidget(CamStreamBusyWidget)
        self.StreamStatusStackedWidget.addWidget(CamStreamSavingWidget)
        self.StreamStatusStackedWidget.setCurrentIndex(0)

        CamStreamBusyWidget.layout = QGridLayout()
        CamStreamBusylabel = QLabel()
        CamStreamBusylabel.setFixedHeight(35)
        CamStreamBusylabel.setAlignment(Qt.AlignVCenter)
        with Icons.Path("progressbar.gif") as path:
            self.StreamBusymovie = QMovie(path)

        CamStreamBusylabel.setMovie(self.StreamBusymovie)
        CamStreamBusyWidget.layout.addWidget(CamStreamBusylabel, 0, 1)

        self.CamStreamingLabel = QLabel("Recording")
        self.CamStreamingLabel.setFixedWidth(135)
        self.CamStreamingLabel.setStyleSheet(
            "QLabel { color : #208000; font: Times New Roman;}"
        )
        self.CamStreamingLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        CamStreamBusyWidget.layout.addWidget(self.CamStreamingLabel, 0, 0)
        CamStreamBusyWidget.setLayout(CamStreamBusyWidget.layout)

        # === Saving prograssbar ===
        CamStreamSavingWidget.layout = QGridLayout()
        CamStreamSavingWidget.layout.addWidget(
            QLabel("File saving progress:"), 0, 0
        )
        self.CamStreamSaving_progressbar = QProgressBar(self)
        self.CamStreamSaving_progressbar.setMaximumWidth(250)
        self.CamStreamSaving_progressbar.setMaximum(100)
        self.CamStreamSaving_progressbar.setStyleSheet(
            """
            QProgressBar {
                color: black;
                border: 1px solid grey;
                border-radius:3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #E6E6FA;
                width: 5px;
                margin: 1px;
            }
            """
        )
        CamStreamSavingWidget.layout.addWidget(
            self.CamStreamSaving_progressbar, 0, 1, 1, 4
        )
        CamStreamSavingWidget.setLayout(CamStreamSavingWidget.layout)

        CameraAcquisitionTab_2.layout.addWidget(
            self.StreamStatusStackedWidget, 6, 0, 1, 4
        )
        self.CamStreamActionContainer.setLayout(CamStreamActionLayout)

        CameraAcquisitionTab_2.layout.addWidget(
            self.CamStreamActionContainer, 0, 0, 5, 4
        )

        CameraAcquisitionTab_2.setLayout(CameraAcquisitionTab_2.layout)

        CameraAcquisitionTab.addTab(CameraAcquisitionTab_1, "Live")
        CameraAcquisitionTab.addTab(CameraAcquisitionTab_2, "Stream")
        CameraAcquisitionTab.setStyleSheet(
            """
            QTabBar {
                width: 200px;
                font-size: 8pt;
                font: bold;
                color: #003366
            }
            """
        )
        self.AcquisitionROIstackedWidget.addWidget(CameraAcquisitionTab)

        """
        # === Check ROI ===
        """
        ShowROIWidgetContainer = QGroupBox()
        ShowROIWidgetContainerLayout = QGridLayout()

        self.ShowROIWidget = pg.ImageView()
        self.ShowROIitem = self.ShowROIWidget.getImageItem()  # setLevels
        self.ShowROIview = self.ShowROIWidget.getView()
        self.ShowROIitem.setAutoDownsample("subsample")

        self.ShowROIWidget.ui.roiBtn.hide()
        self.ShowROIWidget.ui.menuBtn.hide()
        self.ShowROIWidget.ui.normGroup.hide()
        self.ShowROIWidget.ui.roiPlot.hide()
        self.ShowROIWidget.ui.histogram.hide()

        ShowROIWidgetContainerLayout.addWidget(self.ShowROIWidget, 1, 0)
        ShowROIWidgetContainer.setLayout(ShowROIWidgetContainerLayout)

        self.AcquisitionROIstackedWidget.addWidget(ShowROIWidgetContainer)
        self.AcquisitionROIstackedWidget.setCurrentIndex(0)
        CameraAcquisitionLayout.addWidget(
            self.AcquisitionROIstackedWidget, 2, 0
        )

        CameraAcquisitionContainer.setLayout(CameraAcquisitionLayout)

        # === Live Screen ===
        # Initiating an imageview object for the main Livescreen. Hiding the
        # pre existing ROI and menubuttons.
        LiveWidgetContainer = QGroupBox()
        LiveWidgetContainer.setMinimumHeight(920)
        LiveWidgetContainer.setMinimumWidth(950)
        self.LiveWidgetLayout = QGridLayout()

        self.LiveWidget = pg.ImageView()
        self.Live_item = self.LiveWidget.getImageItem()  # setLevels
        self.Live_view = self.LiveWidget.getView()
        self.Live_item.setAutoDownsample(True)

        self.LiveWidget.ui.roiBtn.hide()
        self.LiveWidget.ui.menuBtn.hide()
        self.LiveWidget.ui.normGroup.hide()
        self.LiveWidget.ui.roiPlot.hide()

        self.LiveWidgetLayout.addWidget(self.LiveWidget, 1, 0)

        LiveWidgetContainer.setLayout(self.LiveWidgetLayout)
        main_layout.addWidget(LiveWidgetContainer, 0, 1, 2, 2)

        MainWinCentralWidget.setLayout(main_layout)
        self.setCentralWidget(MainWinCentralWidget)

        # === Once open GUI, try to connect the camera ===
        try:
            self.ConnectCamera()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)

        # Create a vertical layout for the left side containers
        left_layout = QtWidgets.QVBoxLayout()

        left_layout.addWidget(CameraSettingContainer)
        left_layout.addWidget(tiffImportContainer)
        left_layout.addWidget(CameraImageInspectionContainer)
        left_layout.addWidget(CameraAcquisitionContainer)

        # Add a stretch at the end to push any remaining space below the lowest
        # container
        left_layout.addStretch()

        # Add the left layout to the main layout
        main_layout.addLayout(left_layout, 0, 0, 2, 1)

        # Add the LiveWidgetContainer to the main layout
        main_layout.addWidget(LiveWidgetContainer, 0, 1, 2, 2)

        MainWinCentralWidget.setLayout(main_layout)
        self.setCentralWidget(MainWinCentralWidget)

        """
        # === END of GUI ===
        """

    def ConnectCamera(self):
        """
        # Initialization of the camera.
        # Load dcamapi.dll version: 19.12.641.5901
        """

        files = importlib.resources.files(sys.modules[__package__])
        traversable = files.joinpath("19_12/dcamapi.dll")
        with importlib.resources.as_file(traversable) as path:
            self.dcam = ctypes.WinDLL(str(path))

        paraminit = HamamatsuDCAM.DCAMAPI_INIT(0, 0, 0, 0, None, None)
        paraminit.size = ctypes.sizeof(paraminit)
        error_code = self.dcam.dcamapi_init(
            ctypes.byref(paraminit)
        )  # TODO unused
        # if (error_code != DCAMERR_NOERROR):
        # raise DCAMException(
        # f"DCAM initialization failed with error code {error_code}")

        n_cameras = paraminit.iDeviceCount

        logging.info(f"Found: {n_cameras} cameras")

        if n_cameras > 0:
            # Initialization of the camera
            self.hcam = HamamatsuDCAM.HamamatsuCameraMR(camera_id=0)
            self.CamStatusLabel.setText(self.hcam.getModelInfo(0))

            # Set the camera to the default settings
            self.hcam.setPropertyValue("defect_correct_mode", 2)
            self.hcam.setPropertyValue("readout_speed", 2)
            self.hcam.setPropertyValue("binning", "1x1")

            self.CamExposureTime = self.hcam.getPropertyValue("exposure_time")[
                0
            ]
            self.CamExposureTimeText = str(self.CamExposureTime).replace(
                ".", "p"
            )
            self.CamExposureBox.setValue(round(self.CamExposureTime, 5))

            self.GetKeyCameraProperties()

            if self.subarray_hsize == 2048 and self.subarray_vsize == 2048:
                self.hcam.setPropertyValue("subarray_mode", "OFF")
                self.SubArrayModeSwitchButton.setChecked(False)
            else:
                self.hcam.setPropertyValue("subarray_mode", "ON")
                self.SubArrayModeSwitchButton.setChecked(True)

            self.UpdateHamamatsuSpecsLabels()

            # Get the trigger active button updated
            if self.trigger_source == "INTERNAL":
                self.TriggerButton_1.setChecked(True)
            elif self.trigger_source == "EXTERNAL":
                self.TriggerButton_2.setChecked(True)
            elif self.trigger_source == "MASTER PULSE":
                self.TriggerButton_3.setChecked(True)
            # Get the trigger button updated
            if self.trigger_active == "EDGE":
                self.ExternTriggerSignalComboBox.setCurrentIndex(1)
            elif self.trigger_active == "LEVEL":
                self.ExternTriggerSignalComboBox.setCurrentIndex(2)
            elif self.trigger_active == "SYNCREADOUT":
                self.ExternTriggerSignalComboBox.setCurrentIndex(3)

            # Toggle the switch button.
            self.cam_connect_button.setChecked(True)

    def DisconnectCamera(self):
        self.hcam.shutdown()
        self.dcam.dcamapi_uninit()
        self.CamStatusLabel.setText("Camera disconnected.")

    def cam_connect_switch(self):

        if self.cam_connect_button.isChecked():
            try:
                self.ConnectCamera()
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.cam_connect_button.setChecked(False)
        else:
            self.DisconnectCamera()

        """
        # Properties Settings
        """

    def ListCameraProperties(self):

        logging.info("Supported properties:")
        props = self.hcam.getProperties()
        for i, id_name in enumerate(sorted(props.keys())):
            [p_value, p_type] = self.hcam.getPropertyValue(id_name)
            p_rw = self.hcam.getPropertyRW(id_name)
            read_write = ""
            if p_rw[0]:
                read_write += "read"
            if p_rw[1]:
                read_write += ", write"
            logging.info(
                f"{i}) {id_name} = {p_value} type is: {p_type}, {read_write}"
            )
            text_values = self.hcam.getPropertyText(id_name)
            if len(text_values) > 0:
                logging.info("          option / value")
                for key in sorted(text_values, key=text_values.get):
                    logging.info(f"         {key}/{text_values[key]}")

    def GetKeyCameraProperties(self):
        params = [
            "internal_frame_rate",
            "timing_readout_time",
            "exposure_time",
            "subarray_hsize",
            "subarray_hpos",
            "subarray_vsize",
            "subarray_vpos",
            "subarray_mode",
            "image_framebytes",
            "buffer_framebytes",
            "trigger_source",
            "trigger_active",
        ]

        self.metaData = "Hamamatsu C13440-20CU "

        for param in params:
            if param == "exposure_time":
                self.CamExposureTime = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_exposure_time" + str(self.CamExposureTime)
            if param == "subarray_hsize":
                self.subarray_hsize = self.hcam.getPropertyValue(param)[0]
                self.ROI_hsize_spinbox.setValue(self.subarray_hsize)
                self.metaData += "subarray_hsize" + str(self.subarray_hsize)
            if param == "subarray_hpos":
                self.subarray_hpos = self.hcam.getPropertyValue(param)[0]
                self.ROI_hpos_spinbox.setValue(self.subarray_hpos)
                self.metaData += "subarray_hpos" + str(self.subarray_hpos)
            if param == "subarray_vsize":
                self.subarray_vsize = self.hcam.getPropertyValue(param)[0]
                self.ROI_vsize_spinbox.setValue(self.subarray_vsize)
                self.metaData += "subarray_vsize" + str(self.subarray_vsize)
            if param == "subarray_vpos":
                self.subarray_vpos = self.hcam.getPropertyValue(param)[0]
                self.ROI_vpos_spinbox.setValue(self.subarray_vpos)
                self.metaData += "subarray_vpos" + str(self.subarray_vpos)
            if param == "internal_frame_rate":
                self.internal_frame_rate = self.hcam.getPropertyValue(param)[0]
                self.metaData += "internal_frame_rate" + str(
                    self.internal_frame_rate
                )
            if param == "image_framebytes":
                self.image_framebytes = self.hcam.getPropertyValue(param)[0]
                self.metaData += "image_framebytes" + str(
                    self.image_framebytes
                )
            if param == "buffer_framebytes":
                self.buffer_framebytes = self.hcam.getPropertyValue(param)[0]
                self.metaData += "buffer_framebytes" + str(
                    self.buffer_framebytes
                )
            if param == "timing_readout_time":
                self.timing_readout_time = self.hcam.getPropertyValue(param)[0]
                self.metaData += "timing_readout_time" + str(
                    self.timing_readout_time
                )
            if param == "trigger_source":
                if self.hcam.getPropertyValue(param)[0] == 1:
                    self.trigger_source = "INTERNAL"
                elif self.hcam.getPropertyValue(param)[0] == 2:
                    self.trigger_source = "EXTERNAL"
                elif self.hcam.getPropertyValue(param)[0] == 4:
                    self.trigger_source = "MASTER PULSE"
            if param == "trigger_active":
                if self.hcam.getPropertyValue(param)[0] == 1:
                    self.trigger_active = "EDGE"
                elif self.hcam.getPropertyValue(param)[0] == 2:
                    self.trigger_active = "LEVEL"
                elif self.hcam.getPropertyValue(param)[0] == 3:
                    self.trigger_active = "SYNCREADOUT"

    def UpdateHamamatsuSpecsLabels(self):
        # Get the frame rate and update in the tag
        self.internal_frame_rate = self.hcam.getPropertyValue(
            "internal_frame_rate"
        )[0]
        self.CamFPSLabel.setText(
            "Frame rate: {}".format(round(self.internal_frame_rate, 2))
        )

        # Get the exposure time
        self.CamExposureTime = self.hcam.getPropertyValue("exposure_time")[0]
        self.CamExposureTimeLabel.setText(
            "Exposure time: {}".format(round(self.CamExposureTime, 5))
        )

        # Get the Readout time and update in the tag
        self.timing_readout_time = self.hcam.getPropertyValue(
            "timing_readout_time"
        )[0]
        self.CamReadoutTimeLabel.setText(
            "Readout speed: {}".format(round(1 / self.timing_readout_time, 2))
        )

    def ReadoutSpeedSwitchEvent(self):
        """Set the readout speed.

        Default is fast, corresponding to 2 in "readout_speed".
        """
        if self.ReadoutSpeedSwitchButton.isChecked():
            self.hcam.setPropertyValue("defect_correct_mode", 2)
        else:
            self.hcam.setPropertyValue("defect_correct_mode", 1)

    def DefectCorrectionSwitchEvent(self):
        """Switch defect correction.

        There are a few pixels in CMOS image sensor that have slightly higher
        readout noise performance compared to surrounding pixels.
        And the extended exposures may cause a few white spots which is caused
        by failure in part of the silicon wafer in CMOS image sensor.
        The camera has real-time variant pixel correction features to improve
        image quality.
        The correction is performed in real-time without sacrificing the
        readout speed at all. This function can be turned ON and OFF.
        (Default is ON)
        User can choose the correction level for white spots depend on the
        exposure time.
        """
        if self.DefectCorrectionButton.isChecked():
            self.hcam.setPropertyValue("readout_speed", 1)
        else:
            self.hcam.setPropertyValue("readout_speed", 2)

    def SubArrayModeSwitchEvent(self):
        if self.SubArrayModeSwitchButton.isChecked():
            self.hcam.setPropertyValue("subarray_mode", "ON")
        else:
            self.setRoiParameters("OFF", 2048, 2048, 0, 0)
            self.allowUserInputForExposure = False
            self.SetExposureTimeFromCamera()

    def SetExposureTimeFromCamera(self):
        # Ensure the live update interval is not too short
        readout_time = self.hcam.getPropertyValue("timing_readout_time")[0]

        if self.allowUserInputForExposure:
            ui_exposure_time = self.CamExposureBox.value()

            if ui_exposure_time < readout_time:
                logging.warning(
                    "Trying to set an exposure time that's less than the"
                    "readout time. Setting exposure time to readout time."
                )
                ui_exposure_time = readout_time

            self.CamExposureBox.setValue(ui_exposure_time)
            self.hcam.setPropertyValue("exposure_time", ui_exposure_time)
            self.live_update_interval = (
                self.minimum_live_update_interval
                if ui_exposure_time < self.minimum_live_update_interval
                else ui_exposure_time
            )
        else:
            self.hcam.setPropertyValue("exposure_time", readout_time)
            self.live_update_interval = (
                self.minimum_live_update_interval
                if readout_time < self.minimum_live_update_interval
                else readout_time
            )
            self.allowUserInputForExposure = True

        self.UpdateHamamatsuSpecsLabels()

    def SetBinning(self):
        if self.BinningButtongroup.checkedId() == 1:
            self.hcam.setPropertyValue("binning", "1x1")
        elif self.BinningButtongroup.checkedId() == 2:
            self.hcam.setPropertyValue("binning", "2x2")
        elif self.BinningButtongroup.checkedId() == 3:
            self.hcam.setPropertyValue("binning", "4x4")

    def SetPixelType(self):
        if self.PixelTypeButtongroup.checkedId() == 1:
            self.hcam.setPropertyValue("buffer_pixel_type", "MONO8")
        elif self.PixelTypeButtongroup.checkedId() == 2:
            self.hcam.setPropertyValue("buffer_pixel_type", "MONO12")
        elif self.PixelTypeButtongroup.checkedId() == 3:
            self.hcam.setPropertyValue("buffer_pixel_type", "MONO16")

    def SetTimingTrigger(self):
        if self.TriggerButtongroup.checkedId() == 1:
            self.hcam.setPropertyValue("trigger_source", "INTERNAL")
        elif self.TriggerButtongroup.checkedId() == 2:
            self.hcam.setPropertyValue("trigger_source", "EXTERNAL")
        elif self.TriggerButtongroup.checkedId() == 3:
            self.hcam.setPropertyValue("trigger_source", "MASTER PULSE")

    # https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SCAS0098E_synchronization.pdf
    def SetTriggerActive(self):
        if self.ExternTriggerSignalComboBox.currentText() == "LEVEL":
            self.hcam.setPropertyValue("trigger_active", "LEVEL")
        elif self.ExternTriggerSignalComboBox.currentText() == "EDGE":
            self.hcam.setPropertyValue("trigger_active", "EDGE")
        elif self.ExternTriggerSignalComboBox.currentText() == "SYNCREADOUT":
            self.hcam.setPropertyValue("trigger_active", "SYNCREADOUT")

        """
        # ROI functions
        """

    def ShowROISelector(self):
        if self.ShowROISelectorButton.isChecked():
            self.ShowROIImgButton.setEnabled(True)
            self.ROIselector_ispresented = True

            # Wait for ImageView to update a full-sized image
            time.sleep(0.1)

            ROIpen = QPen()  # creates a default pen
            ROIpen.setStyle(Qt.DashDotLine)
            ROIpen.setWidth(0.5)
            ROIpen.setBrush(QColor(0, 191, 255))

            try:  # Initialize the position and size of the ROI widget.
                if (
                    self.hcam.getPropertyValue("subarray_hsize")[0] == 2048
                    and self.hcam.getPropertyValue("subarray_vsize")[0] == 2048
                ):

                    if (
                        self.ROI_vpos_spinbox.value() == 0
                        and self.ROI_vsize_spinbox.value() == 2048
                    ):
                        # If it's the first time opening ROI selector, respawn
                        # it at a imageview center.
                        self.ROIitem = pg.RectROI(
                            [924, 924],
                            [200, 200],
                            centered=True,
                            sideScalers=True,
                            pen=ROIpen,
                        )
                        # Create text object, use HTML tags to specify
                        # color/size
                        self.ROIitemText = pg.TextItem(
                            html="""
                                <div style="text-align: center">
                                    <span style="color: #FFF;"
                                        >Estimated max fps: </span>
                                    <span style="color: #FF0; font-size: 10pt;"
                                        >0</span>
                                </div>
                            """,
                            anchor=(1, 1),
                        )
                        self.ROIitemText.setPos(924, 924)

                    else:
                        # If in the ROI position spinboxes there are numbers
                        # left from last ROI selection
                        self.ROIitem = pg.RectROI(
                            [
                                self.ROI_hpos_spinbox.value(),
                                self.ROI_vpos_spinbox.value(),
                            ],
                            [
                                self.ROI_hsize_spinbox.value(),
                                self.ROI_vsize_spinbox.value(),
                            ],
                            centered=True,
                            sideScalers=True,
                            pen=ROIpen,
                        )
                        # Create text object, use HTML tags to specify
                        # color/size
                        self.ROIitemText = pg.TextItem(
                            html="""
                                <div style="text-align: center">
                                    <span style="color: #FFF;"
                                        >Estimated max fps: </span>
                                    <span style="color: #FF0; font-size: 10pt;"
                                        >0</span>
                                </div>
                            """,
                            anchor=(1, 1),
                        )
                        self.ROIitemText.setPos(
                            self.ROI_hpos_spinbox.value(),
                            self.ROI_vpos_spinbox.value(),
                        )

                else:  # If the camera is already in subarray mode
                    self.ROIitem = pg.RectROI(
                        [
                            self.hcam.getPropertyValue("subarray_hpos")[0],
                            self.hcam.getPropertyValue("subarray_vpos")[0],
                        ],
                        [
                            self.hcam.getPropertyValue("subarray_hsize")[0],
                            self.hcam.getPropertyValue("subarray_vsize")[0],
                        ],
                        centered=True,
                        sideScalers=True,
                        pen=ROIpen,
                    )
                    # Create text object, use HTML tags to specify color/size
                    self.ROIitemText = pg.TextItem(
                        html="""
                                <div style="text-align: center">
                                    <span style="color: #FFF;"
                                        >Estimated max fps: </span>
                                    <span style="color: #FF0; font-size: 10pt;"
                                        >0</span>
                                </div>
                            """,
                        anchor=(1, 1),
                    )
                    self.ROIitemText.setPos(
                        self.hcam.getPropertyValue("subarray_hpos")[0],
                        self.hcam.getPropertyValue("subarray_vpos")[0],
                    )
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.ROIitem = pg.RectROI(
                    [0, 0],
                    [200, 200],
                    centered=True,
                    sideScalers=True,
                    pen=ROIpen,
                )
                # Create text object, use HTML tags to specify color/size
                self.ROIitemText = pg.TextItem(
                    html="""
                                <div style="text-align: center">
                                    <span style="color: #FFF;"
                                        >Estimated max fps: </span>
                                    <span style="color: #FF0; font-size: 10pt;"
                                        >0</span>
                                </div>
                            """,
                    anchor=(0, 0),
                )

            self.Live_view.addItem(self.ROIitem)  # add ROIs to main image
            self.ROIitem.maxBounds = QRectF(0, 0, 2048, 2048)
            # setting the max ROI bounds to be within the camera resolution

            self.ROIitem.sigRegionChanged.connect(
                self.update_ROI_spinbox_coordinates
            )
            # This function ensures the spinboxes show the actual roi
            # coordinates

            # Note that clicking is disabled by default to prevent stealing
            # clicks from objects behind the ROI.
            # self.ROIitem.setAcceptedMouseButtons(Qt.LeftButton)
            # self.ROIitem.sigClicked.connect(self.ShowROIImage)

            self.Live_view.addItem(self.ROIitemText)
        else:
            self.ShowROIImgButton.setEnabled(False)
            self.Live_view.removeItem(self.ROIitem)
            self.Live_view.removeItem(self.ROIitemText)
            self.ROIselector_ispresented = False

    # === Center ROI part from Douwe ===
    def set_roi_flag(self):
        if self.center_roiButton.isChecked():
            self.ROI_vpos_spinbox.setReadOnly(True)
            self.center_frame = 0.5 * 2048
            """
            I've put the center frame in the set_roi_flag so it automatically
            adjusts to the number of pixels (which is dependent on the binning
            settings for example.)
            """
            self.ROIitem.sigRegionChanged.connect(lambda: self.center_roi())
            # setting the ROI to the center every move
            """
            If the ROI centering performs poorly it is also possible to use the
            sigRegionChanged() function. I like this better for now.
            """

        else:
            self.ROI_vpos_spinbox.setReadOnly(False)
            self.ROIitem.sigRegionChanged.disconnect()
            """
            I do not know how to disconnect one specific function, so I
            disconnect both and then reconnect the
            update_ROI_spinbox_coordinates function.
            """
            self.ROIitem.sigRegionChanged.connect(
                self.update_ROI_spinbox_coordinates
            )

    def update_ROI_spinbox_coordinates(self):
        self.ROI_hpos = int(self.ROIitem.pos()[0])
        self.ROI_vpos = int(self.ROIitem.pos()[1])
        self.ROI_vsize = int(self.ROIitem.size()[1])
        self.ROI_hsize = int(self.ROIitem.size()[0])

        self.ROI_hpos_spinbox.setValue(self.ROI_hpos)
        self.ROI_vpos_spinbox.setValue(self.ROI_vpos)
        self.ROI_hsize_spinbox.setValue(self.ROI_hsize)
        self.ROI_vsize_spinbox.setValue(self.ROI_vsize)

        self.update_ROI_estimateMaxFps()

    def update_ROI_estimateMaxFps(self):
        self.ROIupperRowDis = abs(1024 - self.ROI_vpos)
        self.ROIlowerRowDis = abs(1024 - self.ROI_vpos - self.ROI_vsize)
        self.ROIEstimatedMaxFPS = 1 / (
            max(self.ROIupperRowDis, self.ROIlowerRowDis) * 0.0000097
        )

        try:
            self.Live_view.removeItem(self.ROIitemText)
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)

        # Create text object, use HTML tags to specify color/size
        self.ROIitemText = pg.TextItem(
            html="""
                <div style="text-align: center">
                    <span style="color: #FFF;"
                        >Estimated max fps: </span>
                    <span style="color: #FF0; font-size: 10pt;"
                        >{}</span></div>
            """.format(
                round(self.ROIEstimatedMaxFPS, 2)
            ),
            anchor=(1, 1),
        )
        self.ROIitemText.setPos(self.ROI_hpos, self.ROI_vpos)
        self.Live_view.addItem(self.ROIitemText)

    def spin_value_changed(self):
        # Update the ROI item size according to spinbox values.
        if (
            self.ROI_hsize_spinbox.value() != self.ROI_hsize
            or self.ROI_vsize_spinbox.value() != self.ROI_vsize
        ):

            self.ROIitem.setSize(
                [
                    self.ROI_hsize_spinbox.value(),
                    self.ROI_vsize_spinbox.value(),
                ]
            )

        # Update the ROI item position according to spinbox values.
        if self.center_roiButton.isChecked():
            if self.ROI_hpos_spinbox.value() != self.ROI_hpos:
                self.ROIitem.setPos(self.ROI_hpos_spinbox.value())
        else:
            if (
                self.ROI_hpos_spinbox.value() != self.ROI_hpos
                or self.ROI_vpos_spinbox.value() != self.ROI_vpos
            ):
                self.ROIitem.setPos(
                    self.ROI_hpos_spinbox.value(),
                    self.ROI_vpos_spinbox.value(),
                )

        self.UpdateHamamatsuSpecsLabels()
        self.update_ROI_estimateMaxFps()

    # === ROI centering functions ===
    def center_roi(self):

        self.v_center = int(self.center_frame - 0.5 * self.ROI_vsize)
        if self.ROI_vpos != self.v_center:
            self.ROIitem.setPos(self.ROI_hpos, self.v_center)
            self.update_ROI_spinbox_coordinates()

    def SetROI(self):
        # If the camera is live, stop the live acquisition
        if self.cameraIsLive:
            self.StopLIVE()

        # Remove the ROI
        self.Live_view.removeItem(self.ROIitem)
        self.Live_view.removeItem(self.ROIitemText)
        self.ROIselector_ispresented = False

        self.ROI_hsize = self.ROI_hsize_spinbox.value()
        self.ROI_vsize = self.ROI_vsize_spinbox.value()
        self.ROI_hpos = self.ROI_hpos_spinbox.value()
        self.ROI_vpos = self.ROI_vpos_spinbox.value()

        """The Orca Flash 4.0 requires the ROI size to be a multiple of 4."""

        def multiple_of_4(value):
            if value == 0:  # If the ROI size is 0, return 4
                return 4
            return 4 * int(value / 4)

        self.ROI_hsize = multiple_of_4(self.ROI_hsize)
        self.ROI_vsize = multiple_of_4(self.ROI_vsize)
        self.ROI_hpos = multiple_of_4(self.ROI_hpos)
        self.ROI_vpos = multiple_of_4(self.ROI_vpos)

        if self.ROI_hsize == 2048 and self.ROI_vsize == 2048:
            self.setRoiParameters("OFF", 2048, 2048, 0, 0)
            self.SubArrayModeSwitchButton.setChecked(False)

            self.allowUserInputForExposure = False
            self.SetExposureTimeFromCamera()
        else:
            self.setRoiParameters(
                "OFF",
                self.ROI_hsize,
                self.ROI_vsize,
                self.ROI_hpos,
                self.ROI_vpos,
            )
            self.hcam.setPropertyValue("subarray_mode", "ON")
            self.SubArrayModeSwitchButton.setChecked(True)

            self.allowUserInputForExposure = False
            self.SetExposureTimeFromCamera()

        self.UpdateHamamatsuSpecsLabels()
        self.ShowROISelectorButton.setChecked(False)
        self.ShowROIImgButton.setEnabled(False)
        self.center_roiButton.setChecked(False)

        # Log key properties after setting the ROI
        logging.info(
            f"ROI set to: hsize={self.ROI_hsize}, vsize={self.ROI_vsize}, "
            f"hpos={self.ROI_hpos}, vpos={self.ROI_vpos}"
        )
        logging.info(
            f"Subarray mode: {self.hcam.getPropertyValue('subarray_mode')[0]}"
        )
        logging.info(
            "Internal frame rate: "
            f"{self.hcam.getPropertyValue('internal_frame_rate')[0]}"
        )
        logging.info(
            f"Readout speed: "
            f"{self.hcam.getPropertyValue('timing_readout_time')[0]}"
        )
        logging.info(
            f"Exposure time: {self.hcam.getPropertyValue('exposure_time')[0]}"
        )

    def setRoiParameters(self, subarray_mode, hsize, vsize, hpos, vpos):
        self.hcam.setPropertyValue("subarray_mode", subarray_mode)
        self.hcam.setPropertyValue("subarray_hsize", hsize)
        self.hcam.setPropertyValue("subarray_vsize", vsize)
        self.hcam.setPropertyValue("subarray_hpos", hpos)
        self.hcam.setPropertyValue("subarray_vpos", vpos)

        self.subarray_vsize = self.ROI_vsize
        self.subarray_hsize = self.ROI_hsize

        self.ROI_hpos_spinbox.setValue(hpos)
        self.ROI_vpos_spinbox.setValue(vpos)
        self.ROI_hsize_spinbox.setValue(hsize)
        self.ROI_vsize_spinbox.setValue(vsize)

    def SetShowROIImgSwitch(self):
        if self.ShowROIImgButton.isChecked():
            self.AcquisitionROIstackedWidget.setCurrentIndex(1)
            self.ShowROIImgSwitch = True
        else:
            self.AcquisitionROIstackedWidget.setCurrentIndex(0)
            self.ShowROIImgSwitch = False

        """
        # LIVE functions
        """

    def AutoLevelSwitchEvent(self):
        if self.LiveAutoLevelSwitchButton.isChecked():
            self.Live_item_autolevel = True
        else:
            self.Live_item_autolevel = False

        logging.info(f"AutoLevelSwitchEvent: {self.Live_item_autolevel}")

    def LiveSwitchEvent(self):
        if self.LiveButton.isChecked():
            try:
                self.ResetLiveImgView()
            except Exception as exc:
                logging.error("Error resetting live image view:", exc_info=exc)

            self.live_thread = QThread()
            self.live_worker = LiveWorker(self.hcam, self.live_update_interval)
            self.live_worker.moveToThread(self.live_thread)

            self.live_thread.started.connect(self.live_worker.run)
            self.live_worker.update_image.connect(self.refresh_live_image)
            self.live_worker.finished.connect(self.on_live_finished)
            self.live_worker.error.connect(self.on_live_error)
            self.live_worker.finished.connect(self.live_thread.quit)
            self.live_worker.finished.connect(self.live_worker.deleteLater)
            self.live_thread.finished.connect(self.live_thread.deleteLater)

            self.live_thread.start()

            self.SubArrayModeSwitchButton.setEnabled(False)
        else:
            self.StopLIVE()
            self.SubArrayModeSwitchButton.setEnabled(True)

    def StopLIVE(self):
        if hasattr(self, "live_worker") and hasattr(self, "live_thread"):
            if self.live_thread and self.live_thread.isRunning():
                self.live_worker.stop()
                self.live_thread.quit()
                self.live_thread.wait()
            else:
                logging.info("Live thread is not running.")
        else:
            logging.info("Live worker or thread does not exist.")
        logging.info("Live acquisition manually stopped.")

    def on_live_finished(self):
        self.cameraIsLive = False

    def on_live_error(self, error_message):
        logging.error(f"Error: {error_message}")

    def SaveLiveImg(self):
        """Save the latest live image from RAM."""
        self.GetKeyCameraProperties()

        with skimtiff.TiffWriter(
            self.get_file_dir(), append=False, imagej=False
        ) as tif:
            tif.write(self.Live_image, description=self.metaData)

        logging.info(f"Live image saved to {self.get_file_dir()}")
        logging.info(f"Metadata: {self.metaData}")

    def update_displayed_image(
        self, tiff_image, reduce=False, set_viewbox=True
    ):
        """Updates the screen with the latest image."""
        try:
            if reduce:
                if (
                    self.subarray_vsize == 2048
                    and self.subarray_hsize == 2048
                    and self.ROIselector_ispresented is False
                ):
                    tiff_image = block_reduce(
                        tiff_image,
                        block_size=(2, 2),
                        func=np.mean,
                        cval=np.mean(tiff_image),
                    )

            self.Live_item.setImage(
                tiff_image, autoLevels=self.Live_item_autolevel
            )
            self.Live_image = tiff_image
            self.Live_item.resetTransform()

            # Get the ViewBox and adjust its range to fit the image
            self.view_box = self.Live_item.getViewBox()

            if self.view_box and set_viewbox:
                # Automatically scale the image to fit within the screen
                self.view_box.enableAutoRange()

                # Limit the zoom level to prevent excessive zooming out
                self.view_box.setLimits(
                    xMin=0,
                    xMax=tiff_image.shape[1],
                    yMin=0,
                    yMax=tiff_image.shape[0],
                )

            self.enable_pixel_coordinate_display()
            self.clearImageButton.setEnabled(True)

            # Update ROI checking screen
            if (
                hasattr(self, "ShowROIImgSwitch")
                and self.ShowROIImgSwitch is True
            ):
                self.ShowROIitem.setImage(
                    self.ROIitem.getArrayRegion(tiff_image, self.Live_item),
                    autoLevels=None,
                )

        except Exception as exc:
            logging.info("Error displaying TIFF image:", exc_info=exc)

    def refresh_live_image(self, image):
        self.Live_image = image
        self.update_displayed_image(self.Live_image)
        self.output_signal_SnapImg.emit(self.Live_image)
        self.enable_pixel_coordinate_display()

    def SnapImg(self):
        if not self.savingPathValid("SnapImg"):
            return

        if self.cameraIsStreaming:
            logging.info(
                "Trying to snap a picture while streaming. Stop the stream "
                "first."
            )
            return

        if hasattr(self, "live_worker") and self.live_worker.camera_is_live:
            self.StopLIVE()  # Stop live acquisition before snapping
            self.snap_thread = QThread()
            self.snap_worker = LiveWorker(self.hcam, self.live_update_interval)
            self.snap_worker.moveToThread(self.snap_thread)

            self.snap_thread.started.connect(self.snap_worker.snap)
            self.snap_worker.update_image.connect(self.refresh_live_image)
            self.snap_worker.finished.connect(self.on_snap_finished)
            self.snap_worker.error.connect(self.on_snap_error)
            self.snap_worker.finished.connect(self.snap_thread.quit)
            self.snap_worker.finished.connect(self.snap_worker.deleteLater)
            self.snap_thread.finished.connect(self.snap_thread.deleteLater)

            self.snap_thread.start()
        else:
            self.snap_thread = QThread()
            self.snap_worker = LiveWorker(self.hcam, self.live_update_interval)
            self.snap_worker.moveToThread(self.snap_thread)

            self.snap_thread.started.connect(self.snap_worker.snap)
            self.snap_worker.update_image.connect(self.refresh_live_image)
            self.snap_worker.finished.connect(self.on_snap_finished)
            self.snap_worker.error.connect(self.on_snap_error)
            self.snap_worker.finished.connect(self.snap_thread.quit)
            self.snap_worker.finished.connect(self.snap_worker.deleteLater)
            self.snap_thread.finished.connect(self.snap_thread.deleteLater)

            self.snap_thread.start()

    def on_snap_finished(self):
        self.LiveButton.setChecked(False)
        logging.info("Snap finished.")
        self.refresh_live_image(self.Live_image)

    def on_snap_error(self, error_message):
        logging.error(f"Error: {error_message}")

    def ResetLiveImgView(self):
        """Closes the widget nicely.

        Makes sure to clear the graphics scene and release memory.
        """
        self.LiveWidget.close()

        # Replot the imageview
        self.LiveWidget = pg.ImageView()
        self.Live_item = self.LiveWidget.getImageItem()  # setLevels
        self.Live_view = self.LiveWidget.getView()
        self.Live_item.setAutoDownsample(True)

        self.LiveWidget.ui.roiBtn.hide()
        self.LiveWidget.ui.menuBtn.hide()
        self.LiveWidget.ui.normGroup.hide()
        self.LiveWidget.ui.roiPlot.hide()

        self.LiveWidgetLayout.addWidget(self.LiveWidget, 1, 0)

    def closeEvent(self, event):
        try:
            self.hcam.shutdown()
            self.dcam.dcamapi_uninit()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
        self.close()

        """
        # STREAM functions
        """

    def UpdateBufferNumber(self):
        self.BufferNumber = (
            self.desired_fps_spinbox.value()
            * self.StreamTotalTime_spinbox.value()
        )
        self.StreamBufferTotalFrames_spinbox.setValue(self.BufferNumber)

    def desiredFpsValid(self):
        max_fps = 1 / self.hcam.getPropertyValue("timing_readout_time")[0]
        if self.desired_fps_spinbox.value() > max_fps:
            logging.warning(
                "Desired fps is higher than the max internal frame rate. "
                "Please decrease the ROI size or the desired fps."
            )
            return False
        elif (0 < self.desired_fps_spinbox.value()) and (
            self.desired_fps_spinbox.value() <= max_fps
        ):

            if self.TriggerButton_1.isChecked():
                desired_exposure_time = 1 / self.desired_fps_spinbox.value()
                logging.info(
                    "Desired fps is valid. Setting exposure time to "
                    f"{desired_exposure_time} s."
                )
                self.hcam.setPropertyValue(
                    "exposure_time", desired_exposure_time
                )
                self.CamExposureBox.setValue(round(desired_exposure_time, 5))

            return True

    def SetStreamSpecs(self):
        if not self.savingPathValid("SetStreamSpecs"):
            return

        if not self.desiredFpsValid():
            return

        self.UpdateHamamatsuSpecsLabels()

        if self.CamStreamActionContainer.isEnabled():
            self.CamStreamActionContainer.setEnabled(False)
            self.CamSaving_directory_textbox.setEnabled(False)
            self.startOrStopStreamButton.setEnabled(True)
        else:
            self.CamStreamActionContainer.setEnabled(True)
            self.CamSaving_directory_textbox.setEnabled(True)
            self.startOrStopStreamButton.setEnabled(False)

        # Set the number of buffers get prepared.
        self.BufferNumber = self.StreamBufferTotalFrames_spinbox.value()

        if self.StreamStopSignalComBox.currentText() == "Stop signal: Time":
            self.StopSignal = "Time"
            self.StreamDuration = self.StreamTotalTime_spinbox.value()
            self.hcam.acquisition_mode = "fixed_length"

        elif (
            self.StreamStopSignalComBox.currentText() == "Stop signal: Frames"
        ):
            self.StopSignal = "Frames"
            self.StreamDuration = -1
            self.hcam.acquisition_mode = "fixed_length"

        # Emit the stream_parameters signal with desired_fps and total_time
        stream_parameters = {
            "desired_fps": self.desired_fps_spinbox.value(),
            "total_time": self.StreamTotalTime_spinbox.value(),
        }
        self.stream_parameters.emit(stream_parameters)

    def StreamingSwitchEvent(self):
        if self.startOrStopStreamButton.isChecked():
            self.StreamBusymovie.start()
            self.StreamStatusStackedWidget.setCurrentIndex(1)
            with Icons.Path("STOP.png") as path:
                self.startOrStopStreamButton.setIcon(QIcon(path))
            self.StartStreamingThread()
        else:
            with Icons.Path("StartStreaming.png") as path:
                self.startOrStopStreamButton.setIcon(QIcon(path))
            self.StopStreamingThread()

    def StartStreamingThread(self):
        if not self.cameraIsStreaming and not self.cameraIsLive:
            self.streaming_thread = QThread()
            self.streaming_worker = StreamingWorker(
                self.hcam,
                self.StopSignal,
                self.BufferNumber,
                self.StreamDuration,
            )
            self.streaming_worker.moveToThread(self.streaming_thread)

            self.streaming_thread.started.connect(self.streaming_worker.run)
            self.streaming_worker.update_label.connect(
                self.update_streaming_label
            )
            self.streaming_worker.update_progress.connect(
                self.update_progress_bar
            )
            self.streaming_worker.finished.connect(self.on_streaming_finished)
            self.streaming_worker.error.connect(self.on_streaming_error)
            self.streaming_worker.streaming_finished.connect(
                lambda: self.StopStreaming(saveFile=True)
            )
            self.streaming_worker.finished.connect(self.streaming_thread.quit)
            self.streaming_worker.finished.connect(
                self.streaming_worker.deleteLater
            )
            self.streaming_thread.finished.connect(
                self.streaming_thread.deleteLater
            )

            self.streaming_thread.start()

        else:
            logging.warning(
                "Camera is already streaming or live. Stop the stream first."
            )

    def StopStreamingThread(self):
        if (
            hasattr(self, "streaming_worker")
            and hasattr(self, "streaming_thread")
            and self.streaming_thread.isRunning()
        ):
            self.streaming_worker.stop_streaming()
            self.streaming_thread.quit()
            self.streaming_thread.wait()
        logging.info("Streaming manually stopped.")

    def update_streaming_label(self, text):
        self.CamStreamingLabel.setText(text)

    def update_progress_bar(self, value):
        self.CamStreamSaving_progressbar.setValue(value)

    def on_streaming_finished(self):
        self.startOrStopStreamButton.setEnabled(False)
        self.startOrStopStreamButton.setIcon(
            QIcon("./Icons/StartStreaming.png")
        )
        self.CamStreamActionContainer.setEnabled(True)
        self.StreamStatusStackedWidget.setCurrentIndex(2)

        # Display the third frame if available
        if len(self.streaming_worker.video_list) >= 3:
            third_frame = self.streaming_worker.video_list[2]
            if self.streaming_worker.dims:
                third_frame = np.resize(
                    third_frame,
                    (
                        self.streaming_worker.dims[1],
                        self.streaming_worker.dims[0],
                    ),
                )
            self.refresh_live_image(third_frame)

    def on_streaming_error(self, error_message):
        logging.info(f"Error: {error_message}")

    def StopStreaming(self, saveFile):
        # Stop the acquisition
        self.AcquisitionEndTime = time.time()
        streamTime = self.AcquisitionEndTime - self.hcam.AcquisitionStartTime
        logging.info(f"Frames acquired: {self.streaming_worker.image_count}")
        logging.info(f"Total time is: {streamTime} s.")
        hz = round(self.streaming_worker.image_count / streamTime)
        logging.info(f"Estimated fps: {hz} hz.")

        self.hcam.stopAcquisition()
        self.cameraIsStreaming = False
        self.StreamBusymovie.stop()
        self.StreamStatusStackedWidget.setCurrentIndex(2)

        if saveFile:
            self.CamStreamSaving_progressbar.setValue(0)
            self.start_save_thread()

        self.startOrStopStreamButton.setEnabled(True)
        self.CamStreamActionContainer.setEnabled(True)
        self.StreamStatusStackedWidget.setCurrentIndex(0)
        self.CamStreamIsFree.setText(
            "Acquisition done. Frames acquired: {}.".format(
                self.streaming_worker.image_count
            )
        )

    def start_save_thread(self):
        self.GetKeyCameraProperties()
        self.save_thread = QThread()
        self.save_worker = SaveWorker(
            self.streaming_worker.video_list,
            self.streaming_worker.dims,
            self.streaming_worker.image_count,
            self.metaData,
            self.get_file_dir(),
        )

        self.save_worker.moveToThread(self.save_thread)

        self.save_worker.finished.connect(self.on_save_complete)
        self.save_worker.error.connect(self.on_save_error)
        self.save_worker.update_progress.connect(self.update_progress_bar)
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self.save_thread.deleteLater)

        self.save_thread.start()

    def on_save_complete(self):
        logging.info("File saved successfully.")
        self.StreamStatusStackedWidget.setCurrentIndex(0)

    def on_save_error(self, error_message):
        logging.warning(f"Error saving file: {error_message}")

    def SetSavingDirectory(self):
        self.saving_path = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                directory=self.default_folder
            )
        )
        self.CamSaving_directory_textbox.setText(self.saving_path)

    def update_savedirectory(self, new_directory):
        self.CamSaving_directory_textbox.setText(new_directory)
        logging.info(f"Camera save directory updated to: {new_directory}")

    def savingPathValid(self, function_name):
        if (
            not hasattr(self, "saving_path")
            or self.saving_path.strip() == ""
            or self.saving_path.strip() == "Saving folder"
            or self.CamSaving_filename_textbox.text().strip() == ""
            or self.CamSaving_filename_textbox.text().strip()
            == "Tiff file name"
        ):

            QMessageBox.warning(
                self,
                f"{function_name}",
                "Please set the saving path and/or Tiff file name first.",
            )
            return False
        else:
            return True

    def get_file_dir(self):
        return os.path.join(
            self.saving_path,
            self.CamSaving_filename_textbox.text()
            + "_"
            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            + ".tif",
        )

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

    """ Tiff import and analysis functions, added by Nike Celie 8-1-2025 """

    # Function to handle file selection
    def browseTiffFiles(self):
        # Open a file dialog to select a TIFF file
        options = QtWidgets.QFileDialog.Options()
        tiff_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select TIFF file",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=options,
        )

        if tiff_file:
            # Read the TIFF image using skimtiff
            tiff_image = skimtiff.imread(tiff_file)
            self.update_displayed_image(tiff_image)
            self.tiffDirectoryTextbox.setText(tiff_file)

    def clearImage(self):
        try:
            # Clear the image from the display
            self.Live_item.clear()

            # Clear the file path text box as well
            self.tiffDirectoryTextbox.clear()

            # Reset the pixel coordinate display
            self.x_label.setText("X-coordinate: _")
            self.y_label.setText("Y-coordinate: _")
            self.intensity_label.setText("Intensity: _")

            # Reset any other states (like the stored live image)
            self.Live_image = None

            # Disconnect the mouse click event handler to prevent updates
            # after clearing the image
            if self.mouse_click_connection is not None:
                try:
                    self.Live_item.scene().sigMouseClicked.disconnect(
                        self.mouse_click_connection
                    )
                except Exception as exc:
                    logging.info(
                        "Error disconnecting mouse click event:", exc_info=exc
                    )

                self.mouse_click_connection = None

            logging.info("Image cleared successfully.")
            self.clearImageButton.setEnabled(False)
            self.find_camera_registration_point.setEnabled(False)
            self._set_registration_params(False)
            self._remove_registration_gaussian(remove=True)

        except Exception as exc:
            logging.info("Error clearing image:", exc_info=exc)

    def enable_pixel_coordinate_display(self):
        if self.Live_image is None:
            logging.info("No image loaded for interaction.")
            return

        # Disconnect the previous connection if it exists
        if self.mouse_click_connection is not None:
            try:
                self.Live_item.scene().sigMouseClicked.disconnect(
                    self.mouse_click_connection
                )
            except Exception as exc:
                logging.info(
                    "Error disconnecting mouse click event:", exc_info=exc
                )

        # Connect the mouse click event to the handler
        self.mouse_click_connection = (
            self.Live_item.scene().sigMouseClicked.connect(
                self.handleMouseClick
            )
        )

    def handleMouseClick(self, event):
        if self.Live_image is None:
            logging.info("No image loaded for interaction.")
            return

        height, width = self.Live_image.shape[:2]
        self.mouseClickedAndValid = False

        # Get the mouse click position in scene coordinates
        scene_pos = event.scenePos()

        # Map scene coordinates to image pixel coordinates
        image_pos = self.Live_item.mapFromScene(scene_pos)
        self.x_pixel = int(image_pos.x())
        self.y_pixel = int(image_pos.y())

        if (
            hasattr(self, "ROI_hpos")
            and hasattr(self, "ROI_vpos")
            and self.ROI_hpos is not None
            and self.ROI_vpos is not None
        ):
            logging.info(
                f"Viewbox coordinates: ({self.x_pixel}, {self.y_pixel})"
            )

            self.x_pixel += self.ROI_hpos
            self.y_pixel += self.ROI_vpos

        self.pixel_intensity = self.Live_image[self.y_pixel, self.x_pixel]

        if 0 <= self.x_pixel < width and 0 <= self.y_pixel < height:
            self.x_label.setText(f"X-coordinate: {self.x_pixel}")
            self.y_label.setText(f"Y-coordinate: {self.y_pixel}")
            self.intensity_label.setText(f"Intensity: {self.pixel_intensity}")
            self.mouseClickedAndValid = True

            # Trigger point collection if contour creation is active
            if self.createContour:
                self.addContourPoint()
        else:
            logging.info("Click is outside the image bounds.")

    def createCustomContour(self):
        self.createContour = True
        self.userInputContourPoints = np.zeros(
            (2, self.contourSize), dtype=int
        )
        self.current_index = 0
        self.contourCreationButton.setEnabled(False)
        self.contourCreationProgress.setVisible(True)
        logging.info(
            f"Contour creation started. Collect {self.contourSize} points."
        )
        self.contourCreationProgress.setText("Waiting on user input.")

    def addContourPoint(self):
        if (
            self.mouseClickedAndValid
            and self.createContour
            and self.current_index < self.contourSize
        ):
            # Add the clicked point to the contour
            self.userInputContourPoints[:, self.current_index] = [
                self.x_pixel,
                self.y_pixel,
            ]
            self.current_index += 1
            self.x_label.setText(f"X-coordinate: {self.x_pixel}")
            self.y_label.setText(f"Y-coordinate: {self.y_pixel}")
            self.contourCreationProgress.setText(
                f"Point {self.current_index}/{self.contourSize} added: "
                f"({self.x_pixel}, {self.y_pixel})"
            )

            # Update the ROI contour dynamically
            self.drawROIContour()

            # Check if all points are collected
            if self.current_index == self.contourSize:
                logging.info("All contour points collected.")
                self.onContourComplete()

    def drawROIContour(self):
        # Use only the collected points to draw the contour
        collected_points = self.userInputContourPoints[
            :, : self.current_index
        ].T
        logging.info(collected_points)
        self.previous_positions = collected_points

        # Remove existing contour if it exists
        if (
            hasattr(self, "ROIContour")
            and self.ROIContour in self.view_box.addedItems
        ):
            self.view_box.removeItem(self.ROIContour)

        # Create the PolyLineROI for the contour
        self.ROIContour = pg.PolyLineROI(
            collected_points, closed=True, movable=True
        )

        # Add names to the handles
        for i, handle in enumerate(self.ROIContour.handles):
            handle["name"] = f"Point {i + 1}"

        self.initializeROIContour()

    def onHandleMoved(self):
        ROIHandlePositions = self.ROIContour.getLocalHandlePositions()
        current_positions = np.array(
            [
                [np.round(pos[1].x(), 0), np.round(pos[1].y(), 0)]
                for pos in ROIHandlePositions
            ]
        )

        # Track changes in real-time or finalized movement
        for i, (prev, curr) in enumerate(
            zip(self.previous_positions, current_positions)
        ):
            if not np.allclose(
                prev, curr
            ):  # Check if the position has changed
                if self.is_final_update:
                    logging.info(f"Point {i+1} finalized at {curr}")
                    self.contourCreationProgress.setText(
                        f"Point {i+1} moved to {curr}"
                    )
                else:
                    self.contourCreationProgress.setText(
                        f"Moving Point {i+1} {curr}"
                    )

        # Update the stored positions if movement is finalized
        if self.is_final_update:
            self.previous_positions = np.copy(current_positions)

    def initializeROIContour(self):
        self.is_final_update = (
            False  # Flag to differentiate between live and final updates
        )
        self.ROIContour.sigRegionChanged.connect(self.trackLiveUpdates)
        self.ROIContour.sigRegionChangeFinished.connect(self.finalizeUpdates)
        self.view_box.addItem(self.ROIContour)

    def trackLiveUpdates(self):
        self.is_final_update = False
        self.onHandleMoved()

    def finalizeUpdates(self):
        self.is_final_update = True
        self.onHandleMoved()

    def checkContourSizeInput(self):
        user_input = self.contourSizeInput.text().strip()
        inputIsValid = False
        self.contourCreationProgress.setVisible(False)
        self.saveContourButton.setVisible(False)

        try:
            if not user_input:
                inputIsValid = False
                self.contourCreationButton.setEnabled(False)
                return

            self.contourSize = int(user_input)

            if self.contourSize < 3:
                # QMessageBox.warning(self, "Error", "Please enter a positive
                # integer greater than 2.")
                inputIsValid = False
            else:
                inputIsValid = True
        except ValueError:
            QMessageBox.warning(
                self,
                "Error",
                "Invalid input! Please enter a positive integer greater than "
                "2.",
            )
            inputIsValid = False

        if inputIsValid:
            self.contourCreationButton.setEnabled(True)
        else:
            self.contourCreationButton.setEnabled(False)

    def checkContourIntensityInput(self):
        user_input = self.contourIntensityInput.text().strip()
        self.lower_intensity_bound, self.upper_intensity_bound = (
            self.image_analyzer.determine_thresholds(self.Live_image)
        )

        try:
            if not user_input:
                self.contourGenerationButton.setEnabled(False)
                return

            self.contour_intensity_input_value = int(user_input)

            if (
                self.lower_intensity_bound < self.contour_intensity_input_value
            ) and (
                self.contour_intensity_input_value < self.upper_intensity_bound
            ):
                self.contourGenerationButton.setEnabled(True)
            else:
                self.contourGenerationButton.setEnabled(False)
                return
        except ValueError:
            self.contourGenerationButton.setEnabled(False)
            QMessageBox.warning(
                self,
                "Error",
                "Invalid input! Please enter a valid intensity value.",
            )
            return

    def generateContour(self, zoom_in_on_roi=False):
        self.generated_contour = self.image_analyzer.find_contour(
            self.Live_image, self.contour_intensity_input_value, num_points=500
        )

        if zoom_in_on_roi:
            roi_generated_contour = self.image_analyzer.zoom_in_on_roi(
                None, self.generated_contour, factor=2
            )

            # Set the limit parameters for the view box
            x_min, x_max = roi_generated_contour[0]
            x_min, x_max = np.clip([x_min, x_max], 0, self.Live_image.shape[1])
            y_min, y_max = roi_generated_contour[1]
            y_min, y_max = np.clip([y_min, y_max], 0, self.Live_image.shape[0])

            # Set the limits for the view box
            self.view_box.setLimits(
                xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max
            )

            # Set the range to update the view box to the new limits
            self.view_box.setRange(
                xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0
            )

        # Reset the sliders
        self.contourSizeSlider.setValue(int(1.0 * 100))
        self.contourSmoothnessSlider.setValue(0)
        minimum_intensity = max(
            self.lower_intensity_bound,
            self.contour_intensity_input_value * 0.7,
        )
        maximum_intensity = min(
            self.upper_intensity_bound,
            self.contour_intensity_input_value * 1.3,
        )
        self.contourIntensitySlider.setMinimum(int(minimum_intensity))
        self.contourIntensitySlider.setMaximum(int(maximum_intensity))
        self.contourIntensitySlider.setValue(
            self.contour_intensity_input_value
        )

        # Plot the contour over the live image
        self.plotContour(self.generated_contour)

    def plotContour(self, contour):
        # Remove any existing contour plot
        if hasattr(self, "contour_plot") and self.contour_plot is not None:
            self.view_box.removeItem(self.contour_plot)

        # Extract the x and y coordinates of the contour
        contour_x = contour[:, 1]
        contour_y = contour[:, 0]

        # Plot the contour over the live image
        self.contour_plot = pg.PlotDataItem(contour_x, contour_y, pen="r")
        self.view_box.addItem(self.contour_plot)

        self.removeGeneratedContourButton.setVisible(True)
        self._setGeneratedContourWidgets(True)
        self.saveContourButton.setVisible(True)

    def removeGeneratedContour(self):
        if hasattr(self, "contour_plot") and self.contour_plot is not None:
            self.view_box.removeItem(self.contour_plot)
            self.contour_plot = None

        self.removeGeneratedContourButton.setVisible(False)
        self.contourIntensityInput.setPlaceholderText(
            "Define intensity threshold"
        )
        self._setGeneratedContourWidgets(False)
        self.saveContourButton.setVisible(False)
        self.contourCreationProgress.setVisible(False)

    def updateGeneratedContour(self):
        intensity, size, smoothness = (
            int(self.contourIntensitySlider.value()),
            self.contourSizeSlider.value() / 100,
            self.contourSmoothnessSlider.value(),
        )
        self._setGeneratedContourWidgets(True, intensity, size, smoothness)
        intensity_contour = self.image_analyzer.find_contour(
            self.Live_image, intensity, num_points=500
        )
        size_contour = self.image_analyzer.resize_contour(
            intensity_contour, size
        )
        smoothness_contour = self.image_analyzer.smoothen_contour(
            size_contour, smoothness
        )

        self.final_contour = smoothness_contour
        self.plotContour(self.final_contour)

    def onContourComplete(self):
        logging.info("Contour creation complete.")
        self.createContour = False
        self.ROIContour.setPen("g")
        self.saveContourButton.setVisible(True)

        # Get the final contour points from the ROIContour handles
        ROIHandlePositions = self.ROIContour.getLocalHandlePositions()
        final_contour = np.array(
            [
                [np.round(pos[1].x(), 0), np.round(pos[1].y(), 0)]
                for pos in ROIHandlePositions
            ]
        )
        logging.info(f"ROI positions: {final_contour}")

        # Interpolate the final contour points
        def interpolate_points(p1, p2, num_points):
            x_values = np.linspace(p1[0], p2[0], num_points)
            y_values = np.linspace(p1[1], p2[1], num_points)
            return np.vstack((x_values, y_values)).T

        num_points_between_vertices = 200
        interpolated_contour = []
        for i in range(len(final_contour)):
            p1 = final_contour[i]
            p2 = final_contour[
                (i + 1) % len(final_contour)
            ]  # Wrap around to the first vertex
            interpolated_points = interpolate_points(
                p1, p2, num_points_between_vertices
            )
            interpolated_contour.extend(interpolated_points)

        # Set self.final_contour to the interpolated contour
        self.final_contour_drawn = np.array(interpolated_contour)

    def saveCustomContour(self):
        # Create an instance of the CameraPmtMapping class and readout
        # calibration parameters
        self.camera_pmt_mapping = CameraPmtMapping()
        cam_vertices, pmt_vertices = (
            self.registrationPoints.camera_vertices,
            self.registrationPoints.pmt_vertices,
        )

        untransformedContour = self.final_contour

        x = untransformedContour[:, 1]
        y = untransformedContour[:, 0]

        untransformedContour = np.vstack((x, y)).T
        # if hasattr(self, 'ROI_hpos') and hasattr(self, 'ROI_vpos') and self.ROI_hpos is not None and self.ROI_vpos is not None:
        # logging.info("Adding ROI offsets to the untransformed contour.")
        # x_coords_with_roi = [c[0] + self.ROI_hpos for c in untransformedContour]
        # y_coords_with_roi = [c[1] + self.ROI_vpos for c in untransformedContour]
        # untransformedContour = np.vstack((x_coords_with_roi, y_coords_with_roi)).T

        # Create affine matrix and transform the final contour
        affine_matrix = (
            self.camera_pmt_mapping.create_affine_transformation_matrix(
                cam_vertices, pmt_vertices
            )
        )
        transformedContour = self.camera_pmt_mapping.transform_contour(
            untransformedContour, affine_matrix
        )

        # Set widgets
        self.contourCreationProgress.setVisible(True)
        self.contourCreationProgress.setText("Contour has been saved!")

        # Emit the signal with the transformed contour
        self.output_signal_camera_pmt_contour.emit(transformedContour)

    def _setGeneratedContourWidgets(
        self, visible, intensity=None, size=None, smoothness=None
    ):
        self.contourIntensityLabel.setVisible(visible)
        self.contourIntensitySlider.setVisible(visible)
        self.contourSizeLabel.setVisible(visible)
        self.contourSizeSlider.setVisible(visible)
        self.contourSmoothnessLabel.setVisible(visible)
        self.contourSmoothnessSlider.setVisible(visible)
        if (
            visible
            and (intensity is not None)
            and (size is not None)
            and (smoothness is not None)
        ):
            self.contourIntensityLabel.setText(f"Intensity: {intensity}")
            self.contourSizeLabel.setText(f"Size: {size}")
            self.contourSmoothnessLabel.setText(f"Smoothness: {smoothness}")

    def handle_galvo_coordinates(self, x, y):
        """Handle the received galvo coordinates."""
        logging.info(f"Received galvo coordinates: x = {x}, y = {y}")

        self.registration_x_coord_pmt_value.setText(str(x))
        self.registration_x_coord_pmt_value.setEnabled(True)
        self.registration_y_coord_pmt_value.setText(str(y))
        self.registration_y_coord_pmt_value.setEnabled(True)
        self.find_camera_registration_point.setEnabled(True)

    def fit_gaussian_over_camera_img(self):
        if self.cameraIsLive or self.cameraIsStreaming:
            logging.warning(
                "Please stop the live or stream. A fixed image is required "
                "for the registration."
            )
            return

        if hasattr(self, "Live_image") and self.Live_image is not None:
            self._set_registration_params(True)
        else:
            logging.warning(
                "No live image available. Please make sure a snapshot of the "
                "registration image is presented."
            )

    def _set_registration_params(self, set_params):

        def set_value(param, value, enabled):
            if enabled:
                param.setText(str(int(value)))
                param.setEnabled(enabled)
            else:
                param.clear()
                param.setEnabled(enabled)

        if not set_params:
            amplitude, xo, yo, sigma_x, sigma_y, offset = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            set_value(self.registration_x_coord_pmt_value, [], set_params)
            set_value(self.registration_y_coord_pmt_value, [], set_params)
        else:
            amplitude, xo, yo, sigma_x, sigma_y, offset = (
                self.camera_pmt_registration.fitDoubleGaussian(self.Live_image)
            )
            registration_params = amplitude, xo, yo, sigma_x, sigma_y, offset
            self._plot_registration_gaussian(registration_params)

        set_value(self.amplitude_value, amplitude, set_params)
        set_value(self.registration_x_coord_camera_value, xo, set_params)
        set_value(self.registration_y_coord_camera_value, yo, set_params)
        set_value(self.sigma_x_value, sigma_x, set_params)
        set_value(self.sigma_y_value, sigma_y, set_params)
        set_value(self.camera_offset_value, offset, set_params)

    def _plot_registration_gaussian(self, registration_params):
        amplitude, xo, yo, sigma_x, sigma_y, offset = registration_params

        # Generate a grid of x and y values
        x = np.linspace(
            0, self.Live_image.shape[1] - 1, self.Live_image.shape[1]
        )
        y = np.linspace(
            0, self.Live_image.shape[0] - 1, self.Live_image.shape[0]
        )
        x, y = np.meshgrid(x, y)

        # Define the 2D Gaussian function
        def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, offset):
            return offset + amplitude * np.exp(
                -(
                    ((x - xo) ** 2) / (2 * sigma_x**2)
                    + ((y - yo) ** 2) / (2 * sigma_y**2)
                )
            )

        # Generate the Gaussian data
        gaussian_data = gaussian(
            x, y, amplitude, xo, yo, sigma_x, sigma_y, offset
        )

        # Add the registration contour plot to visualize the Gaussian fit
        self.registration_contour = pg.IsocurveItem(
            level=amplitude / 2, pen="r"
        )
        self.registration_contour.setOpacity(0.75)
        self.registration_contour.setData(gaussian_data)
        self.Live_view.addItem(self.registration_contour)

        # Add a cross at the center of the Gaussian
        self.registration_cross = pg.ScatterPlotItem(
            [xo], [yo], symbol="+", size=20, pen=pg.mkPen("r", width=2)
        )
        self.Live_view.addItem(self.registration_cross)

        self.registration_gaussian_plotted = True

    def _remove_registration_gaussian(self, remove=False):
        if remove:
            self.Live_view.removeItem(self.registration_contour)
            self.Live_view.removeItem(self.registration_cross)
            self.registration_gaussian_plotted = False


class StreamingWorker(QObject):
    update_label = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    streaming_finished = pyqtSignal()

    def __init__(
        self, hcam, stop_signal, buffer_number, stream_duration, parent=None
    ):
        super().__init__(parent)
        self.hcam = hcam
        self.stop_signal = stop_signal
        self.buffer_number = buffer_number
        self.stream_duration = stream_duration
        self.camera_is_streaming = False
        self.video_list = []
        self.image_count = 0
        self.dims = None

    def run(self):
        try:
            self.camera_is_streaming = True
            self.hcam.setACQMode(
                "fixed_length",
                number_frames=self.buffer_number,
                additional_buffer_factor=1.0,
            )
            self.hcam.startAcquisition()

            if self.stop_signal == "Time":
                # Wait for the first frame to be acquired
                while True:
                    frames, dims = self.hcam.getFrames()
                    if frames:
                        self.dims = dims
                        self.video_list.append(frames[0].np_array)
                        self.image_count += 1
                        self.update_label.emit(
                            f"Recording, {self.image_count} frames.."
                        )
                        start_time = time.time()
                        break
                    time.sleep(0.01)  # Sleep briefly to avoid busy-waiting

                while time.time() - start_time < self.stream_duration:
                    frames, dims = self.hcam.getFrames()
                    self.dims = dims
                    for aframe in frames:
                        self.video_list.append(aframe.np_array)
                        self.image_count += 1
                        self.update_label.emit(
                            f"Recording, {self.image_count} frames.."
                        )
                self.stop_streaming()

            elif self.stop_signal == "Frames":
                for _ in range(self.buffer_number):
                    frames, dims = self.hcam.getFrames()
                    self.dims = dims
                    for aframe in frames:
                        self.video_list.append(aframe.np_array)
                        self.image_count += 1
                        self.update_label.emit(
                            f"Recording, {self.image_count} frames.."
                        )
                self.stop_streaming()
        except Exception as exc:
            logging.info("Error while streaming", exc_info=exc)
            self.error.emit("streaming error")
        finally:
            self.streaming_finished.emit()
            self.finished.emit()

    def stop_streaming(self):
        self.hcam.stopAcquisition()
        self.camera_is_streaming = False


class LiveWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_image = pyqtSignal(np.ndarray)

    def __init__(self, hcam, live_update_interval, parent=None):
        super().__init__(parent)
        self.hcam = hcam
        self.live_update_interval = live_update_interval
        self.camera_is_live = False
        self.snapping = False

    def run(self):
        self.camera_is_live = True
        self.hcam.acquisition_mode = "run_till_abort"
        self.hcam.startAcquisition()

        logging.info("Live acquisition started.")

        while self.camera_is_live:
            try:
                frames, dims = self.hcam.getFrames()
                if frames:
                    live_image = np.resize(
                        frames[-1].np_array, (dims[1], dims[0])
                    )
                    self.update_image.emit(live_image)
                    logging.debug(f"Frame acquired: {live_image.shape}")
                else:
                    logging.warning("No frames retrieved from the camera.")
            except Exception as exc:
                logging.error("Error during live acquisition:", exc_info=exc)
                self.camera_is_live = False

            time.sleep(self.live_update_interval)

        self.hcam.stopAcquisition()
        logging.info("Live acquisition stopped.")
        self.finished.emit()

    def stop(self):
        self.camera_is_live = False

    def snap(self):
        try:
            self.snapping = True
            self.hcam.stopAcquisition()  # Ensure live acquisition is stopped
            self.hcam.acquisition_mode = "snap"
            self.hcam.startAcquisition()
            logging.info("Snapping image.")
            frames, dims = self.hcam.getFrames()
            if frames:
                snap_image = np.resize(frames[-1].np_array, (dims[1], dims[0]))
                self.update_image.emit(snap_image)
                logging.debug(f"Snap image acquired: {snap_image.shape}")
            else:
                logging.warning("No frames retrieved from the camera.")
        except Exception as exc:
            logging.error("Error during snapping:", exc_info=exc)
        finally:
            self.hcam.stopAcquisition()
            self.snapping = False
            self.finished.emit()


class SaveWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(
        self, video_list, dims, image_count, metaData, file_dir, parent=None
    ):
        super().__init__(parent)
        self.video_list = video_list
        self.dims = dims
        self.image_count = image_count
        self.metaData = metaData
        self.file_dir = file_dir

    def run(self):
        try:
            logging.info("Starting save operation.")
            batch_size = max(100, min(1000, int(self.image_count * 0.05)))

            write_starttime = time.time()

            with skimtiff.TiffWriter(
                self.file_dir, append=True, imagej=False
            ) as tif:
                for i in range(0, self.image_count, batch_size):
                    batch_frames = self.video_list[i : i + batch_size]

                    images = np.array(
                        [
                            np.resize(frame, (self.dims[1], self.dims[0]))
                            for frame in batch_frames
                        ]
                    )
                    tif.write(
                        images,
                        description=self.metaData,
                        photometric="minisblack",
                    )

                    progress = int((i + batch_size) / self.image_count * 100)
                    self.update_progress.emit(progress)

            totaltime = round(time.time() - write_starttime, 2)
            logging.info(f"Done writing frames. Total time: {totaltime} sec.")
            self.finished.emit()

        except Exception as exc:
            logging.error("Error during save operation:", exc_info=exc)
            self.error.emit("error during save")


if __name__ == "__main__":

    def run_app(pmt_widget_ui=None, waveform_widget_ui=None):
        app = QtWidgets.QApplication(sys.argv)
        QtWidgets.QApplication.setStyle(QStyleFactory.create("Fusion"))

        if pmt_widget_ui is None:
            pmt_widget_ui = PMTWidget.PMTWidgetUI()

        if waveform_widget_ui is None:
            waveform_widget_ui = WaveformWidget.WaveformGenerator()

        mainwin = CameraUI(pmt_widget_ui, waveform_widget_ui)
        mainwin.show()
        app.exec_()

    run_app()
