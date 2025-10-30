# c -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:54:40 2019

@author: xinmeng
    ============================== ==============================================

    For general experiments in Dr. Daan's lab ゴゴゴ ゴ ゴ ゴ  ゴ  ゴ
    ============================== ==============================================
    == Widget descriptions ==

    - HamamatsuCam.HamamatsuUI: Hamamatsu orca flash 4.0 camera user interface.
    - PatchClamp.ui_patchclamp_sealtest: The GUI for patch clamp.
    - NIDAQ.WaveformWidget: The GUI for configuring and executing waveforms in National Instrument Data Acquisition (DAQ) device.
    - GalvoWidget.PMTWidget: For PMT scanning imaging.
    - ImageAnalysis.AnalysisWidget: Data Analysis widget.
    - SampleStageControl.StageMoveWidget: The GUI for sample stage movement control.
    - NIDAQ.AOTFWidget: To control AOTF, which is controlled by NI-daq.
    - ThorlabsFilterSlider.FilterSliderWidget: Controller for Thorlabs filter slider.
    - PI_ObjectiveMotor.ObjMotorWidget: Widget for objective motor control.
    - CoordinatesManager.CoordinatesWidget: Widget to create mask based on widefield image. Project mask with DMD or galvos
    ============================== ==============================================
"""
import os
import sys

import pyqtgraph as pg
import pyqtgraph.console
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QTextCursor

from . import (
    NIDAQ,
    CoordinatesManager,
    GalvoWidget,
    HamamatsuCam,
    Icons,
    ImageAnalysis,
    InsightX3,
    PatchClamp,
    PI_ObjectiveMotor,
    SampleStageControl,
    ScreeningWidget,
    StylishQT,
    ThorlabsFilterSlider,
    ThorlabsKCube,
)


class Mainbody(QtWidgets.QWidget):
    savedirectory_changed = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with Icons.Path("Logo.png") as path:
            self.setWindowIcon(QIcon(path))

        self.setFont(QFont("Arial"))

        # === GUI ===
        self.setMinimumSize(800, 600)
        # self.setMaximumHeight(1080)
        self.setWindowTitle("Fiumicino")
        self.layout = QtWidgets.QGridLayout(self)

        """
        # GUI for right tabs panel-Creating instances of each widget showing on right side tabs.
        """
        self.tabs = QtWidgets.QTabWidget()
        self.Galvo_WidgetInstance = GalvoWidget.PMTWidget.PMTWidgetUI()
        self.Waveformer_WidgetInstance = (
            NIDAQ.WaveformWidget.WaveformGenerator()
        )
        self.Camera_WidgetInstance = HamamatsuCam.HamamatsuUI.CameraUI(
            self.Galvo_WidgetInstance, self.Waveformer_WidgetInstance
        )
        self.savedirectory_changed.connect(
            self.Camera_WidgetInstance.update_savedirectory
        )
        self.PatchClamp_WidgetInstance = (
            PatchClamp.ui_patchclamp_sealtest.PatchclampSealTestUI()
        )
        self.Analysis_WidgetInstance = (
            ImageAnalysis.AnalysisWidget.AnalysisWidgetUI()
        )
        self.Coordinate_WidgetInstance = (
            CoordinatesManager.CoordinateWidget.CoordinatesWidgetUI()
        )

        # === Add tab widgets ===
        self.tabs.addTab(self.Camera_WidgetInstance, "Camera imaging")
        self.tabs.addTab(self.Galvo_WidgetInstance, "PMT imaging")
        self.tabs.addTab(self.Waveformer_WidgetInstance, "Waveform")
        # self.tabs.addTab(self.PatchClamp_WidgetInstance,"Patch clamp")
        self.tabs.addTab(self.Coordinate_WidgetInstance, "Coordinates")
        self.tabs.addTab(self.Analysis_WidgetInstance, "Image analysis")

        self.savedirectory = ""

        """
        # GUI for left panel.
        """
        # GUI for set directory
        setdirectoryContainer = StylishQT.roundQGroupBox(title="Set directory")
        self.setdirectorycontrolLayout = QtWidgets.QGridLayout()

        self.saving_prefix = ""
        self.savedirectorytextbox = QtWidgets.QLineEdit(self)
        self.savedirectorytextbox.setPlaceholderText("Saving directory")
        self.savedirectorytextbox.returnPressed.connect(
            self.update_saving_directory
        )
        self.setdirectorycontrolLayout.addWidget(
            self.savedirectorytextbox, 0, 1, 1, 2
        )

        self.prefixtextbox = QtWidgets.QLineEdit(self)
        self.prefixtextbox.setPlaceholderText("Prefix")
        self.prefixtextbox.returnPressed.connect(self.set_prefix)
        self.setdirectorycontrolLayout.addWidget(self.prefixtextbox, 0, 0)

        # self.setdirectorycontrolLayout.addWidget(QLabel("Saving prefix:"), 0, 0)

        self.toolButtonOpenDialog = QtWidgets.QPushButton()
        with Icons.Path("Browse.png") as path:
            self.toolButtonOpenDialog.setIcon(QIcon(path))
        self.toolButtonOpenDialog.setObjectName("toolButtonOpenDialog")
        self.toolButtonOpenDialog.clicked.connect(self.set_saving_directory)

        self.setdirectorycontrolLayout.addWidget(
            self.toolButtonOpenDialog, 0, 3
        )

        # Console massage
        self.console_text_edit = QtWidgets.QTextEdit()
        # self.console_text_edit.setFontItalic(True)
        self.console_text_edit.setPlaceholderText("Notice board from console.")
        self.console_text_edit.setFixedHeight(150)
        self.setdirectorycontrolLayout.addWidget(
            self.console_text_edit, 1, 0, 5, 3
        )

        self.setMetaTextButton = QtWidgets.QPushButton()
        with Icons.Path("write.png") as path:
            self.setMetaTextButton.setIcon(QIcon(path))
        self.setMetaTextButton.setObjectName("Init. Meta Text")
        self.setdirectorycontrolLayout.addWidget(
            self.setMetaTextButton, 1, 3, 1, 1
        )
        self.setMetaTextButton.clicked.connect(self.Init_Meta_Text)

        self.saveMetaTextButton = QtWidgets.QPushButton()
        with Icons.Path("save.png") as path:
            self.saveMetaTextButton.setIcon(QIcon(path))
        self.saveMetaTextButton.setObjectName("Save Meta Text")
        self.setdirectorycontrolLayout.addWidget(
            self.saveMetaTextButton, 2, 3, 1, 1
        )
        self.saveMetaTextButton.clicked.connect(self.Save_Meta_Text)

        setdirectoryContainer.setLayout(self.setdirectorycontrolLayout)
        setdirectoryContainer.setMaximumHeight(220)
        setdirectoryContainer.setMaximumWidth(420)

        self.layout.addWidget(setdirectoryContainer, 2, 0, 1, 4)

        # GUI for general buttons

        with Icons.Path("shutter.png") as path:
            self.shutter2PButton = StylishQT.checkableButton(Icon_path=path)
        self.shutter2PButton.clicked.connect(self.shutter2Paction)
        self.shutter2PButton.setFixedWidth(46)
        self.shutter2PButton.setToolTip("Open/Close InsightX3 shutter")

        with Icons.Path("LED.png") as path:
            self.LEDButton = StylishQT.checkableButton(Icon_path=path)
        self.LEDButton.clicked.connect(self.LEDaction)
        self.LEDButton.setFixedWidth(46)
        self.LEDButton.setToolTip("Turn on/off LED")

        self.shutter2PButton.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.LEDButton.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )

        self.layout.addWidget(self.shutter2PButton, 5, 0)
        self.layout.addWidget(self.LEDButton, 5, 1)

        # GUI for weather
        # self.layout.addWidget(GeneralWidgets.Weather_GUI.WeatherUI(), 1, 0)

        # GUI for Thorlabs motor
        self.KCubeWidgetInstance = (
            SampleStageControl.StageMoveWidget.StageWidgetUI()
        )
        self.layout.addWidget(self.KCubeWidgetInstance, 9, 0, 1, 4)

        # GUI for sample stage
        self.StageMoveWidgetInstance = (
            ThorlabsKCube.KCubeWidget.KCubeWidgetUI()
        )
        self.layout.addWidget(self.StageMoveWidgetInstance, 10, 0, 1, 4)

        # GUI for AOTF
        self.AOTFWidgetInstance = NIDAQ.AOTFWidget.AOTFWidgetUI()
        self.layout.addWidget(self.AOTFWidgetInstance, 6, 0, 1, 3)

        # GUI for DMD-mini
        self.DMDminiWidgetInstance = (
            CoordinatesManager.DMDminiWidget.DMDminiWidgetUI()
        )
        self.layout.addWidget(self.DMDminiWidgetInstance, 6, 3, 1, 1)

        # GUI for fliter silder
        FilterSliderWidgetInstance = (
            ThorlabsFilterSlider.FilterSliderWidget.FilterSliderWidgetUI()
        )
        self.layout.addWidget(FilterSliderWidgetInstance, 8, 0, 1, 4)

        # GUI for objective motor
        ObjMotorInstance = PI_ObjectiveMotor.ObjMotorWidget.ObjMotorWidgetUI()
        self.layout.addWidget(ObjMotorInstance, 7, 0, 1, 4)

        # GUI for camera button
        # self.open_cam = StylishQT.FancyPushButton(50, 50, color1=(255,153,255), color2=(204,208,255))

        # self.open_cam.setIcon(QIcon('./Icons/Hamamatsu.png'))
        # self.open_cam.setToolTip("Open camera widget")
        # self.open_cam.setIconSize(QSize(60, 60))
        # self.open_cam.clicked.connect(self.open_camera)
        # self.layout.addWidget(self.open_cam, 4, 0, 1, 1)
        # self.open_cam.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        # GUI for Insight X3
        self.open_Insight = StylishQT.FancyPushButton(
            40, 50, color1=(176, 224, 230), color2=(135, 206, 250)
        )
        with Icons.Path("two_photon.png") as path:
            self.open_Insight.setIcon(QIcon(path))
        self.open_Insight.setToolTip("Open 2-p laser widget")
        self.open_Insight.setIconSize(QSize(45, 45))
        self.open_Insight.setFixedWidth(65)
        self.open_Insight.clicked.connect(self.open_Insight_UI)
        self.layout.addWidget(self.open_Insight, 3, 1, 2, 1)
        self.open_Insight.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        # GUI for evolution screening
        self.open_screening_button = StylishQT.FancyPushButton(
            40, 50, color1=(245, 245, 220), color2=(255, 228, 196)
        )
        with Icons.Path("Screening1.png") as path:
            self.open_screening_button.setIcon(QIcon(path))
        self.open_screening_button.setToolTip("Open screening widget")
        self.open_screening_button.setIconSize(QSize(45, 45))
        self.open_screening_button.setFixedWidth(65)
        self.open_screening_button.clicked.connect(self.open_screening)
        self.layout.addWidget(self.open_screening_button, 3, 0, 2, 1)
        self.open_screening_button.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        # GUI for open_sealtest
        self.open_sealtest_button = StylishQT.FancyPushButton(
            40, 50, color1=(255, 153, 255), color2=(204, 208, 255)
        )
        with Icons.Path("patchclamp.png") as path:
            self.open_sealtest_button.setIcon(QIcon(path))
        # self.open_sealtest_button.setText("Seal test")
        self.open_sealtest_button.setToolTip("Open sealtest widget")
        self.open_sealtest_button.setIconSize(QSize(45, 45))
        self.open_sealtest_button.setFixedWidth(65)
        self.open_sealtest_button.clicked.connect(self.open_sealtest)
        self.layout.addWidget(self.open_sealtest_button, 3, 2, 2, 1)
        self.open_sealtest_button.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )

        # GUI for Auto-patcher
        self.open_AutoPatcher_button = StylishQT.FancyPushButton(
            40, 50, color1=(50, 50, 255), color2=(100, 100, 255)
        )
        with Icons.Path("patchclamp.png") as path:
            self.open_AutoPatcher_button.setIcon(QIcon(path))
        # self.open_sealtest_button.setText("Seal test")
        self.open_AutoPatcher_button.setToolTip("Open AutoPatcher")
        self.open_AutoPatcher_button.setIconSize(QSize(45, 45))
        self.open_AutoPatcher_button.setFixedWidth(65)
        self.open_AutoPatcher_button.clicked.connect(self.open_AutoPatcher)
        self.layout.addWidget(self.open_AutoPatcher_button, 3, 3, 2, 1)
        self.open_sealtest_button.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )

        # self.setLayout(pmtmaster)

        self.layout.addWidget(self.tabs, 0, 4, 12, 4)
        self.setLayout(self.layout)

        # Establishing communication between widgets.
        self.Galvo_WidgetInstance.SignalForContourScanning.connect(
            self.PassVariable_GalvoWidget_to_Waveformer
        )
        self.Galvo_WidgetInstance.MessageBack.connect(self.normalOutputWritten)

        self.Coordinate_WidgetInstance.sig_start_registration.connect(
            lambda: self.AOTFWidgetInstance.set_registration_mode(True)
        )
        self.Coordinate_WidgetInstance.sig_finished_registration.connect(
            lambda: self.AOTFWidgetInstance.set_registration_mode(False)
        )
        self.Coordinate_WidgetInstance.MessageBack.connect(
            self.normalOutputWritten
        )
        self.Waveformer_WidgetInstance.waveform_started.connect(
            self.AOTFWidgetInstance.reset_sliders
        )

        self.Analysis_WidgetInstance.MessageBack.connect(
            self.normalOutputWritten
        )
        self.Analysis_WidgetInstance.Cellselection_DMD_mask_contour.connect(
            self.Coordinate_WidgetInstance.DMDWidget.receive_mask_coordinates
        )

        # Connect camera with DMD widget, so that snapped images are shown in
        # DMD widget.
        self.Camera_WidgetInstance.output_signal_SnapImg.connect(
            self.Coordinate_WidgetInstance.receive_image_from_camera
        )

        """
        === END of GUI ===
        """

        self.Init_Meta_Text()

    # %%
    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

        """
        === Functions to pass variables across widges ===
        """

    def PassVariable_GalvoWidget_to_Waveformer(
        self,
        contour_point_number,
        Daq_sample_rate_pmt,
        time_per_contour,
        handle_viewbox_coordinate_x,
        handle_viewbox_coordinate_y,
    ):

        # Number of points in single round of contour scan
        self.Waveformer_WidgetInstance.galvo_contour_label_1.setText(
            "Points in contour: %.d" % contour_point_number
        )
        self.Waveformer_WidgetInstance.galvo_contour_label_2.setText(
            "Sampling rate: %.d" % Daq_sample_rate_pmt
        )
        self.Waveformer_WidgetInstance.Daq_sample_rate_pmt = (
            Daq_sample_rate_pmt
        )
        self.Waveformer_WidgetInstance.handle_viewbox_coordinate_position_array_expanded_x = (
            handle_viewbox_coordinate_x
        )
        self.Waveformer_WidgetInstance.handle_viewbox_coordinate_position_array_expanded_y = (
            handle_viewbox_coordinate_y
        )

        # Time needed for single round of contour scanning
        self.Waveformer_WidgetInstance.time_per_contour = time_per_contour

    # Fucs for set directory
    # Set the savedirectory and prefix of Waveform widget in syn.
    def set_saving_directory(self):
        self.savedirectory = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                caption="Set saving directory", directory=""
            )
        )
        self.savedirectorytextbox.setText(self.savedirectory)

        # Emit the signal to notify other widgets
        self.savedirectory_changed.emit(self.savedirectory)

        # Assert saving directories in other widgets
        self.Galvo_WidgetInstance.savedirectory = self.savedirectory
        self.Waveformer_WidgetInstance.savedirectory = self.savedirectory
        self.Analysis_WidgetInstance.savedirectory = self.savedirectory
        self.PatchClamp_WidgetInstance.saving_dir = self.savedirectory
        self.Camera_WidgetInstance.default_folder = self.savedirectory

        self.set_prefix()

    def update_saving_directory(self):
        self.savedirectory = str(self.savedirectorytextbox.text())

        # Emit the signal to notify other widgets
        self.savedirectory_changed.emit(self.savedirectory)

    def set_prefix(self):
        self.saving_prefix = str(self.prefixtextbox.text())

        self.Galvo_WidgetInstance.prefixtextboxtext = self.saving_prefix
        self.Waveformer_WidgetInstance.saving_prefix = self.saving_prefix

    def shutter2Paction(self):
        daq = NIDAQ.DAQoperator.DAQmission()
        # For 2P shutter
        if self.shutter2PButton.isChecked():
            daq.sendSingleDigital("2Pshutter", True)
        else:
            daq.sendSingleDigital("2Pshutter", False)

    def LEDaction(self):
        daq = NIDAQ.DAQoperator.DAQmission()
        # For LED
        if self.LEDButton.isChecked():
            daq.sendSingleDigital("LED", True)
        else:
            daq.sendSingleDigital("LED", False)

    def open_Insight_UI(self):
        self.open_Insight_UIWindow = (
            InsightX3.TwoPhotonLaserUI.InsightWidgetUI()
        )
        self.open_Insight_UIWindow.show()

    def open_screening(self):
        try:
            self.open_screening_UIWindow.deleteLater()
        except AttributeError:
            pass

        self.open_screening_UIWindow = (
            ScreeningWidget.Evolution_GUI_v7.Mainbody()
        )
        self.open_screening_UIWindow.show()

    def open_sealtest(self):
        self.PatchClamp_WidgetInstance.show()

    def open_AutoPatcher(self):
        self.open_AutoPatcher_UIWindow = (
            PatchClamp.smartpatcher_frontend.PatchClampUI()
        )
        self.open_AutoPatcher_UIWindow.show()

    # Fucs for console display
    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.console_text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.console_text_edit.setTextCursor(cursor)
        self.console_text_edit.ensureCursorVisible()

    def Init_Meta_Text(self):
        Init_Meta_Text = "=========Meta Text======== \n\nConstruct: \n\nPipette pulling program: \nHeat: \nVelocity: \nDelay: \nPressure: \nRamp: \n\nTwo-photon wavelength: \nTwo-photon power: \nTwo-photon ND value: \nPMT amplification: \n\
=======================\n"

        self.console_text_edit.clear()
        self.console_text_edit.setPlainText(Init_Meta_Text)

    def Save_Meta_Text(self):

        meta_text = self.console_text_edit.toPlainText()
        with open(
            os.path.join(self.savedirectory, "meta_text.txt"), "w"
        ) as output_file:
            output_file.write(meta_text)

    def closeEvent(self, event):
        QtWidgets.QApplication.quit()
        event.accept()


def run_app():
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    pg.setConfigOptions(imageAxisOrder="row-major")
    mainwin = Mainbody()
    mainwin.show()
    app.exec_()


if __name__ == "__main__":
    run_app()
