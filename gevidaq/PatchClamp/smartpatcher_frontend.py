# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:34:10 2022

@author: TvdrBurgt
"""

import logging
import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPalette, QPen
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QWidget,
)
from skimage import io

from ..NIDAQ.constants import MeasurementConstants
from .camerathread import CameraThread
from .micromanipulator import ScientificaPatchStar
from .pressurethread import PressureThread
from .sealtestthread import SealTestThread
from .smartpatcher_backend import SmartPatcher
from .stage import LudlStage

try:
    from .objective import PIMotor
except Exception as exc:
    logging.critical("caught exception", exc_info=exc)
    logging.info("pipython not configured.")


class PatchClampUI(QWidget):
    def __init__(self):
        super().__init__()
        """
        === Start of GUI ===
        """
        """
        # === General widget settings ===
        """
        self.setWindowTitle("Smartpatcher")

        """
        === Hardware container ===
        """
        hardwareContainer = QGroupBox("Hardware devices")
        hardwareLayout = QGridLayout()

        # Button to (dis)connect camera
        self.connect_camerathread_button = QPushButton(
            text="Camera", clicked=self.connect_camerathread
        )
        self.connect_camerathread_button.setCheckable(True)

        # Button to (dis)connect objective motor
        self.connect_objectivemotor_button = QPushButton(
            text="Objective motor", clicked=self.connect_objectivemotor
        )
        self.connect_objectivemotor_button.setCheckable(True)

        # Button to (dis)connect micromanipulator
        self.connect_micromanipulator_button = QPushButton(
            text="Micromanipulator", clicked=self.connect_micromanipulator
        )
        self.connect_micromanipulator_button.setCheckable(True)

        # Button to (dis)connect XYstage
        self.connect_XYstage_button = QPushButton(
            text="XY stage", clicked=self.connect_XYstage
        )
        self.connect_XYstage_button.setCheckable(True)

        # Button to (dis)connect NIDAQ for sealtest
        self.connect_sealtestthread_button = QPushButton(
            text="NIDAQ/Sealtest", clicked=self.connect_sealtestthread
        )
        self.connect_sealtestthread_button.setCheckable(True)

        # Button to (dis)connect pressure controller
        self.connect_pressurecontroller_button = QPushButton(
            text="Pressure controller", clicked=self.connect_pressurethread
        )
        self.connect_pressurecontroller_button.setCheckable(True)

        hardwareLayout.addWidget(self.connect_camerathread_button, 0, 0, 1, 1)
        hardwareLayout.addWidget(
            self.connect_objectivemotor_button, 1, 0, 1, 1
        )
        hardwareLayout.addWidget(
            self.connect_micromanipulator_button, 2, 0, 1, 1
        )
        hardwareLayout.addWidget(self.connect_XYstage_button, 3, 0, 1, 1)
        hardwareLayout.addWidget(
            self.connect_sealtestthread_button, 4, 0, 1, 1
        )
        hardwareLayout.addWidget(
            self.connect_pressurecontroller_button, 5, 0, 1, 1
        )
        hardwareContainer.setLayout(hardwareLayout)

        """
        === Stage movements ===
        """
        stagemoveContainer = QGroupBox("Stage move")
        stagemoveLayout = QGridLayout()

        # Stage translation
        request_stage_up_button = QPushButton(
            text="up", clicked=self.request_stage_up
        )
        request_stage_down_button = QPushButton(
            text="down", clicked=self.request_stage_down
        )
        request_stage_right_button = QPushButton(
            text="right", clicked=self.request_stage_right
        )
        request_stage_left_button = QPushButton(
            text="left", clicked=self.request_stage_left
        )

        # Button for moving target to center
        request_target2center_button = QPushButton(
            text="Target to center", clicked=self.request_target2center
        )
        request_target2center_button.setToolTip(
            "Brings target ROI to the center of the field-of-view.\n"
            + "Make sure that a target is set."
        )

        stagemoveLayout.addWidget(request_stage_up_button, 0, 1, 1, 1)
        stagemoveLayout.addWidget(request_stage_down_button, 2, 1, 1, 1)
        stagemoveLayout.addWidget(request_stage_right_button, 1, 2, 1, 1)
        stagemoveLayout.addWidget(request_stage_left_button, 1, 0, 1, 1)
        stagemoveLayout.addWidget(request_target2center_button, 3, 0, 1, 3)
        stagemoveContainer.setLayout(stagemoveLayout)

        """
        === Calibration options ===
        """
        calibrationContainer = QGroupBox("Calibration options")
        calibrationLayout = QGridLayout()

        # Calibration algorithms
        request_calibrate_xy_button = QPushButton(
            text="Calibrate XY", clicked=self.request_calibrate_xy
        )
        request_calibrate_pixelsize_button = QPushButton(
            text="Calibrate pixelsize",
            clicked=self.request_calibrate_pixelsize,
        )
        request_calibrate_xy_button.setToolTip(
            "Aligns the micromanipulator axes with the pixel rows and columns "
            "on screen.\n"
            "Performs best with the pipette tip ~10μm below focus"
        )
        request_calibrate_pixelsize_button.setToolTip(
            "Calibrates the pixelsize. Use this if 'Target to center' and"
            " 'Start target approach' miss their target.\n"
            "Performs best with the pipette tip ~10μm below focus"
        )

        # Calibration statistics
        rotation_angles_label = QLabel("Rotation axes (α,β,γ):")
        self.rotation_angles_value_label = QLabel("-")
        self.rotation_angles_value_label.setFont(
            QFont("Times", weight=QFont.Bold)
        )
        rotation_angles_units_label = QLabel("degrees")
        pixelsize_label = QLabel("Pixelsize (hxw):")
        self.pixelsize_unit_label = QLabel("-")
        self.pixelsize_unit_label.setFont(QFont("Times", weight=QFont.Bold))
        pixelsize_units_label = QLabel("nm")

        calibrationLayout.addWidget(request_calibrate_xy_button, 0, 0, 1, 4)
        calibrationLayout.addWidget(
            request_calibrate_pixelsize_button, 1, 0, 1, 4
        )
        calibrationLayout.addWidget(rotation_angles_label, 2, 0, 1, 1)
        calibrationLayout.addWidget(
            self.rotation_angles_value_label, 2, 1, 1, 2
        )
        calibrationLayout.addWidget(rotation_angles_units_label, 2, 3, 1, 1)
        calibrationLayout.addWidget(pixelsize_label, 3, 0, 1, 1)
        calibrationLayout.addWidget(self.pixelsize_unit_label, 3, 1, 1, 2)
        calibrationLayout.addWidget(pixelsize_units_label, 3, 3, 1, 1)
        calibrationContainer.setLayout(calibrationLayout)

        """
        === Camera view display ===
        """
        liveContainer = QGroupBox("Field-of-view")
        liveContainer.setMinimumWidth(700)
        liveLayout = QGridLayout()

        # Display to project live camera view
        liveWidget = pg.ImageView()
        liveWidget.ui.roiBtn.hide()
        liveWidget.ui.menuBtn.hide()
        liveWidget.ui.histogram.hide()
        self.liveView = liveWidget.getView()
        self.liveImageItem = liveWidget.getImageItem()
        self.liveImageItem.setAutoDownsample(True)

        # Button for pausing camera view
        self.request_pause_button = QPushButton(
            text="Pause live", clicked=self.toggle_pauselive
        )
        self.request_pause_button.setCheckable(True)

        # Button for clearing phantoms
        request_clear_drawings = QPushButton(
            text="Clear phantoms", clicked=self.clearROIs
        )

        # Button for reseting the field of view
        request_reset_fov = QPushButton(
            text="Reset view", clicked=self.reset_imageview
        )

        # Button for snapshotting
        self.request_snap_button = QPushButton(
            text="Take snap", clicked=self.request_snap
        )

        liveLayout.addWidget(liveWidget, 0, 0, 1, 4)
        liveLayout.addWidget(self.request_pause_button, 1, 0, 1, 1)
        liveLayout.addWidget(request_clear_drawings, 1, 1, 1, 1)
        liveLayout.addWidget(request_reset_fov, 1, 2, 1, 1)
        liveLayout.addWidget(self.request_snap_button, 1, 3, 1, 1)
        liveContainer.setLayout(liveLayout)

        """
        === Autopatch control buttons ===
        """
        autopatchContainer = QGroupBox("Autoptach algorithms")
        autopatchLayout = QGridLayout()

        # Buttons for autopatch algorithms
        request_prechecks_button = QPushButton(
            text="Pre-checks", clicked=self.request_prechecks
        )
        request_selecttarget_button = QPushButton(
            text="Select target", clicked=self.request_selecttarget
        )
        request_confirmtarget_button = QPushButton(
            text="Confirm target", clicked=self.request_confirmtarget
        )
        request_start_autopatch_button = QPushButton(
            text="Start autopatch", clicked=self.request_start_autopatch
        )
        request_manual_adjustment_button = QPushButton(
            text="Correct tip", clicked=self.request_manual_adjustment
        )
        request_adjustment_confirm_button = QPushButton(
            text="Confirm tip", clicked=self.request_adjustment_confirm
        )
        request_start_approach_button = QPushButton(
            text="Start target approach", clicked=self.request_start_approach
        )
        request_start_gigaseal_button = QPushButton(
            text="Start gigaseal", clicked=self.request_start_gigaseal
        )
        request_start_breakin_button = QPushButton(
            text="Start break-in", clicked=self.request_start_breakin
        )
        self.request_stop_button = QPushButton(
            text="STOP!", clicked=self.request_emergency_stop
        )
        self.request_stop_button.setCheckable(True)
        self.request_stop_button.setToolTip(
            "Do not spam! Leave it checked and the algorithm will\n"
            + " safely stop while stopping all motion devices."
        )

        # Labels for autopatch status and progress updates
        autopatch_status_label = QLabel("Autopatch status:")
        autopatch_progress_label = QLabel("Progress updates:")
        self.autopatch_status = QLabel("Ready")
        self.autopatch_progress = QLabel("-")
        self.autopatch_status.setFont(QFont("Times", weight=QFont.Bold))
        self.autopatch_progress.setFont(QFont("Times", weight=QFont.Bold))

        autopatchLayout.addWidget(request_prechecks_button, 0, 0, 1, 2)
        autopatchLayout.addWidget(request_selecttarget_button, 1, 0, 1, 1)
        autopatchLayout.addWidget(request_confirmtarget_button, 1, 1, 1, 1)
        autopatchLayout.addWidget(request_start_autopatch_button, 2, 0, 1, 2)
        autopatchLayout.addWidget(request_manual_adjustment_button, 3, 0, 1, 1)
        autopatchLayout.addWidget(
            request_adjustment_confirm_button, 3, 1, 1, 1
        )
        autopatchLayout.addWidget(request_start_approach_button, 4, 0, 1, 2)
        autopatchLayout.addWidget(request_start_gigaseal_button, 5, 0, 1, 2)
        autopatchLayout.addWidget(request_start_breakin_button, 6, 0, 1, 2)
        autopatchLayout.addWidget(self.request_stop_button, 7, 0, 1, 2)
        autopatchLayout.addWidget(autopatch_status_label, 8, 0, 1, 2)
        autopatchLayout.addWidget(self.autopatch_status, 9, 0, 1, 2)
        autopatchLayout.addWidget(autopatch_progress_label, 10, 0, 1, 2)
        autopatchLayout.addWidget(self.autopatch_progress, 11, 0, 1, 2)
        autopatchContainer.setLayout(autopatchLayout)

        """
        === Pressure control layout ===
        """
        pressurecontrolContainer = QGroupBox(title="Pressure control")
        pressurecontrolLayout = QGridLayout()

        # Labels set and read pressure
        pressure_set_label = QLabel("Request pressure:")
        pressure_read_label = QLabel("Readout pressure:")
        pressure_units_label1 = QLabel("mBar")
        pressure_units_label2 = QLabel("mBar")

        # Spinbox to set pressure
        self.set_pressure_spinbox = QDoubleSpinBox()
        self.set_pressure_spinbox.setMinimum(-300)
        self.set_pressure_spinbox.setMaximum(200)
        self.set_pressure_spinbox.setDecimals(0)
        self.set_pressure_spinbox.setValue(0)
        self.set_pressure_spinbox.setSingleStep(10)

        # Label pressure readout value
        self.pressure_value_Label = QLabel("-")
        self.pressure_value_Label.setFont(QFont("Times", weight=QFont.Bold))

        # Label pressure status
        pressure_statustext_label = QLabel("Pressure status:")
        self.pressure_status_label = QLabel("-")
        self.pressure_status_label.setFont(QFont("Times", weight=QFont.Bold))

        # Buttons for relatively valued pressures
        low_positive_button = QPushButton(
            text="Low +", clicked=self.request_low_positive
        )
        low_negative_button = QPushButton(
            text="Low -", clicked=self.request_low_negative
        )
        high_positive_button = QPushButton(
            text="High +", clicked=self.request_high_positive
        )
        high_negative_button = QPushButton(
            text="High -", clicked=self.request_high_negative
        )
        stop_regulating_button = QPushButton(
            text="Set once", clicked=self.request_stop_regulating
        )
        atmoshpere_button = QPushButton(
            text="ATM", clicked=self.request_atmosphere
        )

        # Buttons for operating modes
        apply_pressure_button = QPushButton(
            text="Apply pressure", clicked=self.request_apply_pressure
        )
        spike_pulse_button = QPushButton(
            text="Apply pulse", clicked=self.request_apply_pulse
        )
        self.toggle_lcd_button = QPushButton(
            text="LCD off", clicked=self.toggle_lcd
        )
        self.toggle_lcd_button.setCheckable(True)

        pressurecontrolLayout.addWidget(pressure_set_label, 0, 0, 1, 2)
        pressurecontrolLayout.addWidget(self.set_pressure_spinbox, 0, 2, 1, 1)
        pressurecontrolLayout.addWidget(pressure_units_label1, 0, 3, 1, 1)
        pressurecontrolLayout.addWidget(pressure_read_label, 1, 0, 1, 2)
        pressurecontrolLayout.addWidget(self.pressure_value_Label, 1, 2, 1, 1)
        pressurecontrolLayout.addWidget(pressure_units_label2, 1, 3, 1, 1)
        pressurecontrolLayout.addWidget(pressure_statustext_label, 2, 0, 1, 2)
        pressurecontrolLayout.addWidget(self.pressure_status_label, 2, 2, 1, 2)
        pressurecontrolLayout.addWidget(low_positive_button, 3, 0, 1, 1)
        pressurecontrolLayout.addWidget(low_negative_button, 3, 1, 1, 1)
        pressurecontrolLayout.addWidget(high_positive_button, 4, 0, 1, 1)
        pressurecontrolLayout.addWidget(high_negative_button, 4, 1, 1, 1)
        pressurecontrolLayout.addWidget(stop_regulating_button, 5, 0, 1, 1)
        pressurecontrolLayout.addWidget(atmoshpere_button, 5, 1, 1, 1)
        pressurecontrolLayout.addWidget(apply_pressure_button, 3, 2, 1, 2)
        pressurecontrolLayout.addWidget(spike_pulse_button, 4, 2, 1, 2)
        pressurecontrolLayout.addWidget(self.toggle_lcd_button, 5, 2, 1, 2)
        pressurecontrolContainer.setLayout(pressurecontrolLayout)

        """
        === Electrophysiology ===
        """
        electrophysiologyContainer = QGroupBox("Electrophysiology")
        electrophysiologyLayout = QGridLayout()

        # Labels for electrophysiology
        resistance_label = QLabel("Resistance: ")
        self.resistance_value_label = QLabel("-")
        self.resistance_value_label.setFont(QFont("Times", weight=QFont.Bold))
        resistance_unit_label = QLabel("MΩ")
        capacitance_label = QLabel("Capacitance: ")
        self.capacitance_value_label = QLabel("-")
        self.capacitance_value_label.setFont(QFont("Times", weight=QFont.Bold))
        capacitance_unit_label = QLabel("F")
        ratio_label = QLabel("Ratio: ")
        self.ratio_value_label = QLabel("-")
        membranevoltage_label = QLabel("Membrane potential:")
        self.membranevoltage_value_label = QLabel("-")
        membranevoltage_unit_label = QLabel("V")

        request_zap_button = QPushButton(text="ZAP!", clicked=self.request_zap)

        electrophysiologyLayout.addWidget(resistance_label, 0, 0, 1, 1)
        electrophysiologyLayout.addWidget(
            self.resistance_value_label, 0, 1, 1, 1
        )
        electrophysiologyLayout.addWidget(resistance_unit_label, 0, 2, 1, 1)
        electrophysiologyLayout.addWidget(capacitance_label, 1, 0, 1, 1)
        electrophysiologyLayout.addWidget(
            self.capacitance_value_label, 1, 1, 1, 1
        )
        electrophysiologyLayout.addWidget(capacitance_unit_label, 1, 2, 1, 1)
        electrophysiologyLayout.addWidget(ratio_label, 2, 0, 1, 1)
        electrophysiologyLayout.addWidget(self.ratio_value_label, 2, 1, 1, 1)
        electrophysiologyLayout.addWidget(membranevoltage_label, 3, 0, 1, 1)
        electrophysiologyLayout.addWidget(
            self.membranevoltage_value_label, 3, 1, 1, 1
        )
        electrophysiologyLayout.addWidget(
            membranevoltage_unit_label, 3, 2, 1, 1
        )
        electrophysiologyLayout.addWidget(request_zap_button, 4, 0, 1, 1)
        electrophysiologyContainer.setLayout(electrophysiologyLayout)

        """
        === Sensor display ===
        """
        sensorContainer = QGroupBox()
        sensorLayout = QGridLayout()

        # Algorithm plot
        algorithmPlotWidget = pg.PlotWidget()
        algorithmPlotWidget.setTitle("Sharpness graph")
        algorithmPlotWidget.setLabel("left", text="a.u.")
        algorithmPlotWidget.setLabel("bottom", text="height", units="μm")
        self.algorithmPlot = algorithmPlotWidget.plot(pen=(1, 4))

        # Current plot
        currentPlotWidget = pg.PlotWidget()
        currentPlotWidget.setTitle("Current")
        currentPlotWidget.setLabel("left", units="A")
        currentPlotWidget.setLabel("bottom", text="20 ms")
        self.currentPlot = currentPlotWidget.plot(pen=(2, 4))

        # Resistance plot
        resistancePlotWidget = pg.PlotWidget()
        resistancePlotWidget.setTitle("Resistance")
        resistancePlotWidget.setLabel("left", units="Ω")
        resistancePlotWidget.setLabel("bottom", text="20 ms")
        self.resistancePlot = resistancePlotWidget.plot(pen=(3, 4))

        # Pressure plot
        pressurePlotWidget = pg.PlotWidget()
        pressurePlotWidget.setTitle("Pressure")
        pressurePlotWidget.setLabel("left", units="mBar")
        pressurePlotWidget.setLabel("bottom", text="time", units="s")
        self.pressurePlot = pressurePlotWidget.plot(pen=(4, 4))

        # Reset plot button
        request_resetplots_button = QPushButton(
            text="Reset all plots", clicked=self.reset_plots
        )

        # Create plot tabs
        self.displayWidget = QTabWidget()
        self.displayWidget.addTab(algorithmPlotWidget, "Algorithm")
        self.displayWidget.addTab(resistancePlotWidget, "Resistance")
        self.displayWidget.addTab(currentPlotWidget, "Current")
        self.displayWidget.addTab(pressurePlotWidget, "Pressure")

        sensorLayout.addWidget(self.displayWidget, 0, 0, 1, 1)
        sensorLayout.addWidget(request_resetplots_button, 1, 0, 1, 1)
        sensorContainer.setLayout(sensorLayout)

        """
        === Add widgets and set Layout ===
        """
        layout = QGridLayout()
        layout.addWidget(hardwareContainer, 0, 0, 2, 1)
        layout.addWidget(stagemoveContainer, 2, 0, 2, 1)
        layout.addWidget(calibrationContainer, 4, 0, 1, 1)
        layout.addWidget(liveContainer, 0, 1, 5, 5)
        layout.addWidget(autopatchContainer, 0, 6, 2, 1)
        layout.addWidget(pressurecontrolContainer, 0, 7, 1, 1)
        layout.addWidget(electrophysiologyContainer, 1, 7, 1, 1)
        layout.addWidget(sensorContainer, 2, 6, 3, 2)
        self.setLayout(layout)

        """
        === Stack old GUI's ===
        """
        """
        === Start up roi manager ===
        """
        self.roimanager = ROIManagerGUI(offset=len(self.liveView.addedItems))

        """
        === Start up backend and connect signals/slots ===
        """
        self.backend = SmartPatcher()
        self.backend.worker.draw.connect(self.draw_roi)
        self.backend.worker.graph1.connect(self.update_algorithmplot)
        self.backend.worker.graph2.connect(self.update_resistanceplot)
        self.backend.worker.status.connect(self.update_autopatch_status)
        self.backend.worker.progress.connect(self.update_autopatch_progress)
        self.backend.worker.finished.connect(
            self.update_constants_from_backend
        )

        """
        === Update constant from backend ===
        """
        self.update_constants_from_backend()

        """
        === End of GUI ===
        """

    def request_imagexygrid(self):  # FLAG: relevant for MSc thesis
        self.backend.request(
            name="request_imagexygrid"
        )  # FLAG: relevant for MSc thesis

    def request_imagezstack(self):  # FLAG: relevant for MSc thesis
        self.backend.request(
            name="request_imagezstack"
        )  # FLAG: relevant for MSc thesis

    def connect_camerathread(self):
        """
        We initiate the camera by creating the CameraThread object, then we
        connect signals with slots to govern inter-thread communication, i.e.
        to recieve images to display. Afterwards we move the camera thread to
        the backend where the thread is immediately started.
        """
        if self.connect_camerathread_button.isChecked():
            try:
                camerathread = CameraThread(camerahandle=None)
                self.signal_camera_live = camerathread.livesignal
                self.signal_camera_live.connect(self.update_live)
                self.backend.camerathread = camerathread
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.connect_camerathread_button.setChecked(False)
                raise
        else:
            del self.backend.camerathread
            del self.signal_camera_live

    def connect_objectivemotor(self):
        """
        We initiate the objective motor by creating the PIMotor object. This
        takes time in which the GUI freezes, this issue can be solved by
        passing the objective motor instance from Tupolev/Fiumicino as an
        objective_motor_handle. Deleting the object disconnects the objective
        motor.
        """
        if self.connect_objectivemotor_button.isChecked():
            try:
                objectivemotor = PIMotor(objective_motor_handle=None)
                self.backend.objectivemotor = objectivemotor
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.connect_objectivemotor_button.setChecked(False)
                raise
        else:
            del self.backend.objectivemotor

    def connect_micromanipulator(self):
        """
        We initiate the micromanipulator by creating the ScientificaPatchStar
        serial object. Creating the object object automatically opens the USB
        port that the device is connected to, similarly, deleting the object
        stops micromanipulator movement and closes the port.
        """
        if self.connect_micromanipulator_button.isChecked():
            try:
                micromanipulator = ScientificaPatchStar(
                    address="COM16", baud=38400
                )
                self.backend.micromanipulator = micromanipulator
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.connect_micromanipulator_button.setChecked(False)
                raise
        else:
            del self.backend.micromanipulator

    def connect_XYstage(self):
        """
        We initiate the XYstage (sample stage) by creating the LudlStage object
        Creating the object does not automatically open the USB port to which
        the device is connected to. Deleting the object stops stage movement.
        """
        if self.connect_XYstage_button.isChecked():
            try:
                ludlStage = LudlStage("COM12")
                self.backend.XYstage = ludlStage
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.connect_XYstage_button.setChecked(False)
                raise
        else:
            del self.backend.XYstage

    def connect_sealtestthread(self):
        """
        We initiate the sealtest thread by creating the SealTestThread object,
        then we connect the measurement signal to update the graph for voltage
        and current. Afterwards we move the sealtest thread to the backend
        where the thread is immediately started.
        """
        if self.connect_sealtestthread_button.isChecked():
            try:
                sealtestthread = SealTestThread()
                self.signal_sealtest = sealtestthread.measurement
                self.signal_sealtest.connect(self.update_current_voltage)
                self.backend.sealtestthread = sealtestthread
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.connect_sealtestthread_button.setChecked(False)
                raise
        else:
            del self.backend.sealtestthread
            del self.signal_sealtest

    def connect_pressurethread(self):
        """
        We initiate the pressure thread by creating the PressureThread object,
        then we connect the measurement signal to update the pressure evolution
        graph. Afterwards we move the sealtest thread to the backend
        where the thread is immediately started.
        """
        if self.connect_pressurecontroller_button.isChecked():
            try:
                pressurethread = PressureThread(pressurecontroller_handle=None)
                self.signal_pressure = pressurethread.measurement
                self.signal_pressure.connect(self.update_pressure)
                self.backend.pressurethread = pressurethread
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                self.connect_sealtestthread_button.setChecked(False)
                raise
        else:
            del self.backend.pressurethread
            del self.signal_pressure

    def request_stage_up(self):
        # Convert pixels to ludl indices (1577 ludl index = 340 um)
        step_height = int(
            self.backend.image_size[0]
            * self.backend.pixel_size
            / 1000
            * 1577
            / 340
        )
        self.backend.XYstage.moveRel(xRel=step_height, yRel=0)

    def request_stage_down(self):
        # Convert pixels to ludl indices (1577 ludl index = 340 um)
        step_height = int(
            self.backend.image_size[0]
            * self.backend.pixel_size
            / 1000
            * 1577
            / 340
        )
        self.backend.XYstage.moveRel(xRel=-step_height, yRel=0)

    def request_stage_right(self):
        # Convert pixels to ludl indices (1577 ludl index = 340 um)
        step_width = int(
            self.backend.image_size[1]
            * self.backend.pixel_size
            / 1000
            * 1577
            / 340
        )
        self.backend.XYstage.moveRel(xRel=0, yRel=-step_width)

    def request_stage_left(self):
        # Convert pixels to ludl indices (1577 ludl index = 340 um)
        step_width = int(
            self.backend.image_size[1]
            * self.backend.pixel_size
            / 1000
            * 1577
            / 340
        )
        self.backend.XYstage.moveRel(xRel=0, yRel=step_width)

    def request_target2center(self):
        self.backend.request(name="target2center")

    def request_calibrate_xy(self):
        orientations = ["North", "East", "South", "West"]
        orientations_inradians = [np.pi / 2, np.pi, 3 * np.pi / 2, 0]
        orientation, ok = QInputDialog().getItem(
            self,
            "Calibration options",
            "Pipette orientation",
            orientations,
            3,
            False,
        )
        if ok:
            self.backend.pipette_orientation = orientations_inradians[
                orientations.index(orientation)
            ]
            self.backend.request(name="hardcalibration", mode="XY")

    def request_calibrate_pixelsize(self):
        diameter, ok = QInputDialog().getInt(
            self,
            "Calibration options",
            "Pipette diamter (in pixels)",
            value=16,
        )
        if ok:
            self.backend.pipette_diameter = diameter
            self.backend.request(name="hardcalibration", mode="pixelsize")

    def toggle_pauselive(self):
        if hasattr(self, "signal_camera_live"):
            try:
                self.signal_camera_live.disconnect()
            except TypeError:
                pass
            if self.request_pause_button.isChecked():
                self.request_pause_button.setChecked(True)
            else:
                self.signal_camera_live.connect(self.update_live)
                self.request_pause_button.setChecked(False)
        else:
            self.request_pause_button.setChecked(False)

    def clearROIs(self):
        self.roimanager.removeallROIs()
        for roi in self.liveView.addedItems[self.roimanager.offset : :]:
            self.liveView.removeItem(roi)

    def reset_imageview(self):
        self.liveView.autoRange()

    def request_snap(self):
        if self.backend.camerathread is not None:
            img = self.backend.camerathread.snap()
            io.imsave("snapshot.tif", img, check_contrast=False)
            self.toggle_pauselive()
            self.update_live(img)
        else:
            raise AttributeError("no camera connected")

    def request_prechecks(self):
        self.displayWidget.setCurrentIndex(3)
        self.backend.request(name="pre-checks")

    def request_selecttarget(self):
        """
        The user drags a circular ROI on top of the target cell. The ROI center
        is the target. We first check if a target already exists, if so, we
        recycle it. If the target ROI got removed then we place it back to its
        last known position.
        """
        self.request_pause_button.setChecked(True)
        self.toggle_pauselive()

        coords = self.backend.target_coordinates
        if coords[0] is None and coords[1] is None:
            coords = (0, 0)
        else:
            coords = (coords[0] - 60, coords[1] - 60)

        if not self.roimanager.contains("target"):
            target = pg.CircleROI(
                pos=coords,
                radius=60,
                movable=True,
                pen=QPen(QColor(255, 255, 0), 0),
            )
            target.setZValue(10)
            self.roimanager.addROI("target")
            self.liveView.addItem(target)
        else:
            idx = self.roimanager.giveROIindex("target")[-1]
            self.liveView.addedItems[idx].translatable = True
            self.liveView.addedItems[idx].setPen(QPen(QColor(255, 255, 0), 0))

    def request_confirmtarget(self):
        """
        If a target is selected, we save the center coordinates of the ROI in
        the camera field of reference.
        """
        if self.roimanager.contains("target"):
            idx = self.roimanager.giveROIindex("target")[-1]
            x, y = (
                self.liveView.addedItems[idx].state["pos"]
                + self.liveView.addedItems[idx].state["size"] / 2
            )
            self.liveView.addedItems[idx].translatable = False
            self.liveView.addedItems[idx].setPen(
                QPen(QColor(193, 245, 240), 0)
            )
            self.backend.target_coordinates = np.array([x, y, None])

    def request_start_autopatch(self):
        check = QMessageBox.question(
            self,
            "Start autopatch",
            "Is the pipette tip inside the field-of-view?",
            QMessageBox.No | QMessageBox.Yes,
        )
        if check == QMessageBox.Yes:
            self.displayWidget.setCurrentIndex(0)
            self.backend.request(name="autopatch")

    def request_manual_adjustment(self):
        """
        The user drags a circular ROI from the detected tip to the real tip.The
        ROI center is the tip location. We first check if a roi already exists,
        if so, we recycle it. If the tip ROI got removed then we place it back
        to its last known position.
        """
        self.request_pause_button.setChecked(True)
        self.toggle_pauselive()
        self.clearROIs()

        coords = self.backend.pipette_coordinates_pair[1]
        if all(values is None for values in coords):
            coords = (0, 0)
        else:
            coords = (coords[0] - 30, coords[1] - 30)

        if not self.roimanager.contains("tip"):
            target = pg.CircleROI(
                pos=coords,
                radius=30,
                movable=True,
                pen=QPen(QColor(255, 0, 255), 0),
            )
            target.setZValue(10)
            self.roimanager.addROI("tip")
            self.liveView.addItem(target)
        else:
            idx = self.roimanager.giveROIindex("tip")[-1]
            self.liveView.addedItems[idx].translatable = True
            self.liveView.addedItems[idx].setPen(QPen(QColor(255, 0, 255), 0))

    def request_adjustment_confirm(self):
        """
        If an adjustment is made, we save the center coordinates of the ROI in
        the camera field of reference.
        """
        if self.roimanager.contains("tip"):
            idx = self.roimanager.giveROIindex("tip")[-1]
            x, y = (
                self.liveView.addedItems[idx].state["pos"]
                + self.liveView.addedItems[idx].state["size"] / 2
            )
            self.liveView.addedItems[idx].translatable = False
            self.liveView.addedItems[idx].setPen(
                QPen(QColor(193, 245, 240), 0)
            )
            coords_pair = self.backend.pipette_coordinates_pair
            coords_pair[1, 0] = x
            coords_pair[1, 1] = y
            self.backend.pipette_coordinates_pair = coords_pair

    def request_start_approach(self):
        self.displayWidget.setCurrentIndex(1)
        self.backend.request(name="approach")

    def request_start_gigaseal(self):
        self.displayWidget.setCurrentIndex(1)
        self.backend.request(name="gigaseal")

    def request_start_breakin(self):
        self.displayWidget.setCurrentIndex(2)
        self.backend.request(name="break-in")

    def request_emergency_stop(self):
        if self.request_stop_button.isChecked():
            self.backend.emergency_stop(True)
            self.backend.stop_moving_hardware()
        else:
            self.backend.emergency_stop(False)

    def request_low_positive(self):
        pressure = 30
        self.backend.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_low_negative(self):
        pressure = -30
        self.backend.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_high_positive(self):
        pressure = 100
        self.backend.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_high_negative(self):
        pressure = -100
        self.backend.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_stop_regulating(self):
        pressure = self.set_pressure_spinbox.value()
        self.backend.pressurethread.set_pressure_hold_stop_waveform(pressure)
        self.pressure_status_label.setText("Not regulating")

    def request_atmosphere(self):
        pressure = 0
        self.backend.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Not regulating")

    def request_apply_pressure(self):
        pressure = self.set_pressure_spinbox.value()
        self.backend.pressurethread.set_pressure_stop_waveform(pressure)
        self.pressure_status_label.setText("Regulating")

    def request_apply_pulse(self):
        pressure = self.set_pressure_spinbox.value()
        self.backend.pressurethread.set_pulse_stop_waveform(pressure)
        self.pressure_status_label.setText("Doing pulse")

    def toggle_lcd(self):
        if self.backend.pressurethread is not None:
            if self.toggle_lcd_button.isChecked():
                self.backend.pressurethread.pressurecontroller.LCDoff()
            else:
                self.backend.pressurethread.pressurecontroller.LCDon()
        else:
            self.toggle_lcd_button.setChecked(False)

    def request_zap(self):
        self.backend.sealtestthread.zap()

    def update_live(self, image):
        """
        Image data is retrieved per image and directly displayed.
        """
        self.liveImageItem.setImage(image)

    def update_algorithmplot(self, data):
        self.algorithmPlot.setData(data[0, :], data[1, :])

    def update_resistanceplot(self, data):
        self.resistancePlot.setData(data)

    def update_current_voltage(self, voltOut, curOut):
        """
        Voltage and current are retrieved, converted to their right units, and
        processed in the backend. We retrieve the voltage and current as an
        array that we can display and/or further process to calculate the
        resistance and capacitance from.
        """
        self.backend._voltage_append_(voltOut / 10)
        self.backend._current_append_(curOut / 1 / (100 * 10**6))
        # voltage = self.backend.voltage
        current = self.backend.current
        self.currentPlot.setData(current)
        self.update_resistance_capacitance(
            self.backend.current, self.backend.voltage
        )

    def update_resistance_capacitance(self, curOut, voltOut):
        """Update the resistance and capacitance labels.
        http://scipy-lectures.org/intro/scipy/auto_examples/plot_curve_fit.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        This function was taken from lhuismans and modified by TvdrBurgt.
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
            dIss = np.mean(
                curData[
                    int(np.floor(0.15 * sampPerCyc)) : int(
                        np.floor(sampPerCyc / 2)
                    )
                    - 2
                ]
            ) - np.mean(
                curData[int(np.floor(0.65 * sampPerCyc)) : sampPerCyc - 2]
            )  # Computing the current distance
            membraneResistance = dV / (dIss * 1000000)  # Ohms law (MegaOhm)
            self.resistance_value_label.setText(
                "{:.2f}".format(membraneResistance)
            )
            R_to_append = membraneResistance * 1e6

            # The resistance of a typical patch of membrane,
            # RM is 10000 Omega/{cm}^2
            estimated_size_resistance = 10000 / (membraneResistance * 1000000)
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.resistance_value_label.setText("NaN")
            R_to_append = np.nan

        try:
            measured_vlotage = np.mean(voltData) * 1000
            self.membranevoltage_value_label.setText(
                "{:.1f}".format(measured_vlotage)
            )
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.membranevoltage_value_label.setText("NaN")
        try:
            # Computing capacitance
            points = 10
            maxCur = np.amax(curData)
            maxCurIndex = np.where(curData == maxCur)[0][0]
            curFit = curData[
                int(maxCurIndex + 1) : int(maxCurIndex + 1 + points - 1)
            ] - 0.5 * (
                np.mean(
                    curData[
                        int(np.floor(0.15 * sampPerCyc)) : int(
                            np.floor(sampPerCyc / 2)
                        )
                        - 2
                    ]
                )
                + np.mean(
                    curData[int(np.floor(0.65 * sampPerCyc)) : sampPerCyc - 2]
                )
            )
            timepoints = (
                1000
                * np.arange(3, points - 1 + 3)
                / constants.patchSealSampRate
            )

            # Fitting the data to an exponential of the form y=a*exp(-b*x)
            # where b = 1/tau and tau = RC

            # I(t)=I0*e^−t/τ, y=a*exp(-b*x), get log of both sides:
            # log y = -bx + log a

            # Converting the exponential to a linear function and fitting it
            fit = np.polyfit(timepoints, curFit, 1)

            # Extracting data
            current = fit[0]
            resistance = dV * 1000 / current / 2  # Getting the resistance
            tau = -1 / fit[1]
            capacitance = 1000 * tau / resistance
            self.capacitance_value_label.setText("{:.1f}".format(capacitance))
            C_to_append = capacitance

            estimated_size_capacitance = capacitance * (10**-12) * (10**6)

            if estimated_size_capacitance > estimated_size_resistance:
                estimated_ratio = (
                    estimated_size_capacitance / estimated_size_resistance
                )
            else:
                estimated_ratio = (
                    estimated_size_resistance / estimated_size_capacitance
                )

            self.ratio_value_label.setText(
                "{:.1f}".format(estimated_ratio)
            )  # http://www.cnbc.cmu.edu/~bard/passive2/node5.html

        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.capacitance_value_label.setText("NaN")
            C_to_append = np.nan
            self.ratio_value_label.setText("NaN")

        # store resistance and capacitance values in backend
        self.backend._resistance_append_(R_to_append)
        self.backend._capacitance_append_(C_to_append)

    def update_pressure(self, data):
        """
        Pressure data is retrieved as [pressure, timing] and processed in the
        backend. We retrieve the processed pressure as an array that we can
        display.
        """
        self.backend._pressure_append_(data[0], data[1])
        pressuredata = self.backend.pressure
        self.pressurePlot.setData(pressuredata[1], pressuredata[0])
        self.pressure_value_Label.setText("{:.1f}".format(pressuredata[0, -1]))

    def reset_plots(self):
        del self.backend.current
        del self.backend.voltage
        del self.backend.pressure
        self.algorithmPlot.setData([0])
        self.currentPlot.setData([0])
        self.resistancePlot.setData([0])
        self.pressurePlot.setData([0])

    def update_autopatch_status(self, message):
        self.autopatch_status.setText(message)

    def update_autopatch_progress(self, message):
        self.autopatch_progress.setText(message)

    def update_constants_from_backend(self):
        # retrieve constants from backend
        alpha = self.backend.rotation_angles[0] * 180 / np.pi
        beta = self.backend.rotation_angles[1] * 180 / np.pi
        gamma = self.backend.rotation_angles[2] * 180 / np.pi
        pixelsize = self.backend.pixel_size

        # update labels
        self.rotation_angles_value_label.setText(
            "({:.2f}".format(alpha)
            + ", {:.2f}, ".format(beta)
            + "{:.2f})".format(gamma)
        )
        self.pixelsize_unit_label.setText(
            "{:.1f}".format(pixelsize) + "x{:.1f}".format(pixelsize)
        )

    def draw_roi(self, *args):
        label = args[0][0]
        if label == "cross":
            xpos, ypos = args[0][1:3]
            vertical = pg.ROI(pos=(xpos, ypos - 15), size=[1, 30], pen=(1, 3))
            horizontal = pg.ROI(
                pos=(xpos - 15, ypos), size=[30, 1], pen=(1, 3)
            )
            vertical.setZValue(10)
            horizontal.setZValue(10)
            self.roimanager.addROI("cross" + "_vertical")
            self.roimanager.addROI("cross" + "_horizontal")
            self.liveView.addItem(vertical)
            self.liveView.addItem(horizontal)
        elif label == "calibrationline":
            xpos, ypos, orientation = args[0][1:4]
            if len(self.roimanager.giveROIindex("calibrationline")) == 0:
                P = (0, 3)
            elif len(self.roimanager.giveROIindex("calibrationline")) == 1:
                P = (2, 3)
            else:
                P = (1, 3)
            horizontal = pg.ROI(
                pos=(xpos - 15, ypos), angle=orientation, size=[500, 1], pen=P
            )
            horizontal.setZValue(10)
            self.roimanager.addROI("calibrationline")
            self.liveView.addItem(horizontal)
        elif label == "image":
            img = args[0][1]
            self.update_live(img)
        elif label == "target":
            dx, dy = args[0][1:3]
            idx = self.roimanager.giveROIindex("target")[-1]
            x, y = self.liveView.addedItems[idx].state["pos"]
            self.liveView.addedItems[idx].setPos([x - dx, y - dy])
        else:
            logging.info(f"{label} is not a known draw-label")

    def closeEvent(self, event):
        """Close event
        This method is called when the GUI is shut down. First we need to stop
        the threads that are still running, then we disconnect all hardware to
        be reused in the main widget, only then we accept the close event.
        and quit the widget.
        """
        self.backend.emergency_stop(True)

        try:
            del self.backend.camerathread
        except AttributeError:
            pass
        try:
            del self.backend.objectivemotor
        except AttributeError:
            pass
        try:
            del self.backend.micromanipulator
        except AttributeError:
            pass
        try:
            del self.backend.XYstage
        except AttributeError:
            pass
        try:
            del self.backend.sealtestthread
        except AttributeError:
            pass
        try:
            del self.backend.pressurethread
        except AttributeError:
            pass

        event.accept()

        # Frees the console by quitting the application entirely
        # QtWidgets.QApplication.quit() # remove when part of Tupolev!!


class ROIManagerGUI:
    def __init__(self, offset):
        self.offset = offset
        self.ROInumber = 0
        self.ROIdictionary = {}

    def giveROIindex(self, name):
        if self.contains(name):
            index_array = [
                x + self.offset for x in self.ROIdictionary[name][:]
            ]
        else:
            index_array = []
        return index_array

    def addROI(self, name):
        if name in self.ROIdictionary:
            self.ROIdictionary[name] = self.ROIdictionary[name] + [
                self.ROInumber
            ]
            self.ROInumber += 1
        else:
            self.ROIdictionary[name] = [self.ROInumber]
            self.ROInumber += 1

    def removeROI(self, name, args):
        if args == "all":
            self.ROInumber -= len(self.ROIdictionary[name])
            del self.ROIdictionary[name]
        else:
            try:
                index = int(args)
            except (ValueError, TypeError):
                raise ValueError(
                    f"args should be and index or 'all' not '{args}'"
                ) from None

            entries = len(self.ROIdictionary[name])
            if entries > index:
                self.ROInumber -= index
                self.ROIdictionary[name] = self.ROIdictionary[name][:-index]
            elif entries == index:
                self.ROInumber -= index
                del self.ROIdictionary[name]
            else:
                raise ValueError(f"index out of range ({index} > {entries})")

    def removeallROIs(self):
        names = list(self.ROIdictionary)
        for name in names:
            self.removeROI(name, args="all")

    def contains(self, name):
        return name in self.ROIdictionary


if __name__ == "__main__":
    # Create a palette to switch to dark mode colors
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setPalette(palette)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = PatchClampUI()
        mainwin.show()
        app.exec_()

    run_app()
