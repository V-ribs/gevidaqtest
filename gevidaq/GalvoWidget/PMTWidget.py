# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:34:56 2020

@author: xinmeng
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pyqtgraph as pg
import tifffile as skimtiff
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)
from scipy.interpolate import splev, splprep

from .. import StylishQT
from ..GeneralUsage.ThreadingFunc import run_in_thread
from ..NIDAQ.constants import HardwareConstants
from ..NIDAQ.DAQoperator import DAQmission
from .GalvoScan_backend import PMT_zscan
from .pmt_thread import pmtimagingTest, pmtimagingTest_contour


class PMTWidgetUI(QWidget):

    SignalForContourScanning = pyqtSignal(
        int, int, int, np.ndarray, np.ndarray
    )
    GalvoCoordinatesCommand = pyqtSignal(int, int)
    MessageBack = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        self.setMinimumSize(1200, 850)
        self.setWindowTitle("PMTWidget")
        self.layout = QGridLayout(self)
        # === Initiating class ===
        self.pmtTest = pmtimagingTest()
        self.pmtTest_contour = pmtimagingTest_contour()

        self.savedirectory = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\pmt_image_default_dump"  # TODO hardcoded path
        self.prefixtextboxtext = "_fromGalvoWidget"

        self.contour_ROI_signals_dict = {}
        self.contour_ROI_handles_dict = {}

        self.clicked_points_list = []
        self.flag_is_drawing = False

        # === GUI for PMT tab ===
        pmtimageContainer = StylishQT.roundQGroupBox(title="PMT image")
        self.pmtimageLayout = QGridLayout()

        # Configure the PMT video ImageItem and viewbox
        self.pmtvideoWidget = pg.ImageView()
        self.pmtvideoWidget.ui.roiBtn.hide()
        self.pmtvideoWidget.ui.menuBtn.hide()
        self.pmtvideoWidget.resize(500, 500)
        self.pmtimageLayout.addWidget(self.pmtvideoWidget, 0, 0)

        self.pmt_viewbox = self.pmtvideoWidget.getView()
        self.pmtimageitem = self.pmtvideoWidget.getImageItem()
        self.pmt_viewbox.scene().sigMouseClicked.connect(
            self.generate_poly_roi
        )
        self.pmt_viewbox.scene().sigMouseClicked.connect(
            self.update_pixel_coords_and_intensity
        )

        pmtroiContainer = StylishQT.roundQGroupBox(title="PMT ROI")
        self.pmtimageroiLayout = QGridLayout()

        self.pmt_roiwidget = pg.GraphicsLayoutWidget()
        self.pmt_roiwidget.resize(150, 150)
        self.pmt_roiwidget.addLabel("ROI", row=0, col=0)

        self.pmtimageroiLayout.addWidget(self.pmt_roiwidget, 0, 0)
        # === create ROI ===
        self.vb_2 = self.pmt_roiwidget.addViewBox(
            row=1, col=0, lockAspect=True, colspan=1
        )
        self.vb_2.name = "ROI"

        self.pmtimgroi = pg.ImageItem()
        self.vb_2.addItem(self.pmtimgroi)
        self.ROIpen = QPen()  # creates a default pen
        self.ROIpen.setStyle(Qt.DashDotLine)
        self.ROIpen.setWidthF(0.03)
        self.ROIpen.setBrush(QColor(0, 161, 255))

        self.roi = pg.PolyLineROI(
            [[0, 0], [80, 0], [80, 80], [0, 80]], closed=True, pen=self.ROIpen
        )  # , maxBounds=r1
        self.roi.sigHoverEvent.connect(
            lambda: self.show_handle_num(self.roi)
        )  # update handle numbers

        pmtimageContainer.setMinimumWidth(850)
        pmtroiContainer.setFixedHeight(320)

        pmtimageContainer.setLayout(self.pmtimageLayout)
        pmtroiContainer.setLayout(self.pmtimageroiLayout)

        self.camera_generated_contour_voltages = None

        # === Contour ===
        pmtContourContainer = StylishQT.roundQGroupBox(
            title="Contour selection"
        )
        pmtContourContainer.setFixedHeight(220)
        self.pmtContourLayout = QGridLayout()

        self.pmt_handlenum_Label = QLabel("Handle number: _")
        self.pmtContourLayout.addWidget(self.pmt_handlenum_Label, 1, 2)

        self.contour_strategy = QComboBox()
        self.contour_strategy.addItems(["Uniform", "Evenly between"])
        self.contour_strategy.setToolTip(
            "Even in-between: points evenly distribute inbetween handles; "
            "Uniform: evenly distribute regardless of handles"
        )
        self.pmtContourLayout.addWidget(self.contour_strategy, 0, 3)
        self.pmtContourLayout.addWidget(QLabel("Contour strategy:"), 0, 2)

        self.pointsinContour = QSpinBox(self)
        self.pointsinContour.setMinimum(1)
        self.pointsinContour.setMaximum(10000)
        self.pointsinContour.setValue(100)
        self.pointsinContour.setSingleStep(100)
        self.pmtContourLayout.addWidget(self.pointsinContour, 0, 1)
        self.pmtContourLayout.addWidget(QLabel("Points in contour:"), 0, 0)

        self.contour_samplerate = QSpinBox(self)
        self.contour_samplerate.setMinimum(0)
        self.contour_samplerate.setMaximum(1000000)
        self.contour_samplerate.setValue(50000)
        self.contour_samplerate.setSingleStep(50000)
        self.pmtContourLayout.addWidget(self.contour_samplerate, 0, 5)
        self.pmtContourLayout.addWidget(QLabel("Sampling rate:"), 0, 4)

        self.pmtContourLayout.addWidget(QLabel("Contour index:"), 1, 0)
        self.roi_index_spinbox = QSpinBox(self)
        self.roi_index_spinbox.setMinimum(1)
        self.roi_index_spinbox.setMaximum(20)
        self.roi_index_spinbox.setValue(1)
        self.roi_index_spinbox.setSingleStep(1)
        self.pmtContourLayout.addWidget(self.roi_index_spinbox, 1, 1)

        self.go_to_first_handle_button = StylishQT.GeneralFancyButton(
            label="Go to 1st point"
        )
        self.go_to_first_handle_button.setFixedHeight(32)
        self.go_to_first_handle_button.clicked.connect(self.go_to_first_point)
        self.go_to_first_handle_button.setToolTip(
            "Set galvo initial positions in advance"
        )
        self.go_to_first_handle_button.setEnabled(True)
        self.go_to_first_handle_button.setVisible(True)
        self.pmtContourLayout.addWidget(self.go_to_first_handle_button, 2, 3)

        # ROI_interaction_tips = QLabel("Hover for tips. Key F:en/disable drawing ROI")
        # ROI_interaction_tips.setToolTip("Left drag moves the ROI\n\
        # Left drag + Ctrl moves the ROI with position snapping\n\
        # Left drag + Alt rotates the ROI\n\
        # Left drag + Alt + Ctrl rotates the ROI with angle snapping\n\
        # Left drag + Shift scales the ROI\n\
        # Left drag + Shift + Ctrl scales the ROI with size snapping")
        # self.pmtContourLayout.addWidget(ROI_interaction_tips, 4, 0, 1, 2)

        self.regenerate_roi_handle_button = StylishQT.GeneralFancyButton(
            label="Regain ROI"
        )
        self.regenerate_roi_handle_button.setFixedHeight(32)
        self.pmtContourLayout.addWidget(
            self.regenerate_roi_handle_button, 2, 4
        )
        self.regenerate_roi_handle_button.clicked.connect(
            self.regenerate_roi_handles
        )

        self.reset_roi_handle_button = StylishQT.GeneralFancyButton(
            label="Reset handles"
        )
        self.reset_roi_handle_button.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.reset_roi_handle_button, 1, 3)
        self.reset_roi_handle_button.clicked.connect(self.reset_roi_handles)

        # Button to add roi to stack
        self.add_roi_to_stack_button = StylishQT.addButton("Add ROI")
        self.add_roi_to_stack_button.setToolTip(
            "Add current ROI to the stack at the contour index"
        )
        self.add_roi_to_stack_button.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.add_roi_to_stack_button, 1, 4)
        self.add_roi_to_stack_button.clicked.connect(
            self.add_coordinates_to_list
        )

        self.del_roi_in_stack_button = StylishQT.stop_deleteButton(
            "Delete ROI"
        )
        self.del_roi_in_stack_button.setToolTip(
            "Delete current ROI from the stack at the contour index"
        )
        self.del_roi_in_stack_button.setFixedHeight(32)
        self.del_roi_in_stack_button.clicked.connect(
            self.del_coordinates_from_list
        )
        self.pmtContourLayout.addWidget(self.del_roi_in_stack_button, 1, 5)

        self.reset_roi_stack_button = StylishQT.cleanButton("Clear all")
        self.reset_roi_stack_button.setFixedHeight(32)
        self.reset_roi_stack_button.setToolTip(
            "Clear all the ROIs in the stack"
        )
        self.pmtContourLayout.addWidget(self.reset_roi_stack_button, 2, 5)
        self.reset_roi_stack_button.clicked.connect(
            self.reset_coordinates_dict
        )

        self.generate_contour_scan = StylishQT.generateButton(
            "Generate contour"
        )
        self.pmtContourLayout.addWidget(self.generate_contour_scan, 2, 0)
        self.generate_contour_scan.clicked.connect(
            lambda: self.generate_final_contour_signals()
        )

        self.do_contour_scan = StylishQT.runButton("Execute scan")
        self.do_contour_scan.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.do_contour_scan, 2, 1)
        self.do_contour_scan.clicked.connect(
            lambda: self.buttonenabled("contourscan", "start")
        )
        self.do_contour_scan.clicked.connect(
            lambda: self.measure_pmt_contourscan()
        )
        self.do_contour_scan.setEnabled(False)

        self.stopButton_contour = StylishQT.stop_deleteButton("Stop scan")
        self.stopButton_contour.setFixedHeight(32)
        self.stopButton_contour.clicked.connect(
            lambda: self.buttonenabled("contourscan", "stop")
        )
        self.stopButton_contour.clicked.connect(
            lambda: self.stopMeasurement_pmt_contour()
        )
        self.stopButton_contour.setVisible(False)
        self.pmtContourLayout.addWidget(self.stopButton_contour, 2, 2)

        pmtContourContainer.setLayout(self.pmtContourLayout)

        # === Control ===
        self.scanning_tabs = QTabWidget()
        self.scanning_tabs.setFixedWidth(290)
        self.scanning_tabs.setFixedHeight(320)

        # === Continuous scanning ===
        Continuous_widget = QWidget()
        controlLayout = QGridLayout()

        self.pmt_fps_Label = QLabel("Per frame: ")
        controlLayout.addWidget(self.pmt_fps_Label, 5, 0)

        self.saveButton_pmt = StylishQT.saveButton()
        self.saveButton_pmt.clicked.connect(lambda: self.saveimage_pmt())
        controlLayout.addWidget(self.saveButton_pmt, 5, 1)

        self.startButton_pmt = StylishQT.runButton("")
        self.startButton_pmt.setFixedHeight(32)
        self.startButton_pmt.setCheckable(True)
        self.startButton_pmt.clicked.connect(
            lambda: self.buttonenabled("rasterscan", "start")
        )
        self.startButton_pmt.clicked.connect(lambda: self.measure_pmt())

        controlLayout.addWidget(self.startButton_pmt, 6, 0)

        self.stopButton = StylishQT.stop_deleteButton()
        self.stopButton.setFixedHeight(32)
        self.stopButton.clicked.connect(
            lambda: self.buttonenabled("rasterscan", "stop")
        )
        self.stopButton.clicked.connect(lambda: self.stopMeasurement_pmt())
        self.stopButton.setEnabled(False)
        controlLayout.addWidget(self.stopButton, 6, 1)

        # === Galvo scanning ===
        self.continuous_scanning_sr_spinbox = QSpinBox(self)
        self.continuous_scanning_sr_spinbox.setMinimum(0)
        self.continuous_scanning_sr_spinbox.setMaximum(1000000)
        self.continuous_scanning_sr_spinbox.setValue(250000)
        self.continuous_scanning_sr_spinbox.setSingleStep(100000)
        controlLayout.addWidget(self.continuous_scanning_sr_spinbox, 1, 1)
        controlLayout.addWidget(QLabel("Sampling rate:"), 1, 0)

        # controlLayout.addWidget(QLabel("Galvo raster scanning : "), 1, 0)
        self.continuous_scanning_Vrange_spinbox = QSpinBox(self)
        self.continuous_scanning_Vrange_spinbox.setMinimum(-10)
        self.continuous_scanning_Vrange_spinbox.setMaximum(10)
        self.continuous_scanning_Vrange_spinbox.setValue(3)
        self.continuous_scanning_Vrange_spinbox.setSingleStep(1)
        controlLayout.addWidget(self.continuous_scanning_Vrange_spinbox, 2, 1)
        controlLayout.addWidget(QLabel("Volt range:"), 2, 0)

        self.Scanning_pixel_num_combobox = QSpinBox(self)
        self.Scanning_pixel_num_combobox.setMinimum(0)
        self.Scanning_pixel_num_combobox.setMaximum(1000)
        self.Scanning_pixel_num_combobox.setValue(500)
        self.Scanning_pixel_num_combobox.setSingleStep(244)
        controlLayout.addWidget(self.Scanning_pixel_num_combobox, 3, 1)
        controlLayout.addWidget(QLabel("Pixel number:"), 3, 0)

        self.continuous_scanning_average_spinbox = QSpinBox(self)
        self.continuous_scanning_average_spinbox.setMinimum(1)
        self.continuous_scanning_average_spinbox.setMaximum(20)
        self.continuous_scanning_average_spinbox.setValue(1)
        self.continuous_scanning_average_spinbox.setSingleStep(1)
        controlLayout.addWidget(self.continuous_scanning_average_spinbox, 4, 1)
        controlLayout.addWidget(QLabel("average over:"), 4, 0)

        Continuous_widget.setLayout(controlLayout)

        # === stack scanning ===
        Zstack_widget = QWidget()
        Zstack_Layout = QGridLayout()

        self.stack_scanning_sampling_rate_spinbox = QSpinBox(self)
        self.stack_scanning_sampling_rate_spinbox.setMinimum(0)
        self.stack_scanning_sampling_rate_spinbox.setMaximum(1000000)
        self.stack_scanning_sampling_rate_spinbox.setValue(250000)
        self.stack_scanning_sampling_rate_spinbox.setSingleStep(100000)
        Zstack_Layout.addWidget(
            self.stack_scanning_sampling_rate_spinbox, 1, 1
        )
        Zstack_Layout.addWidget(QLabel("Sampling rate:"), 1, 0)

        self.stack_scanning_Vrange_spinbox = QSpinBox(self)
        self.stack_scanning_Vrange_spinbox.setMinimum(-10)
        self.stack_scanning_Vrange_spinbox.setMaximum(10)
        self.stack_scanning_Vrange_spinbox.setValue(3)
        self.stack_scanning_Vrange_spinbox.setSingleStep(1)
        Zstack_Layout.addWidget(self.stack_scanning_Vrange_spinbox, 2, 1)
        Zstack_Layout.addWidget(QLabel("Volt range:"), 2, 0)

        self.stack_scanning_Pnumber_spinbox = QSpinBox(self)
        self.stack_scanning_Pnumber_spinbox.setMinimum(0)
        self.stack_scanning_Pnumber_spinbox.setMaximum(1000)
        self.stack_scanning_Pnumber_spinbox.setValue(500)
        self.stack_scanning_Pnumber_spinbox.setSingleStep(244)
        Zstack_Layout.addWidget(self.stack_scanning_Pnumber_spinbox, 3, 1)
        Zstack_Layout.addWidget(QLabel("Pixel number:"), 3, 0)

        self.stack_scanning_Avgnumber_spinbox = QSpinBox(self)
        self.stack_scanning_Avgnumber_spinbox.setMinimum(1)
        self.stack_scanning_Avgnumber_spinbox.setMaximum(20)
        self.stack_scanning_Avgnumber_spinbox.setValue(1)
        self.stack_scanning_Avgnumber_spinbox.setSingleStep(1)
        Zstack_Layout.addWidget(self.stack_scanning_Avgnumber_spinbox, 4, 1)
        Zstack_Layout.addWidget(QLabel("average over:"), 4, 0)

        self.stack_scanning_stepsize_spinbox = QDoubleSpinBox(self)
        self.stack_scanning_stepsize_spinbox.setMinimum(-10000)
        self.stack_scanning_stepsize_spinbox.setMaximum(10000)
        self.stack_scanning_stepsize_spinbox.setDecimals(6)
        self.stack_scanning_stepsize_spinbox.setSingleStep(0.001)
        self.stack_scanning_stepsize_spinbox.setValue(0.004)
        Zstack_Layout.addWidget(self.stack_scanning_stepsize_spinbox, 5, 1)
        Zstack_Layout.addWidget(QLabel("Step size(mm):"), 5, 0)

        self.stack_scanning_depth_spinbox = QDoubleSpinBox(self)
        self.stack_scanning_depth_spinbox.setMinimum(-10000)
        self.stack_scanning_depth_spinbox.setMaximum(10000)
        self.stack_scanning_depth_spinbox.setDecimals(6)
        self.stack_scanning_depth_spinbox.setSingleStep(0.001)
        self.stack_scanning_depth_spinbox.setValue(0.012)
        Zstack_Layout.addWidget(self.stack_scanning_depth_spinbox, 6, 1)

        depth_label = QLabel("Depth(mm):")
        Zstack_Layout.addWidget(depth_label, 6, 0)
        depth_label.setToolTip(
            "In case of not changing z-position, set here to 0."
        )

        self.startButton_stack_scanning = StylishQT.runButton("")
        self.startButton_stack_scanning.setFixedHeight(32)
        self.startButton_stack_scanning.setCheckable(True)
        self.startButton_stack_scanning.clicked.connect(
            lambda: self.buttonenabled("stackscan", "start")
        )
        self.startButton_stack_scanning.clicked.connect(
            lambda: run_in_thread(self.start_Zstack_scanning)
        )
        Zstack_Layout.addWidget(self.startButton_stack_scanning, 7, 0)

        self.stopButton_stack_scanning = StylishQT.stop_deleteButton()
        self.stopButton_stack_scanning.setFixedHeight(32)
        self.stopButton_stack_scanning.clicked.connect(
            lambda: self.buttonenabled("stackscan", "stop")
        )
        self.stopButton_stack_scanning.clicked.connect(
            lambda: run_in_thread(self.stop_Zstack_scanning)
        )
        self.stopButton_stack_scanning.setEnabled(False)
        Zstack_Layout.addWidget(self.stopButton_stack_scanning, 7, 1)

        Zstack_widget.setLayout(Zstack_Layout)

        self.scanning_tabs.addTab(Continuous_widget, "Raster scanning")
        self.scanning_tabs.addTab(Zstack_widget, "Stack scanning")

        # === Galvo control layout ===
        galvoControlWidget = QWidget()
        galvoControlLayout = QGridLayout()

        self.desired_galvo_x_label = QLabel("Desired x-position:")
        self.desired_galvo_x_spinbox = QDoubleSpinBox(self)
        self.desired_galvo_x_spinbox.setMinimum(0)
        self.desired_galvo_x_spinbox.setMaximum(500)
        self.desired_galvo_x_spinbox.setSingleStep(1)
        self.desired_galvo_x_spinbox.setValue(250)
        self.desired_galvo_x_spinbox.setDecimals(0)
        galvoControlLayout.addWidget(self.desired_galvo_x_label, 0, 0)
        galvoControlLayout.addWidget(self.desired_galvo_x_spinbox, 0, 1)

        self.desired_galvo_y_label = QLabel("Desired y-position:")
        self.desired_galvo_y_spinbox = QDoubleSpinBox(self)
        self.desired_galvo_y_spinbox.setMinimum(0)
        self.desired_galvo_y_spinbox.setMaximum(500)
        self.desired_galvo_y_spinbox.setSingleStep(1)
        self.desired_galvo_y_spinbox.setValue(250)
        self.desired_galvo_y_spinbox.setDecimals(0)
        galvoControlLayout.addWidget(self.desired_galvo_y_label, 1, 0)
        galvoControlLayout.addWidget(self.desired_galvo_y_spinbox, 1, 1)

        self.move_galvo_button = StylishQT.runButton("Move galvos")
        self.move_galvo_button.setFixedHeight(32)
        self.move_galvo_button.clicked.connect(
            lambda: self.move_galvos_to_specified_point(
                self.desired_galvo_x_spinbox.value(),
                self.desired_galvo_y_spinbox.value(),
            )
        )
        galvoControlLayout.addWidget(self.move_galvo_button, 2, 0, 1, 2)

        galvoControlWidget.setLayout(galvoControlLayout)
        self.scanning_tabs.addTab(galvoControlWidget, "Galvo control")

        # ---------------------------PMT-image inspection layout---------------------------
        pmt_image_inspection_widget = StylishQT.roundQGroupBox(
            title="PMT-image inspection"
        )
        pmt_image_inspection_widget.setFixedHeight(100)
        pmt_image_inspection_widget.setFixedWidth(290)
        pmt_image_inspection_layout = QGridLayout()

        self.pmt_image_x_coordinate_label = QLabel("X-coordinate: _")
        self.pmt_image_y_coordinate_label = QLabel("Y-coordinate: _")
        self.pmt_image_intensity_label = QLabel("Intensity: _")

        pmt_image_inspection_layout.addWidget(
            self.pmt_image_x_coordinate_label, 0, 0
        )
        pmt_image_inspection_layout.addWidget(
            self.pmt_image_y_coordinate_label, 0, 1
        )
        pmt_image_inspection_layout.addWidget(
            self.pmt_image_intensity_label, 0, 2
        )

        import_pmt_image_button = QPushButton("Import PMT image")
        import_pmt_image_button.clicked.connect(self.import_pmt_image)
        pmt_image_inspection_layout.addWidget(import_pmt_image_button, 1, 0)

        self.tiff_directory_textbox = QLineEdit()
        self.tiff_directory_textbox.setPlaceholderText("Tiff directory")
        pmt_image_inspection_layout.addWidget(
            self.tiff_directory_textbox, 1, 1
        )

        clear_pmt_image_button = QPushButton("Clear PMT image")
        clear_pmt_image_button.clicked.connect(self.clear_pmt_image)
        pmt_image_inspection_layout.addWidget(clear_pmt_image_button, 1, 2)

        pmt_image_inspection_widget.setLayout(pmt_image_inspection_layout)

        # Create a container for scanning_tabs and pmt_image_inspection_widget
        scanning_and_inspection_container = QWidget()
        scanning_and_inspection_container.setFixedHeight(330)
        scanning_and_inspection_layout = QHBoxLayout()
        scanning_and_inspection_container.setLayout(
            scanning_and_inspection_layout
        )

        # Add scanning_tabs and pmt_image_inspection_widget to the horizontal layout
        scanning_and_inspection_layout.addWidget(self.scanning_tabs)
        scanning_and_inspection_layout.addWidget(pmt_image_inspection_widget)

        # Add the containers to the main layout
        self.layout.addWidget(pmtimageContainer, 0, 0, 3, 1)
        self.layout.addWidget(pmtroiContainer, 0, 1)
        self.layout.addWidget(pmtContourContainer, 1, 1)
        self.layout.addWidget(scanning_and_inspection_container, 2, 1, 1, 1)

    # === Functions for TAB 'PMT' ===

    def generate_poly_roi(self, event):
        """Generate a polygon roi."""
        if not self.flag_is_drawing:
            return

        x, y = int(event.pos().x()), int(event.pos().y())
        qpoint_viewbox = self.pmt_viewbox.mapSceneToView(QPoint(x, y))
        point = [qpoint_viewbox.x(), qpoint_viewbox.y()]

        self.clicked_points_list.append(point)

        if len(self.clicked_points_list) == 1:
            self.starting_point = self.clicked_points_list[0]
            self.starting_point_handle_position = [x, y]

        elif len(self.clicked_points_list) == 2:
            self.click_poly_roi = pg.PolyLineROI(
                positions=[self.starting_point, point]
            )
            self.click_poly_roi.sigHoverEvent.connect(
                lambda: self.show_handle_num(self.click_poly_roi)
            )  # update handle numbers
            self.pmt_viewbox.addItem(self.click_poly_roi)
            self.new_roi = False

        else:
            self.click_poly_roi.addFreeHandle(point)

            # Remove closing segment of previous mouse movement
            if len(self.click_poly_roi.segments) > 1:
                self.click_poly_roi.removeSegment(
                    self.click_poly_roi.segments[-1]
                )

            self.click_poly_roi.addSegment(
                self.click_poly_roi.handles[-1]["item"],
                self.click_poly_roi.handles[-2]["item"],
            )

            # Add new closing segment
            self.click_poly_roi.addSegment(
                self.click_poly_roi.handles[0]["item"],
                self.click_poly_roi.handles[-1]["item"],
            )

    def keyPressEvent(self, event):
        # Toggle between drawing and not drawing roi states.
        if event.key() == 70:  # If the 'f' key is pressed
            if self.flag_is_drawing:
                self.flag_is_drawing = False
            else:
                self.flag_is_drawing = True
                self.new_roi = True

    def buttonenabled(self, button, switch):

        if button == "rasterscan":
            if switch == "start":
                self.startButton_pmt.setEnabled(False)
                self.stopButton.setEnabled(True)

            elif switch == "stop":
                self.startButton_pmt.setEnabled(True)
                self.stopButton.setEnabled(False)

        elif button == "contourscan":
            if (
                switch == "start"
            ):  # disable start button and enable stop button
                self.do_contour_scan.setEnabled(False)
                self.stopButton_contour.setEnabled(True)
            elif switch == "stop":
                self.do_contour_scan.setEnabled(True)
                self.stopButton_contour.setEnabled(False)

        elif button == "stackscan":
            if switch == "start":
                self.startButton_stack_scanning.setEnabled(False)
                self.stopButton_stack_scanning.setEnabled(True)

            elif switch == "stop":
                self.startButton_stack_scanning.setEnabled(True)
                self.stopButton_stack_scanning.setEnabled(False)

    def measure_pmt(self):
        """Do raster scan and update the graph."""
        try:
            self.Daq_sample_rate_pmt = int(
                self.continuous_scanning_sr_spinbox.value()
            )

            # Voltage settings, by default it's equal range square.
            self.Value_voltXMax = (
                self.continuous_scanning_Vrange_spinbox.value()
            )
            self.Value_voltXMin = self.Value_voltXMax * -1
            Value_voltYMin = self.Value_voltXMin
            Value_voltYMax = self.Value_voltXMax

            self.Value_xPixels = int(self.Scanning_pixel_num_combobox.value())
            Value_yPixels = self.Value_xPixels
            self.averagenum = int(
                self.continuous_scanning_average_spinbox.value()
            )

            Totalscansamples = self.pmtTest.setWave(
                self.Daq_sample_rate_pmt,
                self.Value_voltXMin,
                self.Value_voltXMax,
                Value_voltYMin,
                Value_voltYMax,
                self.Value_xPixels,
                Value_yPixels,
                self.averagenum,
            )
            time_per_frame_pmt = Totalscansamples / self.Daq_sample_rate_pmt

            self.pmtTest.pmtimagingThread.measurement.connect(
                self.update_pmt_Graphs
            )  # Connecting to the measurement signal
            self.pmt_fps_Label.setText(
                "Per frame:  %.4f s" % time_per_frame_pmt
            )
            self.pmtTest.start()

        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("NI-Daq not connected.")
            self.update_pmt_Graphs(
                data=np.zeros((Value_yPixels, Value_yPixels))
            )

    def measure_pmt_contourscan(self):

        self.Daq_sample_rate_pmt = int(self.contour_samplerate.value())

        self.pmtTest_contour.setWave_contourscan(
            self.Daq_sample_rate_pmt,
            self.final_stacked_voltage_signals,
            self.points_per_round,
        )
        contour_freq = self.Daq_sample_rate_pmt / self.points_per_round

        self.pmtTest_contour.start()
        self.MessageToMainGUI("---!! Continuous contour scanning !!---" + "\n")
        logging.info("Contour frequency:  %.4f Hz" % contour_freq)

        self.stopButton_contour.setVisible(True)

    def saveimage_pmt(self):
        Localimg = Image.fromarray(
            self.data_pmtcontineous
        )  # generate an image object
        Localimg.save(
            os.path.join(
                self.savedirectory,
                "PMT_"
                + self.prefixtextboxtext
                + "_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".tif",
            )
        )  # save as tif

    def update_pmt_Graphs(self, data):
        """Update graphs."""

        self.data_pmtcontineous = data
        self.pmtvideoWidget.setImage(data, autoLevels=None)

    def show_handle_num(self, roi_item):
        """Show the number of handles."""

        self.ROI_handles = roi_item.getHandles()
        self.number_of_ROI_handles = len(self.ROI_handles)
        self.pmt_handlenum_Label.setText(
            "Handle number: %.d" % self.number_of_ROI_handles
        )

    def regenerate_roi_handles(self):
        """Regenerate the handles from desired roi in sequence."""

        current_roi_handles_list = self.contour_ROI_handles_dict[
            "handles_{}".format(self.roi_index_spinbox.value())
        ]

        self.pmt_viewbox.removeItem(self.click_poly_roi)

        self.click_poly_roi = pg.PolyLineROI(
            current_roi_handles_list, pen=self.ROIpen, closed=True
        )
        self.click_poly_roi.sigHoverEvent.connect(
            lambda: self.show_handle_num(self.click_poly_roi)
        )

        self.pmt_viewbox.addItem(self.click_poly_roi)

    def add_coordinates_to_list(self):
        """Add one coordinate signals to the loop."""

        # Record roi handle positions
        roi_handles_scene_list = []

        has_camera_generated_contour_voltages = (
            hasattr(self, "camera_generated_contour_voltages")
            and self.camera_generated_contour_voltages is not None
        )
        has_click_poly_roi = (
            hasattr(self, "click_poly_roi") and self.click_poly_roi is not None
        )

        if (
            has_camera_generated_contour_voltages
            and has_click_poly_roi is False
        ):
            logging.info("Adding camera generated coordinates to the list.\n")
            self.current_stacked_voltage_signals = (
                self.camera_generated_contour_voltages
            )

        elif (
            has_click_poly_roi
            and has_camera_generated_contour_voltages is False
        ):
            logging.info("Adding UI-PMT coordinates to the list.\n")
            self.current_stacked_voltage_signals = (
                self.generate_contour_coordinates(self.click_poly_roi)
            )

            # From QPoint to list
            for each_item in self.local_handle_positions:
                roi_handles_scene_list.append(
                    [each_item[1].x(), each_item[1].y()]
                )

            self.contour_ROI_handles_dict[
                "handles_{}".format(self.roi_index_spinbox.value())
            ] = roi_handles_scene_list

        elif has_camera_generated_contour_voltages and has_click_poly_roi:
            logging.info(
                "Trying to add a camera generated contour as well as a UI-PMT "
                "contour at the same time. This is not allowed."
            )

        else:
            logging.info(
                "Missing contour coordinates. Please generate or create a "
                "contour."
            )

        # Place the signals to the corresponding dictionary position
        self.contour_ROI_signals_dict[
            "roi_{}".format(self.roi_index_spinbox.value())
        ] = self.current_stacked_voltage_signals

    def del_coordinates_from_list(self):
        """Remove the last mask from the list."""
        del self.contour_ROI_signals_dict[
            "roi_{}".format(self.roi_index_spinbox.value())
        ]
        del self.contour_ROI_handles_dict[
            "handles_{}".format(self.roi_index_spinbox.value())
        ]

    def generate_final_contour_signals(self):
        """Add together all the signals and emit it to other widgets."""

        if len(self.contour_ROI_signals_dict) == 0:
            QMessageBox.warning(
                self,
                "generate_final_contour_signals",
                "No contour coordinates found. Please add some coordinates.\n",
            )
            return
        if len(self.contour_ROI_signals_dict) == 1:
            # With only one roi in list
            self.final_stacked_voltage_signals = self.contour_ROI_signals_dict[
                "roi_1"
            ]
        else:
            # With multiple rois added
            temp_list = []
            for each_roi_coordinate in self.contour_ROI_signals_dict:
                temp_list.append(
                    self.contour_ROI_signals_dict[each_roi_coordinate]
                )

            self.final_stacked_voltage_signals = np.concatenate(
                temp_list, axis=1
            )

        # Number of points in single round of contour scan
        self.points_per_round = len(self.final_stacked_voltage_signals[0])

        # To the main widget Fiumicino
        self.emit_contour_signal()

        self.do_contour_scan.setEnabled(True)

    def go_to_first_point(self):
        """Before executing contour scanning, preset galvo positions to first
        point.
        """

        first_point_x = self.final_stacked_voltage_signals[:, 0][0]
        first_point_y = self.final_stacked_voltage_signals[:, 0][1]

        logging.info(
            "galvo move to: {}, {}".format(first_point_x, first_point_y)
        )

        daq = DAQmission()
        daq.sendSingleAnalog("galvosx", first_point_x)
        daq.sendSingleAnalog("galvosy", first_point_y)

    def move_galvos_to_specified_point(self, x, y):
        """Move galvos to specified point."""

        logging.info("Moving the galvos to: {}, {}".format(x, y))

        self.Value_xPixels = int(self.Scanning_pixel_num_combobox.value())
        self.Value_voltXMax = self.continuous_scanning_Vrange_spinbox.value()

        x_voltage_galvo, y_voltage_galvo = self.convert_coordinates_to_voltage(
            self.Value_xPixels, self.Value_voltXMax, 1, np.array([[x, y]])
        )

        logging.info(
            "Corresponding voltages: {}, {}".format(
                x_voltage_galvo, y_voltage_galvo
            )
        )

        daq = DAQmission()
        daq.sendSingleAnalog("galvosx", x_voltage_galvo)
        daq.sendSingleAnalog("galvosy", y_voltage_galvo)

        self.GalvoCoordinatesCommand.emit(x, y)

    def reset_roi_handles(self):
        """Reset_roi_handles positions."""

        if hasattr(self, "scatter_plot") and self.scatter_plot is not None:
            self.pmt_viewbox.removeItem(self.scatter_plot)
            self.scatter_plot = None
            logging.info("Removed scatter plot from viewbox")

        if hasattr(self, "click_poly_roi") and self.click_poly_roi is not None:
            self.pmt_viewbox.removeItem(self.click_poly_roi)
            self.click_poly_roi = None
            logging.info("Removed UI-plot from viewbox")

        self.clicked_points_list = []
        self.pmt_handlenum_Label.setText("Handle number: _")

    def reset_coordinates_dict(self):

        self.final_stacked_voltage_signals = None
        self.contour_ROI_signals_dict = {}

        if (
            hasattr(self, "camera_generated_contour_voltages")
            and self.camera_generated_contour_voltages is not None
        ):
            self.camera_generated_contour_voltages = None

        self.reset_roi_handles()

    def generate_contour_coordinates(self, roi_item):
        """Generate the voltage signals according to current ROI's handle
        positions.
        """

        self.Daq_sample_rate_pmt = int(self.contour_samplerate.value())
        self.ROI_handles = roi_item.getHandles()
        self.number_of_ROI_handles = len(self.ROI_handles)
        contour_length = int(self.pointsinContour.value())

        # Get the handle positions in the viewbox coordinates
        self.viewbox_handle_positions = [
            handle.pos() for handle in self.ROI_handles
        ]
        self.viewbox_handle_coords = np.vstack(
            [[pos.x(), pos.y()] for pos in self.viewbox_handle_positions]
        )

        # Get the handle positions in the local coordinates
        self.local_handle_positions = roi_item.getLocalHandlePositions()

        expanded_viewbox_coords = None

        # Even interpolation between handles
        if self.contour_strategy.currentText() == "Evenly between":
            if contour_length % self.number_of_ROI_handles == 0:
                self.points_per_handle = (
                    contour_length // self.number_of_ROI_handles
                )
            else:
                QMessageBox.warning(
                    self,
                    "generate_contour_coordinates",
                    "Please input a multiple of the handle number for the "
                    "contour points.",
                )
                return

            evenly_expanded_viewbox_coords = self.interpolate_handles_evenly(
                self.viewbox_handle_coords, self.points_per_handle
            )
            expanded_viewbox_coords = evenly_expanded_viewbox_coords

        # Uniform interpolation through handles
        if self.contour_strategy.currentText() == "Uniform":
            uniformly_expanded_viewbox_coords = self.interpolate_uniformly(
                self.viewbox_handle_coords, contour_length
            )
            expanded_viewbox_coords = uniformly_expanded_viewbox_coords

        self.Value_xPixels = int(self.Scanning_pixel_num_combobox.value())
        self.Value_voltXMax = self.continuous_scanning_Vrange_spinbox.value()

        # Transform into Voltages to galvos
        self.expanded_viewbox_x_coords, self.expanded_viewbox_y_coords = (
            self.convert_coordinates_to_voltage(
                self.Value_xPixels,
                self.Value_voltXMax,
                contour_length,
                expanded_viewbox_coords,
            )
        )

        # Speed and acceleration check
        if self.is_galvo_speed_and_acceleration_within_limits(
            self.Daq_sample_rate_pmt,
            self.expanded_viewbox_x_coords,
            self.expanded_viewbox_y_coords,
        ):

            # The final stacked voltage signals, ready to be sent to the DAQ
            current_stacked_voltage_signals = np.vstack(
                (
                    self.expanded_viewbox_x_coords,
                    self.expanded_viewbox_y_coords,
                )
            )
            resequenced_stacked_voltage_signals = (
                current_stacked_voltage_signals
            )

            return resequenced_stacked_voltage_signals

    def interpolate_handles_evenly(self, handle_positions, points_per_handle):
        """Interpolate evenly between handles."""

        def interpolate_between_two_points(p1, p2, num_points):
            """Helper function to interpolate points between two given points."""
            x_diff = p2[0] - p1[0]
            y_diff = p2[1] - p1[1]
            x_step = x_diff / num_points
            y_step = y_diff / num_points
            return np.array(
                [
                    [p1[0] + i * x_step, p1[1] + i * y_step]
                    for i in range(1, num_points)
                ]
            )

        interpolated_points = []

        # Interpolate between consecutive nodes
        for i in range(handle_positions.shape[0] - 1):
            p1 = handle_positions[i]
            p2 = handle_positions[i + 1]
            interpolated_points.append(p1)
            interpolated_points.extend(
                interpolate_between_two_points(p1, p2, points_per_handle)
            )

        # Interpolate between the last and the first node to close the loop
        p1 = handle_positions[-1]
        p2 = handle_positions[0]
        interpolated_points.append(p1)
        interpolated_points.extend(
            interpolate_between_two_points(p1, p2, points_per_handle)
        )

        # Convert to numpy array
        interpolated_array = np.array(interpolated_points)

        # Reorder points so the first placed handle is at the beginning
        interpolated_array = np.roll(interpolated_array, 1, axis=0)

        return interpolated_array

    def interpolate_uniformly(self, handle_positions, contour_length):
        """Interpolate uniformly between ROI handles using a B-spline."""

        # Create a B-spline representation of the curve
        tck, _ = splprep(
            [handle_positions[:, 0], handle_positions[:, 1]], s=0, per=True
        )

        # Generate uniformly spaced parameter values
        u_new = np.linspace(0, 1, contour_length)

        # Evaluate the B-spline at the new parameter values
        x_new, y_new = splev(u_new, tck)

        # Combine the new x and y coordinates
        expanded_handle_coords = np.vstack((x_new, y_new)).T

        return expanded_handle_coords

    def convert_coordinates_to_voltage(
        self, Value_xPixels, Value_voltXMax, contour_length, viewbox_coords
    ):
        """Convert coordinates to voltages."""

        if Value_xPixels == 500:
            logging.info(
                f"Converting coordinates within range [-{Value_voltXMax}, "
                f"{Value_voltXMax}] V. "
            )

            # Scale coordinates from pixels to voltages
            viewbox_coords[:, 0] = (
                (viewbox_coords[:, 0] / Value_xPixels) * 2 - 1
            ) * Value_voltXMax
            viewbox_coords[:, 1] = (
                (viewbox_coords[:, 1] / Value_xPixels) * 2 - 1
            ) * Value_voltXMax
            viewbox_coords = np.around(viewbox_coords, decimals=3)

            # Ensure the coordinates are resized to match the contour length
            transformed_x = np.resize(viewbox_coords[:, 0], (contour_length,))
            transformed_y = np.resize(viewbox_coords[:, 1], (contour_length,))
        else:
            QMessageBox.warning(
                self,
                "convert_coordinates_to_voltage",
                "Please input the correct pixel number for the conversion.",
            )
            return

        return transformed_x, transformed_y

    def is_galvo_speed_and_acceleration_within_limits(
        self, sampling_rate, trace_x, trace_y
    ):
        """Check the speed and acceleration of the galvo mirrors."""

        def calculate_speed_acceleration(trace, time_gap):
            """Calculate speed and acceleration for a given trace."""
            speed = np.diff(trace) / time_gap
            acceleration = np.diff(speed) / time_gap
            return speed, acceleration

        def log_and_display(value, label, unit, scientific_notation="0f"):
            """Log and display the mean value for a given label."""
            mean_value = int(np.round(np.mean(np.abs(value))))
            logging.info(
                f"Mean {label}: {mean_value:.{scientific_notation}} [{unit}]"
            )
            return mean_value

        # Constants and parameters
        time_gap = 1 / sampling_rate
        constants = HardwareConstants()
        max_speed = constants.maxGalvoSpeed  # Maximum galvo speed (Volt/s)
        max_acceleration = (
            constants.maxGalvoAccel
        )  # Maximum galvo acceleration (Volt/s^2)

        # Compute speed and acceleration
        contour_x_speed, contour_x_acceleration = calculate_speed_acceleration(
            trace_x, time_gap
        )
        contour_y_speed, contour_y_acceleration = calculate_speed_acceleration(
            trace_y, time_gap
        )

        # Log speed and acceleration values
        log_and_display(contour_x_speed, "x speed", "volt/s")
        log_and_display(contour_y_speed, "y speed", "volt/s")
        log_and_display(
            contour_x_acceleration, "x acceleration", "volt/s^2", "2e"
        )
        log_and_display(
            contour_y_acceleration, "y acceleration", "volt/s^2", "2e"
        )

        # Check galvo speed
        exceeded_speeds = {
            "X": np.where(np.abs(contour_x_speed) > max_speed)[0] + 1,
            "Y": np.where(np.abs(contour_y_speed) > max_speed)[0] + 1,
        }

        if not exceeded_speeds["X"].size and not exceeded_speeds["Y"].size:
            self.MessageToMainGUI("Contour speed is OK\n")
            logging.info("Contour speed is OK")
        else:
            message = "Speed too high in direction(s):"
            if exceeded_speeds["X"].size:
                message += f" X ({np.amax(np.abs(contour_x_speed)):.2f} V/s) "
                message += f"at handles {list(exceeded_speeds['X'])}"
            if exceeded_speeds["Y"].size:
                message += f" Y ({np.amax(np.abs(contour_y_speed)):.2f} V/s) "
                message += f"at handles {list(exceeded_speeds['Y'])}"

            QMessageBox.warning(self, "Overload", message, QMessageBox.Ok)

        # Check galvo acceleration
        exceeded_accelerations = {
            "X": np.where(np.abs(contour_x_acceleration) > max_acceleration)[0]
            + 1,
            "Y": np.where(np.abs(contour_y_acceleration) > max_acceleration)[0]
            + 1,
        }

        if (
            not exceeded_accelerations["X"].size
            and not exceeded_accelerations["Y"].size
        ):
            self.MessageToMainGUI("Contour acceleration is OK\n")
            logging.info("Contour acceleration is OK")
        else:
            message = "Acceleration too high in direction(s): \n"
            if exceeded_accelerations["X"].size:
                message += f" X ({np.amax(np.abs(contour_x_acceleration)):.2e}"
                message += " V/s²) at handles "
                message += f"{list(exceeded_accelerations['X'])} \n"
            if exceeded_accelerations["Y"].size:
                message += f" Y ({np.amax(np.abs(contour_y_acceleration)):.2e}"
                message += " V/s²) at handles "
                message += f"{list(exceeded_accelerations['Y'])}"
            QMessageBox.warning(self, "Overload", message, QMessageBox.Ok)

        return not (
            exceeded_speeds["X"].size
            or exceeded_speeds["Y"].size
            or exceeded_accelerations["X"].size
            or exceeded_accelerations["Y"].size
        )

    def import_pmt_image(self):
        # Open a file dialog to select a TIFF file
        options = QFileDialog.Options()
        tiff_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF file",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=options,
        )

        if tiff_file:
            self.displayTiffImage(tiff_file)
            self.tiff_directory_textbox.setText(tiff_file)

    def displayTiffImage(self, file_path):
        try:
            tiff_image = skimtiff.imread(file_path)
            self.pmtvideoWidget.setImage(tiff_image, autoLevels=None)
            self.pmtimgroi.setImage(tiff_image)
            self.pmtvideoWidget.getView().resetTransform()
            self.pmt_viewbox.enableAutoRange()
            self.pmt_viewbox.setLimits(
                xMin=0,
                xMax=tiff_image.shape[1],
                yMin=0,
                yMax=tiff_image.shape[0],
            )

        except Exception as exc:
            logging.info("Error displaying TIFF image:", exc_info=exc)

    def update_pixel_coords_and_intensity(self, event):
        """Update the pixel coordinates and intensity values."""

        def show_pixel_coords():
            x, y = int(event.pos().x()), int(event.pos().y())
            qpoint_viewbox = self.pmt_viewbox.mapSceneToView(QPoint(x, y))
            self.pmt_image_x_coordinate_label.setText(
                "X-coordinate: %.0f" % qpoint_viewbox.x()
            )
            self.pmt_image_y_coordinate_label.setText(
                "Y-coordinate: %.0f" % qpoint_viewbox.y()
            )
            return int(qpoint_viewbox.x()), int(qpoint_viewbox.y())

        if hasattr(self, "data_pmtcontineous"):
            x, y = show_pixel_coords()
            self.pmt_image_intensity_label.setText(
                "Intensity: %.4f" % self.data_pmtcontineous[y, x]
            )
        elif hasattr(self, "pmtimgroi") and self.pmtimgroi.image is not None:
            x, y = show_pixel_coords()
            self.pmt_image_intensity_label.setText(
                "Intensity: %.4f" % self.pmtimgroi.image[y, x]
            )

    def clear_pmt_image(self):
        self.pmtvideoWidget.clear()
        self.pmtimgroi.clear()
        self.pmt_viewbox.enableAutoRange()
        self.pmt_viewbox.setLimits(xMin=0, xMax=500, yMin=0, yMax=500)

        self.pmt_image_x_coordinate_label.setText("X-coordinate: _")
        self.pmt_image_y_coordinate_label.setText("Y-coordinate: _")
        self.pmt_image_intensity_label.setText("Intensity: _")

    def handle_received_camera_contour(self, contour):
        """Plot and transform the contour emitted by the camera."""

        # Remove the previous contour plot if it exists
        if hasattr(self, "scatter_plot") and self.scatter_plot is not None:
            self.pmt_viewbox.removeItem(self.scatter_plot)
            logging.info("Removed previous contour plot")

        # Scatter plot the points
        scatter_points = [
            {
                "pos": (x, y),
                "size": 2,
                "pen": {"color": "r", "width": 2},
                "brush": "r",
            }
            for x, y in contour
        ]
        self.scatter_plot = pg.ScatterPlotItem(scatter_points)

        # Add the ScatterPlotItem to the viewbox
        self.pmt_viewbox.addItem(self.scatter_plot)
        self.pmt_viewbox.enableAutoRange()

        self.Value_xPixels = int(self.Scanning_pixel_num_combobox.value())
        self.Value_voltXMax = self.continuous_scanning_Vrange_spinbox.value()

        converted_x_coords, converted_y_coords = (
            self.convert_coordinates_to_voltage(
                self.Value_xPixels, self.Value_voltXMax, len(contour), contour
            )
        )

        self.Daq_sample_rate_pmt = int(self.contour_samplerate.value())

        if self.is_galvo_speed_and_acceleration_within_limits(
            self.Daq_sample_rate_pmt, converted_x_coords, converted_y_coords
        ):
            self.camera_generated_contour_voltages = np.vstack(
                (converted_x_coords, converted_y_coords)
            )

    def emit_contour_signal(self):
        """Emit generated contour signals to the main widget, then pass to
        waveform widget.
        """

        self.SignalForContourScanning.emit(
            int(self.points_per_round),
            self.Daq_sample_rate_pmt,
            (1 / int(self.contour_samplerate.value()) * 1000)
            * self.points_per_round,  # time per contour scan
            self.final_stacked_voltage_signals[0],
            self.final_stacked_voltage_signals[1],
        )

    def start_Zstack_scanning(self):
        """Create the stack scanning instance and run."""

        saving_dir = self.savedirectory
        z_depth = self.stack_scanning_depth_spinbox.value()
        z_step_size = self.stack_scanning_stepsize_spinbox.value()
        sampling_rate = self.stack_scanning_sampling_rate_spinbox.value()
        imaging_conditions = {
            "Daq_sample_rate": sampling_rate,
            "edge_volt": self.stack_scanning_Vrange_spinbox.value(),
            "pixel_number": self.stack_scanning_Pnumber_spinbox.value(),
            "average_number": self.stack_scanning_Avgnumber_spinbox.value(),
        }

        self.zstack_ins = PMT_zscan(
            saving_dir, z_depth, z_step_size, imaging_conditions
        )
        self.zstack_ins.start_scan()

    def stop_Zstack_scanning(self):
        self.zstack_ins.stop_scan()

    def MessageToMainGUI(self, text):
        self.MessageBack.emit(text)

    def stopMeasurement_pmt(self):
        """Stop the seal test."""
        self.pmtTest.aboutToQuitHandler()

    def stopMeasurement_pmt_contour(self):
        """Stop the seal test."""
        self.pmtTest_contour.aboutToQuitHandler()
        self.MessageToMainGUI("---!! Contour stopped !!---" + "\n")
        self.stopButton_contour.setVisible(False)


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = PMTWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
