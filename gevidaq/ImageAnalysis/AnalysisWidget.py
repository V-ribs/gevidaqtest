# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:27:04 2020

@author: xinmeng

                                Image analysis GUI
"""

import csv
import logging
import os
import sys
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import tifffile as skimtiff
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)
from scipy import signal
from scipy.optimize import curve_fit
from skimage.io import imread

from .. import StylishQT
from ..NIDAQ import waveform_specification
from .ImageProcessing import PatchAnalysis, ProcessImage


class AnalysisWidgetUI(QWidget):
    # waveforms_generated = pyqtSignal(object, object, list, int)
    # SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    MessageBack = pyqtSignal(str)
    Cellselection_DMD_mask_contour = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        self.setMinimumSize(1250, 850)
        self.setWindowTitle("AnalysisWidget")
        self.layout = QGridLayout(self)
        self.savedirectory = ""
        self.OC = 0.1  # Patch clamp constant
        # === GUI for Data analysis tab ===
        readimageContainer = QGroupBox("Readin images")
        self.readimageLayout = QGridLayout()

        self.Construct_name = QLineEdit(self)
        self.Construct_name.setPlaceholderText("Enter construct name")
        self.Construct_name.setFixedWidth(150)
        self.readimageLayout.addWidget(self.Construct_name, 1, 0)

        self.switch_Vp_or_camtrace = QComboBox()
        self.switch_Vp_or_camtrace.addItems(
            ["Correlate to Vp", "Correlate to video", "Photocurrent"]
        )
        self.readimageLayout.addWidget(self.switch_Vp_or_camtrace, 1, 1)

        self.textbox_directory_name = QLineEdit(self)
        self.readimageLayout.addWidget(self.textbox_directory_name, 1, 6)

        # self.button_browse = QPushButton('Set data folder', self)
        # self.readimageLayout.addWidget(self.button_browse, 1, 5)

        # self.button_browse.clicked.connect(self.getfile)

        self.button_load = StylishQT.loadButton()
        self.button_load.setFixedWidth(100)
        self.button_load.setToolTip("Choose data folder and load")
        self.readimageLayout.addWidget(self.button_load, 1, 7)
        self.button_load.clicked.connect(self.getfile)

        self.Spincamsamplingrate = QSpinBox(self)
        self.Spincamsamplingrate.setMaximum(20000)
        self.Spincamsamplingrate.setValue(1000)
        self.Spincamsamplingrate.setSingleStep(500)
        self.readimageLayout.addWidget(self.Spincamsamplingrate, 1, 3)
        self.readimageLayout.addWidget(QLabel("Camera FPS:"), 1, 2)

        self.VoltageStepFrequencyQSpinBox = QSpinBox(self)
        self.VoltageStepFrequencyQSpinBox.setMaximum(20000)
        self.VoltageStepFrequencyQSpinBox.setValue(5)
        self.VoltageStepFrequencyQSpinBox.setSingleStep(5)
        self.readimageLayout.addWidget(self.VoltageStepFrequencyQSpinBox, 1, 5)
        self.readimageLayout.addWidget(QLabel("Voltage step frequency:"), 1, 4)

        self.run_analysis_button = StylishQT.runButton()
        self.run_analysis_button.setFixedWidth(100)
        self.run_analysis_button.setEnabled(False)
        self.readimageLayout.addWidget(self.run_analysis_button, 1, 8)

        self.run_analysis_button.clicked.connect(self.finish_analysis)

        self.button_clearpolts = StylishQT.cleanButton()
        self.button_clearpolts.setFixedWidth(100)
        self.readimageLayout.addWidget(self.button_clearpolts, 1, 9)

        self.button_clearpolts.clicked.connect(self.clearplots)

        readimageContainer.setLayout(self.readimageLayout)
        readimageContainer.setMaximumHeight(120)

        # === Image analysis display Tab ===
        Display_Container = QGroupBox("Image analysis display")
        Display_Layout = QGridLayout()
        # Setting tabs
        Display_Container_tabs = QTabWidget()

        # ------------------------------------------------------Image Analysis-Average window-------------------------------------------------------
        image_display_container_layout = QGridLayout()

        imageanalysis_average_Container = QGroupBox("Background selection")
        self.imageanalysisLayout_average = QGridLayout()

        # self.pw_averageimage = averageimagewindow()
        self.pw_averageimage = pg.ImageView()
        self.pw_averageimage.ui.roiBtn.hide()
        self.pw_averageimage.ui.menuBtn.hide()

        self.roi_average = pg.PolyLineROI(
            [[0, 0], [0, 30], [30, 30], [30, 0]], closed=True
        )
        self.roi_average = pg.RectROI(
            [0, 0], [30, 30], centered=True, sideScalers=True
        )
        self.pw_averageimage.view.addItem(self.roi_average)
        # self.pw_weightimage = weightedimagewindow()
        self.imageanalysisLayout_average.addWidget(
            self.pw_averageimage, 0, 0, 5, 3
        )

        imageanalysis_average_Container.setLayout(
            self.imageanalysisLayout_average
        )
        imageanalysis_average_Container.setMinimumHeight(180)
        # ------------------------------------------------------Image Analysis-weighV window-------------------------------------------------------
        imageanalysis_weight_Container = QGroupBox("Weighted image")
        self.imageanalysisLayout_weight = QGridLayout()

        # self.pw_averageimage = averageimagewindow()
        self.pw_weightimage = pg.ImageView()
        self.pw_weightimage.ui.roiBtn.hide()
        self.pw_weightimage.ui.menuBtn.hide()

        self.roi_weighted = pg.PolyLineROI(
            [[0, 0], [0, 30], [30, 30], [30, 0]], closed=True
        )
        self.pw_weightimage.view.addItem(self.roi_weighted)
        # self.pw_weightimage = weightedimagewindow()
        self.imageanalysisLayout_weight.addWidget(
            self.pw_weightimage, 0, 0, 5, 3
        )

        imageanalysis_weight_Container.setLayout(
            self.imageanalysisLayout_weight
        )
        imageanalysis_weight_Container.setMinimumHeight(180)

        image_display_container_layout.addWidget(
            imageanalysis_average_Container, 0, 0
        )
        image_display_container_layout.addWidget(
            imageanalysis_weight_Container, 0, 1
        )

        Display_Container_tabs_tab3 = PlotAnalysisGUI()
        # Display_Container_tabs_tab3.setLayout(self.Curvedisplay_Layout)

        # Display_Container_tabs_tab2 = QWidget()
        # Display_Container_tabs_tab2.setLayout(self.VIdisplay_Layout)

        Display_Container_tabs_Galvo_WidgetInstance = QWidget()
        Display_Container_tabs_Galvo_WidgetInstance.setLayout(
            image_display_container_layout
        )

        # self.Display_Container_tabs_Cellselection = QWidget()
        # self.Display_Container_tabs_Cellselection_layout = QGridLayout()

        # self.show_cellselection_gui_button = QPushButton('show')
        # self.show_cellselection_gui_button.clicked.connect(self.show_cellselection_gui)
        # self.Display_Container_tabs_Cellselection_layout.addWidget(self.show_cellselection_gui_button, 0,0)
        # self.Display_Container_tabs_Cellselection.setLayout(self.Display_Container_tabs_Cellselection_layout)

        # === Show trace ===
        Display_Container_tabs_tab4 = QWidget()
        Display_Container_tabs_tab4_layout = QGridLayout()

        self.waveform_samplingrate_box = QSpinBox(self)
        self.waveform_samplingrate_box.setMaximum(250000)
        self.waveform_samplingrate_box.setValue(50000)
        self.waveform_samplingrate_box.setSingleStep(500)
        Display_Container_tabs_tab4_layout.addWidget(
            self.waveform_samplingrate_box, 0, 1
        )
        Display_Container_tabs_tab4_layout.addWidget(
            QLabel("Sampling rate:"), 0, 0
        )

        self.textbox_single_waveform_filename = QLineEdit(self)
        Display_Container_tabs_tab4_layout.addWidget(
            self.textbox_single_waveform_filename, 0, 2
        )

        self.button_browse_tab4 = QPushButton("Browse", self)
        Display_Container_tabs_tab4_layout.addWidget(
            self.button_browse_tab4, 0, 3
        )

        self.button_browse_tab4.clicked.connect(self.get_single_waveform)

        Display_Container_tabs_tab4.setLayout(
            Display_Container_tabs_tab4_layout
        )

        # Add tabs
        Display_Container_tabs.addTab(
            Display_Container_tabs_Galvo_WidgetInstance, "Patch clamp display"
        )
        # Display_Container_tabs.addTab(Display_Container_tabs_tab2,"Patch display")
        Display_Container_tabs.addTab(
            Display_Container_tabs_tab3, "Patch perfusion"
        )
        # Display_Container_tabs.addTab(self.Display_Container_tabs_Cellselection,"Cell selection")
        Display_Container_tabs.addTab(
            Display_Container_tabs_tab4, "Display NIdaq trace"
        )

        Display_Layout.addWidget(Display_Container_tabs, 0, 0)
        Display_Container.setLayout(Display_Layout)

        self.layout.addWidget(readimageContainer, 0, 0, 1, 2)
        self.layout.addWidget(Display_Container, 1, 0, 1, 2)

    # master_data_analysis.addWidget(imageanalysis_average_Container, 2, 0, 1,1)
    # master_data_analysis.addWidget(imageanalysis_weight_Container, 2, 1, 1,1)

    # === Functions for Data analysis Tab ===
    def getfile(self):
        self.main_directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.textbox_directory_name.setText(self.main_directory)

        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            for file in os.listdir(self.main_directory):
                # For Labview generated data.
                if file.endswith(".tif") or file.endswith(".TIF"):
                    self.fileName = self.main_directory + "/" + file
                    logging.info(self.fileName)

            self.start_analysis()

        elif self.switch_Vp_or_camtrace.currentIndex() == 2:
            # For photocurrent analysis
            ProcessImage.PhotoCurrent(self.main_directory)

    def start_analysis(self):
        """
        Getting the data folder, load the video

        Returns
        None.

        """
        try:
            get_ipython = sys.modules["IPython"].get_ipython
        except KeyError:
            pass
        else:
            get_ipython().run_line_magic("matplotlib", "qt")

        if not os.path.exists(
            os.path.join(self.main_directory, "Patch analysis")
        ):
            # If the folder is not there, create the folder
            os.mkdir(os.path.join(self.main_directory, "Patch analysis"))

        logging.info("Loading data...")
        self.MessageToMainGUI("Loading data..." + "\n")

        t1 = threading.Thread(target=self.load_data_thread)
        t1.start()

        try:
            get_ipython = sys.modules["IPython"].get_ipython
        except KeyError:
            pass
        else:
            get_ipython().run_line_magic("matplotlib", "inline")

    def finish_analysis(self):
        t2 = threading.Thread(target=self.finish_analysis_thread)
        t2.start()

    def load_data_thread(self):
        self.button_load.setEnabled(False)
        # Load tif video file.
        self.videostack = imread(self.fileName)
        logging.info(self.videostack.shape)
        self.MessageToMainGUI(
            "Video size: " + str(self.videostack.shape) + "\n"
        )
        self.roi_average.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        self.roi_weighted.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        logging.info(
            "============ Loading complete, ready to fire ============ "
        )
        self.MessageToMainGUI("=== Loading complete, ready to fire ===" + "\n")

        # Load wave files.
        self.loadcurve()

        # display electrical signals
        try:
            self.display_electrical_signals()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("No electrical recordings.")

        time.sleep(0.5)
        # Calculate the mean intensity of video, for background substraction.
        self.video_mean()

        logging.info("=========== Ready for analyse. =============")
        self.button_load.setEnabled(False)
        self.run_analysis_button.setEnabled(True)

    def finish_analysis_thread(self):
        self.MessageToMainGUI("=== Analysis start.. ===" + "\n")
        # Calculate the background
        self.calculate_background_from_ROI_average()

        # Substract background
        self.substract_background()

        # calculate_weight
        self.calculate_weight()

        # display_weighted_trace
        self.display_weighted_trace()

        # Fit on weighted trace and sumarize the statistics.
        self.fit_on_trace()

        logging.info("============ Analysis done. ============")
        self.MessageToMainGUI("=== Analysis done. ===" + "\n")

        self.button_load.setEnabled(True)
        self.run_analysis_button.setEnabled(False)

    def ReceiveVideo(self, videosentin):
        self.videostack = videosentin
        logging.info(self.videostack.shape)
        self.MessageToMainGUI(
            "Video size: " + str(self.videostack.shape) + "\n"
        )
        self.roi_average.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        self.roi_weighted.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        logging.info("Loading complete, ready to fire")
        self.MessageToMainGUI("Loading complete, ready to fire" + "\n")

    def loadcurve(self):
        """
        Load the 1D array files, like voltage, current recordings or waveform information.

        By default the waveforms configured contain 4 extra samples at the front
        and 1 in the end, while the recorded waveforms contain 8 extra samples at
        the front and 1 in the end.

        Returns
        None.

        """
        found_correct_waveform_spelling = False
        for file in os.listdir(self.main_directory):
            # For Labview generated data.
            if file.endswith(".Ip"):
                self.Ipfilename = self.main_directory + "/" + file
                curvereadingobjective_i = ProcessImage.readbinaryfile(
                    self.Ipfilename
                )
                (
                    self.Ip,
                    self.samplingrate_display_curve,
                ) = curvereadingobjective_i.readbinarycurve()
                self.Ip = self.Ip[0 : len(self.Ip) - 2]

            elif file.endswith(".Vp"):
                self.Vpfilename = self.main_directory + "/" + file
                curvereadingobjective_V = ProcessImage.readbinaryfile(
                    self.Vpfilename
                )
                (
                    self.Vp,
                    self.samplingrate_display_curve,
                ) = curvereadingobjective_V.readbinarycurve()
                self.Vp = self.Vp[
                    0 : len(self.Vp) - 2
                ]  # Here -2 because there are two extra recording points in the recording file.

            # === Load configured waveform ===
            elif waveform_specification.is_waveform(file):
                if found_correct_waveform_spelling:
                    if waveform_specification.is_misspelled_wavefrom(file):
                        continue
                elif not waveform_specification.is_misspelled_wavefrom(file):
                    found_correct_waveform_spelling = True

                self.Waveform_filename_npy = self.main_directory + "/" + file
                # Read in configured waveforms
                configwave_wavenpfileName = self.Waveform_filename_npy
                self.configured_waveforms_container = (
                    waveform_specification.load(configwave_wavenpfileName)
                )

                # Get the sampling rate
                self.samplingrate_display_curve = (
                    waveform_specification.get_sample_rate(
                        configwave_wavenpfileName
                    )
                )
                logging.info(
                    "Waveforms_sampling rate: {}".format(
                        self.samplingrate_display_curve
                    )
                )

                # Load the configured patch voltage waveform
                for each_waveform in self.configured_waveforms_container:
                    if each_waveform["Specification"] == "patchAO":
                        self.configured_Vp = each_waveform["Waveform"]

                        logging.info(
                            "Length of configured_Vp: {}".format(
                                len(self.configured_Vp)
                            )
                        )

                        if len(self.configured_Vp) % 10 == 5:
                            # First 4 numbers and the last one in the recording
                            # are padding values outside the real camera recording.

                            # Extra camera triggers: False True True False | Real signals: True
                            # Extra patch voltage:   -0.7  -0.7 -0.7 -0.7  | Real signals: 0.3
                            self.configured_Vp = (
                                self.configured_Vp[4:][0:-1] / 10
                            )

                            logging.info(
                                "Length of cut Vp: {}".format(
                                    len(self.configured_Vp)
                                )
                            )

                        elif len(self.configured_Vp) % 10 == 2:
                            # The first number and the last one in the recording
                            # are padding values outside the real camera recording.
                            self.configured_Vp = (
                                self.configured_Vp[1:][0:-1] / 10
                            )

                            logging.info(
                                "Length of cut Vp: {}".format(
                                    len(self.configured_Vp)
                                )
                            )

            # For python generated data
            elif file.startswith("Vp"):
                self.Vpfilename_npy = self.main_directory + "/" + file
                curvereadingobjective_V = np.load(self.Vpfilename_npy)

                # In the recorded Vp trace, the 0 data is sampling rate,
                # 1 to 4 are NiDaq scaling coffecients,
                # 5 to 8 are extra samples for extra camera trigger,
                # The last one is padding 0 to reset NIDaq channels.
                self.Vp = curvereadingobjective_V[9:]
                self.samplingrate_display_curve = curvereadingobjective_V[0]
                self.Vp = self.Vp[0:-1]

                # This is from Fixed output from the patch amplifier, unfiltered, x10 already.
                # Ditched x10 voltage channel in amplifier from 22.12.2021
                # self.Vp = self.Vp/10

            elif file.startswith("Ip"):
                self.Ipfilename_npy = self.main_directory + "/" + file
                curvereadingobjective_I = np.load(self.Ipfilename_npy)
                # print("I raw: {}".format(curvereadingobjective_I[1000]))

                # In the recorded Vp trace, the 0 data is sampling rate,
                # 1 to 4 are NiDaq scaling coffecients,
                # 5 to 8 are extra samples for extra camera trigger,
                # The last one is padding 0 to reset NIDaq channels.
                self.Ip = curvereadingobjective_I[9:]
                self.Ip = self.Ip[0:-1]
                self.samplingrate_display_curve = curvereadingobjective_I[0]

    def getfile_background(self):
        self.fileName_background, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            "M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data",  # TODO hardcoded path
            "Image files (*.jpg *.tif)",
        )
        self.textbox_Background_filename.setText(self.fileName_background)

    def substract_background(self):
        """
        Substract the background from the original video.

        Returns
        None.

        """
        logging.info("Loading...")

        self.background_rolling_average_number = int(self.samplingrate_cam / 5)
        logging.info(
            "background_rolling_average_number is : {}".format(
                self.background_rolling_average_number
            )
        )

        unique, counts = np.unique(
            self.averageimage_ROI_mask, return_counts=True
        )
        count_dict = dict(zip(unique, counts))
        logging.info(f"number of 1 and 0: {count_dict}")

        self.background_trace = []

        for i in range(self.videostack.shape[0]):
            ROI_bg = (
                self.videostack[i][
                    self.roi_avg_coord_raw_start : self.roi_avg_coord_raw_start
                    + self.averageimage_ROI_mask.shape[0],
                    self.roi_avg_coord_col_start : self.roi_avg_coord_col_start
                    + self.averageimage_ROI_mask.shape[1],
                ]
                * self.averageimage_ROI_mask
            )

            # Sum of all pixel values and devided by non-zero pixel number
            bg_mean = np.sum(ROI_bg) / count_dict[1]

            self.background_trace.append(bg_mean)

        fig, ax0 = plt.subplots(figsize=(8.0, 5.8))
        fig.suptitle("Raw ROI background trace")
        plt.plot(self.cam_time_label, self.background_trace)
        ax0.set_xlabel("time(s)")
        ax0.set_ylabel("Pixel values")
        fig.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//ROI raw background trace.png",
            ),
            dpi=1000,
        )
        plt.show()

        # Use rolling average to smooth the background trace
        # self.background_trace = uniform_filter1d(uniform_filter1d
        # (self.background_trace, size=self.background_rolling_average_number), size=self.background_rolling_average_number*2)

        # Bi-exponential curve to fit the background
        def bg_func(t, a, t1, b, t2):
            return a * np.exp(-(t / t1)) + b * np.exp(-(t / t2))

        # Parameter bounds for the parameters in the bi-exponential function
        parameter_bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

        # popt   = Optimal parameters for the curve fit
        # pcov   = The estimated covariance of popt
        popt, pcov = curve_fit(
            bg_func,
            self.cam_time_label,
            self.background_trace,
            bounds=parameter_bounds,
            maxfev=500000,
        )

        self.background_trace_smoothed = bg_func(self.cam_time_label, *popt)

        fig, ax1 = plt.subplots(figsize=(8.0, 5.8))
        fig.suptitle("Smoothed background trace")

        (p01,) = ax1.plot(
            self.cam_time_label,
            self.background_trace,
            color=(0, 0, 0.4),
            linestyle="None",
            marker="o",
            markersize=0.5,
            markerfacecolor=(0, 0, 0.9),
            label="Raw background",
        )
        (p02,) = ax1.plot(
            self.cam_time_label,
            self.background_trace_smoothed,
            color=(0.9, 0, 0),
            label="Smoothed background",
        )
        ax1.legend([p01, p02], ["Raw background", "Smoothed background"])
        ax1.set_xlabel("time(s)")
        ax1.set_ylabel("Pixel values")
        fig.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Smoothed background trace.png",
            ),
            dpi=1000,
        )
        plt.show()

        # For each frame in video, substract the background
        container = np.empty(
            (
                self.videostack.shape[0],
                self.videostack.shape[1],
                self.videostack.shape[2],
            )
        )

        for i in range(self.videostack.shape[0]):
            raw_frame = self.videostack[i]

            background_mean = self.background_trace_smoothed[i]

            background_mean_2d = background_mean * np.ones(
                (self.videostack.shape[1], self.videostack.shape[2])
            )

            temp_diff = raw_frame - background_mean_2d
            temp_diff[temp_diff < 0] = 0

            container[i] = temp_diff
            self.videostack[i, :, :] = temp_diff

        logging.info("ROI background correction done.")

        saveBgSubstractedVideop = False
        # Save the file.
        if saveBgSubstractedVideop is True:
            with skimtiff.TiffWriter(
                os.path.join(
                    self.main_directory,
                    "Patch analysis//saveBgSubstractedVideo.tif",
                ),
                append=True,
            ) as tif:
                tif.save(self.videostack, compress=0)

        self.videostack = container
        # Show the background corrected trace.
        mean_camera_counts_backgroubd_substracted = []
        for i in range(self.videostack.shape[0]):
            mean_camera_counts_backgroubd_substracted.append(
                np.mean(self.videostack[i])
            )

        fig2, ax2 = plt.subplots(figsize=(8.0, 5.8))
        fig2.suptitle("Mean camera intensity after backgroubd substracted")
        plt.plot(
            self.cam_time_label, mean_camera_counts_backgroubd_substracted
        )
        ax2.set_xlabel("time(s)")
        ax2.set_ylabel("Pixel values")
        fig2.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Mean camera intensity after backgroubd substracted.png",
            ),
            dpi=1000,
        )
        plt.show()

        # Updates the mean intensity
        self.imganalysis_averageimage = np.mean(self.videostack, axis=0)
        self.pw_averageimage.setImage(self.imganalysis_averageimage)

        fig3 = plt.figure(figsize=(8.0, 5.8))
        fig3.suptitle("Mean intensity")
        plt.imshow(self.imganalysis_averageimage)
        fig3.savefig(
            os.path.join(
                self.main_directory, "Patch analysis//Mean intensity.png"
            ),
            dpi=1000,
        )
        plt.show()

    def display_electrical_signals(self):
        """
        Display the patch clamp electrode recored membrane potential and current signals.

        Returns
        None.

        """
        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            self.patchcurrentlabel = (
                np.arange(len(self.Ip)) / self.samplingrate_display_curve
            )

            self.patchvoltagelabel = (
                np.arange(len(self.configured_Vp))
                / self.samplingrate_display_curve
            )

            self.electrical_signals_figure, (ax1, ax2) = plt.subplots(2, 1)
            # plt.title('Electrode recording')
            # Current here is already
            # Probe gain: low-100M ohem
            # [DAQ recording / 10**8 (voltage to current)]* 10**12 (A to pA) == pA
            ax1.plot(
                self.patchcurrentlabel,
                self.Ip * 10000,
                label="Current",
                color="b",
            )
            ax1.set_title("Electrode recording")
            ax1.set_xlabel("time(s)")
            ax1.set_ylabel("Current (pA)")
            # ax1.legend()

            # ax2 = self.electrical_signals_figure.add_subplot(212)
            # *1000: convert to mV; /10 is to correct for the *10 add on at patch amplifier.
            ax2.plot(
                self.patchvoltagelabel,
                self.configured_Vp * 1000,
                label="Set voltage",
                color="b",
            )
            # ax2.set_title('Voltage')
            ax2.set_xlabel("time(s)")
            ax2.set_ylabel("Set voltage(mV)")
            # ax2.legend()

            plt.show()
            self.electrical_signals_figure.savefig(
                os.path.join(
                    self.main_directory,
                    "Patch analysis//Electrode recording.png",
                ),
                dpi=1000,
            )
        else:
            pass

    def video_mean(self):
        """
        Calculating the average 2d image from the video.

        Returns
        None.

        """
        self.imganalysis_averageimage = np.mean(self.videostack, axis=0)
        self.pw_averageimage.setImage(self.imganalysis_averageimage)
        # self.pw_averageimage.setLevels((50, 200))

        self.samplingrate_cam = self.Spincamsamplingrate.value()
        self.cam_time_label = (
            np.arange(self.videostack.shape[0]) / self.samplingrate_cam
        )

        fig = plt.figure(figsize=(8.0, 5.8))
        fig.suptitle("Mean intensity of raw video")
        plt.imshow(self.imganalysis_averageimage)
        fig.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Mean intensity of raw video.png",
            ),
            dpi=1000,
        )
        plt.show()

        self.mean_camera_counts = []
        for i in range(self.videostack.shape[0]):
            self.mean_camera_counts.append(np.mean(self.videostack[i]))

        fig2, ax2 = plt.subplots(figsize=(8.0, 5.8))
        fig2.suptitle("Mean intensity trace of raw video")
        plt.plot(self.cam_time_label, self.mean_camera_counts)
        ax2.set_xlabel("time(s)")
        ax2.set_ylabel("Pixel values")
        fig2.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Mean intensity trace of raw video.png",
            ),
            dpi=1000,
        )
        plt.show()

    def calculate_background_from_ROI_average(self):
        """
        Calculate the mean background value from the ROI selector.

        Returns
        None.

        """
        self.averageimage_imageitem = self.pw_averageimage.getImageItem()
        self.averageimage_ROI = self.roi_average.getArrayRegion(
            self.imganalysis_averageimage, self.averageimage_imageitem
        )
        self.averageimage_ROI_mask = np.where(self.averageimage_ROI > 0, 1, 0)

        # self.roi_average_pos = self.roi_average.pos()
        self.roi_average_Bounds = self.roi_average.parentBounds()
        self.roi_avg_coord_col_start = round(
            self.roi_average_Bounds.topLeft().x()
        )
        self.roi_avg_coord_col_end = round(
            self.roi_average_Bounds.bottomRight().x()
        )
        self.roi_avg_coord_raw_start = round(
            self.roi_average_Bounds.topLeft().y()
        )
        self.roi_avg_coord_raw_end = round(
            self.roi_average_Bounds.bottomRight().y()
        )

    def calculateweight(self):
        t2 = threading.Thread(target=self.calculate_weight)
        t2.start()

    def calculate_weight(self):
        """
        Calculate the pixels weights using correlation between the video and voltage reocrding.

        Returns
        None.

        """
        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            self.samplingrate_cam = self.Spincamsamplingrate.value()
            self.downsample_ratio = int(
                self.samplingrate_display_curve / self.samplingrate_cam
            )

            logging.info(
                "Vp downsampling ratio: {}".format(self.downsample_ratio)
            )
            logging.info(
                "Sampling rate camera: {}".format(self.samplingrate_cam)
            )
            logging.info(
                "Sampling rate DAQ: {}".format(self.samplingrate_display_curve)
            )

            try:
                self.Vp_downsample = self.configured_Vp.reshape(
                    -1, self.downsample_ratio
                ).mean(axis=1)

                # self.Vp_downsample = self.Vp_downsample[0 : len(self.videostack)]
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                logging.info("Vp downsampling ratio is not an integer.")
                small_ratio = int(
                    np.floor(
                        self.samplingrate_display_curve / self.samplingrate_cam
                    )
                )
                resample_length = int(len(self.videostack) * small_ratio)
                self.Vp_downsample = signal.resample(
                    self.configured_Vp, resample_length
                )
                plt.figure()
                plt.plot(self.Vp_downsample)
                plt.show()

                self.Vp_downsample = self.Vp_downsample.reshape(
                    -1, small_ratio
                ).mean(axis=1)

            # If there's a missing frame in camera video(probably in the beginning),
            # cut the Vp from the beginning
            self.Vp_downsample = self.Vp_downsample[
                len(self.Vp_downsample) - len(self.videostack) :
            ]

            logging.info(f"{self.videostack.shape} {self.Vp_downsample.shape}")

            (
                self.corrimage,
                self.weightimage,
                self.sigmaimage,
            ) = ProcessImage.extractV(
                self.videostack, self.Vp_downsample * 1000
            )
            # *1000: convert to mV; /10 is to correct for the *10 add on at patch amplifier(not anymore).

            self.pw_weightimage.setImage(self.weightimage)

        elif self.switch_Vp_or_camtrace.currentIndex() == 1:
            (
                self.corrimage,
                self.weightimage,
                self.sigmaimage,
            ) = ProcessImage.extractV(
                self.videostack, self.camsignalsum * 1000
            )

            self.pw_weightimage.setImage(self.weightimage)

        fig = plt.figure(figsize=(8.0, 5.8))
        fig.suptitle("Weighted pixels")
        plt.imshow(self.weightimage)
        fig.savefig(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted pixel image.png"
            ),
            dpi=1000,
        )
        np.save(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted pixel image.npy"
            ),
            self.weightimage,
        )
        plt.show()

    def display_weighted_trace(self):
        """
        Display the mean sum weight and display frame by frame.

        Returns
        None.

        """
        self.videolength = len(self.videostack)

        # datv = squeeze(mean(mean(mov.*repmat(Wv./sum(Wv(:))*movsize(1)*movsize(2), [1 1 length(sig)]))));
        k = np.tile(
            self.weightimage
            / np.sum(self.weightimage)
            * self.videostack.shape[1]
            * self.videostack.shape[2],
            (self.videolength, 1, 1),
        )
        self.weighttrace_tobetime = self.videostack * k

        self.weight_trace_data = np.zeros(self.videolength)
        for i in range(self.videolength):
            self.weight_trace_data[i] = np.mean(self.weighttrace_tobetime[i])

        self.patch_camtrace_label_weighted = (
            np.arange(self.videolength) / self.samplingrate_cam
        )

        np.save(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted_trace.npy"
            ),
            self.weight_trace_data,
        )

        fig, ax1 = plt.subplots(figsize=(8.0, 5.8))
        fig.suptitle("Weighted pixel trace")
        plt.plot(self.patch_camtrace_label_weighted, self.weight_trace_data)
        ax1.set_xlabel("time(s)")
        ax1.set_ylabel("Weighted trace(counts)")
        fig.savefig(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted pixel trace.png"
            ),
            dpi=1000,
        )
        plt.show()

    def fit_on_trace(self):
        """
        Using curve fit function to calculate all the statistics.

        Returns
        None.

        """
        fit = PatchAnalysis(
            weighted_trace=self.weight_trace_data,
            DAQ_waveform=self.configured_Vp,
            voltage_step_frequency=self.VoltageStepFrequencyQSpinBox.value(),
            camera_fps=self.samplingrate_cam,
            DAQ_sampling_rate=self.samplingrate_display_curve,
            main_directory=self.main_directory,
            rhodopsin=self.Construct_name.text(),
        )
        fit.Photobleach()
        fit.ExtractPeriod()
        fit.FitSinglePeriod()
        fit.ExtractSensitivity()
        fit.Statistics()

    def save_analyzed_image(self, catag):
        if catag == "weight_image":
            Localimg = Image.fromarray(
                self.weightimage
            )  # generate an image object
            Localimg.save(
                os.path.join(
                    self.savedirectory,
                    "Weight_"
                    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    + ".tif",
                )
            )  # save as tif

    def clearplots(self):
        self.button_load.setEnabled(True)
        self.run_analysis_button.setEnabled(False)

        self.pw_weightimage.clear()
        self.pw_averageimage.clear()

        self.videostack = None

    def MessageToMainGUI(self, text):
        self.MessageBack.emit(text)

    def send_DMD_mask_contour(self, contour_from_cellselection):
        self.Cellselection_DMD_mask_contour.emit(contour_from_cellselection)

    def get_single_waveform(self):
        """
        Display a single trace.

        Returns
        None.

        """
        (
            self.single_waveform_fileName,
            _,
        ) = QtWidgets.QFileDialog.getOpenFileName(self, "Single File")
        self.textbox_single_waveform_filename.setText(
            self.single_waveform_fileName
        )

        self.single_waveform = np.load(
            self.single_waveform_fileName, allow_pickle=True
        )

        wave_sampling_rate = self.waveform_samplingrate_box.value()

        time_axis = (
            np.arange(len(self.single_waveform[9:-1])) / wave_sampling_rate
        )

        try:
            if "Ip" in self.single_waveform_fileName:
                # If plotting the patch current
                self.Ip = self.single_waveform[9:-1]

                fig, ax = plt.subplots()
                plt.plot(time_axis, self.Ip * 10000)
                ax.set_title("Patch current")
                ax.set_ylabel("Current (pA)")
                ax.set_xlabel("time(s)")
            elif "Vp" in self.single_waveform_fileName:
                # If plotting the patch voltage
                self.Vp = self.single_waveform[9:-1]

                fig, ax = plt.subplots()
                plt.plot(time_axis, self.Vp * 1000)
                ax.set_title("Patch voltage")
                ax.set_ylabel("Membrane potential (mV)")
                ax.set_xlabel("time(s)")

                logging.info(
                    "For Vp recored earlier than 22.12.2021, pls devided by 10 as the amplifier channel multipliesby 10 by default."
                )
            elif "PMT" in self.single_waveform_fileName:
                self.PMT_recording = self.single_waveform[9:-1]

                fig, ax = plt.subplots()
                plt.plot(time_axis, self.PMT_recording)
                ax.set_title("PMT voltage")
                ax.set_ylabel("Volt (V)")
                ax.set_xlabel("time(s)")

            else:
                plt.figure()
                plt.plot(self.single_waveform_fileName)

            plt.show()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)


# %%
class PlotAnalysisGUI(QWidget):
    waveforms_generated = pyqtSignal(object, object, list, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # === Initiating patchclamp class ===
        # === GUI ===
        self.setMinimumSize(200, 200)
        self.setWindowTitle("Plot display")
        self.layout = QGridLayout(self)

        pmtimageContainer = QGroupBox("Read-in")
        self.pmtimageLayout = QGridLayout()

        self.checkboxWaveform = QCheckBox("Waveform")
        self.checkboxWaveform.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )
        self.checkboxWaveform.setChecked(True)
        self.layout.addWidget(self.checkboxWaveform, 0, 0)

        self.checkboxTrace = QCheckBox("Recorded trace")
        self.checkboxTrace.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )
        self.layout.addWidget(self.checkboxTrace, 1, 0)

        self.checkboxCam = QCheckBox("Cam trace")
        self.checkboxCam.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )

        self.Spincamsamplingrate = QSpinBox(self)
        self.Spincamsamplingrate.setMaximum(2000)
        self.Spincamsamplingrate.setValue(250)
        self.Spincamsamplingrate.setSingleStep(250)
        self.layout.addWidget(self.Spincamsamplingrate, 2, 2)
        self.layout.addWidget(QLabel("Camera FPS:"), 2, 1)

        self.layout.addWidget(self.checkboxCam, 2, 0)

        self.savedirectorytextbox = QtWidgets.QLineEdit(self)
        self.pmtimageLayout.addWidget(self.savedirectorytextbox, 1, 0)

        # self.v_directorytextbox = QtWidgets.QLineEdit(self)
        # self.pmtimageLayout.addWidget(self.v_directorytextbox, 2, 0)

        self.toolButtonOpenDialog = QtWidgets.QPushButton("Select folder")

        self.toolButtonOpenDialog.clicked.connect(self._open_file_dialog)

        self.pmtimageLayout.addWidget(self.toolButtonOpenDialog, 1, 1)

        self.toolButtonLoad = QtWidgets.QPushButton("Graph")

        self.toolButtonLoad.clicked.connect(self.show_graphy)
        self.pmtimageLayout.addWidget(self.toolButtonLoad, 1, 2)

        pmtimageContainer.setLayout(self.pmtimageLayout)
        self.layout.addWidget(pmtimageContainer, 3, 0, 1, 3)

    def _open_file_dialog(self):
        self.Nest_data_directory = str(
            QtWidgets.QFileDialog.getExistingDirectory()
        )
        self.savedirectorytextbox.setText(self.Nest_data_directory)

    def show_graphy(self):
        try:
            get_ipython = sys.modules["IPython"].get_ipython
        except KeyError:
            pass
        else:
            get_ipython().run_line_magic("matplotlib", "qt")

        self.cam_trace_fluorescence_dictionary = {}
        self.cam_trace_fluorescence_filename_dictionary = {}
        self.region_file_name = []

        found_correct_waveform_spelling = False
        for file in os.listdir(self.Nest_data_directory):
            if waveform_specification.is_waveform(file):
                if found_correct_waveform_spelling:
                    if waveform_specification.is_misspelled_wavefrom(file):
                        continue
                elif not waveform_specification.is_misspelled_wavefrom(file):
                    found_correct_waveform_spelling = True

                self.wave_fileName = os.path.join(
                    self.Nest_data_directory, file
                )
            elif file.endswith("csv"):  # Quick dirty fix
                self.recorded_cam_fileName = os.path.join(
                    self.Nest_data_directory, file
                )

                self.samplingrate_cam = self.Spincamsamplingrate.value()
                self.cam_trace_time_label = np.array([])
                self.cam_trace_fluorescence_value = np.array([])

                with open(self.recorded_cam_fileName, newline="") as csvfile:
                    spamreader = csv.reader(
                        csvfile, delimiter=" ", quotechar="|"
                    )
                    for column in spamreader:
                        coords = column[0].split(",")
                        if coords[0] != "X":  # First row and column is 'x, y'
                            self.cam_trace_time_label = np.append(
                                self.cam_trace_time_label, int(coords[0])
                            )
                            self.cam_trace_fluorescence_value = np.append(
                                self.cam_trace_fluorescence_value,
                                float(coords[1]),
                            )
                self.cam_trace_fluorescence_dictionary[
                    "region_{0}".format(len(self.region_file_name) + 1)
                ] = self.cam_trace_fluorescence_value
                self.cam_trace_fluorescence_filename_dictionary[
                    "region_{0}".format(len(self.region_file_name) + 1)
                ] = file
                self.region_file_name.append(file)
            elif "Vp" in file:
                self.recorded_wave_fileName = os.path.join(
                    self.Nest_data_directory, file
                )

        # Read in configured waveforms
        configwave_wavenpfileName = self.wave_fileName
        temp_loaded_container = waveform_specification.load(
            configwave_wavenpfileName
        )

        Daq_sample_rate = waveform_specification.get_sample_rate(
            configwave_wavenpfileName
        )

        self.Checked_display_list = ["Waveform"]
        if self.checkboxTrace.isChecked():
            self.Checked_display_list = np.append(
                self.Checked_display_list, "Recorded_trace"
            )
        if self.checkboxCam.isChecked():
            self.Checked_display_list = np.append(
                self.Checked_display_list, "Cam_trace"
            )

        reference_length = len(temp_loaded_container[0]["Waveform"])
        xlabel_all = np.arange(reference_length) / Daq_sample_rate

        # === For patch perfusion ===
        if len(self.region_file_name) == 0:
            if len(self.Checked_display_list) == 2:
                figure, (ax1, ax2) = plt.subplots(2, 1)

            elif len(self.Checked_display_list) == 3:
                figure, (ax1, ax2, ax3) = plt.subplots(3, 1)

            for i in range(len(temp_loaded_container)):
                if temp_loaded_container[i]["Specification"] == "640AO":
                    pass
                elif temp_loaded_container[i]["Specification"] == "488AO":
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="488AO",
                        color="b",
                    )
                elif (
                    temp_loaded_container[i]["Specification"] == "Perfusion_8"
                ):
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="KCL",
                    )
                elif (
                    temp_loaded_container[i]["Specification"] == "Perfusion_7"
                ):
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="EC",
                    )
                elif (
                    temp_loaded_container[i]["Specification"] == "Perfusion_2"
                ):
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="Suction",
                    )
            ax1.set_title("Output waveforms")
            ax1.set_xlabel("time(s)")
            ax1.set_ylabel("Volt")
            ax1.legend()
            figure.tight_layout()
            plt.show()  # TODO cleanup
        # === For plots with camera regions ===
        if len(self.region_file_name) != 0:
            for region_number in range(len(self.region_file_name)):
                if len(self.Checked_display_list) == 2:
                    figure, (ax1, ax2) = plt.subplots(2, 1)
                    logging.info(1111)
                elif len(self.Checked_display_list) == 3:
                    figure, (ax1, ax2, ax3) = plt.subplots(3, 1)

                for i in range(len(temp_loaded_container)):
                    if temp_loaded_container[i]["Specification"] == "640AO":
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="640AO",
                            color="r",
                        )
                    elif temp_loaded_container[i]["Specification"] == "488AO":
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="488AO",
                            color="b",
                        )
                    elif (
                        temp_loaded_container[i]["Specification"]
                        == "Perfusion_8"
                    ):
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="KCL",
                        )
                    elif (
                        temp_loaded_container[i]["Specification"]
                        == "Perfusion_7"
                    ):
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="EC",
                        )
                    elif (
                        temp_loaded_container[i]["Specification"]
                        == "Perfusion_2"
                    ):
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="Suction",
                        )
                ax1.set_title("Output waveforms")
                ax1.set_xlabel("time(s)")
                ax1.set_ylabel("Volt")
                ax1.legend()

                if "Recorded_trace" in self.Checked_display_list:
                    # Read in recorded waves
                    Readin_fileName = self.recorded_wave_fileName

                    if (
                        "Vp" in os.path.split(Readin_fileName)[1]
                    ):  # See which channel is recorded
                        Vm = np.load(Readin_fileName, allow_pickle=True)
                        Vm = Vm[9:-1]  # first 5 are sampling rate, Daq coffs

                    ax2.set_xlabel("time(s)")
                    ax2.set_title("Recording")
                    ax2.set_ylabel("V (Vm*10)")
                    ax2.plot(xlabel_all, Vm, label="Vm")
                    ax2.legend()
                elif (
                    "Recorded_trace" not in self.Checked_display_list
                    and len(self.Checked_display_list) == 2
                ):
                    ax2.plot(
                        self.cam_trace_time_label / self.samplingrate_cam,
                        self.cam_trace_fluorescence_dictionary[
                            "region_{0}".format(region_number + 1)
                        ],
                        label="Fluorescence",
                    )
                    ax2.set_xlabel("time(s)")
                    ax2.set_title(
                        "ROI Fluorescence"
                        + " ("
                        + str(
                            self.cam_trace_fluorescence_filename_dictionary[
                                "region_{0}".format(region_number + 1)
                            ]
                        )
                        + ")"
                    )
                    ax2.set_ylabel("CamCounts")
                    ax2.legend()

                if len(self.Checked_display_list) == 3:
                    ax3.plot(
                        self.cam_trace_time_label / self.samplingrate_cam,
                        self.cam_trace_fluorescence_dictionary[
                            "region_{0}".format(region_number + 1)
                        ],
                        label="Fluorescence",
                    )
                    ax3.set_xlabel("time(s)")
                    ax3.set_title(
                        "ROI Fluorescence"
                        + " ("
                        + str(
                            self.cam_trace_fluorescence_filename_dictionary[
                                "region_{0}".format(region_number + 1)
                            ]
                        )
                        + ")"
                    )
                    ax3.set_ylabel("CamCounts")
                    ax3.legend()
                figure.tight_layout()
                plt.show()

    def closeEvent(self, event):
        try:
            get_ipython = sys.modules["IPython"].get_ipython
        except KeyError:
            pass
        else:
            get_ipython().run_line_magic("matplotlib", "inline")


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = AnalysisWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
