# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:32:50 2020

@author: xinmeng
"""

import logging
import os
import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
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
    QTextEdit,
    QWidget,
)
from skimage.io import imread

from .. import StylishQT
from ..CoordinatesManager import CoordinateTransformations
from ..SampleStageControl.stage import LudlStage
from .ImageProcessing import ProcessImage



def run_in_thread(fn):
    """
    https://stackoverflow.com/questions/23944657/typeerror-method-takes-1-positional-argument-but-2-were-given
    """

    @staticmethod
    def run(*k):
        thread = threading.Thread(target=fn, args=(*k,), daemon=False)
        thread.start()

        return thread  # <-- return the thread

    return run


class MainGUI(QWidget):
    waveforms_generated = pyqtSignal(object, object, list, int)

    # %%
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        # self.setMinimumSize(900, 1020)
        self.setWindowTitle("Screening Analysis")
        self.layout = QGridLayout(self)

        pg.setConfigOptions(imageAxisOrder="row-major")

        self.pop_next_top_cell_counter = 0
        self.picked_cell_index = 1
        self.Tag_folder = None
        self.Lib_folder = None
        self.Tag_round_infor = []
        self.Lib_round_infor = []
        # === GUI for Billboard display ===
        ImageDisplayContainer = QGroupBox("Billboard")
        ImageDisplayContainerLayout = QGridLayout()

        self.GraphyDisplayTab = QTabWidget()

        MatDsiplayPart = QWidget()
        MatDsiplayPart.layout = QGridLayout()

        # a figure instance to plot on
        self.Matdisplay_Figure = Figure()
        self.Matdisplay_Canvas = FigureCanvas(self.Matdisplay_Figure)

        self.Matdisplay_toolbar = NavigationToolbar(
            self.Matdisplay_Canvas, self
        )
        MatDsiplayPart.layout.addWidget(self.Matdisplay_toolbar, 0, 0)
        MatDsiplayPart.layout.addWidget(self.Matdisplay_Canvas, 1, 0)
        MatDsiplayPart.setLayout(MatDsiplayPart.layout)

        self.OriginalImgWidget = pg.ImageView()
        self.OriginalImg_item = (
            self.OriginalImgWidget.getImageItem()
        )  # setLevels
        self.OriginalImg_view = self.OriginalImgWidget.getView()
        self.OriginalImg_item.setAutoDownsample(True)

        self.OriginalImgWidget.ui.roiBtn.hide()
        self.OriginalImgWidget.ui.menuBtn.hide()
        self.OriginalImgWidget.ui.normGroup.hide()
        self.OriginalImgWidget.ui.roiPlot.hide()

        self.GraphyDisplayTab.addTab(self.OriginalImgWidget, "Image loaded")
        self.GraphyDisplayTab.addTab(MatDsiplayPart, "Scatter")

        ImageDisplayContainerLayout.addWidget(self.GraphyDisplayTab, 1, 1)

        ImageButtonContainer = QGroupBox()
        ImageButtonContainerLayout = QGridLayout()

        ButtonRankResetCoordImg = QPushButton("Reset index", self)
        ButtonRankResetCoordImg.clicked.connect(self.ResetRankIndex)
        ImageButtonContainerLayout.addWidget(ButtonRankResetCoordImg, 0, 6)

        ButtonPickedResetCoordImg = QPushButton("Reset picked index", self)
        ButtonPickedResetCoordImg.clicked.connect(self.ResetPickedIndex)
        ImageButtonContainerLayout.addWidget(ButtonPickedResetCoordImg, 0, 7)

        ButtonRankPreviousCoordImg = QPushButton("Previous ←", self)
        ButtonRankPreviousCoordImg.setShortcut("a")
        ButtonRankPreviousCoordImg.clicked.connect(
            lambda: self.GoThroughTopCells("previous")
        )
        ImageButtonContainerLayout.addWidget(ButtonRankPreviousCoordImg, 1, 6)

        self.SwitchMaskButton = StylishQT.MySwitch(
            "Mask", "#FFFFCC", "PMT", "#FFE5CC", width=65
        )
        # self.SwitchMaskButton.setChecked(True)
        self.SwitchMaskButton.clicked.connect(self.display_ML_mask)
        ImageButtonContainerLayout.addWidget(self.SwitchMaskButton, 2, 6)

        self.ShowLibImgButton = StylishQT.MySwitch(
            "Lib/KC", "#CCE5FF", "Tag/EC", "#E5FFCC", width=65
        )
        self.ShowLibImgButton.setChecked(True)
        self.ShowLibImgButton.clicked.connect(
            lambda: self.GoThroughTopCells("null")
        )
        ImageButtonContainerLayout.addWidget(self.ShowLibImgButton, 2, 7)

        ButtonRankNextCoordImg = QPushButton("Next →", self)
        ButtonRankNextCoordImg.setShortcut("d")
        ButtonRankNextCoordImg.clicked.connect(
            lambda: self.GoThroughTopCells("next")
        )
        ImageButtonContainerLayout.addWidget(ButtonRankNextCoordImg, 1, 7)

        GoSeqButton = QPushButton("Go to Cell_: ", self)
        GoSeqButton.clicked.connect(lambda: self.GoThroughTopCells("IDNumber"))
        ImageButtonContainerLayout.addWidget(GoSeqButton, 3, 6)

        self.CellSequenceBox = QSpinBox(self)
        self.CellSequenceBox.setMaximum(9000)
        self.CellSequenceBox.setMinimum(1)
        self.CellSequenceBox.setValue(1)
        self.CellSequenceBox.setSingleStep(1)
        self.CellSequenceBox.setFixedWidth(200)
        ImageButtonContainerLayout.addWidget(self.CellSequenceBox, 3, 7)

        MoveToCoordButton = QPushButton("Move FOV", self)
        MoveToCoordButton.clicked.connect(self.MoveToCoordinate)
        ImageButtonContainerLayout.addWidget(MoveToCoordButton, 4, 6)

        MoveToCellButton = QPushButton("Move to cell", self)
        MoveToCellButton.clicked.connect(self.MoveCellToCamCentre)
        ImageButtonContainerLayout.addWidget(MoveToCellButton, 4, 7)

        SaveCellInforButton = QPushButton("Save cell infor", self)
        SaveCellInforButton.clicked.connect(self.saveCellInfor)
        ImageButtonContainerLayout.addWidget(SaveCellInforButton, 5, 6)
        # ButtonRankDeleteFromList = QPushButton('Delete', self)
        # ButtonRankDeleteFromList.clicked.connect(self.DeleteFromTopCells)
        # ImageButtonContainerLayout.addWidget(ButtonRankDeleteFromList, 2, 7)

        # ButtonRankSaveList = QPushButton('Save Excel', self)
        # ButtonRankSaveList.clicked.connect(self.SaveCellsDataframetoExcel)
        # ImageButtonContainerLayout.addWidget(ButtonRankSaveList, 2, 7)

        self.ConsoleTextDisplay = QTextEdit()
        self.ConsoleTextDisplay.setFontItalic(True)
        self.ConsoleTextDisplay.setPlaceholderText(
            "Notice board from console."
        )
        self.ConsoleTextDisplay.setFixedHeight(150)
        ImageButtonContainerLayout.addWidget(
            self.ConsoleTextDisplay, 5, 6, 3, 2
        )

        ImageButtonContainer.setLayout(ImageButtonContainerLayout)

        ImageDisplayContainer.setLayout(ImageDisplayContainerLayout)
        ImageDisplayContainer.setMinimumHeight(700)
        ImageDisplayContainer.setMinimumWidth(700)

        self.layout.addWidget(ImageDisplayContainer, 0, 0, 2, 2)
        self.layout.addWidget(ImageButtonContainer, 0, 2)
        # === GUI for Image processing settings ===
        self.PostProcessTab = QTabWidget()
        self.PostProcessTab.setMaximumWidth(400)
        self.PostProcessTab.setFixedHeight(250)

        ImageProcessingContainer = QGroupBox()
        IPLayout = QGridLayout()

        IPLayout.addWidget(QLabel("Contour/soma ratio threshold:"), 0, 0)
        self.Contour_soma_ratio_thres_box = QDoubleSpinBox(self)
        self.Contour_soma_ratio_thres_box.setDecimals(4)
        self.Contour_soma_ratio_thres_box.setMinimum(-10)
        self.Contour_soma_ratio_thres_box.setMaximum(10)
        self.Contour_soma_ratio_thres_box.setValue(0.950)
        self.Contour_soma_ratio_thres_box.setSingleStep(0.15)
        IPLayout.addWidget(self.Contour_soma_ratio_thres_box, 0, 1)

        IPLayout.addWidget(
            QLabel("Mean intensity in contour threshold:"), 0, 2
        )
        self.Mean_intensity_in_contour_thres_box = QDoubleSpinBox(self)
        self.Mean_intensity_in_contour_thres_box.setDecimals(4)
        self.Mean_intensity_in_contour_thres_box.setMinimum(-10)
        self.Mean_intensity_in_contour_thres_box.setMaximum(10)
        self.Mean_intensity_in_contour_thres_box.setValue(0.250)
        self.Mean_intensity_in_contour_thres_box.setSingleStep(0.0001)
        IPLayout.addWidget(self.Mean_intensity_in_contour_thres_box, 0, 3)

        ImageProcessingContainer.setLayout(IPLayout)

        # === Loading ===
        LoadSettingContainer = QGroupBox()
        LoadSettingLayout = QGridLayout()

        self.FilepathSwitchBox = QComboBox()
        self.FilepathSwitchBox.addItems(["Tag", "Lib", "Cam Z-max"])
        LoadSettingLayout.addWidget(self.FilepathSwitchBox, 1, 0)
        self.FilepathSwitchBox.setToolTip(
            "For camera screening and generating z-max, choose cam z-max.\nFor normal analysis in folder, configure the path is enough."
        )

        self.AnalysisRoundBox = QSpinBox(self)
        self.AnalysisRoundBox.setMaximum(2000)
        self.AnalysisRoundBox.setValue(1)
        self.AnalysisRoundBox.setSingleStep(1)
        LoadSettingLayout.addWidget(self.AnalysisRoundBox, 1, 2)

        self.AddAnalysisRoundButton = QtWidgets.QPushButton("Add Round:")
        self.AddAnalysisRoundButton.clicked.connect(self.SetAnalysisRound)
        LoadSettingLayout.addWidget(self.AddAnalysisRoundButton, 1, 1)

        self.datasavedirectorytextbox = QLineEdit(self)
        self.datasavedirectorytextbox.setPlaceholderText("Data directory")
        LoadSettingLayout.addWidget(self.datasavedirectorytextbox, 0, 0, 1, 4)

        self.toolButtonOpenDialog = QtWidgets.QPushButton("Set path")
        self.toolButtonOpenDialog.clicked.connect(self.SetAnalysisPath)
        LoadSettingLayout.addWidget(self.toolButtonOpenDialog, 0, 4)

        readExcelButtonOpenDialog = QtWidgets.QPushButton("Read excel")
        readExcelButtonOpenDialog.clicked.connect(self.ReadEexcel)
        LoadSettingLayout.addWidget(readExcelButtonOpenDialog, 0, 5)

        ExecuteAnalysisButton = QPushButton("Load", self)
        # ExecuteAnalysisButton.setObjectName('Startbutton')
        ExecuteAnalysisButton.clicked.connect(
            lambda: self.run_in_thread(self.ScreeningAnalysis)
        )
        LoadSettingLayout.addWidget(ExecuteAnalysisButton, 1, 3)

        self.ClearAnalysisInforButton = QtWidgets.QPushButton("Clear infor")
        self.ClearAnalysisInforButton.clicked.connect(self.ClearAnalysisInfor)
        LoadSettingLayout.addWidget(self.ClearAnalysisInforButton, 1, 4)

        Re_plot_Button = QtWidgets.QPushButton("Replot data")
        Re_plot_Button.clicked.connect(self.ReplotExcel)
        LoadSettingLayout.addWidget(Re_plot_Button, 1, 5)

        self.X_axisBox = QComboBox()
        self.X_axisBox.addItems(
            [
                "Lib_Tag_contour_ratio",
                "Contour_soma_ratio_Lib",
                "Contour_soma_ratio",
                "KC_EC_LibTag_contour_ratio",
                "KC_EC_Mean_intensity_in_contour_ratio",
            ]
        )
        LoadSettingLayout.addWidget(self.X_axisBox, 2, 1, 1, 3)
        LoadSettingLayout.addWidget(QLabel("X axis: "), 2, 0)

        self.WeightBoxSelectionFactor_1 = QDoubleSpinBox(self)
        self.WeightBoxSelectionFactor_1.setDecimals(2)
        self.WeightBoxSelectionFactor_1.setMinimum(0)
        self.WeightBoxSelectionFactor_1.setMaximum(1)
        self.WeightBoxSelectionFactor_1.setValue(0.6)
        self.WeightBoxSelectionFactor_1.setSingleStep(0.1)
        LoadSettingLayout.addWidget(self.WeightBoxSelectionFactor_1, 2, 5)
        LoadSettingLayout.addWidget(QLabel("Weight:"), 2, 4)

        self.Y_axisBox = QComboBox()
        self.Y_axisBox.addItems(
            [
                "Mean_intensity_in_contour_Lib",
                "Contour_soma_ratio_Lib",
                "Mean_intensity_in_contour",
                "Mean_intensity_in_contour_Lib_EC",
                "Lib_Tag_contour_ratio_EC",
                "Contour_soma_ratio_Lib_EC",
            ]
        )
        LoadSettingLayout.addWidget(self.Y_axisBox, 3, 1, 1, 3)
        LoadSettingLayout.addWidget(QLabel("Y axis: "), 3, 0)

        self.WeightBoxSelectionFactor_2 = QDoubleSpinBox(self)
        self.WeightBoxSelectionFactor_2.setDecimals(2)
        self.WeightBoxSelectionFactor_2.setMinimum(0)
        self.WeightBoxSelectionFactor_2.setMaximum(1)
        self.WeightBoxSelectionFactor_2.setValue(0.5)
        self.WeightBoxSelectionFactor_2.setSingleStep(0.1)
        LoadSettingLayout.addWidget(self.WeightBoxSelectionFactor_2, 3, 5)
        LoadSettingLayout.addWidget(QLabel("Weight:"), 3, 4)

        self.Z_axisBox = QComboBox()
        self.Z_axisBox.addItems(
            [
                "None",
                "Contour_soma_ratio_Lib",
                "Contour_soma_ratio",
                "Contour_soma_ratio_Lib_EC",
            ]
        )
        LoadSettingLayout.addWidget(self.Z_axisBox, 4, 1, 1, 3)
        LoadSettingLayout.addWidget(QLabel("Z axis: "), 4, 0)

        self.WeightBoxSelectionFactor_3 = QDoubleSpinBox(self)
        self.WeightBoxSelectionFactor_3.setDecimals(2)
        self.WeightBoxSelectionFactor_3.setMinimum(0)
        self.WeightBoxSelectionFactor_3.setMaximum(1)
        self.WeightBoxSelectionFactor_3.setValue(0.0)
        self.WeightBoxSelectionFactor_3.setSingleStep(0.5)
        LoadSettingLayout.addWidget(self.WeightBoxSelectionFactor_3, 4, 5)
        LoadSettingLayout.addWidget(QLabel("Weight:"), 4, 4)

        LoadSettingContainer.setLayout(LoadSettingLayout)

        # === GUI for Selection threshold settings ===

        self.PostProcessTab.addTab(LoadSettingContainer, "Loading settings")
        self.PostProcessTab.addTab(
            ImageProcessingContainer, "Image analysis thresholds"
        )

        self.layout.addWidget(self.PostProcessTab, 1, 2)

        self.setLayout(self.layout)

    # === functions for console display ===
    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.ConsoleTextDisplay.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.ConsoleTextDisplay.setTextCursor(cursor)
        self.ConsoleTextDisplay.ensureCursorVisible()

    # %%
    """
    # FUNCTIONS FOR DATA ANALYSIS AND DISPLAY
    """

    def SetAnalysisPath(self):
        """
        Set the directory information for rounds.

        Returns
        None.

        """
        self.Analysis_saving_directory = str(
            QtWidgets.QFileDialog.getExistingDirectory()
        )
        self.datasavedirectorytextbox.setText(self.Analysis_saving_directory)

        self.Tag_folder = self.Analysis_saving_directory
        self.Lib_folder = self.Analysis_saving_directory

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

    def ClearAnalysisInfor(self):
        self.Tag_folder = None
        self.Lib_folder = None
        self.Tag_round_infor = []
        self.Lib_round_infor = []

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

    def ScreeningAnalysis(self):
        """
        Main function running through the whole brightness screening data analysis.
        1) Create MaskRCNN instance.
        2) Get the Dataframe from tag and library screening round.
        3) Merge the two rounds based on bounding box intersection between cell images.
        4) Filter out results under threadhold.

        Returns
        None.

        """
        self.Mean_intensity_in_contour_thres = (
            self.Mean_intensity_in_contour_thres_box.value()
        )
        self.Contour_soma_ratio_thres = (
            self.Contour_soma_ratio_thres_box.value()
        )

        self.normalOutputWritten("Start loading images...\n")

        
        
       

    # %%
    def ReadEexcel(self):
        """
        Read in existing excel file and do the ranking and graph generating.

        Returns
        None.

        """
        self.Mean_intensity_in_contour_thres = (
            self.Mean_intensity_in_contour_thres_box.value()
        )
        self.Contour_soma_ratio_thres = (
            self.Contour_soma_ratio_thres_box.value()
        )

        self.ExcelfileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope",  # TODO hardcoded path
            "(*.xlsx)",
        )
        try:
            self.Excelfile = pd.read_excel(self.ExcelfileName)
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.Excelfile = pd.read_excel(
                self.ExcelfileName, engine="openpyxl"
            )
        # Return the directory name of pathname path.
        self.Tag_folder = os.path.dirname(self.ExcelfileName)
        self.Lib_folder = os.path.dirname(self.ExcelfileName)
        self.Analysis_saving_directory = os.path.dirname(self.ExcelfileName)
        DataFrames_filtered = ProcessImage.FilterDataFrames(
            self.Excelfile,
            self.Mean_intensity_in_contour_thres,
            self.Contour_soma_ratio_thres,
        )

        self.DataFrame_sorted = ProcessImage.sort_on_axes(
            DataFrames_filtered,
            axis_1=self.X_axisBox.currentText(),
            axis_2=self.Y_axisBox.currentText(),
            axis_3=self.Z_axisBox.currentText(),
            weight_1=self.WeightBoxSelectionFactor_1.value(),
            weight_2=self.WeightBoxSelectionFactor_2.value(),
            weight_3=self.WeightBoxSelectionFactor_3.value(),
        )
        # try:
        # print("Save CellsDataframe to Excel...")
        # self.DataFrame_sorted.to_excel(os.path.join(self.Tag_folder, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_ReloadedEexcel_CellsProperties.xlsx'))
        # print("Saved.")
        # except:
        # pass

        self.UpdateSelectionScatter()

    def ReplotExcel(self):
        """
        Replot the data based on current hyperparameters.

        Returns
        None.

        """

        DataFrames_filtered = ProcessImage.FilterDataFrames(
            self.Cell_DataFrame_Merged,
            self.Mean_intensity_in_contour_thres,
            self.Contour_soma_ratio_thres,
        )

        self.DataFrame_sorted = ProcessImage.sort_on_axes(
            DataFrames_filtered,
            axis_1=self.X_axisBox.currentText(),
            axis_2=self.Y_axisBox.currentText(),
            axis_3=self.Z_axisBox.currentText(),
            weight_1=self.WeightBoxSelectionFactor_1.value(),
            weight_2=self.WeightBoxSelectionFactor_2.value(),
            weight_3=self.WeightBoxSelectionFactor_3.value(),
        )
        logging.info("Save CellsDataframe to Excel...")
        self.DataFrame_sorted.to_excel(
            os.path.join(
                self.Tag_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + "_Replot_CellsProperties.xlsx",
            )
        )

        self.UpdateSelectionScatter()

    def UpdateSelectionScatter(self):
        """
        Plot the scatter graph.

        Returns
        None.

        """
        if self.Z_axisBox.currentText() == "None":
            # if two axes analysed
            self.EvaluatingPara_list = [
                self.X_axisBox.currentText(),
                self.Y_axisBox.currentText(),
            ]
        else:
            self.EvaluatingPara_list = [
                self.X_axisBox.currentText(),
                self.Y_axisBox.currentText(),
                self.Z_axisBox.currentText(),
            ]

        self.Matdisplay_Figure.clear()

        if len(self.EvaluatingPara_list) == 2:
            ax1 = self.Matdisplay_Figure.add_subplot(111)
            ax1.scatter(
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[1]],
                s=np.pi * 3,
                c="blue",
                alpha=0.5,
            )
            ax1.set_xlabel(self.EvaluatingPara_list[0])
            ax1.set_ylabel(self.EvaluatingPara_list[1])
            self.Matdisplay_Figure.tight_layout()
            self.Matdisplay_Canvas.draw()

            # Some numbers ready for tracing back
            self.TotaNumofCellSelected = len(self.DataFrame_sorted)
            self.TotalCellNum = len(self.DataFrame_sorted)
            self.normalOutputWritten(
                "---- Total cells selected: {}; Total cells: {}----\n".format(
                    self.TotaNumofCellSelected, self.TotalCellNum
                )
            )

            # saving_directory = os.path.join(self.Tag_folder, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_Screening scatters.html')
            # self.ProcessML.showPlotlyScatter\
            # (self.DataFrame_sorted, x_axis=self.EvaluatingPara_list[0], y_axis=self.EvaluatingPara_list[1], saving_directory = saving_directory)

        elif len(self.EvaluatingPara_list) == 3:
            ax1 = self.Matdisplay_Figure.add_subplot(111, projection="3d")
            ax1.scatter(
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[1]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[2]],
                s=np.pi * 3,
                c="blue",
                alpha=0.5,
            )
            ax1.set_xlabel(self.EvaluatingPara_list[0])
            ax1.set_ylabel(self.EvaluatingPara_list[1])
            ax1.set_zlabel(self.EvaluatingPara_list[2])
            self.Matdisplay_Figure.tight_layout()
            self.Matdisplay_Canvas.draw()

            # Some numbers ready for tracing back
            self.TotaNumofCellSelected = len(self.DataFrame_sorted)
            self.TotalCellNum = len(self.DataFrame_sorted)
            self.normalOutputWritten(
                "---- Total cells selected: {}; Total cells: {}----\n".format(
                    self.TotaNumofCellSelected, self.TotalCellNum
                )
            )

    def GoThroughTopCells(self, direction):
        """
        According to the direction, get the cell index in dataframe and display
        the cell image and position in scatter plot.

        Cell dataframe index starts from Cell 1, which corresponds to pop_next_top_cell_counter = 0.

        Parameters
        direction : string
            In which direction show the next cell.

        Returns
        None.

        """
        # Show the next ranked cell
        if direction == "next":
            if self.pop_next_top_cell_counter > (
                self.TotaNumofCellSelected - 1
            ):  # Make sure it doesn't go beyond the last coords.
                self.pop_next_top_cell_counter -= 1

            self.CurrentRankCellpProperties = self.DataFrame_sorted.iloc[
                self.pop_next_top_cell_counter
            ]

            self.display_selected_image()

            self.pop_next_top_cell_counter += (
                1  # Alwasy plus 1 to get it ready for next move.
            )

        # Show the previous ranked cell
        elif direction == "previous":
            self.pop_next_top_cell_counter -= 2
            if self.pop_next_top_cell_counter >= 0:
                self.CurrentRankCellpProperties = self.DataFrame_sorted.iloc[
                    self.pop_next_top_cell_counter
                ]

                self.display_selected_image()

                if self.pop_next_top_cell_counter < (
                    self.TotaNumofCellSelected - 1
                ):
                    self.pop_next_top_cell_counter += 1
            else:
                self.pop_next_top_cell_counter = 0

        # Show the current cell
        elif direction == "null":
            self.pop_next_top_cell_counter -= 1

            self.CurrentRankCellpProperties = self.DataFrame_sorted.iloc[
                self.pop_next_top_cell_counter
            ]

            self.display_selected_image()

            self.pop_next_top_cell_counter += 1

        # if go to specific cell
        elif direction == "IDNumber":
            self.GotoSequence()

        # Show cell in the scatter
        if direction == "IDNumber":
            self.ShowSequenceScatter()
        else:
            self.ShowScatterPos()

    def GotoSequence(self):
        """
        Go to a specific cell
        """
        self.SpecificIndexInArray = "Cell " + str(self.CellSequenceBox.value())
        self.CurrentRankCellpProperties = self.DataFrame_sorted.loc[
            self.SpecificIndexInArray, :
        ]

        self.display_selected_image()

    def display_selected_image(self):
        """
        Display the current selected cell image and output some text information
        """
        # === Show image with cell in box ===
        # === readin image ===

        if self.ShowLibImgButton.isChecked():
            # === Display the library image ===
            if "ImgNameInfor" in self.CurrentRankCellpProperties.index:
                # In brightness lib/tag dataframe
                self.meta_data = self.CurrentRankCellpProperties.loc[
                    "ImgNameInfor"
                ]
                Each_bounding_box = self.CurrentRankCellpProperties.loc[
                    "BoundingBox"
                ]
            elif "ImgNameInfor_Lib" in self.CurrentRankCellpProperties.index:
                # In brightness lib/tag dataframe
                self.meta_data = self.CurrentRankCellpProperties.loc[
                    "ImgNameInfor_Lib"
                ]
                Each_bounding_box = self.CurrentRankCellpProperties.loc[
                    "BoundingBox_Lib"
                ]
            elif (
                "ImgNameInfor_Lib_EC" in self.CurrentRankCellpProperties.index
            ):
                # In Kcl assay, show the KC lib image.
                self.meta_data = self.CurrentRankCellpProperties.loc[
                    "ImgNameInfor_Lib_KC"
                ]
                Each_bounding_box = self.CurrentRankCellpProperties.loc[
                    "BoundingBox_Lib_KC"
                ]

            if ".tif" in self.meta_data:
                self.lib_imagefilename = os.path.join(
                    self.Lib_folder, self.meta_data
                )
            else:
                if self.FilepathSwitchBox.currentText() != "Cam Z-max":
                    self.lib_imagefilename = os.path.join(
                        self.Lib_folder, self.meta_data + "_PMT_0Zmax.tif"
                    )
                else:
                    # In case of camera screening
                    self.lib_imagefilename = os.path.join(
                        self.Lib_folder, self.meta_data + "_Cam_Zmax.tif"
                    )

            logging.info(
                self.lib_imagefilename[
                    len(self.lib_imagefilename)
                    - 50 : len(self.lib_imagefilename)
                ]
            )
            self.loaded_image_display = imread(
                self.lib_imagefilename, as_gray=True
            )

        else:
            # === Display the tag protein image ===
            if "ImgNameInfor" in self.CurrentRankCellpProperties.index:
                # In brightness lib/tag dataframe
                self.meta_data = self.CurrentRankCellpProperties.loc[
                    "ImgNameInfor"
                ]
                Each_bounding_box = self.CurrentRankCellpProperties.loc[
                    "BoundingBox"
                ]
            elif "ImgNameInfor_Tag" in self.CurrentRankCellpProperties.index:
                # In brightness lib/tag dataframe
                self.meta_data = self.CurrentRankCellpProperties.loc[
                    "ImgNameInfor_Tag"
                ]
                Each_bounding_box = self.CurrentRankCellpProperties.loc[
                    "BoundingBox_Tag"
                ]
            elif (
                "ImgNameInfor_Lib_EC" in self.CurrentRankCellpProperties.index
            ):
                # In Kcl assay, show the KC lib image.
                self.meta_data = self.CurrentRankCellpProperties.loc[
                    "ImgNameInfor_Lib_EC"
                ]
                Each_bounding_box = self.CurrentRankCellpProperties.loc[
                    "BoundingBox_Lib_EC"
                ]

            if ".tif" in self.meta_data:
                self.tag_imagefilename = os.path.join(
                    self.Tag_folder, self.meta_data
                )
            else:
                self.tag_imagefilename = os.path.join(
                    self.Tag_folder, self.meta_data + "_PMT_0Zmax.tif"
                )
            logging.info(
                self.tag_imagefilename[
                    len(self.tag_imagefilename)
                    - 50 : len(self.tag_imagefilename)
                ]
            )
            self.loaded_image_display = imread(
                self.tag_imagefilename, as_gray=True
            )

        # Get stage coordinates information.
        self.coordinate_text = self.meta_data[
            self.meta_data.index("_R") + 1 : len(self.meta_data)
        ]

        # Retrieve boundingbox information
        minr = int(
            Each_bounding_box[
                Each_bounding_box.index("minr")
                + 4 : Each_bounding_box.index("_maxr")
            ]
        )
        maxr = (
            int(
                Each_bounding_box[
                    Each_bounding_box.index("maxr")
                    + 4 : Each_bounding_box.index("_minc")
                ]
            )
            - 1
        )
        minc = int(
            Each_bounding_box[
                Each_bounding_box.index("minc")
                + 4 : Each_bounding_box.index("_maxc")
            ]
        )
        maxc = (
            int(
                Each_bounding_box[
                    Each_bounding_box.index("maxc")
                    + 4 : len(Each_bounding_box)
                ]
            )
            - 1
        )

        self.currentCellCentre_PMTimgCoordinates = [
            (minr + maxr) // 2,
            (minc + maxc) // 2,
        ]
        logging.info(
            "current CellCentre_PMTimgCoordinates: {}".format(
                self.currentCellCentre_PMTimgCoordinates
            )
        )

        if self.FilepathSwitchBox.currentText() != "Cam Z-max":
            # Show the bounding box
            self.loaded_image_display[minr, minc:maxc] = 4
            self.loaded_image_display[maxr, minc:maxc] = 4
            self.loaded_image_display[minr:maxr, minc] = 4
            self.loaded_image_display[minr:maxr, maxc] = 4
        elif self.FilepathSwitchBox.currentText() == "Cam Z-max":
            # In case of camera
            self.loaded_image_display[minc:maxc, minr] = 10**4
            self.loaded_image_display[minc:maxc, maxr] = 10**4
            self.loaded_image_display[minc, minr:maxr] = 10**4
            self.loaded_image_display[maxc, minr:maxr] = 10**4

        # === Show image in imageview ===
        if self.FilepathSwitchBox.currentText() != "Cam Z-max":
            # === In case of PMT image ===
            self.OriginalImg_item.setImage(
                np.fliplr(np.rot90(self.loaded_image_display)), autoLevels=True
            )
            self.OriginalImg_item.setLevels((0, 3))
        elif self.FilepathSwitchBox.currentText() == "Cam Z-max":
            self.OriginalImg_item.setImage(
                self.loaded_image_display, autoLevels=True
            )
            self.OriginalImg_item.setLevels((50, 190))

        # === Print details of cell of interest ===
        self.normalOutputWritten(
            "------------------IDNumber {}----------------\n".format(
                self.CurrentRankCellpProperties.name
            )
        )
        self.normalOutputWritten(
            "ID: {}\n{}: {}\n{}: {}\n{}: {}\n".format(
                self.meta_data,
                self.EvaluatingPara_list[0],
                round(
                    self.CurrentRankCellpProperties.loc[
                        self.EvaluatingPara_list[0]
                    ],
                    4,
                ),
                self.EvaluatingPara_list[1],
                round(
                    self.CurrentRankCellpProperties.loc[
                        self.EvaluatingPara_list[1]
                    ],
                    4,
                ),
                "IDNumber",
                self.CurrentRankCellpProperties.name,
            )
        )

    def display_ML_mask(self):
        """
        Display the ML mask.
        """
        if self.SwitchMaskButton.isChecked():
            if self.ShowLibImgButton.isChecked():
                roundnum = self.meta_data[0 : self.meta_data.index("_Grid")]

                if not os.path.exists(
                    os.path.join(self.Tag_folder, "ML_masks")
                ):
                    # All the mask images from MaskRCNN are saved in tag folder
                    # Normal screening, masks saved under different folder names.
                    self.lib_mask_imagefilename = os.path.join(
                        self.Tag_folder,
                        "MLimages_{}".format(roundnum),
                        self.meta_data + ".tif",
                    )
                else:
                    self.lib_mask_imagefilename = os.path.join(
                        self.Tag_folder,
                        "ML_masks",
                        "ML_mask_"
                        + self.meta_data[0 : len(self.meta_data) - 4]
                        + ".png",
                    )

                self.loaded_image_display = imread(self.lib_mask_imagefilename)

            else:
                roundnum = self.meta_data[0 : self.meta_data.index("_Grid")]

                if not os.path.exists(
                    os.path.join(self.Tag_folder, "ML_masks")
                ):
                    # All the mask images from MaskRCNN are saved in tag folder
                    # Normal screening, masks saved under different folder names.
                    self.tag_mask_imagefilename = os.path.join(
                        self.Tag_folder,
                        "MLimages_{}".format(roundnum),
                        self.meta_data + ".tif",
                    )
                else:
                    self.tag_mask_imagefilename = os.path.join(
                        self.Tag_folder,
                        "ML_masks",
                        "ML_mask_"
                        + self.meta_data[0 : len(self.meta_data) - 4]
                        + ".png",
                    )

                self.loaded_image_display = imread(self.tag_mask_imagefilename)

            # === Show image in imageview ===
            self.OriginalImg_item.setImage(
                np.fliplr(np.rot90(self.loaded_image_display)), autoLevels=True
            )

        else:
            self.display_selected_image()

    def ShowScatterPos(self):
        """
        Show the scatter position of cell in ranked sequence.

        Returns
        None.

        """
        # if self.ButtonShowInScatter.isChecked():
        self.Matdisplay_Figure.clear()
        if len(self.EvaluatingPara_list) == 2:
            ax1 = self.Matdisplay_Figure.add_subplot(111)
            ax1.scatter(
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[1]],
                s=np.pi * 3,
                c="blue",
                alpha=0.5,
            )
            ax1.scatter(
                self.DataFrame_sorted.iloc[
                    self.pop_next_top_cell_counter - 1, :
                ].loc[self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.iloc[
                    self.pop_next_top_cell_counter - 1, :
                ].loc[self.EvaluatingPara_list[1]],
                s=np.pi * 8,
                c="red",
                alpha=0.5,
            )
            ax1.set_xlabel(self.EvaluatingPara_list[0])
            ax1.set_ylabel(self.EvaluatingPara_list[1])
            self.Matdisplay_Figure.tight_layout()
            self.Matdisplay_Canvas.draw()
        # else:
        # self.GoThroughTopCells('null')
        elif len(self.EvaluatingPara_list) == 3:
            ax1 = self.Matdisplay_Figure.add_subplot(111, projection="3d")
            ax1.scatter(
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[1]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[2]],
                s=np.pi * 3,
                c="blue",
                alpha=0.5,
            )
            ax1.scatter(
                self.DataFrame_sorted.iloc[
                    self.pop_next_top_cell_counter - 1, :
                ].loc[self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.iloc[
                    self.pop_next_top_cell_counter - 1, :
                ].loc[self.EvaluatingPara_list[1]],
                self.DataFrame_sorted.iloc[
                    self.pop_next_top_cell_counter - 1, :
                ].loc[self.EvaluatingPara_list[2]],
                s=np.pi * 8,
                c="red",
                alpha=0.5,
            )
            ax1.set_xlabel(self.EvaluatingPara_list[0])
            ax1.set_ylabel(self.EvaluatingPara_list[1])
            ax1.set_zlabel(self.EvaluatingPara_list[2])
            self.Matdisplay_Figure.tight_layout()
            self.Matdisplay_Canvas.draw()

    def ShowSequenceScatter(self):
        """
        Show the scatter position of cell which is searched through index IDnum.

        Returns
        None.

        """
        # if self.ShowSequenceScatterButton.isChecked():
        self.Matdisplay_Figure.clear()

        if len(self.EvaluatingPara_list) == 2:
            ax1 = self.Matdisplay_Figure.add_subplot(111)
            ax1.scatter(
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[0]],
                self.DataFrame_sorted.loc[:, self.EvaluatingPara_list[1]],
                s=np.pi * 3,
                c="blue",
                alpha=0.5,
            )
            ax1.scatter(
                self.DataFrame_sorted.loc[self.SpecificIndexInArray, :].loc[
                    self.EvaluatingPara_list[0]
                ],
                self.DataFrame_sorted.loc[self.SpecificIndexInArray, :].loc[
                    self.EvaluatingPara_list[1]
                ],
                s=np.pi * 8,
                c="red",
                alpha=0.5,
            )

            ax1.set_xlabel(self.EvaluatingPara_list[0])
            ax1.set_ylabel(self.EvaluatingPara_list[1])
            self.Matdisplay_Figure.tight_layout()
            self.Matdisplay_Canvas.draw()
        # else:
        # self.GoThroughTopCells('sequence')
        elif len(self.EvaluatingPara_list) == 3:
            ax1 = self.Matdisplay_Figure.add_subplot(111, projection="3d")
            ax1.scatter(
                self.DataFrame_sorted.loc[self.SpecificIndexInArray, :].loc[
                    self.EvaluatingPara_list[0]
                ],
                self.DataFrame_sorted.loc[self.SpecificIndexInArray, :].loc[
                    self.EvaluatingPara_list[1]
                ],
                self.DataFrame_sorted.loc[self.SpecificIndexInArray, :].loc[
                    self.EvaluatingPara_list[2]
                ],
                s=np.pi * 8,
                c="red",
                alpha=0.5,
            )
            ax1.set_xlabel(self.EvaluatingPara_list[0])
            ax1.set_ylabel(self.EvaluatingPara_list[1])
            ax1.set_zlabel(self.EvaluatingPara_list[2])
            self.Matdisplay_Figure.tight_layout()
            self.Matdisplay_Canvas.draw()

    def MoveToCoordinate(self):
        """
        Move to the stage coordinate of current inspecting cell.
        """

        coordinate_row = int(
            self.coordinate_text[
                self.coordinate_text.index("R")
                + 1 : self.coordinate_text.index("C")
            ]
        )
        coordinate_col = int(
            self.coordinate_text[
                self.coordinate_text.index("C") + 1 : len(self.coordinate_text)
            ]
        )

        ludlStage = LudlStage("COM12")
        ludlStage.moveAbs(coordinate_row, coordinate_col)

    def MoveCellToCamCentre(self):
        """
        Move the identified cell to the centre of camera FOV.
        """
        # Find the corresponding centre coordinates in camera image.
        camera_centre_coordinates = (
            CoordinateTransformations.general_coordinates_transformation(
                [self.currentCellCentre_PMTimgCoordinates],
                "PMT2Camera",
                scanning_config=[5, 500],
            )
        )
        logging.info(
            "Corresponding coordinates in camera image: {}".format(
                camera_centre_coordinates[0]
            )
        )

        # Calculate the relative stage move coordinates
        # Camera pixel offset
        camera_image_size = 2048
        relative_cam_pixel_offset_row = (
            camera_image_size / 2 - camera_centre_coordinates[0][0]
        )
        relative_cam_pixel_offset_col = (
            camera_image_size / 2 - camera_centre_coordinates[0][1]
        )

        # Transform into stage coordinates
        relative_stage_move_row = int(relative_cam_pixel_offset_row * -1.132)
        relative_stage_move_col = int(relative_cam_pixel_offset_col * 1.132)
        logging.info(
            "Stage relative moving steps: {}".format(
                [relative_stage_move_row, relative_stage_move_col]
            )
        )

        ludlStage = LudlStage("COM12")
        ludlStage.moveRel(relative_stage_move_row, relative_stage_move_col)

    def SaveCellsDataframetoExcel(self):
        """
        Save the data sheet into excel file.
        """

        self.DataFrame_sorted.to_excel(
            os.path.join(
                self.Tag_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + "_CellsProperties.xlsx",
            )
        )

    def saveCellInfor(self):
        # create an exporter instance, as an argument give it
        # the item you wish to export
        exporter = pg.exporters.ImageExporter(self.OriginalImg_item)

        # set export parameters if needed
        exporter.parameters()[
            "width"
        ] = 500  # (note this also affects height parameter)

        # save to file
        if self.picked_cell_index == 1:
            if not os.path.exists(
                os.path.join(self.Analysis_saving_directory, "Picked cells")
            ):
                # If the folder is not there, create the folder
                logging.info("Create Picked Cells folder.")
                os.mkdir(
                    os.path.join(
                        self.Analysis_saving_directory, "Picked cells"
                    )
                )

        exporter.export(
            os.path.join(
                self.Analysis_saving_directory,
                "Picked cells\\"
                + "Picked cell_"
                + str(self.picked_cell_index)
                + " "
                + self.coordinate_text
                + " "
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".png",
            )
        )

        self.Matdisplay_Figure.savefig(
            os.path.join(
                self.Analysis_saving_directory,
                "Picked cells\\"
                + "Picked cell_"
                + str(self.picked_cell_index)
                + " scatter"
                + self.coordinate_text
                + " "
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".png",
            )
        )

        self.picked_cell_index += 1

    def ResetRankIndex(self):
        self.pop_next_top_cell_counter = 0

    def ResetPickedIndex(self):
        self.picked_cell_index = 1

    # %%


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = MainGUI()
        mainwin.show()
        app.exec_()

    run_app()
