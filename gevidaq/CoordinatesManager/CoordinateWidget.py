# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:17:02 2020

@author: ideheer

Modified by Xin, adding machine learning portal.
"""

import colorsys
import logging
import random
import sys
import threading

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)
from skimage.io import imread
from skimage.measure import find_contours

from .. import Icons
from ..ImageAnalysis.ImageProcessing import ProcessImage
from ..StylishQT import SquareImageView, cleanButton, roundQGroupBox
from . import (
    DMDWidget,
    GalvoWidget,
    ManualRegistration,
    StageRegistrationWidget,
)
from .ui_widgets.DrawingWidget import DrawingWidget


class CoordinatesWidgetUI(QWidget):
    sig_cast_mask_coordinates_to_dmd = pyqtSignal(dict)
    sig_cast_mask_coordinates_to_galvo = pyqtSignal(list)
    sig_start_registration = pyqtSignal()
    sig_finished_registration = pyqtSignal()
    sig_cast_camera_image = pyqtSignal(np.ndarray)

    MessageBack = pyqtSignal(str)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main_application = parent
        self.init_gui()
        self.sig_to_calling_widget = {}
        self.untransformed_mask_dict = {}
        # HamamatsuUI.CameraUI.signal_SnapImg.connect(self.receive_image_from_camera)

    def closeEvent(self, event):
        try:
            self.DMD
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
        else:
            self.DMD.disconnect_DMD()

        QtWidgets.QApplication.quit()
        event.accept()

    # %%
    def init_gui(self):
        self.setWindowTitle("Coordinate control")

        self.layout = QGridLayout()
        self.setMinimumSize(1250, 1000)
        self.setLayout(self.layout)

        self.image_mask_stack = QTabWidget()

        # === ROIs win ===
        self.selection_view = DrawingWidget(self)
        self.selection_view.setMinimumWidth(900)
        self.selection_view.enable_drawing(True)
        # self.selection_view.getView().setLimits(xMin = 0, xMax = 2048, yMin = 0, yMax = 2048, \
        # minXRange = 2048, minYRange = 2048, maxXRange = 2048, maxYRange = 2048)
        self.selection_view.ui.roiBtn.hide()
        self.selection_view.ui.menuBtn.hide()
        self.selection_view.ui.normGroup.hide()
        self.selection_view.ui.roiPlot.hide()
        # self.selection_view.setImage(plt.imread('CoordinatesManager/Registration_Images/StageRegistration/Distance200_Offset0/A1.png'))

        # === Mask win ===
        self.mask_view = SquareImageView()
        self.mask_view.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.mask_view.ui.roiBtn.hide()
        self.mask_view.ui.menuBtn.hide()
        self.mask_view.ui.normGroup.hide()
        self.mask_view.ui.roiPlot.hide()
        self.mask_view.ui.histogram.hide()

        # === MAsk RCNN ===
        MLmaskviewBox = QWidget()
        MLmaskviewBoxLayout = QGridLayout()

        self.Matdisplay_Figure = Figure()
        self.Matdisplay_Canvas = FigureCanvas(self.Matdisplay_Figure)
        # self.Matdisplay_Canvas.setFixedWidth(900)
        # self.Matdisplay_Canvas.setFixedHeight(900)
        self.Matdisplay_Canvas.mpl_connect("button_press_event", self._onclick)

        self.Matdisplay_toolbar = NavigationToolbar(
            self.Matdisplay_Canvas, self
        )

        MLmaskviewBoxLayout.addWidget(self.Matdisplay_toolbar, 0, 0)
        MLmaskviewBoxLayout.addWidget(self.Matdisplay_Canvas, 1, 0)

        MLmaskviewBox.setLayout(MLmaskviewBoxLayout)

        self.image_mask_stack.addTab(self.selection_view, "Select")
        self.image_mask_stack.addTab(self.mask_view, "Mask")
        self.image_mask_stack.addTab(MLmaskviewBox, "Mask-RCNN")

        self.layout.addWidget(self.image_mask_stack, 0, 0, 4, 7)

        # === Mask generation Container ===

        self.maskGeneratorContainer = roundQGroupBox()
        self.maskGeneratorContainer.setFixedSize(320, 220)
        self.maskGeneratorContainer.setTitle("Mask generator")
        self.maskGeneratorContainerLayout = QGridLayout()

        self.maskGeneratorLayout = QGridLayout()
        self.maskGeneratorContainer.setLayout(
            self.maskGeneratorContainerLayout
        )

        # self.loadMaskFromFileButton = QPushButton('Load mask')
        # self.loadMaskFromFileButton.clicked.connect(self.load_mask_from_file)

        self.addRoiButton = QPushButton("Add ROI")
        self.createMaskButton = QPushButton("Add mask")
        self.deleteMaskButton = QPushButton("Delete mask")
        self.removeSelectionButton = cleanButton()
        self.removeSelectionButton.setFixedHeight(25)
        self.removeSelectionButton.setToolTip("Remove rois and output signals")
        self.addRoiButton.clicked.connect(self.add_polygon_roi)

        self.createMaskButton.clicked.connect(self.create_mask)
        self.deleteMaskButton.clicked.connect(self.delete_mask)
        self.removeSelectionButton.clicked.connect(self.remove_selection)

        self.LoadImageButton = QPushButton("Load image")
        self.LoadImageButton.clicked.connect(self.read_image_to_draw_roi)

        self.maskGeneratorContainerLayout.addWidget(self.addRoiButton, 0, 0)
        self.maskGeneratorContainerLayout.addWidget(self.LoadImageButton, 2, 0)
        self.maskGeneratorContainerLayout.addWidget(
            self.createMaskButton, 2, 1
        )
        self.maskGeneratorContainerLayout.addWidget(
            self.deleteMaskButton, 2, 2
        )
        self.maskGeneratorContainerLayout.addWidget(
            self.removeSelectionButton, 2, 3
        )
        # self.maskGeneratorContainerLayout.addWidget(self.loadMaskFromFileButton, 2, 1)

        self.clearRoiButton = QPushButton("Clear ROI")
        self.clearRoiButton.clicked.connect(
            lambda: self.selection_view.clear_rois()
        )
        self.maskGeneratorContainerLayout.addWidget(self.clearRoiButton, 0, 1)

        self.maskGeneratorContainerLayout.addWidget(
            QLabel("Mask index:"), 1, 0
        )
        self.mask_index_spinbox = QSpinBox()
        self.mask_index_spinbox.setMinimum(1)
        self.mask_index_spinbox.setValue(1)
        self.maskGeneratorContainerLayout.addWidget(
            self.mask_index_spinbox, 1, 1
        )

        self.previous_mask_button = QPushButton()
        self.previous_mask_button.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )
        self.previous_mask_button.setToolTip(
            "Click arrow to enable WASD keyboard control"
        )
        self.previous_mask_button.setFixedWidth(60)
        with Icons.Path("LeftArrow.png") as path:
            self.previous_mask_button.setIcon(QIcon(path))
        self.previous_mask_button.clicked.connect(
            lambda: self.show_mask_with_index(-1)
        )
        self.maskGeneratorContainerLayout.addWidget(
            self.previous_mask_button, 1, 2
        )

        self.next_mask_button = QPushButton()
        self.next_mask_button.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )
        self.next_mask_button.setToolTip(
            "Click arrow to enable WASD keyboard control"
        )
        self.next_mask_button.setFixedWidth(60)
        with Icons.Path("RightArrow.png") as path:
            self.next_mask_button.setIcon(QIcon(path))
        self.next_mask_button.clicked.connect(
            lambda: self.show_mask_with_index(1)
        )
        self.maskGeneratorContainerLayout.addWidget(
            self.next_mask_button, 1, 3
        )

        self.selectionOptionsContainer = roundQGroupBox()
        self.selectionOptionsContainer.setTitle("Options")
        self.selectionOptionsLayout = QGridLayout()
        self.fillContourButton = QCheckBox()
        self.invertMaskButton = QCheckBox()
        self.thicknessSpinBox = QSpinBox()
        self.thicknessSpinBox.setRange(1, 25)
        self.selectionOptionsLayout.addWidget(QLabel("Fill contour:"), 0, 0)
        self.selectionOptionsLayout.addWidget(self.fillContourButton, 0, 1)
        self.selectionOptionsLayout.addWidget(QLabel("Invert mask:"), 1, 0)
        self.selectionOptionsLayout.addWidget(self.invertMaskButton, 1, 1)
        self.selectionOptionsLayout.addWidget(QLabel("Thickness:"), 2, 0)
        self.selectionOptionsLayout.addWidget(self.thicknessSpinBox, 2, 1)

        lasers = ["640", "532", "488"]
        self.transform_for_laser_menu = QListWidget()
        self.transform_for_laser_menu.addItems(lasers)
        self.transform_for_laser_menu.setFixedHeight(48)
        self.transform_for_laser_menu.setFixedWidth(65)
        self.transform_for_laser_menu.setCurrentRow(0)

        self.selectionOptionsLayout.addWidget(
            QLabel("To be used with laser:"), 0, 2
        )
        self.selectionOptionsLayout.addWidget(
            self.transform_for_laser_menu, 1, 2
        )

        self.selectionOptionsContainer.setLayout(self.selectionOptionsLayout)

        self.maskGeneratorContainerLayout.addWidget(
            self.selectionOptionsContainer, 3, 0, 2, 3
        )

        # ----------------------------Mask-RCNN--------------------------------
        self.MLOptionsContainer = roundQGroupBox()
        self.MLOptionsContainer.setTitle("Mask-RCNN")
        self.MLOptionsContainerLayout = QGridLayout()

        self.init_ML_button = QPushButton("Init. ML", self)
        self.MLOptionsContainerLayout.addWidget(self.init_ML_button, 0, 0)
        self.init_ML_button.clicked.connect(
            lambda: self.run_in_thread(self.init_ML)
        )

        self.run_ML_button = QPushButton("Analysis", self)
        self.MLOptionsContainerLayout.addWidget(self.run_ML_button, 1, 0)
        self.run_ML_button.clicked.connect(self.run_ML_onImg_and_display)

        self.generate_MLmask_button = QPushButton("To ROIs", self)
        self.MLOptionsContainerLayout.addWidget(
            self.generate_MLmask_button, 2, 0
        )
        self.generate_MLmask_button.clicked.connect(
            lambda: self.run_in_thread(self.generate_MLmask)
        )

        self.MLOptionsContainer.setLayout(self.MLOptionsContainerLayout)

        self.maskGeneratorContainerLayout.addWidget(
            self.MLOptionsContainer, 3, 3, 2, 1
        )

        self.layout.addWidget(self.maskGeneratorContainer, 0, 8, 1, 3)

        self.DMDWidget = DMDWidget.DMDWidget()
        self.layout.addWidget(self.DMDWidget, 1, 8, 1, 3)

        # Signal sent out from DMDWidget to ask for mask generated here.
        # And then the generated roi list is sent back to
        # function:receive_mask_coordinates in DMDWidget.
        self.DMDWidget.sig_request_mask_coordinates.connect(
            lambda: self.cast_mask_coordinates("dmd")
        )
        self.sig_cast_mask_coordinates_to_dmd.connect(
            self.DMDWidget.receive_mask_coordinates
        )

        self.DMDWidget.sig_start_registration.connect(
            lambda: self.sig_start_registration.emit()
        )
        self.DMDWidget.sig_finished_registration.connect(
            lambda: self.sig_finished_registration.emit()
        )

        # === Galvo control ===
        self.GalvoWidget = GalvoWidget.GalvoWidget()
        self.GalvoWidget.setFixedWidth(200)
        self.GalvoWidget.setFixedHeight(180)
        self.layout.addWidget(self.GalvoWidget, 2, 8, 2, 1)

        self.GalvoWidget.sig_request_mask_coordinates.connect(
            lambda: self.cast_mask_coordinates("galvo")
        )
        self.sig_cast_mask_coordinates_to_galvo.connect(
            self.GalvoWidget.receive_mask_coordinates
        )
        self.GalvoWidget.sig_start_registration.connect(
            lambda: self.sig_start_registration.emit()
        )
        self.GalvoWidget.sig_finished_registration.connect(
            lambda: self.sig_finished_registration.emit()
        )

        # === Manual registration ===
        self.ManualRegistrationWidget = (
            ManualRegistration.ManualRegistrationWidget()
        )
        self.ManualRegistrationWidget.setFixedWidth(100)
        self.ManualRegistrationWidget.sig_request_camera_image.connect(
            self.cast_camera_image
        )
        self.sig_cast_camera_image.connect(
            self.ManualRegistrationWidget.receive_camera_image
        )

        self.layout.addWidget(self.ManualRegistrationWidget, 2, 9, 1, 1)

        # === Stage collect ===
        self.StageRegistrationWidget = StageRegistrationWidget.StageWidget()
        self.StageRegistrationWidget.setFixedWidth(100)
        self.layout.addWidget(self.StageRegistrationWidget, 3, 9, 1, 1)

    # %%

    def cast_transformation_to_DMD(self, transformation, laser):
        self.DMDWidget.transform[laser] = transformation
        self.DMDWidget.save_transformation()

    def cast_transformation_to_galvos(self, transformation):
        self.GalvoWidget.transform = transformation
        self.GalvoWidget.save_transformation()

    def cast_camera_image(self):
        """Send out the image in the image view to ManualRegistration"""
        image = self.selection_view.image
        if image is not None:
            self.sig_cast_camera_image.emit(image)

    def receive_image_from_camera(self, snap_from_camera):
        """
        Receive the emitted snap image singal from camera.
        Signal-slot configured in mian GUI file.
        """
        self.selection_view.setImage(snap_from_camera)

    def read_image_to_draw_roi(self):
        """
        Manually load image to draw rois on.

        Returns
        None.

        """
        loaded_image_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            "",
            "(*.tiff *.tif)",
        )

        loaded_image = imread(loaded_image_name)

        self.selection_view.setImage(loaded_image)

    def cast_mask_coordinates(self, receiver):
        """
        Upon receiving mask rois request signal from DMD or galvo widget, generate list signal including
        list of rois from the current ROIitems in "Select" Drawwidget and send it back to the calling widget.

        Parameters
        receiver : string.
            Specifies which widget is requesting.

        """
        if receiver == "dmd":
            self.sig_cast_mask_coordinates_to_dmd.emit(
                self.sig_to_calling_widget
            )
        else:
            self.sig_cast_mask_coordinates_to_galvo.emit(
                self.sig_to_calling_widget
            )

    def get_list_of_rois(self):
        """
        Return the list of rois from the current ROIitems in "Select" Drawwidget.
        """
        view = self.selection_view
        list_of_rois = []

        for roi in view.roilist:
            roi_handle_positions = roi.getLocalHandlePositions()
            roi_origin = roi.pos()

            for idx, pos in enumerate(roi_handle_positions):
                roi_handle_positions[idx] = pos[1]

            num_vertices = len(roi_handle_positions)
            vertices = np.zeros([num_vertices, 2])

            for idx, vertex in enumerate(roi_handle_positions):
                vertices[idx, :] = np.array(
                    [vertex.x() + roi_origin.x(), vertex.y() + roi_origin.y()]
                )

            list_of_rois.append(vertices)
        # array([[   0,   0],
        # [2048,   0],
        # [2048, 2048,
        # [   0, 2048]])
        return list_of_rois

    def create_mask(self):
        """
        Create untransformed binary mask, sent out the signal to DMD widget for
        further transformation.

        Returns
        None.

        """
        flag_fill_contour = self.fillContourButton.isChecked()
        flag_invert_mode = self.invertMaskButton.isChecked()
        contour_thickness = self.thicknessSpinBox.value()
        target_laser = self.transform_for_laser_menu.selectedItems()[0].text()

        # Get the list of rois from the current ROIitems in "Select" Drawwidget.
        list_of_rois = self.get_list_of_rois()

        # Signal to mask requesting widget.
        current_mask_sig = [
            list_of_rois,
            flag_fill_contour,
            contour_thickness,
            flag_invert_mode,
            target_laser,
        ]

        # === This is the roi list sent to DMD to generate final stack of masks. ===
        self.sig_to_calling_widget[
            "mask_{}".format(self.mask_index_spinbox.value())
        ] = current_mask_sig

        # Show the untransformed mask
        self.current_mask = ProcessImage.CreateBinaryMaskFromRoiCoordinates(
            list_of_rois,
            fill_contour=flag_fill_contour,
            contour_thickness=contour_thickness,
            invert_mask=flag_invert_mode,
        )
        self.untransformed_mask_dict[
            "mask_{}".format(self.mask_index_spinbox.value())
        ] = self.current_mask

        self.mask_view.setImage(self.current_mask)

    def show_mask_with_index(self, direction):
        """
        If direction == 1, then show next mask, elif direction == -1, show previous one.

        Parameters
        direction : TYPE
            -1: Previous; 1: Next.

        Returns
        None.

        """
        try:
            self.mask_index_spinbox.setValue(
                self.mask_index_spinbox.value() + direction
            )
            self.mask_view.setImage(
                self.untransformed_mask_dict[
                    "mask_{}".format(self.mask_index_spinbox.value())
                ]
            )
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("show_mask failed.")

    def delete_mask(self):
        """
        Remove the last mask from the list.
        """
        del self.sig_to_calling_widget[
            "mask_{}".format(self.mask_index_spinbox.value())
        ]
        del self.untransformed_mask_dict[
            "mask_{}".format(self.mask_index_spinbox.value())
        ]

    def remove_selection(self):
        self.sig_to_calling_widget = {}
        self.selection_view.clear_rois()

    def add_polygon_roi(self):
        view = self.selection_view

        x = (view.getView().viewRect().x()) * 0.3
        y = (view.getView().viewRect().y()) * 0.3
        a = (view.getView().viewRect().width() + x) * 0.3
        b = (view.getView().viewRect().height() + y) * 0.3
        c = (view.getView().viewRect().width() + x) * 0.7
        d = (view.getView().viewRect().height() + y) * 0.7
        polygon_roi = pg.PolyLineROI(
            [[a, b], [c, b], [c, d], [a, d]],
            pen=view.pen,
            closed=True,
            movable=True,
            removable=True,
        )

        view.getView().addItem(polygon_roi)
        view.append_to_roilist(polygon_roi)

    def load_mask_from_file(self):
        """
        Open a file manager to browse through files, load image file
        """
        self.loadFileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select file",
            "./CoordinateManager/Images/",
            "Image files (*.jpg *.tif *.png)",
        )
        try:
            image = plt.imread(self.loadFileName)

            self.current_mask = image
            self.mask_view.setImage(self.current_mask)
            self.untransformed_mask_dict[
                "mask_{}".format(self.mask_index_spinbox.value())
            ] = self.current_mask

        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("fail to load file.")

    # MaskRCNN detection part

    def init_ML(self):
        # Initialize the detector instance and load the model.
        from ..ImageAnalysis.ImageProcessing_MaskRCNN import ProcessImageML

        self.ProcessML = ProcessImageML(
            WeigthPath=r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Martijn\SpikingHek.h5"  # TODO hardcoded path
        )
        self.MessageBack.emit("Mask-RCNN environment configured.")

    def run_ML_onImg_and_display(self):
        # self.ResetLiveImgView()
        # For testing
        # snap_from_camera = plt.imread\
        # (r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Vidya\Imaging\Octoscope\2020-10-08 Archon lib V7\V7_gfp_5v_telescope_2.TIF")  # TODO hardcoded path
        # self.selection_view.setImage(snap_from_camera)

        """Run MaskRCNN on input image"""
        self.Matdisplay_Figure.clear()
        self.Matdisplay_Figure_axis = self.Matdisplay_Figure.add_subplot(111)

        self.MLtargetedImg = self.selection_view.image
        logging.info(self.MLtargetedImg.shape)

        # Depends on show_mask or not, the returned figure will be input raw image with mask or not.
        self.MLresults = self.ProcessML.DetectionOnImage(
            self.MLtargetedImg,
            axis=self.Matdisplay_Figure_axis,
            show_result=True,
        )
        self.Mask = self.MLresults["masks"]
        self.Label = self.MLresults["class_ids"]
        self.Score = self.MLresults["scores"]
        self.Bbox = self.MLresults["rois"]

        self.SelectedCellIndex = 0
        self.NumCells = int(len(self.Label))
        self.selected_ML_Index = []
        self.selected_cells_infor_dict = {}

        self.Matdisplay_Figure_axis.imshow(self.MLtargetedImg.astype(np.uint8))

        self.Matdisplay_Figure.tight_layout()
        self.Matdisplay_Canvas.draw()

    # %%
    # Configure click event to add clicked cell mask

    def _onclick(self, event):
        """Highlights the cell selected in the figure by the user when clicked on"""
        if self.NumCells > 0:
            ShapeMask = np.shape(self.Mask)
            # get coorinates at selected location in image coordinates
            if event.xdata is None or event.ydata is None:
                return
            xcoor = min(max(int(event.xdata), 0), ShapeMask[1])
            ycoor = min(max(int(event.ydata), 0), ShapeMask[0])

            # search for the mask coresponding to the selected cell
            for EachCell in range(self.NumCells):
                if self.Mask[ycoor, xcoor, EachCell]:
                    self.SelectedCellIndex = EachCell
                    break

            # highlight selected cell
            if self.SelectedCellIndex not in self.selected_ML_Index:
                # Get the selected cell's contour coordinates and mask patch
                self.contour_verts, self.Cell_patch = self.get_cell_polygon(
                    self.Mask[:, :, self.SelectedCellIndex]
                )

                self.Matdisplay_Figure_axis.add_patch(self.Cell_patch)
                self.Matdisplay_Canvas.draw()

                self.selected_ML_Index.append(self.SelectedCellIndex)
                self.selected_cells_infor_dict[
                    "cell{}_verts".format(str(self.SelectedCellIndex))
                ] = self.contour_verts
            else:
                # If click on the same cell
                self.Cell_patch.remove()
                self.Matdisplay_Canvas.draw()
                self.selected_ML_Index.remove(self.SelectedCellIndex)
                self.selected_cells_infor_dict.pop(
                    "cell{}_verts".format(str(self.SelectedCellIndex))
                )

    def get_cell_polygon(self, mask):
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
        )
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            contour_polygon = mpatches.Polygon(
                verts, facecolor=self.random_colors(1)[0]
            )

        return contours, contour_polygon

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    # %%
    # For mask generation

    def generate_MLmask(self):
        """Generate binary mask with all selected cells"""
        self.MLmask = np.zeros(
            (self.MLtargetedImg.shape[0], self.MLtargetedImg.shape[1])
        )

        if len(self.selected_ML_Index) > 0:
            for selected_index in self.selected_ML_Index:
                self.MLmask = np.add(
                    self.MLmask, self.Mask[:, :, selected_index]
                )

            self.add_rois_of_selected()

    def add_rois_of_selected(self):
        """
        Generate ROI items from ML selected mask.
        Using find_contours to get list of contour coordinates in the binary mask, and then generate polygon rois based on these coordinates.
        """
        handle_downsample_step = 2

        for selected_index in self.selected_ML_Index:
            contours = self.selected_cells_infor_dict[
                "cell{}_verts".format(str(selected_index))
            ]
            # contours = find_contours(self.Mask[:,:,selected_index], 0.5) # Find iso-valued contours in a 2D array for a given level value.

            for n, contour in enumerate(contours):
                contour_coord_array = contours[n]
                # Swap columns
                contour_coord_array[:, 0], contour_coord_array[:, 1] = (
                    contour_coord_array[:, 1],
                    contour_coord_array[:, 0].copy(),
                )

                # Down sample the coordinates otherwise it will be too dense.
                contour_coord_array_del = np.delete(
                    contour_coord_array,
                    np.arange(
                        2,
                        contour_coord_array.shape[0] - 3,
                        handle_downsample_step,
                    ),
                    0,
                )
                for _ in range(3):
                    contour_coord_array_del = np.delete(
                        contour_coord_array_del,
                        np.arange(
                            2,
                            contour_coord_array_del.shape[0] - 3,
                            handle_downsample_step,
                        ),
                        0,
                    )

                self.selected_cells_infor_dict[
                    "cell{}_ROIitem".format(str(selected_index))
                ] = pg.PolyLineROI(
                    positions=contour_coord_array_del, closed=True
                )

                # Add ROI item, and append_to_roilist
                self.selection_view.getView().addItem(
                    self.selected_cells_infor_dict[
                        "cell{}_ROIitem".format(str(selected_index))
                    ]
                )
                self.selection_view.append_to_roilist(
                    self.selected_cells_infor_dict[
                        "cell{}_ROIitem".format(str(selected_index))
                    ]
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

    def ResetLiveImgView(self):
        """Closes the widget nicely, making sure to clear the graphics scene and release memory."""
        self.selection_view.close()

        # Replot the imageview
        self.selection_view = DrawingWidget(self)
        self.selection_view.enable_drawing(True)
        self.selection_view.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.selection_view.ui.roiBtn.hide()
        self.selection_view.ui.menuBtn.hide()
        self.selection_view.ui.normGroup.hide()
        self.selection_view.ui.roiPlot.hide()

        self.image_mask_stack.addTab(self.selection_view, "Select")


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = CoordinatesWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
