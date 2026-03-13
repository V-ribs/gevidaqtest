# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:54:35 2020

@author: xinmeng
"""

import logging
import sys
import threading

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont

from .. import StylishQT
from .filterpyserial import ELL9Filter


class FilterSliderWidgetUI(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        self.resize(265, 130)
        self.setWindowTitle("FilterSliderWidget")
        self.layout = QtWidgets.QGridLayout(self)

        # === GUI for Filter movement ===
        ControlContainer = StylishQT.roundQGroupBox(
            title="Filter Control", background_color="azure"
        )
        ControlContainerLayout = QtWidgets.QGridLayout()

        ND_filtercontrolContainer = StylishQT.roundQGroupBox(
            title="2P ND", background_color="azure"
        )
        self.filtercontrolLayout = QtWidgets.QGridLayout()
        self.filtercontrolLayout.setSpacing(2)

        self.FilterButtongroup_1 = QtWidgets.QButtonGroup(self)

        self.filter1_pos0 = QtWidgets.QPushButton("0")
        self.filter1_pos0.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos0)
        self.filtercontrolLayout.addWidget(self.filter1_pos0, 0, 1)

        self.filter1_pos1 = QtWidgets.QPushButton("1")
        self.filter1_pos1.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos1)
        self.filtercontrolLayout.addWidget(self.filter1_pos1, 0, 2)

        self.filter1_pos2 = QtWidgets.QPushButton("2")
        self.filter1_pos2.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos2)
        self.filtercontrolLayout.addWidget(self.filter1_pos2, 0, 3)

        self.filter1_pos3 = QtWidgets.QPushButton("3")
        self.filter1_pos3.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos3)
        self.filtercontrolLayout.addWidget(self.filter1_pos3, 0, 4)
        self.FilterButtongroup_1.setExclusive(True)
        self.FilterButtongroup_1.buttonClicked[int].connect(
            self.DecodeFilterMove
        )

        self.FilterButtongroup_2 = QtWidgets.QButtonGroup(self)

        self.filter2_pos0 = QtWidgets.QPushButton("0")
        self.filter2_pos0.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos0)
        self.filtercontrolLayout.addWidget(self.filter2_pos0, 1, 1)

        self.filter2_pos1 = QtWidgets.QPushButton("0.1")
        self.filter2_pos1.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos1)
        self.filtercontrolLayout.addWidget(self.filter2_pos1, 1, 2)

        self.filter2_pos2 = QtWidgets.QPushButton("0.3")
        self.filter2_pos2.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos2)
        self.filtercontrolLayout.addWidget(self.filter2_pos2, 1, 3)

        self.filter2_pos3 = QtWidgets.QPushButton("0.5")
        self.filter2_pos3.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos3)
        self.filtercontrolLayout.addWidget(self.filter2_pos3, 1, 4)
        self.FilterButtongroup_2.setExclusive(True)
        self.FilterButtongroup_2.buttonClicked[int].connect(
            self.DecodeFilterMove
        )

        EM_filtercontrolContainer = StylishQT.roundQGroupBox(
            title="Emission", background_color="honeydew"
        )
        self.EM_filtercontrolContainerLayout = QtWidgets.QGridLayout()
        self.EM_filtercontrolContainerLayout.setSpacing(2)

        self.FilterButtongroup_3 = QtWidgets.QButtonGroup(self)

        self.filter3_pos0 = QtWidgets.QPushButton("Arch")
        self.filter3_pos0.setCheckable(True)
        self.FilterButtongroup_3.addButton(self.filter3_pos0)
        self.EM_filtercontrolContainerLayout.addWidget(self.filter3_pos0, 1, 0)

        self.filter3_pos1 = QtWidgets.QPushButton("Citrine")
        self.filter3_pos1.setCheckable(True)
        self.FilterButtongroup_3.addButton(self.filter3_pos1)
        self.EM_filtercontrolContainerLayout.addWidget(self.filter3_pos1, 0, 0)
        self.FilterButtongroup_3.setExclusive(True)
        self.FilterButtongroup_3.buttonClicked[int].connect(
            self.DecodeFilterMove
        )

        EM_filtercontrolContainer.setLayout(
            self.EM_filtercontrolContainerLayout
        )
        EM_filtercontrolContainer.setFixedWidth(65)

        ND_filtercontrolContainer.setLayout(self.filtercontrolLayout)
        ND_filtercontrolContainer.setFixedWidth(200)

        ControlContainerLayout.addWidget(ND_filtercontrolContainer, 0, 0)
        ControlContainerLayout.addWidget(EM_filtercontrolContainer, 0, 1)
        ControlContainer.setLayout(ControlContainerLayout)

        self.layout.addWidget(ControlContainer, 0, 0)

        # === Fucs for filter movement ===

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

    def DecodeFilterMove(self):
        if self.FilterButtongroup_1.checkedId() == -2:
            self.run_in_thread(
                self.filter_move_towards("COM9", 0)
            )  # TODO hardcoded port
        elif self.FilterButtongroup_1.checkedId() == -3:
            self.run_in_thread(self.filter_move_towards("COM9", 1))
        elif self.FilterButtongroup_1.checkedId() == -4:
            self.run_in_thread(self.filter_move_towards("COM9", 2))
        elif self.FilterButtongroup_1.checkedId() == -5:
            self.run_in_thread(self.filter_move_towards("COM9", 3))

        if self.FilterButtongroup_2.checkedId() == -2:
            self.run_in_thread(
                self.filter_move_towards("COM7", 0)
            )  # TODO hardcoded port
        elif self.FilterButtongroup_2.checkedId() == -3:
            self.run_in_thread(self.filter_move_towards("COM7", 1))
        elif self.FilterButtongroup_2.checkedId() == -4:
            self.run_in_thread(self.filter_move_towards("COM7", 2))
        elif self.FilterButtongroup_2.checkedId() == -5:
            self.run_in_thread(self.filter_move_towards("COM7", 3))

        if self.FilterButtongroup_3.checkedId() == -2:
            # Move to Arch
            self.run_in_thread(
                self.filter_move_towards("COM15", 0)
            )  # TODO hardcoded port
        elif self.FilterButtongroup_3.checkedId() == -3:
            self.run_in_thread(self.filter_move_towards("COM15", 1))

    def filter_move_towards(self, COMport, pos):
        ELL9Filter_ins = ELL9Filter(COMport)
        ELL9Filter_ins.moveToPosition(pos)

    def update_slider_current_pos(self, current_pos):
        # .setValue(current_pos)
        logging.info("Slider current position: {}".format(current_pos))


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = FilterSliderWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
