# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:04:41 2020

@author: xinmeng
"""

import sys
import threading

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .. import Icons
from .stage import LudlStage


class StageWidgetUI(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))
        self.ludlStage = LudlStage("COM12")
        # self.setMinimumSize(1350,900)
        self.setWindowTitle("StageWidget")
        self.layout = QGridLayout(self)

        # === GUI for Stage ===
        stagecontrolContainer = QGroupBox("Stage control")
        stagecontrolContainer.setStyleSheet("QGroupBox {\
                                font: bold;\
                                border: 1px solid silver;\
                                border-radius: 6px;\
                                margin-top: 12px;\
                                color:Navy; \
                                background-color: #F8F8FF;}\
                                QGroupBox::title{subcontrol-origin: margin;\
                                                 left: 7px;\
                                                 padding: 5px 5px 5px 5px;}")
        self.stagecontrolLayout = QGridLayout()

        self.stage_upwards = QPushButton()
        self.stage_upwards.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )
        self.stage_upwards.setToolTip(
            "Click arrow to enable WASD keyboard control"
        )
        self.stage_upwards.setFixedWidth(40)
        self.stage_upwards.setFixedHeight(40)
        with Icons.Path("UpArrow.png") as path:
            self.stage_upwards.setIcon(QIcon(path))
        self.stage_upwards.setIconSize(QSize(35, 35))
        self.stagecontrolLayout.addWidget(self.stage_upwards, 1, 4)
        self.stage_upwards.clicked.connect(
            lambda: self.run_in_thread(
                self.sample_stage_move(direction="upwards")
            )
        )
        self.stage_upwards.setShortcut("w")

        self.stage_left = QPushButton()
        self.stage_left.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )
        self.stage_left.setToolTip(
            "Click arrow to enable WASD keyboard control"
        )
        self.stage_left.setFixedWidth(40)
        self.stage_left.setFixedHeight(40)
        with Icons.Path("LeftArrow.png") as path:
            self.stage_left.setIcon(QIcon(path))
        # self.stage_left.setStyleSheet("QPushButton {padding: 10px;}");
        self.stage_left.setIconSize(QSize(35, 35))
        self.stagecontrolLayout.addWidget(self.stage_left, 2, 3)
        self.stage_left.clicked.connect(
            lambda: self.run_in_thread(
                self.sample_stage_move(direction="leftwards")
            )
        )
        self.stage_left.setShortcut("a")

        self.stage_right = QPushButton()
        self.stage_right.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )
        self.stage_right.setToolTip(
            "Click arrow to enable WASD keyboard control"
        )
        self.stage_right.setFixedWidth(40)
        self.stage_right.setFixedHeight(40)
        with Icons.Path("RightArrow.png") as path:
            self.stage_right.setIcon(QIcon(path))
        self.stage_right.setIconSize(QSize(35, 35))
        self.stagecontrolLayout.addWidget(self.stage_right, 2, 5)
        self.stage_right.clicked.connect(
            lambda: self.run_in_thread(
                self.sample_stage_move(direction="rightwards")
            )
        )
        self.stage_right.setShortcut("d")

        self.stage_down = QPushButton()
        self.stage_down.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )
        self.stage_down.setToolTip(
            "Click arrow to enable WASD keyboard control"
        )
        self.stage_down.setFixedWidth(40)
        self.stage_down.setFixedHeight(40)
        with Icons.Path("DownArrow.png") as path:
            self.stage_down.setIcon(QIcon(path))
        self.stage_down.setIconSize(QSize(35, 35))
        self.stagecontrolLayout.addWidget(self.stage_down, 2, 4)
        self.stage_down.clicked.connect(
            lambda: self.run_in_thread(
                self.sample_stage_move(direction="downwards")
            )
        )
        self.stage_down.setShortcut("s")

        self.stage_speed = QSpinBox(self)
        self.stage_speed.setFixedWidth(47)
        self.stage_speed.setMinimum(0)
        self.stage_speed.setMaximum(100000)
        self.stage_speed.setValue(300)
        self.stage_speed.setSingleStep(1650)
        self.stagecontrolLayout.addWidget(self.stage_speed, 2, 1)
        self.stagecontrolLayout.addWidget(QLabel("Step:"), 2, 0)

        # self.stage_current_pos_Label = QLabel("Current position: ")
        # self.stagecontrolLayout.addWidget(self.stage_current_pos_Label, 1, 0)

        self.stage_goto = QPushButton()
        with Icons.Path("move_coord.png") as path:
            self.stage_goto.setIcon(QIcon(path))
        self.stage_goto.setToolTip("Move to absolute position")
        self.stage_goto.setStyleSheet(
            "QPushButton {color:white;background-color: #CCFFFF;}"
            "QPushButton:hover:!pressed {color:white;background-color: #FFE5CC;}"
        )
        self.stage_goto.setFixedWidth(35)
        self.stage_goto.setFixedHeight(35)
        self.stagecontrolLayout.setAlignment(Qt.AlignVCenter)
        # self.stage_goto.setStyleSheet("QPushButton {color:white;background-color: #6495ED; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
        # "QPushButton:pressed {color:red;background-color: white; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
        # "QPushButton:hover:!pressed {color:green;background-color: #6495ED; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}")
        self.stagecontrolLayout.addWidget(self.stage_goto, 1, 0)
        self.stage_goto.clicked.connect(
            lambda: self.run_in_thread(
                self.sample_stage_move(direction="absolute")
            )
        )

        self.stage_goto_x = QLineEdit(self)
        self.stage_goto_x.setFixedWidth(47)
        self.stagecontrolLayout.addWidget(self.stage_goto_x, 1, 1)

        self.stage_goto_y = QLineEdit(self)
        self.stage_goto_y.setFixedWidth(47)
        self.stagecontrolLayout.addWidget(self.stage_goto_y, 1, 2)

        # self.stagecontrolLayout.addWidget(QLabel('Click arrow to enable WASD keyboard control'), 4, 0, 1, 3)

        stagecontrolContainer.setLayout(self.stagecontrolLayout)
        # stagecontrolContainer.setMinimumHeight(206)
        self.layout.addWidget(stagecontrolContainer, 2, 0)

        # === Fucs for stage movement ===

    def sample_stage_move(self, direction):
        self.sample_move_distance_Rel = int(self.stage_speed.value())
        if direction == "upwards":
            self.ludlStage.moveRel(xRel=0, yRel=self.sample_move_distance_Rel)
        elif direction == "downwards":
            self.ludlStage.moveRel(
                xRel=0, yRel=-1 * self.sample_move_distance_Rel
            )
        elif direction == "leftwards":
            self.ludlStage.moveRel(xRel=self.sample_move_distance_Rel, yRel=0)
        elif direction == "rightwards":
            self.ludlStage.moveRel(
                xRel=-1 * self.sample_move_distance_Rel, yRel=0
            )
        elif direction == "absolute":
            self.ludlStage.moveAbs(
                x=int(self.stage_goto_x.text()),
                y=int(self.stage_goto_y.text()),
            )
        self.xPosition, self.yPosition = self.ludlStage.getPos()

        self.stage_goto_x.setText(str(self.xPosition))
        self.stage_goto_y.setText(str(self.yPosition))

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


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = StageWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
