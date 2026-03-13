# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:03:52 2020

@author: xinmeng
"""

import logging
import sys
import threading

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt, QThread
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollBar,
    QWidget,
)

from .. import Icons, StylishQT
from .focuser import PIMotor


class ObjMotorWidgetUI(QWidget):
    # waveforms_generated = pyqtSignal(object, object, list, int)
    # SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    # MessageBack = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        self.setMinimumHeight(150)
        self.setWindowTitle("ObjMotorWidget")
        self.layout = QGridLayout(self)
        self.connect_status = False
        # === GUI for Objective Motor ===

        # Movement based on relative positions.
        self.ObjMotorcontrolContainer = QGroupBox("Objective focus")
        self.ObjMotorcontrolContainer.setStyleSheet("QGroupBox {\
                                font: bold;\
                                border: 1px solid silver;\
                                border-radius: 6px;\
                                margin-top: 12px;\
                                color:Navy; \
                                background-color: #FFFAFA}\
                                QGroupBox::title{subcontrol-origin: margin;\
                                                 left: 7px;\
                                                 padding: 5px 5px 5px 5px;}")
        self.ObjMotorcontrolLayout = QGridLayout()

        self.ObjMotor_connect = StylishQT.connectButton()
        self.ObjMotor_connect.setFixedWidth(70)
        self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_connect, 0, 0)
        self.ObjMotor_connect.clicked.connect(lambda: self.ConnectMotor())

        # self.ObjMotor_disconnect = StylishQT.disconnectButton()
        # self.ObjMotor_disconnect.setFixedWidth(70)
        # self.ObjMotor_disconnect.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        # self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_disconnect, 0, 1)

        # self.ObjMotor_disconnect.clicked.connect(lambda: self.DisconnectMotor())
        # self.ObjMotor_disconnect.setEnabled(False)

        self.ObjMotor_upwards = QPushButton()
        self.ObjMotor_upwards.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )

        with Icons.Path("UpArrow.png") as path:
            self.ObjMotor_upwards.setIcon(QIcon(path))
        self.ObjMotor_upwards.setIconSize(QSize(20, 20))
        self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_upwards, 2, 3)
        self.ObjMotor_upwards.clicked.connect(
            lambda: self.MovingMotorThread("Motor_move_upwards")
        )
        # self.ObjMotor_upwards.setShortcut('w')

        self.ObjMotor_down = QPushButton()
        self.ObjMotor_down.setStyleSheet(
            "QPushButton {color:white;background-color: #FFCCE5;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}"
        )

        with Icons.Path("DownArrow.png") as path:
            self.ObjMotor_down.setIcon(QIcon(path))
        self.ObjMotor_down.setIconSize(QSize(20, 20))
        self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_down, 3, 3)
        self.ObjMotor_down.clicked.connect(
            lambda: self.MovingMotorThread("Motor_move_downwards")
        )
        # self.stage_down.setShortcut('s')

        self.ObjMotor_target = QDoubleSpinBox(self)
        self.ObjMotor_target.setMinimum(-10000)
        self.ObjMotor_target.setMaximum(10000)
        self.ObjMotor_target.setDecimals(6)
        # self.ObjMotor_target.setValue(3.45)
        self.ObjMotor_target.setSingleStep(0.001)
        self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_target, 2, 1)
        self.ObjMotorcontrolLayout.addWidget(QLabel("Target:"), 2, 0)

        self.ObjMotor_current_pos_Label = QLabel("Current position: ")
        self.ObjMotorcontrolLayout.addWidget(
            self.ObjMotor_current_pos_Label, 0, 1, 1, 1
        )

        self.ObjMotor_goto = QPushButton()
        with Icons.Path("move_coord.png") as path:
            self.ObjMotor_goto.setIcon(QIcon(path))
        self.ObjMotor_goto.setToolTip("Move to absolute position")
        self.ObjMotor_goto.setStyleSheet(
            "QPushButton {color:white;background-color: #CCFFFF;}"
            "QPushButton:hover:!pressed {color:white;background-color: #FFE5CC;}"
        )
        self.ObjMotor_goto.setFixedWidth(35)
        self.ObjMotor_goto.setFixedHeight(35)
        self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_goto, 2, 2)
        self.ObjMotor_goto.clicked.connect(
            lambda: self.MovingMotorThread("Motor_move_target")
        )

        self.ObjMotor_step = QDoubleSpinBox(self)
        self.ObjMotor_step.setMinimum(-10000)
        self.ObjMotor_step.setMaximum(10000)
        self.ObjMotor_step.setDecimals(6)
        self.ObjMotor_step.setValue(0.003)
        self.ObjMotor_step.setSingleStep(0.001)
        self.ObjMotorcontrolLayout.addWidget(self.ObjMotor_step, 3, 1)
        self.ObjMotorcontrolLayout.addWidget(QLabel("Step: "), 3, 0)

        self.FocusSlider = QScrollBar(Qt.Horizontal)
        self.FocusSlider.setMinimum(2500000)
        self.FocusSlider.setMaximum(4800000)
        # self.FocusSlider.setTickPosition(QSlider.TicksBothSides)
        # self.FocusSlider.setTickInterval(1000000)
        self.FocusSlider.setStyleSheet("color:white; background: lightblue")
        self.FocusSlider.setSingleStep(10000)
        # self.line640 = QLineEdit(self)
        # self.line640.setFixedWidth(60)
        # self.FocusSlider.sliderReleased.connect(lambda:self.updatelinevalue(640))
        self.FocusSlider.valueChanged.connect(
            lambda: self.MovingMotorThread("Motor_move_slider")
        )
        self.FocusSlider.setTracking(False)
        # self.line640.returnPressed.connect(lambda:self.updatesider(640))
        self.ObjMotorcontrolLayout.addWidget(self.FocusSlider, 4, 0, 1, 4)

        self.ObjMotorcontrolContainer.setLayout(self.ObjMotorcontrolLayout)
        self.ObjMotorcontrolContainer.setMaximumHeight(300)
        self.layout.addWidget(self.ObjMotorcontrolContainer, 4, 0)

        # === Fucs for Motor movement ===

    def ConnectMotor(self):
        if self.ObjMotor_connect.isChecked():
            self.ObjMotor_connect.setEnabled(False)
            self.device_instance = ConnectObj_Thread()
            self.device_instance.start()
            self.device_instance.finished.connect(self.getmotorhandle)

        else:
            self.ObjMotor_connect.setChecked(False)
            self.DisconnectMotor()

    def getmotorhandle(self):
        try:
            self.ObjMotor_connect.setEnabled(True)

            self.pi_device_instance = self.device_instance.getInstance()
            logging.info("Objective motor connected.")
            self.connect_status = True

            self.ObjCurrentPos = self.pi_device_instance.pidevice.qPOS(
                self.pi_device_instance.pidevice.axes
            )
            self.ObjMotor_current_pos_Label.setText(
                "Current position: {:.4f}".format(self.ObjCurrentPos["1"])
            )  # Axis here is a string.
            self.ObjMotor_target.setValue(self.ObjCurrentPos["1"])

            decimal_places = len(str(self.ObjCurrentPos["1"]).split(".")[1])
            logging.info(int(self.ObjCurrentPos["1"] * (10**decimal_places)))
            self.FocusSlider.setValue(int(self.ObjCurrentPos["1"] * (10**6)))
            self.ObjMotorcontrolContainer.setEnabled(True)

            self.ObjMotor_connect.setChecked(True)

        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            self.ObjMotor_connect.setChecked(False)
            QMessageBox.warning(
                self, "Oops", "Failed to connect, try again.", QMessageBox.Ok
            )

    def MovingMotorThread(self, target):
        if target == "Motor_move_target":
            MoveMotorThread = threading.Thread(
                target=self.MoveMotor, args=("Target",)
            )
            MoveMotorThread.start()
        elif target == "Motor_move_upwards":
            MoveMotorThread = threading.Thread(
                target=self.MoveMotor, args=("UP",)
            )
            MoveMotorThread.start()
        elif target == "Motor_move_downwards":
            MoveMotorThread = threading.Thread(
                target=self.MoveMotor, args=("DOWN",)
            )
            MoveMotorThread.start()
        elif target == "Motor_move_slider":
            MoveMotorThread = threading.Thread(
                target=self.MoveMotor, args=("Slider",)
            )
            MoveMotorThread.start()

    def MoveMotor(self, direction):
        if direction == "Target":
            pos = self.pi_device_instance.move(
                self.ObjMotor_target.value()
            )  # TODO unused
        elif direction == "UP":
            self.MotorStep = self.ObjMotor_step.value()
            pos = self.pi_device_instance.move(
                self.ObjCurrentPos["1"] + self.MotorStep
            )  # TODO unused
        elif direction == "DOWN":
            self.MotorStep = self.ObjMotor_step.value()
            pos = self.pi_device_instance.move(
                self.ObjCurrentPos["1"] - self.MotorStep
            )  # TODO unused
        elif direction == "Slider":
            pos = self.pi_device_instance.move(
                self.FocusSlider.value() / 1000000
            )  # TODO unused

        self.ObjCurrentPos = self.pi_device_instance.pidevice.qPOS(
            self.pi_device_instance.pidevice.axes
        )
        self.ObjMotor_current_pos_Label.setText(
            "Current position: {:.4f}".format(self.ObjCurrentPos["1"])
        )  # Axis here is a string.
        self.ObjMotor_target.setValue(self.ObjCurrentPos["1"])

        # decimal_places = len(str(self.ObjCurrentPos['1']).split('.')[1])
        self.FocusSlider.setValue(int(self.ObjCurrentPos["1"] * (10**6)))

    def DisconnectMotor(self):
        self.pi_device_instance.CloseMotorConnection()
        logging.info("Disconnected")
        self.connect_status = False

    # self.normalOutputWritten('Objective motor disconnected.'+'\n')

    def closeEvent(self, event):
        # Because the software combines both PyQt and PyQtGraph, using the
        # closeEvent() from PyQt will cause a segmentation fault. Calling
        # also the exit() from PyQtGraph solves this problem.
        # pg.exit()
        if self.connect_status is True:
            self.DisconnectMotor()

        QtWidgets.QApplication.quit()
        event.accept()


class ConnectObj_Thread(QThread):
    # videostack_signal = pyqtSignal(np.ndarray)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # self.fileName = fileName
    # self.xRel = xRel
    # self.yRel = yRel

    def run(self):
        self.pi_device_instance = PIMotor()

    def getInstance(self):
        return self.pi_device_instance


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = ObjMotorWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
