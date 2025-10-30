# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:18:12 2020

@author: xinmeng
"""

import logging
import threading

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSlider,
    QStackedLayout,
    QWidget,
)

from .. import Icons, StylishQT
from .DAQoperator import DAQmission


class AOTFLaserUI(QWidget):
    def __init__(
        self,
        wavelength,
        colors,
        signal,
        lasers_status,
        *args,
        **kwargs,
    ):
        """ui widget for a single laser control"""
        super().__init__(*args, **kwargs)
        self.wavelength = f"{wavelength}"
        self.channel = f"{wavelength}AO"
        self.blanking_channel = f"{wavelength}blanking"
        self.servo_channel = f"{wavelength} servo"
        self.signal = signal
        self.lasers_status = lasers_status

        with Icons.Path("shutter.png") as path:
            self.shutterButton = StylishQT.checkableButton(
                Icon_path=path, background_color=colors[-1]
            )

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(500)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(100)
        self.slider.setSingleStep(1)
        self.slider.setTracking(False)
        palette = QPalette(QColor(colors[1]))
        palette.setColor(QPalette.Highlight, QColor(colors[0]))
        self.slider.setPalette(palette)

        self.box = QDoubleSpinBox(self)
        self.box.setDecimals(2)
        self.box.setRange(0, 5)
        self.box.setSingleStep(0.2)
        self.box.setButtonSymbols(QDoubleSpinBox.NoButtons)

        self.box.editingFinished.connect(
            lambda: self.slider.setValue(int(self.box.value() * 100))
        )
        self.slider.valueChanged.connect(
            lambda value: self.box.setValue(value / 100)
        )
        self.shutterButton.clicked.connect(self.shutter_CW_action)
        self.connect_signals()

        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.addWidget(self.slider, 0, 0)
        self.layout.addWidget(self.box, 0, 1)
        self.layout.addWidget(self.shutterButton, 0, 2)

    def connect_signals(self):
        self.slider.valueChanged.connect(self.setChannelValue)

    def disconnect_signals(self):
        self.slider.valueChanged.disconnect(self.setChannelValue)

    def setChannelValue(self, value):
        daq = DAQmission()
        self.lasers_status[self.wavelength][1] = value

        daq.sendSingleAnalog(self.channel, value / 100)
        self.signal.emit(self.lasers_status)

    def reset_sliders(self):
        self.disconnect_signals()
        self.slider.setValue(0)
        self.connect_signals()

    def setChannelSwitch(self, value):
        daq = DAQmission()
        self.lasers_status[self.wavelength][0] = value

        daq.sendSingleDigital(self.blanking_channel, value)
        self.sig_lasers_status_changed.emit(self.lasers_status)

    def shutter_CW_action(self):
        daq = DAQmission()
        daq.sendServoSignal(self.servo_channel, self.shutterButton.isChecked())


class AOTFWidgetUI(QWidget):
    sig_lasers_status_changed = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        # self.setMinimumSize(1350,900)
        self.setWindowTitle("StageWidget")
        self.layout = QGridLayout(self)

        # === GUI for AOTF ===

        AOTFcontrolContainer = StylishQT.roundQGroupBox(
            title="AOTF control", background_color="azure"
        )

        self.AOTFstackedLayout = QStackedLayout()

        # self.AOTFdisabledWidget = QWidget()
        self.AOTFdisabledWidget = QLabel(
            "AOTF not available due to running registration procedure"
        )
        self.AOTFdisabledWidget.setWordWrap(True)

        self.AOTFcontrolWidget = QWidget()
        self.AOTFcontrolLayout = QGridLayout()
        self.AOTFcontrolLayout.setSpacing(2)
        self.AOTFcontrolLayout.setContentsMargins(6, 12, 6, 6)
        self.AOTFcontrolWidget.setLayout(self.AOTFcontrolLayout)

        self.switchbutton_blankingAll = StylishQT.MySwitch(
            "Blanking ON",
            "spring green",
            "Blanking OFF",
            "indian red",
            width=60,
        )
        self.AOTFcontrolLayout.addWidget(
            self.switchbutton_blankingAll, 0, 0, 1, 2
        )

        self.lasers_status = {
            wavelength: [False, 0] for wavelength in ("488", "532", "640")
        }

        self.laser640 = AOTFLaserUI(
            640,
            ("red", "indian red", "#DEC8C4"),
            self.sig_lasers_status_changed,
            self.lasers_status,
        )
        self.AOTFcontrolLayout.addWidget(self.laser640, 1, 0)

        self.laser532 = AOTFLaserUI(
            532,
            ("green", "lime green", "#CDDEC4"),
            self.sig_lasers_status_changed,
            self.lasers_status,
        )
        self.AOTFcontrolLayout.addWidget(self.laser532, 2, 0)

        self.laser488 = AOTFLaserUI(
            488,
            ("blue", "corn flower blue", "#C4DDDE"),
            self.sig_lasers_status_changed,
            self.lasers_status,
        )
        self.AOTFcontrolLayout.addWidget(self.laser488, 3, 0)

        self.laserwidgets = self.laser640, self.laser532, self.laser488

        self.AOTFstackedLayout.addWidget(self.AOTFcontrolWidget)
        self.AOTFstackedLayout.addWidget(self.AOTFdisabledWidget)
        self.AOTFstackedLayout.setCurrentIndex(0)

        AOTFcontrolContainer.setLayout(self.AOTFstackedLayout)
        AOTFcontrolContainer.setMaximumHeight(170)
        self.layout.addWidget(AOTFcontrolContainer, 1, 0)

        # blanking switch for red laser is blanks all lasers
        # other blanking switches are not connected
        self.switchbutton_blankingAll.clicked.connect(
            self.laser640.setChannelSwitch
        )

        thread = threading.Thread(target=self.start_up_event)
        thread.start()

    def reset_sliders(self):
        for widget in self.laserwidgets:
            widget.reset_sliders()

    def start_up_event(self):
        try:
            logging.info("Servo position initialization turned off.")
            # servo= Servo()
            # close the blue shutter
            # servo.rotate(target_servo = 'servo_modulation_1', degree = 0)
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("Fail to initialize servo position.")

    def set_registration_mode(self, flag_registration_mode):
        if flag_registration_mode:
            self.AOTFstackedLayout.setCurrentIndex(1)
        else:
            self.AOTFstackedLayout.setCurrentIndex(0)

    def control_for_registration(self, wavelength, value):
        value = int(value)
        daq = DAQmission()

        if value == 0:
            switch = False
        else:
            switch = True

        if wavelength == "640":
            logging.info(f"{wavelength}:{value}")
            logging.info(f"{switch}")
            daq.sendSingleAnalog("640AO", value)

            daq.sendSingleDigital("640blanking", switch)

        elif wavelength == "532":
            logging.info(f"{wavelength}:{value}")
            logging.info(f"{switch}")
            daq.sendSingleAnalog("532AO", value)

            daq.sendSingleDigital("640blanking", switch)

        else:
            logging.info(f"{wavelength}:{value}")
            logging.info(f"{switch}")
            daq.sendSingleAnalog("488AO", value)

            daq.sendSingleDigital("640blanking", switch)


if __name__ == "__main__":
    import sys

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = AOTFWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
