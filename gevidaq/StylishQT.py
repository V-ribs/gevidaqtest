# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:33:21 2020

@author: xinmeng

                For stylish looking
"""

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (
    QAbstractAnimation,
    QPoint,
    QRect,
    QSize,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap

from . import Icons


class roundQGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title=None, background_color=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Round corner group box. Background color can be set e.g. background_color = 'blue'
        """
        if background_color is not None:
            self.background_color = background_color
        else:
            self.background_color = "None"

        if title is not None:
            self.setTitle(title)

        StyleSheet = "QGroupBox {\
                        font: bold;\
                        border: 1px solid silver;\
                        border-radius: 6px;\
                        margin-top: 12px;\
                        color:Navy; \
                        background-color: " + self.background_color + "}QGroupBox::title{subcontrol-origin: margin;\
                                         left: 7px;\
                                         padding: 5px 5px 5px 5px;}"
        self.setStyleSheet(StyleSheet)


class FancyPushButton(QtWidgets.QPushButton):
    clicked = pyqtSignal()
    """
    Button with animation effect. color1 is the color on the right, color2 on the left.
    """

    def __init__(self, width, height, parent=None, *args, **kwargs):
        super().__init__(parent)

        self.setMinimumSize(width, height)

        if len(kwargs) == 0:
            self.color1 = QColor(240, 53, 218)
            self.color2 = QColor(61, 217, 245)
        else:
            self.color1 = QColor(
                kwargs.get("color1", None)[0],
                kwargs.get("color1", None)[1],
                kwargs.get("color1", None)[2],
            )
            self.color2 = QColor(
                kwargs.get("color2", None)[0],
                kwargs.get("color2", None)[1],
                kwargs.get("color2", None)[2],
            )

        self._animation = QVariantAnimation(
            self,
            valueChanged=self._animate,
            startValue=0.00001,
            endValue=0.9999,
            duration=250,
        )

        # self.setStyleSheet("QPushButton:disabled {color:white;background-color: grey; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}")
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )

    def _animate(self, value):
        qss = """
            font: 75 8pt "Microsoft YaHei UI";
            font-weight: bold;
            color: rgb(255, 255, 255);
            border-style: solid;
            border-radius:8px;
        """
        grad = "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:{value} {color2}, stop: 1.0 {color1});".format(
            color1=self.color1.name(), color2=self.color2.name(), value=value
        )
        qss += grad
        self.setStyleSheet(qss)

    def enterEvent(self, event):
        self._animation.setDirection(QAbstractAnimation.Forward)
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.setDirection(QAbstractAnimation.Backward)
        self._animation.start()
        super().enterEvent(event)

    def mousePressEvent(self, event):
        self._animation.setDirection(QAbstractAnimation.Forward)
        self._animation.start()
        self.clicked.emit()
        super().enterEvent(event)


class MySwitch(QtWidgets.QPushButton):
    """
    General switch button widget.
    """

    def __init__(
        self,
        label_1,
        color_1,
        label_2,
        color_2,
        width,
        font_size=8,
        parent=None,
    ):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)
        self.switch_label_1 = label_1
        self.switch_label_2 = label_2
        self.switch_color_1 = color_1
        self.switch_color_2 = color_2
        self.width = width
        self.font_size = font_size
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )

    def paintEvent(self, event):
        label = (
            self.switch_label_1 if self.isChecked() else self.switch_label_2
        )

        if self.isChecked():
            bg_color = QColor(self.switch_color_1)
        else:
            bg_color = QColor(self.switch_color_2)

        radius = 10
        width = self.width
        center = self.rect().center()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QColor(0, 0, 0))

        pen = QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(
            QRect(-width, -radius, 2 * width, 2 * radius), radius, radius
        )
        painter.setBrush(QBrush(bg_color))
        sw_rect = QRect(-radius, -radius, width + radius, 2 * radius)
        if not self.isChecked():
            sw_rect.moveLeft(-width)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.setFont(QFont("Arial", self.font_size, QFont.Bold))
        painter.drawText(sw_rect, Qt.AlignCenter, label)


class GeneralFancyButton(QtWidgets.QPushButton):
    """
    Button style
    """

    def __init__(self, label="", parent=None):
        super().__init__(parent)
        # with Icons.Path("Run_1.png") as path:
        # self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#FFFFFF;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #39C0F0, stop:1 #6666FF);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setText(label)
        self.setFixedHeight(32)
        self.setIconSize(QSize(30, 30))
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        # self.setToolTip("Execute")


class runButton(QtWidgets.QPushButton):
    """
    Button style for 'Run'
    """

    def __init__(self, label="", parent=None):
        super().__init__(parent)
        with Icons.Path("Run_1.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #FF99FF, stop:1 #9ED8FF);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setText(label)
        self.setFixedHeight(32)
        self.setIconSize(QSize(30, 30))
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Execute")


class stop_deleteButton(QtWidgets.QPushButton):
    """
    Button style for 'STOP' or 'Delete'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        with Icons.Path("cross.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #CC0000, stop:1 #FF8000);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #660000;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setFixedHeight(32)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Stop/Delete")


class cleanButton(QtWidgets.QPushButton):
    """
    Button style for 'clean'.
    """

    def __init__(self, label="", parent=None):
        super().__init__(parent)
        with Icons.Path("clean.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #DAECD7, stop:1 #CEECEE);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setText(label)
        self.setFixedHeight(32)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Clear")


class checkableButton(QtWidgets.QPushButton):
    """
    .
    """

    def __init__(
        self, Icon_path="", label="", background_color="#FFE5CC", parent=None
    ):
        super().__init__(parent)
        self.setIcon(QIcon(Icon_path))
        StyleSheet = (
            "QPushButton {color:white;background-color: "
            + background_color
            + ";}QPushButton:hover:!pressed {color:white;background-color: #CCFFFF;}\
                                          QPushButton:checked {color:white;background-color: #B2FF66;}"
        )
        self.setCheckable(True)
        self.setFixedWidth(30)
        self.setFixedHeight(30)
        self.setStyleSheet(StyleSheet)
        self.setText(label)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )


class saveButton(QtWidgets.QPushButton):
    """
    Button style for 'save'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        with Icons.Path("save.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #56F6C6, stop:1 #00CC00);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setFixedHeight(32)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Save")


class addButton(QtWidgets.QPushButton):
    """
    Button style for 'save'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        with Icons.Path("add.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #66FFFF, stop:1 #66FFB2);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setFixedHeight(32)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Add")


class generateButton(QtWidgets.QPushButton):
    """
    Button style for 'generate'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        with Icons.Path("generate.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #39C0F0, stop:1 #CBF0FD);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setFixedHeight(32)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Generate/Configure")


class connectButton(QtWidgets.QPushButton):
    """
    Button style for 'connect' button
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)

        self.setStyleSheet(
            "QPushButton {color:black;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #FFFF00, stop:1 #E5CCFF);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
            "QPushButton:checked {color:black;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #FF9999, stop:1 #FFCC99);border-radius: 8px;}"
        )

        self.setFixedHeight(30)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )

        icon = QIcon()
        with Icons.Path("disconnect.png") as path:
            icon.addPixmap(QPixmap(path))
        with Icons.Path("connect.png") as path:
            icon.addPixmap(QPixmap(path), QIcon.Normal, QIcon.On)
        self.setIcon(icon)
        self.setToolTip("Connect/Disconnect")


class loadButton(QtWidgets.QPushButton):
    """
    Button style for 'load'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        with Icons.Path("Load.png") as path:
            self.setIcon(QIcon(path))
        StyleSheet = (
            "QPushButton {color:#0000CC;font: bold;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #39C0F0, stop:1 #CBF0FD);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )
        self.setStyleSheet(StyleSheet)
        self.setFixedHeight(32)
        self.setIconSize(QSize(30, 30))
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Load from file")


class disconnectButton(QtWidgets.QPushButton):
    """
    Button style for 'generate'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        with Icons.Path("disconnect.png") as path:
            self.setIcon(QIcon(path))
        self.setStyleSheet(
            "QPushButton {color:black;background-color: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #FF9999, stop:1 #FFCC99);border-radius: 8px;}"
            "QPushButton:hover:!pressed {color:white;background-color: #9999FF;border-radius: 8px;}"
            "QPushButton:disabled {color:white;background-color: grey;border-radius: 8px;}"
        )

        self.setFixedHeight(30)
        self.setGraphicsEffect(
            QtWidgets.QGraphicsDropShadowEffect(
                blurRadius=3, xOffset=2, yOffset=2
            )
        )
        self.setToolTip("Disconnect")


class SquareImageView(pg.ImageView):
    """
    ImageView widget that stays square when resized
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event):
        # Create a square base size of 10x10 and scale it to the new size
        # maintaining aspect ratio.
        new_size = QSize(10, 10)
        new_size.scale(event.size(), Qt.KeepAspectRatio)
        self.resize(new_size)


class _Bar(QtWidgets.QWidget):
    clickedValue = QtCore.pyqtSignal(int)

    def __init__(self, steps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )

        if isinstance(steps, list):
            # list of colours.
            self.n_steps = len(steps)
            self.steps = steps

        elif isinstance(steps, int):
            # int number of bars, defaults to red.
            self.n_steps = steps
            self.steps = ["red"] * steps

        else:
            raise TypeError("steps must be a list or int")

        self._bar_solid_percent = 0.8
        self._background_color = QtGui.QColor("black")
        self._padding = 4.0  # n-pixel gap around edge.

    def paintEvent(self, e):
        painter = QtGui.QPainter(self)

        brush = QtGui.QBrush()
        brush.setColor(self._background_color)
        brush.setStyle(Qt.SolidPattern)
        rect = QtCore.QRect(
            0, 0, painter.device().width(), painter.device().height()
        )
        painter.fillRect(rect, brush)

        # Get current state.
        parent = self.parent()
        vmin, vmax = parent.minimum(), parent.maximum()
        value = parent.value()

        # Define our canvas.
        d_height = painter.device().height() - (self._padding * 2)
        d_width = painter.device().width() - (self._padding * 2)

        # Draw the bars.
        step_size = d_height / self.n_steps
        bar_height = step_size * self._bar_solid_percent
        bar_spacer = step_size * (1 - self._bar_solid_percent) / 2

        # Calculate the y-stop position, from the value in range.
        pc = (value - vmin) / (vmax - vmin)
        n_steps_to_draw = int(pc * self.n_steps)

        for n in range(n_steps_to_draw):
            brush.setColor(QtGui.QColor(self.steps[n]))
            rect = QtCore.QRect(
                self._padding,
                self._padding + d_height - ((1 + n) * step_size) + bar_spacer,
                d_width,
                bar_height,
            )
            painter.fillRect(rect, brush)
        painter.drawText(25, 25, "{}-->{}<--{}".format(vmin, value, vmax))
        painter.end()

    def sizeHint(self):
        return QtCore.QSize(40, 120)

    def _trigger_refresh(self):
        self.update()

    def _calculate_clicked_value(self, e):
        parent = self.parent()
        vmin, vmax = parent.minimum(), parent.maximum()
        d_height = self.size().height() + (self._padding * 2)
        step_size = d_height / self.n_steps
        click_y = e.y() - self._padding - step_size / 2

        pc = (d_height - click_y) / d_height
        value = vmin + pc * (vmax - vmin)
        self.clickedValue.emit(value)

    def mouseMoveEvent(self, e):
        self._calculate_clicked_value(e)

    def mousePressEvent(self, e):
        self._calculate_clicked_value(e)


class LabeledSlider(QtWidgets.QWidget):
    def __init__(
        self,
        minimum,
        maximum,
        interval=1,
        orientation=Qt.Horizontal,
        labels=None,
        parent=None,
    ):
        super(LabeledSlider, self).__init__(parent=parent)

        levels = range(minimum, maximum + interval, interval)
        if labels is not None:
            if not isinstance(labels, (tuple, list)):
                raise Exception("<labels> is a list or tuple.")
            if len(labels) != len(levels):
                raise Exception("Size of <labels> doesn't match levels.")
            self.levels = list(zip(levels, labels))
        else:
            self.levels = list(zip(levels, map(str, levels)))

        if orientation == Qt.Horizontal:
            self.layout = QtWidgets.QVBoxLayout(self)
        elif orientation == Qt.Vertical:
            self.layout = QtWidgets.QHBoxLayout(self)
        else:
            raise Exception("<orientation> wrong.")

        # gives some space to print labels
        self.left_margin = 10
        self.top_margin = 10
        self.right_margin = 10
        self.bottom_margin = 10

        self.layout.setContentsMargins(
            self.left_margin,
            self.top_margin,
            self.right_margin,
            self.bottom_margin,
        )

        self.sl = QtWidgets.QSlider(orientation, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        if orientation == Qt.Horizontal:
            self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
            self.sl.setMinimumWidth(300)  # just to make it easier to read
        else:
            self.sl.setTickPosition(QtWidgets.QSlider.TicksLeft)
            self.sl.setMinimumHeight(300)  # just to make it easier to read
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)

        self.layout.addWidget(self.sl)

    def paintEvent(self, e):
        super(LabeledSlider, self).paintEvent(e)

        style = self.sl.style()
        painter = QPainter(self)
        st_slider = QtWidgets.QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation = self.sl.orientation()

        length = style.pixelMetric(
            QtWidgets.QStyle.PM_SliderLength, st_slider, self.sl
        )
        available = style.pixelMetric(
            QtWidgets.QStyle.PM_SliderSpaceAvailable, st_slider, self.sl
        )

        for v, v_str in self.levels:
            # get the size of the label
            rect = painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            if self.sl.orientation() == Qt.Horizontal:
                # I assume the offset is half the length of slider, therefore
                # + length//2
                x_loc = (
                    QtWidgets.QStyle.sliderPositionFromValue(
                        self.sl.minimum(), self.sl.maximum(), v, available
                    )
                    + length // 2
                )

                # left bound of the text = center - half of text width + L_margin
                left = x_loc - rect.width() // 2 + self.left_margin
                bottom = self.rect().bottom()

                # enlarge margins if clipping
                if v == self.sl.minimum():
                    if left <= 0:
                        self.left_margin = rect.width() // 2 - x_loc
                    if self.bottom_margin <= rect.height():
                        self.bottom_margin = rect.height()

                    self.layout.setContentsMargins(
                        self.left_margin,
                        self.top_margin,
                        self.right_margin,
                        self.bottom_margin,
                    )

                if (
                    v == self.sl.maximum()
                    and rect.width() // 2 >= self.right_margin
                ):
                    self.right_margin = rect.width() // 2
                    self.layout.setContentsMargins(
                        self.left_margin,
                        self.top_margin,
                        self.right_margin,
                        self.bottom_margin,
                    )

            else:
                y_loc = QtWidgets.QStyle.sliderPositionFromValue(
                    self.sl.minimum(),
                    self.sl.maximum(),
                    v,
                    available,
                    upsideDown=True,
                )

                bottom = (
                    y_loc
                    + length // 2
                    + rect.height() // 2
                    + self.top_margin
                    - 3
                )
                # there is a 3 px offset that I can't attribute to any metric

                left = self.left_margin - rect.width()
                if left <= 0:
                    self.left_margin = rect.width() + 2
                    self.layout.setContentsMargins(
                        self.left_margin,
                        self.top_margin,
                        self.right_margin,
                        self.bottom_margin,
                    )

            pos = QPoint(left, bottom)
            painter.drawText(pos, v_str)

        return


class PowerBar(QtWidgets.QWidget):
    """
    Custom Qt Widget to show a power bar and dial.
    Demonstrating compound and custom-drawn widget.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = QtCore.pyqtSignal()

    def __init__(self, steps=5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout()
        self._bar = _Bar(steps)
        layout.addWidget(self._bar)

        # Create the QDial widget and set up defaults.
        # - we provide accessors on this class to override.
        self._dial = QtWidgets.QDial()
        self._dial.setNotchesVisible(True)
        self._dial.setWrapping(False)
        self._dial.valueChanged.connect(self._bar._trigger_refresh)

        # Take feedback from click events on the meter.
        self._bar.clickedValue.connect(self._dial.setValue)

        layout.addWidget(self._dial)
        self.setLayout(layout)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self[name]

        return getattr(self._dial, name)

    def setColor(self, color):
        self._bar.steps = [color] * self._bar.n_steps
        self._bar.update()

    def setColors(self, colors):
        self._bar.n_steps = len(colors)
        self._bar.steps = colors
        self._bar.update()

    def setBarPadding(self, i):
        self._bar._padding = int(i)
        self._bar.update()

    def setBarSolidPercent(self, f):
        self._bar._bar_solid_percent = float(f)
        self._bar.update()

    def setBackgroundColor(self, color):
        self._bar._background_color = QtGui.QColor(color)
        self._bar.update()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    frame = QtWidgets.QWidget()
    ha = QtWidgets.QHBoxLayout()
    frame.setLayout(ha)

    w = LabeledSlider(1, 10, 1, orientation=Qt.Horizontal)

    ha.addWidget(w)
    frame.show()
    sys.exit(app.exec_())
