# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:09:56 2021

@author: TvdrBurgt
"""

import time

import numpy as np
import serial


class ScientificaPatchStar:
    """Scientifica PatchStar control through serial communication
    This class is for controlling the Scientifica PatchStar micromanipulator.
    """

    def __init__(self, address, baud):
        self.port = address  # COM port micromanipulator is connected to
        self.baudrate = baud  # Baudrate of the micromanipulator
        self.ENDOFLINE = "\r"  # Carriage return
        self.units = 100  # 1um is 100 PatchStar units

    def send_and_recieve(self, command):
        # Add an end-of-line signature to indicate the end of a command
        command = command + self.ENDOFLINE

        with serial.Serial(self.port, self.baudrate, timeout=3) as patchstar:
            # Encode the command to ascii and send to PatchStar
            patchstar.write(command.encode("ascii"))

            # Wait until all data is written
            patchstar.flush()

            # Read PatchStar response until carriage return
            response = patchstar.read_until(self.ENDOFLINE.encode("ascii"))

        # Decodes response to utf-8
        response = response.decode("utf-8")

        # Strip off the the end-of-line signature
        response = response.rstrip(self.ENDOFLINE)

        return response

    def wait_until_finished(func):
        """
        Decorator that waits until micromanipulator motors are idle.
        """

        def wrapper(self, *args, **kwargs):
            # Execute move
            result = func(self, *args, **kwargs)

            # Wait until motors are idle
            response = "-1"
            while response != "0":
                response = self.send_and_recieve("S")
                time.sleep(0.1)

            return result

        return wrapper

    def getPos(self):
        """
        Reports the position for all three axes separated by tabs, example:
            Send: POS or P
            Response: 1321 543 2
        """
        response = self.send_and_recieve("P")

        # Split response by at the tabs
        [x, y, z] = response.split("\t")

        # Convert coordinates to float and make a numpy array from them
        positionarray = np.array([float(x), float(y), float(z)]) / self.units

        return positionarray

    def setZero(self):
        """
        This sets the current position to (0,0,0) as long as the motion device
        is not moving.
            Send: ZERO
            Response: A (if set is allowed else E)
        """
        response = self.send_and_recieve("ZERO")

        return response

    @wait_until_finished
    def moveAbs(self, x, y, z):
        """
        Moves the patchstar to the given absolute position, example:
            Send: ABS 100 26 3
            Response: A (if move allowed else E)
        """
        x, y, z = np.array([x, y, z]) * self.units

        response = self.send_and_recieve("ABS %d %d %d" % (x, y, z))

        return response

    @wait_until_finished
    def moveRel(self, dx=0, dy=0, dz=0):
        """
        Moves the patchstar to the given relative position, example:
            Send: REL 100 26 3
            Response: A (if move allowed else E)
        """
        dx, dy, dz = np.array([dx, dy, dz]) * self.units

        response = self.send_and_recieve("REL %d %d %d" % (dx, dy, dz))

        return response

    def stop(self):
        """
        Stops any motion.
        """
        self.send_and_recieve("STOP" + self.ENDOFLINE)
