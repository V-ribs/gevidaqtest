# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:41:41 2019

@author: xinmeng
"""

import importlib.resources
import logging
import sys
import time

try:
    from pipython import GCSDevice, pitools
except ImportError:
    logging.info("pipython not configured.")


class PIMotor:
    def __init__(self):
        """
        Provide a device, connected via the PI GCS DLL.

        def pipython.gcsdevice.GCSDevice.__init__	(	 	self,
                devname = '',
                gcsdll = '')

        Parameters
        devname	: Name of device, chooses according DLL which defaults to PI_GCS2_DLL.
        gcsdll	: Name or path to GCS DLL to use, overwrites 'devname'.

        Returns
        None.

        """
        try:
            CONTROLLERNAME = "C-863.11"  # TODO unused
            # STAGES = None
            STAGES = "M-110.1DG"  # TODO unused
            REFMODE = "FNL"  # TODO unused

            # Get the path to dll in the same folder.
            files = importlib.resources.files(sys.modules[__package__])
            traversable = files.joinpath("PI_GCS2_DLL_x64.dll")
            with importlib.resources.as_file(traversable) as path:
                self.pidevice = GCSDevice(gcsdll=str(path))

            logging.info(self.pidevice.EnumerateUSB())
            # InterfaceSetupDlg() is an interactive dialog. There are other methods to
            # connect to an interface without user interaction.
            serialstring = self.pidevice.EnumerateUSB()
            logging.info(serialstring[0])
            # pidevice.InterfaceSetupDlg(key='sample')
            # pidevice.ConnectRS232(comport=1, baudrate=115200)
            self.pidevice.ConnectUSB(
                serialnum="PI C-863 Mercury SN 0185500828"
            )
            # pidevice.ConnectTCPIP(ipaddress='192.168.178.42')

            # Each PI controller supports the qIDN() command which returns an
            # identification string with a trailing line feed character which
            # we "strip" away.

            logging.info("connected: {}".format(self.pidevice.qIDN().strip()))

            # Show the version info which is helpful for PI support when there
            # are any issues.

            if self.pidevice.HasqVER():
                logging.info(
                    "version info: {}".format(self.pidevice.qVER().strip())
                )
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("PI device not initilized.")

    def move(self, target_pos):
        # pidevice.StopAll()
        # pidevice.SVO(pidevice.axes, [True] * len(pidevice.axes))
        # pitools.waitontarget(pidevice, axes=pidevice.axes)

        # pitools.startup(pidevice, stages=STAGES, refmode=REFMODE)

        # Now we query the allowed motion range of all connected stages.
        # GCS commands often return an (ordered) dictionary with axes/channels
        # as "keys" and the according values as "values".

        # rangemin = list(pidevice.qTMN().values())
        # rangemax = list(pidevice.qTMX().values())
        # ranges = zip(rangemin, rangemax)

        targets = [target_pos]
        self.pidevice.MOV(self.pidevice.axes, targets)
        pitools.waitontarget(self.pidevice)
        time.sleep(0.3)
        positions = self.pidevice.qPOS(self.pidevice.axes)
        for axis in self.pidevice.axes:
            logging.info(
                "position of axis {} = {:.5f}".format(axis, positions[axis])
            )

    def GetCurrentPos(self):
        # positions is a dictionary with key being axis name, here '1'.
        positions = self.pidevice.qPOS(self.pidevice.axes)

        return positions["1"]

    def CloseMotorConnection(self):
        self.pidevice.CloseConnection()


if __name__ == "__main__":
    # To see what is going on in the background you can remove the following
    # two hashtags. Then debug messages are shown. This can be helpful if
    # there are any issues.

    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    pi = PIMotor()
    # pi_device = pi.ConnectPIMotor

    # PIMotor.move(pi.pidevice, 3.455)
    logging.info(pi.GetCurrentPos())
    pi.CloseMotorConnection()
