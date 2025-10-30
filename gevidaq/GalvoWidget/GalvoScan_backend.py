# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:35:22 2020

@author: xinmeng
"""

import logging
import os
import time

import nidaqmx
import numpy as np
import tifffile as skimtiff
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogSingleChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter

from ..NIDAQ import wavegenerator
from ..PI_ObjectiveMotor.focuser import PIMotor


class RasterScan:
    def __init__(
        self,
        Daq_sample_rate,
        edge_volt,
        pixel_number=500,
        average_number=1,
        continuous=False,
        return_image=True,
    ):
        """
        Object to run raster PMT scanning.

        Parameters
        Daq_sample_rate : int
            Sampling rate used to generate the waveforms.
        edge_volt : float
            The bounding raster scan voltage.
        pixel_number : int, optional
            Number of pixels in the final image. The default is 500.
        average_number : int, optional
            Number of frames to average on. The default is 1.
        continuous : TYPE, optional
            Whether to do continuous scanning or not. The default is False.
        return_image : TYPE, optional
            Whether return the processed image. The default is True.

        Returns
        None.

        """

        # === Generate the waveforms ===
        self.Daq_sample_rate = Daq_sample_rate
        self.averagenum = average_number
        self.edge_volt = edge_volt
        self.pixel_number = pixel_number
        self.flag_continuous = continuous
        self.flag_return_image = return_image

        # Generate galvo samples
        self.samples_X, self.samples_Y = wavegenerator.waveRecPic(
            sampleRate=self.Daq_sample_rate,
            imAngle=0,
            voltXMin=-1 * self.edge_volt,
            voltXMax=self.edge_volt,
            voltYMin=-1 * self.edge_volt,
            voltYMax=self.edge_volt,
            xPixels=self.pixel_number,
            yPixels=self.pixel_number,
            sawtooth=True,
        )
        # Calculate number of all samples to feed to daq.
        self.Totalscansamples = len(self.samples_X) * self.averagenum
        # Number of samples of each individual line of x scanning, including fly backs.
        # Devided by pixel number as it's repeated for each y line.
        self.total_X_sample_number = int(
            len(self.samples_X) / self.pixel_number
        )

        self.repeated_samples_X = np.tile(self.samples_X, self.averagenum)
        self.repeated_samples_Y = np.tile(self.samples_Y, self.averagenum)

        self.Galvo_samples = np.vstack(
            (self.repeated_samples_X, self.repeated_samples_Y)
        )

    def run(self):
        """
        Starts writing a waveform continuously while reading
        the buffer periodically
        """

        with nidaqmx.Task() as slave_Task, nidaqmx.Task() as master_Task:
            slave_Task.ao_channels.add_ao_voltage_chan("/Dev1/ao0:1")
            master_Task.ai_channels.add_ai_voltage_chan("/Dev1/ai0")

            if self.flag_continuous is False:
                # Timing of analog output channels
                slave_Task.timing.cfg_samp_clk_timing(
                    rate=self.Daq_sample_rate,
                    source="ai/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=self.Totalscansamples,
                )

                # Timing of recording channels
                master_Task.timing.cfg_samp_clk_timing(
                    rate=self.Daq_sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=self.Totalscansamples,
                )
            else:
                # Timing of analog output channels
                slave_Task.timing.cfg_samp_clk_timing(
                    rate=self.Daq_sample_rate,
                    source="ai/SampleClock",
                    sample_mode=AcquisitionType.CONTINUOUS,
                )

                # Timing of recording channels
                master_Task.timing.cfg_samp_clk_timing(
                    rate=self.Daq_sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.Totalscansamples,
                )

            reader = AnalogSingleChannelReader(master_Task.in_stream)
            writer = AnalogMultiChannelWriter(slave_Task.out_stream)

            reader.auto_start = False
            writer.auto_start = False

            writer.write_many_sample(self.Galvo_samples)

            """Reading data from the buffer in a loop.
            The idea is to let the task read more than could be loaded in the buffer for each iteration.
            This way the task will have to wait slightly longer for incoming samples. And leaves the buffer
            entirely clean. This way we always know the correct numpy size and are always left with an empty
            buffer (and the buffer will not slowly fill up)."""
            output = np.zeros(self.Totalscansamples)
            slave_Task.start()  # Will wait for the readtask to start so it can use its clock
            master_Task.start()

            # while not self.isInterruptionRequested():
            reader.read_many_sample(
                data=output,
                number_of_samples_per_channel=self.Totalscansamples,
            )

            Dataholder_average = np.mean(
                output.reshape(self.averagenum, -1), axis=0
            )

            if self.flag_return_image is True:
                # Calculate the mean of average frames.
                self.data_PMT = np.reshape(
                    Dataholder_average,
                    (self.pixel_number, self.total_X_sample_number),
                )

                # Cut off the flying back part.
                if self.Daq_sample_rate == 500000:
                    if self.pixel_number == 500:
                        self.image_PMT = self.data_PMT[:, 50:550] * -1
                    elif self.pixel_number == 256:
                        self.image_PMT = self.data_PMT[:, 70:326] * -1
                elif self.Daq_sample_rate == 250000:
                    if self.pixel_number == 500:
                        self.image_PMT = self.data_PMT[:, 25:525] * -1
                    elif self.pixel_number == 256:
                        self.image_PMT = self.data_PMT[:, 25:525] * -1

                return self.image_PMT


class PMT_zscan:
    def __init__(
        self,
        saving_dir,
        z_depth,
        z_step_size=0.004,
        imaging_conditions={
            "Daq_sample_rate": 500000,
            "edge_volt": 5,
            "pixel_number": 500,
            "average_number": 2,
        },
        motor_handle=None,
        twophoton_handle=None,
        *args,
        **kwargs,
    ):
        """
        Object to run Z-stack scanning.

        Parameters
        saving_dir : str
            Directory to save images.
        z_depth : float
            The depth of scanning.
        z_step_size : float, optional
            Each z step size. The default is 0.004.
        imaging_conditions : dict, optional
            Dictionary containing imaging parameters.
            The default is {'Daq_sample_rate': 500000, 'edge_volt':5, 'pixel_number': 500,'average_number':2}.
        motor_handle : TYPE, optional
            Handle to use objective motor. The default is None.
        twophoton_handle : TYPE, optional
            Handle to control two-photon laser. The default is None.
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        None.

        """

        self.scanning_flag = True
        self.saving_dir = saving_dir

        if motor_handle is None:
            # Connect the objective if the handle is not provided.
            self.pi_device_instance = PIMotor()
        else:
            self.pi_device_instance = motor_handle

        # Current position of the focus.
        self.current_pos = self.pi_device_instance.GetCurrentPos()

        # The step size when first doing coarse searching.
        z_depth_start = self.current_pos
        z_depth_end = z_depth + z_depth_start

        # Number of steps in total to find optimal focus.
        if z_depth != 0:
            self.total_step_number = round(
                (z_depth_end - self.current_pos) / z_step_size
            )
        else:
            # If doing repeating imaging at the same position,
            # z_depth becomes the number of repeats.
            self.total_step_number = int(z_step_size)

        # Generate the sampling positions.
        self.z_stack_positions = np.linspace(
            z_depth_start, z_depth_end, self.total_step_number
        )
        logging.info(self.z_stack_positions)
        # Parameters for imaging.
        self.RasterScanins = RasterScan(
            imaging_conditions["Daq_sample_rate"],
            imaging_conditions["edge_volt"],
            imaging_conditions["pixel_number"],
            imaging_conditions["average_number"],
        )

    def start_scan(self):
        for self.each_pos_index in range(len(self.z_stack_positions)):
            if self.scanning_flag is True:
                # Go through each position and get image.
                self.make_PMT_image(
                    round(self.z_stack_positions[self.each_pos_index], 6)
                )
            else:
                break

        self.pi_device_instance.CloseMotorConnection()

    def stop_scan(self):
        self.scanning_flag = False

    def make_PMT_image(self, obj_position=None):
        """
        Take PMT image at certain objective position.

        Parameters
        obj_position : float, optional
            The target objective position. The default is None.

        """

        if obj_position is not None:
            self.pi_device_instance.move(obj_position)

        # Get the image.
        self.galvo_image = self.RasterScanins.run()

        meta_infor = (
            "index_" + str(self.each_pos_index) + "_pos_" + str(obj_position)
        )

        if True:
            with skimtiff.TiffWriter(
                os.path.join(self.saving_dir, meta_infor + ".tif")
            ) as tif:
                tif.save(self.galvo_image.astype("float32"), compress=0)
        time.sleep(0.5)


if __name__ == "__main__":
    z_stack = PMT_zscan(
        r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Xin\2021-04-18 Filter bleed through\New folder",  # TODO hardcoded path
        z_depth=0.012,
    )
    z_stack.start_scan()
