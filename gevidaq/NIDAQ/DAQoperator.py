# -*- coding: utf-8 -*-
"""
The adaptive NI DAQ tool
Created on Fri May 29 15:29:22 2020

@author: xinmeng
"""

# READIN TASK HAS TO START AHEAD OF READ MANY SAMPLES, OTHERWISE ITS NOT in SYN!!!

import logging
import os
import time
from datetime import datetime

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, LineGrouping
from nidaqmx.stream_readers import AnalogMultiChannelReader
from PyQt5.QtCore import QThread, pyqtSignal

from .constants import NiDaqChannels


class DAQmission(QThread):
    """
    # For all-purpose Nidaq tasks. Use "Dev1/ai22" as reference channel.
    """

    collected_data = pyqtSignal(np.ndarray)
    finishSignal = pyqtSignal()

    def __init__(self, channel_LUT=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Specifiy NI-daq channels. channel_LUT in the form of dictionary, with keys being the
        purpose of the channel (the same as the fields from the input waveforms' "Specification"
        field) and values being the port of the daq. If not specified it will load the dictionary
        from NiDaqChannels class in NIDAQ.constants.
        """
        if channel_LUT is None:
            self.channel_LUT = NiDaqChannels().look_up_table
        else:
            self.channel_LUT = channel_LUT

    def sendSingleAnalog(self, channel, value):
        """
        Write one single digital signal.

        Parameters
        channel : str
            Purpose of the channel.
        value : bool
            Value to send.

        Returns
        None.

        """

        channelname = self.channel_LUT[channel]
        writing_value = value

        # Assume that dev1 is always employed
        with nidaqmx.Task() as writingtask:
            writingtask.ao_channels.add_ao_voltage_chan(channelname)
            writingtask.write(writing_value)

    def sendSingleDigital(self, channel, value):
        """
        Write one single analog signal.

        Parameters
        channel : str
            Purpose of the channel.
        value : float
            Value to send.

        Returns
        None.

        """

        channelname = self.channel_LUT[channel]
        if value is True:
            writing_value = np.array([1], dtype=bool)
        else:
            writing_value = np.array([0], dtype=bool)

        with nidaqmx.Task() as writingtask:
            writingtask.do_channels.add_do_chan(channelname)
            writingtask.write(writing_value)

    def sendServoSignal(self, servo_channel: str, open_servo: bool):
        """Opens or closes the beam path servo_channel.

        servo_channel: the name of the channel as found in the channel_LUT
        open_servo: True means to open the servo, False means close
        """
        sample_rate = 50000
        total_duration = 0.25
        frequency = 50

        # set signal parameters
        if open_servo:
            high_time = 0.001
            low_time = 0.019
        else:
            high_time = 0.002
            low_time = 0.018

        num_cycles = int(total_duration * frequency)

        high_samples = int(high_time * sample_rate)
        low_samples = int(low_time * sample_rate)
        pulse = [True] * high_samples + [False] * low_samples
        signal = pulse * num_cycles

        # write waveform
        channelname = self.channel_LUT[servo_channel]

        with nidaqmx.Task() as writingtask:
            writingtask.do_channels.add_do_chan(
                channelname,
                line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
            )
            writingtask.timing.cfg_samp_clk_timing(
                sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=len(signal),
            )
            writingtask.write(signal, auto_start=False)
            writingtask.start()
            time.sleep(total_duration)

    def runWaveforms(
        self,
        clock_source,
        sampling_rate,
        analog_signals,
        digital_signals,
        readin_channels,
    ):
        """
        Input:
          - samplingrate:
              Sampling rate of the waveforms.

          - analogsignals:
              Signals for the analog channels.
              It's a structured array with two fields:
              1) 'Waveform': Raw 1-D np.array of actual float voltage signals.
              2) 'Specification': string telling which device to control that help to specify the NI-daq port, check self.channel_LUT.

              Multiple waveforms should be stack on top of each other using np.stack.

              if empty, input can be a blank {}.

           -digitalsignals:
              Signals for the digital channels.
              It's a structured array with two fields: 1) 'Waveform': Raw 1-D np.array of type bool.
                                                       2) 'Specification': string that specifies the NI-daq port.
                                                          for example: dtype = np.dtype([('Waveform', float, (self.reference_length,)), ('Specification', 'U20')])
           -readinchannels:
              A list that contains the readin channels wanted.
        """

        # Setting up waveforms

        Analog_channel_number = len(analog_signals)
        Digital_channel_number = len(digital_signals)
        self.readin_channels = readin_channels
        self.sampling_rate = sampling_rate

        # galvosx and galvosy as Sepcification key words are already enough.

        # Get the average number and y pixel number information from data
        self.galvosx_originalkey = "galvosx"
        self.galvosy_originalkey = "galvosy"

        # If there are kyes with information like 'galvosxavgnum', extract the
        # information and then convert the key to 'galvosx'.
        if Analog_channel_number != 0:
            for i in range(len(analog_signals["Specification"])):
                if "galvosxavgnum" in analog_signals["Specification"][i]:
                    self.averagenumber = int(
                        analog_signals["Specification"][i][
                            analog_signals["Specification"][i].index("_")
                            + 1 : len(analog_signals["Specification"][i])
                        ]
                    )
                    self.galvosx_originalkey = analog_signals["Specification"][
                        i
                    ]
                    analog_signals["Specification"][i] = "galvosx"
                elif "galvosyypixels" in analog_signals["Specification"][i]:
                    self.ypixelnumber = int(
                        analog_signals["Specification"][i][
                            analog_signals["Specification"][i].index("_")
                            + 1 : len(analog_signals["Specification"][i])
                        ]
                    )
                    self.galvosy_originalkey = analog_signals["Specification"][
                        i
                    ]
                    analog_signals["Specification"][i] = "galvosy"
                elif "galvos_X_contour" in analog_signals["Specification"][i]:
                    self.galvosx_originalkey = analog_signals["Specification"][
                        i
                    ]
                    analog_signals["Specification"][i] = "galvosx"

                elif "galvos_Y_contour" in analog_signals["Specification"][i]:
                    self.galvosy_originalkey = analog_signals["Specification"][
                        i
                    ]
                    analog_signals["Specification"][i] = "galvosy"

        # === Devide samples from Dev1 or 2 ===
        self.Dev1_analog_channel_list = []
        self.Dev2_analog_channel_list = []

        Dev1_analog_waveforms_list = []
        Dev2_analog_waveforms_list = []

        if Analog_channel_number != 0:
            if len(analog_signals["Waveform"]) != 0:
                num_rows, num_cols = analog_signals["Waveform"].shape
                for i in range(int(num_rows)):
                    specification = analog_signals["Specification"][i]
                    channel = self.channel_LUT[specification]
                    waveform = analog_signals["Waveform"][i]
                    if "Dev1" in channel:
                        self.Dev1_analog_channel_list.append(channel)
                        Dev1_analog_waveforms_list.append(waveform)
                    else:
                        self.Dev2_analog_channel_list.append(channel)
                        Dev2_analog_waveforms_list.append(waveform)

        Dev1_analog_channel_number = len(self.Dev1_analog_channel_list)
        Dev2_analog_channel_number = len(self.Dev2_analog_channel_list)

        # See if only digital signal is involved.
        if Analog_channel_number == 0 and Digital_channel_number != 0:
            self.Only_Digital_signals = True
        else:
            self.Only_Digital_signals = False

        # See if Dev1 is involved. If only Dev2 is involved in sending analog
        # signals then the timing configs are different.

        # === Number of samples in each waveform ===
        if self.Only_Digital_signals is False:
            self.Waveforms_length = len(analog_signals["Waveform"][0])
            num_rows, num_cols = analog_signals["Waveform"].shape
            logging.info(f"row number of analog signals:  {num_rows}")
        elif self.Only_Digital_signals is True:
            self.Waveforms_length = len(digital_signals["Waveform"][0])

        # === Stack the Analog samples of dev1 and dev2 individually ===
        # IN CASE OF ONLY ONE ARRAY, WE NEED TO CONVERT THE SHAPE TO (1,N) BY USING np.array([]) OUTSIDE THE ARRAY!!
        if Dev1_analog_channel_number == 1:
            Dev1_analog_samples_to_write = np.array(
                [Dev1_analog_waveforms_list[0]]
            )

        elif Dev1_analog_channel_number == 0:
            Dev1_analog_samples_to_write = []
        else:
            Dev1_analog_samples_to_write = Dev1_analog_waveforms_list[0]
            for i in range(1, Dev1_analog_channel_number):
                Dev1_analog_samples_to_write = np.vstack(
                    (
                        Dev1_analog_samples_to_write,
                        Dev1_analog_waveforms_list[i],
                    )
                )

        if Dev2_analog_channel_number == 1:
            Dev2_analog_samples_to_write = np.array(
                [Dev2_analog_waveforms_list[0]]
            )
        elif Dev2_analog_channel_number == 0:
            Dev2_analog_samples_to_write = []
        else:
            Dev2_analog_samples_to_write = Dev2_analog_waveforms_list[0]
            for i in range(1, Dev2_analog_channel_number):
                Dev2_analog_samples_to_write = np.vstack(
                    (
                        Dev2_analog_samples_to_write,
                        Dev2_analog_waveforms_list[i],
                    )
                )

        # Stack the digital samples
        if Digital_channel_number == 1:
            Digital_samples_to_write = np.array(
                [digital_signals["Waveform"][0]]
            )

        elif Digital_channel_number == 0:
            Digital_samples_to_write = []
        else:
            Digital_samples_to_write = digital_signals["Waveform"][0]
            for i in range(1, Digital_channel_number):
                Digital_samples_to_write = np.vstack(
                    (Digital_samples_to_write, digital_signals["Waveform"][i])
                )

        # === Set the dtype of digital signals ===
        # For each digital waveform sample, it 0 or 1. To write to NI-daq, you
        # need to send int number corresponding to the channel binary value,
        # like write 8(2^3, 0001) to channel 4.
        # The same as (0b1 << n)
        Digital_samples_to_write = np.array(
            Digital_samples_to_write, dtype="uint32"
        )

        for i in range(Digital_channel_number):
            specification = digital_signals["Specification"][i]
            channel = self.channel_LUT[specification]
            lineindex = channel.index("line")
            convernum = int(channel[lineindex + 4 : len(channel)])
            Digital_samples_to_write[i] = Digital_samples_to_write[i] * (
                2 ** (convernum)
            )

        # For example, to send commands to line 0 and line 3, you hava to write
        # 1001 to digital port, convert to uint32 that is 9.
        if Digital_channel_number > 1:
            Digital_samples_to_write = np.sum(
                Digital_samples_to_write, axis=0
            )  # sum along the columns, for multiple lines
            Digital_samples_to_write = np.array(
                [Digital_samples_to_write]
            )  # here convert the shape from (n,) to (1,n)

        # === Set up data holder for recording data ===
        if len(self.readin_channels) != 0:
            self.has_recording_channel = True
        else:
            self.has_recording_channel = False

        if self.has_recording_channel is True:
            self.Dataholder = np.zeros(
                (len(self.readin_channels), self.Waveforms_length)
            )
        else:
            self.Dataholder = np.zeros((1, self.Waveforms_length))

        # Configure DAQ channels and execute waveforms

        """
        # Analog signal in Dev 1 is involved
        """
        if Dev1_analog_channel_number != 0:
            with (
                nidaqmx.Task() as slave_Task_1_analog_dev1,
                nidaqmx.Task() as slave_Task_1_analog_dev2,
                nidaqmx.Task() as master_Task_readin,
                nidaqmx.Task() as slave_Task_2_digitallines,
            ):
                # === adding channels ===
                # Set tasks from different devices apart
                for i in range(Dev1_analog_channel_number):
                    slave_Task_1_analog_dev1.ao_channels.add_ao_voltage_chan(
                        self.Dev1_analog_channel_list[i]
                    )

                slave_Task_2_digitallines.do_channels.add_do_chan(
                    "/Dev1/port0",
                    line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
                )

                if self.has_recording_channel is True:
                    self.Dataholder = np.zeros(
                        (len(self.readin_channels), self.Waveforms_length)
                    )
                else:
                    self.Dataholder = np.zeros((1, self.Waveforms_length))
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["Vp"]
                    )  # If no read-in channel is added, vp channel is added to keep code alive.

                # print(self.Dataholder.shape)
                if "PMT" in self.readin_channels:
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["PMT"]
                    )
                if "Vp" in self.readin_channels:
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["Vp"]
                    )
                # For the current measurement, we use the voltage channel in DAQ
                # and convert to current later devided by current gain and patch
                # probe resistance.
                if "Ip" in self.readin_channels:
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["Ip"]
                    )

                # === get scaling coefficients ===
                self.aichannelnames = (
                    master_Task_readin.ai_channels.channel_names
                )

                self.ai_dev_scaling_coeff_vp = []
                self.ai_dev_scaling_coeff_ip = []
                if "Vp" in self.readin_channels:
                    self.ai_dev_scaling_coeff_vp = (
                        nidaqmx._task_modules.channels.ai_channel.AIChannel(
                            master_Task_readin._handle, self.channel_LUT["Vp"]
                        )
                    )
                    # https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019TuoSAE&l=nl-NL
                    # self.ai_dev_scaling_coeff.ai_dev_scaling_coeff
                    self.ai_dev_scaling_coeff_vp = np.array(
                        self.ai_dev_scaling_coeff_vp.ai_dev_scaling_coeff
                    )

                if "Ip" in self.readin_channels:
                    self.ai_dev_scaling_coeff_ip = (
                        nidaqmx._task_modules.channels.ai_channel.AIChannel(
                            master_Task_readin._handle, self.channel_LUT["Ip"]
                        )
                    )
                    # https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019TuoSAE&l=nl-NL
                    # self.ai_dev_scaling_coeff.ai_dev_scaling_coeff
                    self.ai_dev_scaling_coeff_ip = np.array(
                        self.ai_dev_scaling_coeff_ip.ai_dev_scaling_coeff
                    )

                self.ai_dev_scaling_coeff_list = np.append(
                    self.ai_dev_scaling_coeff_vp, self.ai_dev_scaling_coeff_ip
                )

                # === setting clock ===
                if (
                    clock_source == "DAQ"
                ):  # If NI-DAQ is set as master clock source
                    # Analog clock  USE clock on Dev1 as center clock
                    slave_Task_1_analog_dev1.timing.cfg_samp_clk_timing(
                        self.sampling_rate,
                        source="ai/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.Waveforms_length,
                    )
                    master_Task_readin.timing.cfg_samp_clk_timing(
                        self.sampling_rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.Waveforms_length,
                    )

                    # Export the clock timing of Dev1 to specific port, use BNC cable to bridge this port
                    # and clock receiving port on Dev2.
                    master_Task_readin.export_signals.samp_clk_output_term = (
                        self.channel_LUT["clock1Channel"]
                    )  # '/Dev1/PFI1'#
                    master_Task_readin.export_signals.start_trig_output_term = self.channel_LUT[
                        "trigger1Channel"
                    ]  # '/Dev1/PFI2'

                    # If dev2 is involved, set the timing for dev2.
                    if Dev2_analog_channel_number != 0:
                        # By default assume that read master task is in dev1

                        for i in range(Dev2_analog_channel_number):
                            slave_Task_1_analog_dev2.ao_channels.add_ao_voltage_chan(
                                self.Dev2_analog_channel_list[i]
                            )

                        # Set the clock of Dev2 to the receiving port from Dev1.
                        dev2Clock = self.channel_LUT[
                            "clock2Channel"
                        ]  # /Dev2/PFI1
                        slave_Task_1_analog_dev2.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source=dev2Clock,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )

                        AnalogWriter = (
                            nidaqmx.stream_writers.AnalogMultiChannelWriter(
                                slave_Task_1_analog_dev1.out_stream,
                                auto_start=False,
                            )
                        )
                        AnalogWriter.auto_start = False

                        AnalogWriter_dev2 = (
                            nidaqmx.stream_writers.AnalogMultiChannelWriter(
                                slave_Task_1_analog_dev2.out_stream,
                                auto_start=False,
                            )
                        )
                        AnalogWriter_dev2.auto_start = False

                    # === Digital clock ===
                    if Digital_channel_number != 0:
                        slave_Task_2_digitallines.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source="ai/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )
                        # slave_Task_2_digitallines.triggers.sync_type.SLAVE = True

                elif (
                    clock_source == "Camera"
                ):  # All the clock should refer to camera output trigger
                    # Set all clock source to camera trigger receiving port.
                    self.cam_trigger_receiving_port = "/Dev1/PFI0"

                    slave_Task_1_analog_dev1.timing.cfg_samp_clk_timing(
                        self.sampling_rate,
                        source=self.cam_trigger_receiving_port,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.Waveforms_length,
                    )
                    slave_Task_1_analog_dev1.triggers.start_trigger.cfg_dig_edge_start_trig(
                        self.cam_trigger_receiving_port
                    )

                    master_Task_readin.timing.cfg_samp_clk_timing(
                        self.sampling_rate,
                        source=self.cam_trigger_receiving_port,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.Waveforms_length,
                    )
                    master_Task_readin.export_signals.samp_clk_output_term = (
                        self.channel_LUT["clock1Channel"]
                    )  # '/Dev1/PFI1'#
                    master_Task_readin.export_signals.start_trig_output_term = self.channel_LUT[
                        "trigger1Channel"
                    ]  # '/Dev1/PFI2'
                    master_Task_readin.triggers.start_trigger.cfg_dig_edge_start_trig(
                        self.cam_trigger_receiving_port
                    )

                    if Dev2_analog_channel_number != 0:
                        # By default assume that read master task is in dev1

                        for i in range(Dev2_analog_channel_number):
                            slave_Task_1_analog_dev2.ao_channels.add_ao_voltage_chan(
                                self.Dev2_analog_channel_list[i]
                            )

                        # Set clock of dev2 to the input from dev1 timing output.
                        dev2Clock = self.channel_LUT[
                            "clock2Channel"
                        ]  # /Dev2/PFI1
                        slave_Task_1_analog_dev2.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source=dev2Clock,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )
                        # slave_Task_1_analog_dev2.triggers.sync_type.SLAVE = True

                        # slave_Task_1_analog_dev2.triggers.start_trigger.cfg_dig_edge_start_trig(self.channel_LUT["trigger2Channel"])#'/Dev2/PFI7'

                        AnalogWriter = (
                            nidaqmx.stream_writers.AnalogMultiChannelWriter(
                                slave_Task_1_analog_dev1.out_stream,
                                auto_start=False,
                            )
                        )
                        AnalogWriter.auto_start = False

                        AnalogWriter_dev2 = (
                            nidaqmx.stream_writers.AnalogMultiChannelWriter(
                                slave_Task_1_analog_dev2.out_stream,
                                auto_start=False,
                            )
                        )
                        AnalogWriter_dev2.auto_start = False

                    # === Digital clock ===
                    if Digital_channel_number != 0:
                        slave_Task_2_digitallines.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source=self.cam_trigger_receiving_port,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )
                        slave_Task_2_digitallines.triggers.start_trigger.cfg_dig_edge_start_trig(
                            self.cam_trigger_receiving_port
                        )
                        # slave_Task_2_digitallines.triggers.sync_type.SLAVE = True

                # === Configure the writer and reader ===
                AnalogWriter = nidaqmx.stream_writers.AnalogMultiChannelWriter(
                    slave_Task_1_analog_dev1.out_stream, auto_start=False
                )
                AnalogWriter.auto_start = False
                if Digital_channel_number != 0:
                    DigitalWriter = (
                        nidaqmx.stream_writers.DigitalMultiChannelWriter(
                            slave_Task_2_digitallines.out_stream,
                            auto_start=False,
                        )
                    )
                    DigitalWriter.auto_start = False
                reader = AnalogMultiChannelReader(master_Task_readin.in_stream)
                reader.auto_start = False

                # === Begin to execute in DAQ ===
                AnalogWriter.write_many_sample(
                    Dev1_analog_samples_to_write, timeout=605.0
                )

                if Dev2_analog_channel_number != 0:
                    AnalogWriter_dev2.write_many_sample(
                        Dev2_analog_samples_to_write, timeout=605.0
                    )

                if Digital_channel_number != 0:
                    DigitalWriter.write_many_sample_port_uint32(
                        Digital_samples_to_write, timeout=605.0
                    )

                logging.info(
                    "^^^^^^^^^^^^^^^^^^Daq tasks start^^^^^^^^^^^^^^^^^^"
                )
                if Dev2_analog_channel_number != 0:
                    slave_Task_1_analog_dev2.start()
                slave_Task_1_analog_dev1.start()

                if Digital_channel_number != 0:
                    slave_Task_2_digitallines.start()

                master_Task_readin.start()  # !!!!!!!!!!!!!!!!!!!! READIN TASK HAS TO START AHEAD OF READ MANY SAMPLES, OTHERWISE ITS NOT SYN!!!

                reader.read_many_sample(
                    data=self.Dataholder,
                    number_of_samples_per_channel=self.Waveforms_length,
                    timeout=605.0,
                )
                # self.data_PMT = []

                slave_Task_1_analog_dev1.wait_until_done()
                if Dev2_analog_channel_number != 0:
                    slave_Task_1_analog_dev2.wait_until_done()
                if Digital_channel_number != 0:
                    slave_Task_2_digitallines.wait_until_done()
                master_Task_readin.wait_until_done()

                slave_Task_1_analog_dev1.stop()
                if Dev2_analog_channel_number != 0:
                    slave_Task_1_analog_dev2.stop()
                if Digital_channel_number != 0:
                    slave_Task_2_digitallines.stop()
                master_Task_readin.stop()

                if self.has_recording_channel is True:
                    self.collected_data.emit(self.Dataholder)
                self.finishSignal.emit()
                logging.info(
                    "^^^^^^^^^^^^^^^^^^Daq tasks finish^^^^^^^^^^^^^^^^^^"
                )

            # set the keys of galvos back for next round
            if Analog_channel_number != 0:
                for i in range(len(analog_signals["Specification"])):
                    if "galvosx" in analog_signals["Specification"][i]:
                        analog_signals["Specification"][
                            i
                        ] = self.galvosx_originalkey
                    elif "galvosy" in analog_signals["Specification"][i]:
                        analog_signals["Specification"][
                            i
                        ] = self.galvosy_originalkey

            """
            # Only Dev 2 is involved  in sending analog signals
            """
        elif Dev2_analog_channel_number != 0:
            with (
                nidaqmx.Task() as slave_Task_1_analog_dev2,
                nidaqmx.Task() as master_Task_readin,
                nidaqmx.Task() as slave_Task_2_digitallines,
            ):
                # adding channels
                # Set tasks from different devices apart
                slave_Task_2_digitallines.do_channels.add_do_chan(
                    "/Dev1/port0",
                    line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
                )

                if self.has_recording_channel is True:
                    self.Dataholder = np.zeros(
                        (len(self.readin_channels), self.Waveforms_length)
                    )
                else:
                    self.Dataholder = np.zeros((1, self.Waveforms_length))
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["Vp"]
                    )  # If no read-in channel is added, vp channel is added to keep code alive.

                # print(self.Dataholder.shape)
                if "PMT" in self.readin_channels:
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["PMT"]
                    )
                if "Vp" in self.readin_channels:
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["Vp"]
                    )
                if "Ip" in self.readin_channels:
                    master_Task_readin.ai_channels.add_ai_voltage_chan(
                        self.channel_LUT["Ip"]
                    )

                # get scaling coefficients
                self.aichannelnames = (
                    master_Task_readin.ai_channels.channel_names
                )

                self.ai_dev_scaling_coeff_vp = []
                self.ai_dev_scaling_coeff_ip = []
                if "Vp" in self.readin_channels:
                    self.ai_dev_scaling_coeff_vp = (
                        nidaqmx._task_modules.channels.ai_channel.AIChannel(
                            master_Task_readin._handle, self.channel_LUT["Vp"]
                        )
                    )
                    # self.ai_dev_scaling_coeff.ai_dev_scaling_coeff
                    self.ai_dev_scaling_coeff_vp = np.array(
                        self.ai_dev_scaling_coeff_vp.ai_dev_scaling_coeff
                    )

                if "Ip" in self.readin_channels:
                    self.ai_dev_scaling_coeff_ip = (
                        nidaqmx._task_modules.channels.ai_channel.AIChannel(
                            master_Task_readin._handle, self.channel_LUT["Ip"]
                        )
                    )
                    # self.ai_dev_scaling_coeff.ai_dev_scaling_coeff
                    self.ai_dev_scaling_coeff_ip = np.array(
                        self.ai_dev_scaling_coeff_ip.ai_dev_scaling_coeff
                    )

                self.ai_dev_scaling_coeff_list = np.append(
                    self.ai_dev_scaling_coeff_vp, self.ai_dev_scaling_coeff_ip
                )

                # === setting clock ===
                if (
                    clock_source == "DAQ"
                ):  # If NI-DAQ is set as master clock source
                    # setting clock
                    master_Task_readin.timing.cfg_samp_clk_timing(
                        self.sampling_rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.Waveforms_length,
                    )

                    master_Task_readin.export_signals.samp_clk_output_term = (
                        self.channel_LUT["clock1Channel"]
                    )  # '/Dev1/PFI1'#
                    master_Task_readin.export_signals.start_trig_output_term = self.channel_LUT[
                        "trigger1Channel"
                    ]  # '/Dev1/PFI2'

                    if Dev2_analog_channel_number != 0:
                        # By default assume that read master task is in dev1

                        for i in range(Dev2_analog_channel_number):
                            slave_Task_1_analog_dev2.ao_channels.add_ao_voltage_chan(
                                self.Dev2_analog_channel_list[i]
                            )

                        dev2Clock = self.channel_LUT[
                            "clock2Channel"
                        ]  # /Dev2/PFI1
                        slave_Task_1_analog_dev2.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source=dev2Clock,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )

                        AnalogWriter_dev2 = (
                            nidaqmx.stream_writers.AnalogMultiChannelWriter(
                                slave_Task_1_analog_dev2.out_stream,
                                auto_start=False,
                            )
                        )
                        AnalogWriter_dev2.auto_start = False

                    # Digital clock
                    if Digital_channel_number != 0:
                        slave_Task_2_digitallines.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source="ai/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )

                elif clock_source == "Camera":
                    # All the clock should refer to camera output trigger
                    self.cam_trigger_receiving_port = "/Dev1/PFI0"

                    master_Task_readin.timing.cfg_samp_clk_timing(
                        self.sampling_rate,
                        source=self.cam_trigger_receiving_port,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.Waveforms_length,
                    )
                    master_Task_readin.triggers.start_trigger.cfg_dig_edge_start_trig(
                        self.cam_trigger_receiving_port
                    )
                    master_Task_readin.export_signals.samp_clk_output_term = (
                        self.channel_LUT["clock1Channel"]
                    )  # '/Dev1/PFI1'#
                    master_Task_readin.export_signals.start_trig_output_term = self.channel_LUT[
                        "trigger1Channel"
                    ]  # '/Dev1/PFI2'

                    if Dev2_analog_channel_number != 0:
                        # By default assume that read master task is in dev1

                        for i in range(Dev2_analog_channel_number):
                            slave_Task_1_analog_dev2.ao_channels.add_ao_voltage_chan(
                                self.Dev2_analog_channel_list[i]
                            )

                        dev2Clock = self.channel_LUT[
                            "clock2Channel"
                        ]  # /Dev2/PFI1
                        slave_Task_1_analog_dev2.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source=dev2Clock,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )
                        slave_Task_1_analog_dev2.triggers.start_trigger.cfg_dig_edge_start_trig(
                            self.cam_trigger_receiving_port
                        )
                        AnalogWriter_dev2 = (
                            nidaqmx.stream_writers.AnalogMultiChannelWriter(
                                slave_Task_1_analog_dev2.out_stream,
                                auto_start=False,
                            )
                        )
                        AnalogWriter_dev2.auto_start = False

                    # === Digital clock ===
                    if Digital_channel_number != 0:
                        slave_Task_2_digitallines.timing.cfg_samp_clk_timing(
                            self.sampling_rate,
                            source=self.cam_trigger_receiving_port,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=self.Waveforms_length,
                        )
                        slave_Task_2_digitallines.triggers.start_trigger.cfg_dig_edge_start_trig(
                            self.cam_trigger_receiving_port
                        )

                # Configure the writer and reader

                if Digital_channel_number != 0:
                    DigitalWriter = (
                        nidaqmx.stream_writers.DigitalMultiChannelWriter(
                            slave_Task_2_digitallines.out_stream,
                            auto_start=False,
                        )
                    )
                    DigitalWriter.auto_start = False
                reader = AnalogMultiChannelReader(master_Task_readin.in_stream)
                reader.auto_start = False

                # === Begin to execute in DAQ ===

                if Dev2_analog_channel_number != 0:
                    AnalogWriter_dev2.write_many_sample(
                        Dev2_analog_samples_to_write, timeout=605.0
                    )

                if Digital_channel_number != 0:
                    DigitalWriter.write_many_sample_port_uint32(
                        Digital_samples_to_write, timeout=605.0
                    )

                logging.info(
                    "^^^^^^^^^^^^^^^^^^Daq tasks start^^^^^^^^^^^^^^^^^^"
                )
                if Dev2_analog_channel_number != 0:
                    slave_Task_1_analog_dev2.start()

                if Digital_channel_number != 0:
                    slave_Task_2_digitallines.start()

                master_Task_readin.start()  # !!!!!!!!!!!!!!!!!!!! READIN TASK HAS TO START AHEAD OF READ MANY SAMPLES, OTHERWISE ITS NOT SYN!!!

                reader.read_many_sample(
                    data=self.Dataholder,
                    number_of_samples_per_channel=self.Waveforms_length,
                    timeout=605.0,
                )
                # self.data_PMT = []

                if Dev2_analog_channel_number != 0:
                    slave_Task_1_analog_dev2.wait_until_done()
                if Digital_channel_number != 0:
                    slave_Task_2_digitallines.wait_until_done()
                master_Task_readin.wait_until_done()

                if Dev2_analog_channel_number != 0:
                    slave_Task_1_analog_dev2.stop()
                if Digital_channel_number != 0:
                    slave_Task_2_digitallines.stop()
                master_Task_readin.stop()

                if self.has_recording_channel is True:
                    self.collected_data.emit(self.Dataholder)
                self.finishSignal.emit()
                logging.info(
                    "^^^^^^^^^^^^^^^^^^Daq tasks finish^^^^^^^^^^^^^^^^^^"
                )

            # set the keys of galvos back for next round
            if Analog_channel_number != 0:
                for i in range(len(analog_signals["Specification"])):
                    if "galvosx" in analog_signals["Specification"][i]:
                        analog_signals["Specification"][
                            i
                        ] = self.galvosx_originalkey
                    elif "galvosy" in analog_signals["Specification"][i]:
                        analog_signals["Specification"][
                            i
                        ] = self.galvosy_originalkey

            """
            # Only digital signals
            """
        elif self.Only_Digital_signals is True:
            # some preparations for digital lines
            Waveforms_length = len(digital_signals["Waveform"][0])

            digitalsignalslinenumber = len(
                digital_signals["Waveform"]
            )  # TODO unused

            # Assume that dev1 is always employed
            with nidaqmx.Task() as slave_Task_2_digitallines:
                # adding channels
                # Set tasks from different devices apart
                slave_Task_2_digitallines.do_channels.add_do_chan(
                    "/Dev1/port0",
                    line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
                )
                # Digital clock
                slave_Task_2_digitallines.timing.cfg_samp_clk_timing(
                    self.sampling_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=Waveforms_length,
                )

                # Configure the writer and reader
                DigitalWriter = (
                    nidaqmx.stream_writers.DigitalMultiChannelWriter(
                        slave_Task_2_digitallines.out_stream, auto_start=False
                    )
                )
                DigitalWriter.auto_start = False

                # === Begin to execute in DAQ ===
                logging.info(
                    "^^^^^^^^^^^^^^^^^^Daq tasks start^^^^^^^^^^^^^^^^^^"
                )
                DigitalWriter.write_many_sample_port_uint32(
                    Digital_samples_to_write, timeout=605.0
                )

                slave_Task_2_digitallines.start()

                slave_Task_2_digitallines.wait_until_done(timeout=605.0)

                slave_Task_2_digitallines.stop()
                logging.info(
                    "^^^^^^^^^^^^^^^^^^Daq tasks finish^^^^^^^^^^^^^^^^^^"
                )

    def get_raw_data(self):
        return self.Dataholder

    def save_as_binary(self, directory):
        # print(self.ai_dev_scaling_coeff_vp)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.has_recording_channel is True:
            if "Vp" in self.readin_channels:
                if "PMT" not in self.readin_channels:
                    self.binaryfile_vp_data = np.concatenate(
                        (
                            np.array([self.sampling_rate]),
                            np.array(self.ai_dev_scaling_coeff_vp),
                            self.Dataholder[0, :],
                        )
                    )
                    np.save(
                        os.path.join(
                            directory,
                            "Vp" + timestamp,
                        ),
                        self.binaryfile_vp_data,
                    )

                    if "Ip" in self.readin_channels:
                        self.binaryfile_Ip_data = np.concatenate(
                            (
                                np.array([self.sampling_rate]),
                                np.array(self.ai_dev_scaling_coeff_ip),
                                self.Dataholder[1, :],
                            )
                        )
                        np.save(
                            os.path.join(
                                directory,
                                "Ip" + timestamp,
                            ),
                            self.binaryfile_Ip_data,
                        )
                else:
                    self.binaryfile_vp_data = np.concatenate(
                        (
                            np.array([self.sampling_rate]),
                            np.array(self.ai_dev_scaling_coeff_vp),
                            self.Dataholder[1, :],
                        )
                    )
                    np.save(
                        os.path.join(
                            directory,
                            "Vp" + timestamp,
                        ),
                        self.binaryfile_vp_data,
                    )

                    if "Ip" in self.readin_channels:
                        self.binaryfile_Ip_data = np.concatenate(
                            (
                                np.array([self.sampling_rate]),
                                np.array(self.ai_dev_scaling_coeff_ip),
                                self.Dataholder[2, :],
                            )
                        )
                        np.save(
                            os.path.join(
                                directory,
                                "Ip" + timestamp,
                            ),
                            self.binaryfile_Ip_data,
                        )

            if "Ip" in self.readin_channels:
                self.binaryfile_Ip_data = np.concatenate(
                    (
                        np.array([self.sampling_rate]),
                        np.array(self.ai_dev_scaling_coeff_ip),
                        self.Dataholder[0, :],
                    )
                )
                np.save(
                    os.path.join(
                        directory,
                        "Ip" + timestamp,
                    ),
                    self.binaryfile_Ip_data,
                )

            if "PMT" in self.readin_channels:
                self.data_PMT = self.Dataholder[0, :] * -1
                np.save(
                    os.path.join(
                        directory,
                        "PMT_array_" + timestamp,
                    ),
                    self.data_PMT,
                )


# self.pmtimage = Image.fromarray(self.data_PMT) #generate an image object
# self.pmtimage.save(os.path.join(directory, 'PMT'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.tif')) #save as tif


if __name__ == "__main__":
    daq = DAQmission()
    # daq.sendSingleDigital('DMD_trigger', False)
    # daq.sendSingleDigital('LED', False)
    daq.sendSingleAnalog("galvosx", 0)
    daq.sendSingleAnalog("galvosy", 0)
