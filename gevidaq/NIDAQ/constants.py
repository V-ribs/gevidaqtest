# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:54:58 2019

@author: lhuismans

This class contains all the constants that are frequently used and will stay unchanged
(for a long time).
"""


class MeasurementConstants:
    def __init__(self):
        # Seal test
        self.patchSealSampRate = 10000  # Samples/s
        self.patchSealMinVol = 0
        self.patchSealMaxVol = 0.01
        self.patchSealFreq = 100
        self.patchSealDuty = 0.5


class HardwareConstants:
    def __init__(self):
        self.maxGalvoSpeed = 20000.0  # Volt/s
        self.maxGalvoAccel = 1.54 * 10**8  # Acceleration galvo in volt/s^2

        self.pmt_3v_indentation_pixels = 52


class NiDaqChannels:
    def __init__(self):
        # Ports specification
        self.look_up_table = {
            "galvosx": "Dev1/ao0",
            "galvosy": "Dev1/ao1",
            "640AO": "Dev1/ao3",
            "488AO": "Dev2/ao1",
            "532AO": "Dev2/ao0",
            "patchAO": "Dev2/ao2",  # Output to patch clamp probe, patchVoltInChannel
            "fieldstimulator": "Dev2/ao3",  # Output to fieldstimulator electrodes
            "cameratrigger": "Dev1/port0/line25",
            "galvotrigger": "Dev1/port0/line25",
            "blankingall": "Dev1/port0/line4",
            "640blanking": "Dev1/port0/line4",
            "532blanking": "Dev1/port0/line6",
            "488blanking": "Dev1/port0/line3",
            "DMD_trigger": "Dev1/port0/line0",
            "PMT": "Dev1/ai0",
            "Vp": "Dev1/ai20",  # patchVoltOutChannel Vp and Ip are from the same channel(22.12.2021)
            "VpPatch": "Dev1/ai22",  # For sealtest only
            "Ip": "Dev1/ai20",  # patchCurOutChannel
            "Perfusion_8": "Dev1/port0/line21",  # line21 is perfusion_8, set to 19-LED for test
            "LED": "Dev1/port0/line19",  # Need to assign new port.
            "Perfusion_7": "Dev1/port0/line22",
            "Perfusion_6": "Dev1/port0/line23",
            "Perfusion_2": "Dev1/port0/line24",
            "2Pshutter": "Dev1/port0/line18",
            "servo_modulation_1": "Dev1/port0/line11",
            "clock1Channel": "/Dev1/PFI1",
            "trigger1Channel": "/Dev1/PFI2",
            "clock2Channel": "/Dev2/PFI1",
            "trigger2Channel": "/Dev2/PFI7",
            "532 servo": "/Dev1/port0/line8",
            "640 servo": "/Dev1/port0/line9",
            "488 servo": "/Dev1/port0/line10",
        }

        # self.patchVoltOutChannel = "Dev1/ai22"
        # self.patchCurOutChannel = "Dev1/ai20"
        # self.patchVoltInChannel = 'Dev2/ao2'
