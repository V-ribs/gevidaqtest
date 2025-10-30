"""global settings for NIDAQ"""

from ..Settings.settings import settings

defaults = {
    "Measurement": {
        # Seal test
        "patchSealSampRate": 10000,  # Samples/s
        "patchSealMinVol": 0,
        "patchSealMaxVol": 0.01,
        "patchSealFreq": 100,
        "patchSealDuty": 0.5,
    },
    "Hardware": {
        "maxGalvoSpeed": 20000.0,  # Volt/s
        "maxGalvoAccel": 1.54 * 10**8,  # Acceleration galvo in volt/s^2
        "pmt_3v_indentation_pixels": 52,
    },
    "Channels":
    # Ports specification
    {
        "galvosx": "Dev1/ao0",
        "galvosy": "Dev1/ao1",
        "640AO": "Dev1/ao3",
        "488AO": "Dev2/ao1",
        "532AO": "Dev2/ao0",
        "patchAO": "Dev2/ao2",  # Output to patch clamp probe, patchVoltInChannel
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
    },
    # self.patchVoltOutChannel = "Dev1/ai22"
    # self.patchCurOutChannel = "Dev1/ai20"
    # self.patchVoltInChannel = 'Dev2/ao2'
}

NIDAQSettings = settings().add("NIDAQ", defaults)
