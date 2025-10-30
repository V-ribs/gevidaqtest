import re
from datetime import datetime

import numpy as np

WAVEFORM_SR_RX = re.compile(r"(.*)Wavef(or|ro)ms_sr_(\d+)\.npy")


def make_dtype(length, type_=bool):
    return np.dtype(
        [
            ("Waveform", type_, (length,)),
            ("Specification", "<U20"),
        ]
    )


def fix_sepcification(array):
    """fix misspelling of Specification in array"""
    new_array = np.empty(len(array), dtype="O")
    for i, item in enumerate(array):
        try:
            channel_keyword = item["Sepcification"]
        except KeyError:
            return array  # array contents are not misspelled

        waveform = item["Waveform"]
        dtype = make_dtype(len(waveform), waveform.dtype)
        # this is not the same as
        # np.array((waveform, channel_keyword), dtype=dtype)
        # I don't know why
        new_array[i] = np.array([(waveform, channel_keyword)], dtype=dtype)[0]

    return new_array


def is_waveform(filename):
    """check if filename could be a waveform file"""
    return WAVEFORM_SR_RX.match(filename) is not None


def is_misspelled_wavefrom(filename):
    """check if filename is misspelled as wavefrom instead of waveform"""
    match = WAVEFORM_SR_RX.match(filename)
    if match is None:
        raise ValueError(
            f"{filename} is not a waveform specification filename"
        )

    return match.group(2) == "ro"


def get_sample_rate(filename):
    """get sample rate from waveform filename"""
    match = WAVEFORM_SR_RX.match(filename)
    if match is None:
        raise ValueError(
            f"{filename} is not a waveform specification filename"
        )

    return int(float(match.group(3)))


def load(filename):
    return fix_sepcification(np.load(filename, allow_pickle=True))


def create_filename(prefix, sample_rate):
    timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestr}_{prefix}_Waveforms_sr_{int(sample_rate)}"
