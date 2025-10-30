import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from . import waveform_specification
from .constants import HardwareConstants


def xValuesSingleSawtooth(
    sampleRate=1000, voltXMin=0, voltXMax=5, xPixels=1024, sawtooth=True
):
    """
    This function generates the xValues for !!!ONE PERIOD!!! of the sawtooth/triangle wave.

    First part: linearly moving up
    Second part: accelerating to the rampdown speed (maximum galvo speed for sawtooth)
    Third part: linearly moving down
    Fourth part: accelerating to rampup speed
    """
    # === Defining standard variables ===
    constants = HardwareConstants()
    speedGalvo = constants.maxGalvoSpeed  # Volt/s
    aGalvo = constants.maxGalvoAccel  # Acceleration galvo in volt/s^2
    aGalvoPix = aGalvo / (sampleRate**2)  # Acceleration galvo in volt/pixel^2
    xArray = np.array([])  # Array for x voltages
    rampUpSpeed = (
        voltXMax - voltXMin
    ) / xPixels  # Ramp up speed in volt/pixel
    rampDownSpeed = (
        -speedGalvo / sampleRate
    )  # Ramp down speed in volt/pixel (Default sawtooth)

    # === Checking for triangle wave ===
    if sawtooth is False:
        rampDownSpeed = -rampUpSpeed

    # === x pixel wave function ===

    # === Defining the ramp up (x) ===
    rampUp = np.linspace(voltXMin, voltXMax, xPixels)
    xArray = np.append(
        xArray, rampUp
    )  # Adding the voltage values for the ramp up

    # === Defining the inertial part ===
    inertialPart = np.array(
        []
    )  # Making a temporary array for storing the voltage values of the inertial part
    vIn = rampUpSpeed  # Speed of "incoming" ramp (volt/pixel)
    vOut = rampDownSpeed  # Speed of "outgoing" ramp (volt/pixel)
    a = -aGalvoPix  # Acceleration in volt/pixel^2
    timespanInertial = abs(
        math.floor((vOut - vIn) / a)
    )  # Calculating the timespan needed
    t = np.arange(timespanInertial)
    inertialPart = (
        0.5 * a * t[1::] ** 2 + vIn * t[1::] + xArray[-1]
    )  # Making the array with the voltage values, we are not taking into acount the first value as this is the value of the previous sample
    xArray = np.append(
        xArray, inertialPart
    )  # Adding the array to the total path

    if sawtooth is False:
        lineSizeStepFunction = (
            xArray.size
        )  # Defining the linesize for the yArray in case of a triangle wave

    # === Defining the ramp down ===
    a = aGalvoPix
    startVoltage = xArray[-1] + rampDownSpeed
    # We calculate the endvoltage by using the timespan for the intertial part and
    # the starting voltage
    endVoltage = (
        0.5 * a * timespanInertial**2
        - rampUpSpeed * timespanInertial
        + voltXMin
    )

    if sawtooth is True:
        timespanRampDown = abs(
            math.ceil((endVoltage - startVoltage) / rampDownSpeed)
        )
        rampDownSpeed = (
            endVoltage - startVoltage
        ) / timespanRampDown  # Above line changed the rampDownSpeed so we have to recalculate
    else:
        timespanRampDown = (
            rampUp.size
        )  # If it is a triangle wave the ramp down part should be as big as the ramp up part

    rampDown = np.linspace(
        startVoltage, endVoltage, timespanRampDown
    )  # Specifying the linear path
    xArray = np.append(xArray, rampDown)  # Adding the array to the total path

    # === Defining the second inertial part ===
    inertialPart2 = np.array([])
    vIn = rampDownSpeed  # Speed of "incoming" ramp (volt/pixel)
    a = aGalvoPix  # Acceleration in volt/pixel^2
    inertialPart2 = (
        0.5 * a * t[1::] ** 2 + vIn * t[1::] + xArray[-1]
    )  # We can use the same time units as the first inertial part but not including the last value, as this is part of the next iteration
    xArray = np.append(xArray, inertialPart2)

    if sawtooth is True:
        lineSizeStepFunction = xArray.size

    return xArray, lineSizeStepFunction


def yValuesFullSawtooth(
    sampleRate, voltYMin, voltYMax, xPixels, yPixels, lineSize
):
    """
    This functiong generates the !!!FULL!!! yArray (stepfunction) for the sawtooth or triangle wave.

    lineSize defines the length of each step.
    For the trianglewave this is ~half the wavelength and for the sawtooth it is
    the full wavelength.
    """
    stepSize = (voltYMax - voltYMin) / yPixels

    # Creating the 'stairs'
    extendedYArray = (
        np.ones(xPixels) * voltYMin
    )  # The first line is created manually as this is shorter
    # The step is starting at the beginning of the intertial part
    for i in np.arange(yPixels - 1) + 1:
        extendedYArray = np.append(
            extendedYArray, np.ones(lineSize) * i * stepSize + voltYMin
        )

    extraPixels = (
        lineSize * yPixels - extendedYArray.size
    )  # Some extra pixels are needed to make x and y the same size
    extendedYArray = np.append(extendedYArray, np.ones(extraPixels) * voltYMin)

    return extendedYArray


def rotateXandY(
    xArray, yArray, voltXMin, voltXMax, voltYMin, voltYMax, imAngle
):
    """
    Rotates x and corresponding y array for galvos around its center point.
    """
    radAngle = math.pi / 180 * imAngle  # Converting degrees to radians

    # Shifting to the center
    xArray = xArray - ((voltXMax - voltXMin) / 2 + voltXMin)
    yArray = yArray - ((voltYMax - voltYMin) / 2 + voltYMin)

    # Converting the x and y arrays
    rotatedXArray = xArray * math.cos(radAngle) - yArray * math.sin(radAngle)
    rotatedYArray = xArray * math.sin(radAngle) + yArray * math.cos(radAngle)

    # Shifting it back
    finalXArray = rotatedXArray + ((voltXMax - voltXMin) / 2 + voltXMin)
    finalYArray = rotatedYArray + ((voltYMax - voltYMin) / 2 + voltYMin)

    return finalXArray, finalYArray


def repeatWave(wave, repeats):
    """
    Repeats the wave a set number of times and returns a new repeated wave.
    """
    extendedWave = np.array([])
    for i in range(repeats):
        extendedWave = np.append(extendedWave, wave)
    return extendedWave


def waveRecPic(
    sampleRate=4000,
    imAngle=0,
    voltXMin=0,
    voltXMax=5,
    voltYMin=0,
    voltYMax=5,
    xPixels=1024,
    yPixels=512,
    sawtooth=True,
):
    """
    Generates the x and y values for making rectangular picture with a scanning laser.
    """
    xArray, lineSize = xValuesSingleSawtooth(
        sampleRate, voltXMin, voltXMax, xPixels, sawtooth
    )
    yArray = yValuesFullSawtooth(
        sampleRate, voltYMin, voltYMax, xPixels, yPixels, lineSize
    )

    # Looping it to get the desired amount of periods for x
    if sawtooth is True:
        extendedXArray = repeatWave(xArray, yPixels)
    else:
        repeats = int(math.ceil(yPixels / 2))
        extendedXArray = repeatWave(xArray, repeats)

        # Checking if we should remove the last ramp down
        if yPixels % 2 == 1:
            extendedXArray = extendedXArray[0:-lineSize]

    # Rotatin
    finalX, finalY = rotateXandY(
        extendedXArray, yArray, voltXMin, voltXMax, voltYMin, voltYMax, imAngle
    )
    return finalX, finalY


def blockWave(sampleRate, frequency, voltMin, voltMax, dutycycle):
    """
    Generates a one period blockwave.
    sampleRate      samplerate set on the DAQ (int)
    frequency       frequency you want for the block wave (int)
    voltMin         minimum value of the blockwave (float)
    voltMax         maximum value of the blockwave (float)
    dutycycle       duty cycle of the wave (wavelength at voltMax) (float)
    """
    wavelength = int(sampleRate / frequency)  # Wavelength in number of samples
    # The high values
    high = np.ones(math.ceil(wavelength * dutycycle)) * voltMax
    # Low values
    low = np.ones(math.floor(wavelength * (1 - dutycycle))) * voltMin
    # Adding them
    return np.append(high, low)


def testSawtooth():
    sRate = 2000000
    imAngle = 0
    VxMax = 4
    VxMin = 0.0
    VyMax = 10
    VyMin = 2
    xPixels = 1024
    yPixels = 1
    sawtooth = True

    xValues, yValues = waveRecPic(
        sRate, imAngle, VxMin, VxMax, VyMin, VyMax, xPixels, yPixels, sawtooth
    )

    plt.plot(np.arange(xValues.size), xValues)
    plt.plot(np.arange(yValues.size), yValues)
    plt.show()

    # %%


class generate_AO_for640:
    def __init__(
        self,
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
        value7,
        value8,
        value9,
        value10,
        value11,
    ):
        self.Daq_sample_rate = value1
        self.wavefrequency_2 = value2
        self.waveoffset_2 = value3
        self.waveperiod_2 = value4
        self.waveDC_2 = value5
        self.waverepeat_2 = value6
        self.wavegap_2 = value7
        self.wavestartamplitude_2 = value8
        self.wavebaseline_2 = value9
        self.wavestep_2 = value10
        self.wavecycles_2 = value11

    def generate(self):
        # Offset samples
        # By default one 0 is added so that we have a rising edge at the beginning.
        self.offsetsamples_number_2 = int(
            (self.waveoffset_2 / 1000) * self.Daq_sample_rate
        )
        self.offsetsamples_2 = self.wavebaseline_2 * np.ones(
            self.offsetsamples_number_2
        )  # Be default offsetsamples_number is an integer.

        self.sample_num_singleperiod_2 = round(
            self.Daq_sample_rate / self.wavefrequency_2
        )  # round((int((self.waveperiod_2/1000)*self.Daq_sample_rate))/self.wavefrequency_2)
        self.true_sample_num_singleperiod_2 = round(
            (self.waveDC_2 / 100) * self.sample_num_singleperiod_2
        )
        self.false_sample_num_singleperiod_2 = (
            self.sample_num_singleperiod_2
            - self.true_sample_num_singleperiod_2
        )

        self.sample_singleperiod_2 = np.append(
            self.wavestartamplitude_2
            * np.ones(self.true_sample_num_singleperiod_2),
            self.wavebaseline_2
            * np.ones(self.false_sample_num_singleperiod_2),
        )
        self.repeatnumberintotal_2 = int(
            self.wavefrequency_2 * (self.waveperiod_2 / 1000)
        )
        # In default, pulses * sample_singleperiod_2 = period
        self.sample_singlecycle_2 = np.tile(
            self.sample_singleperiod_2, int(self.repeatnumberintotal_2)
        )  # At least 1 rise and fall during one cycle

        self.waveallcycle_2 = []
        # Adding steps to cycles
        for i in range(self.wavecycles_2):
            cycle_roof_value = self.wavestep_2 * i
            self.cycleappend = np.where(
                self.sample_singlecycle_2 < self.wavestartamplitude_2,
                self.wavebaseline_2,
                self.wavestartamplitude_2 + cycle_roof_value,
            )
            self.waveallcycle_2 = np.append(
                self.waveallcycle_2, self.cycleappend
            )

        if self.wavegap_2 != 0:
            self.gapsample = self.wavebaseline_2 * np.ones(self.wavegap_2)
            self.waveallcyclewithgap_2 = np.append(
                self.waveallcycle_2, self.gapsample
            )
        else:
            self.waveallcyclewithgap_2 = self.waveallcycle_2

        self.waverepeated = np.tile(
            self.waveallcyclewithgap_2, self.waverepeat_2
        )

        AO_output = np.append(self.offsetsamples_2, self.waverepeated)

        return AO_output


class generate_digital_waveform:
    def __init__(
        self, value1, value2, value3, value4, value5, value6, value7, value8
    ):
        self.Daq_sample_rate = value1
        self.wavefrequency = value2
        self.waveoffset = value3
        self.waveperiod = value4
        self.waveDC = value5
        self.waverepeat = value6
        self.wavegap = value7
        self.fiveVolts = value8

    def generate(self):
        self.offsetsamples_number = int(
            (self.waveoffset / 1000) * self.Daq_sample_rate
        )  # By default one 0 is added so that we have a rising edge at the beginning.
        self.offsetsamples = np.zeros(
            self.offsetsamples_number, dtype=bool
        )  # Be default offsetsamples_number is an integer.

        self.sample_num_singleperiod = round(
            self.Daq_sample_rate / self.wavefrequency
        )
        self.true_sample_num_singleperiod = round(
            (self.waveDC / 100) * self.sample_num_singleperiod
        )
        self.false_sample_num_singleperiod = (
            self.sample_num_singleperiod - self.true_sample_num_singleperiod
        )

        self.sample_singleperiod = np.append(
            np.ones(self.true_sample_num_singleperiod, dtype=bool),
            np.zeros(self.false_sample_num_singleperiod, dtype=bool),
        )
        self.repeatnumberintotal = int(
            self.wavefrequency * (self.waveperiod / 1000)
        )
        # In default, pulses * sample_singleperiod_2 = period
        self.sample_singlecycle = np.tile(
            self.sample_singleperiod, int(self.repeatnumberintotal)
        )  # At least 1 rise and fall during one cycle

        if self.wavegap != 0:
            self.gapsample = np.zeros(self.wavegap, dtype=bool)
            self.waveallcyclewithgap = np.append(
                self.sample_singlecycle, self.gapsample
            )
        else:
            self.waveallcyclewithgap = self.sample_singlecycle

        self.waverepeated = np.tile(self.waveallcyclewithgap, self.waverepeat)

        # Append a False in the end
        self.finalwave = np.append(self.offsetsamples, self.waverepeated)
        # self.finalwave = np.append(self.finalwave, False)

        if self.fiveVolts:
            self.finalwave = 5 * self.finalwave
        return self.finalwave


class generate_ramp:
    def __init__(
        self,
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
        value7,
        value8,
        value9,
        value10,
        value11,
    ):
        self.Daq_sample_rate = value1
        self.wavefrequency = value2
        self.waveoffset = value3
        self.waveperiod = value4
        self.wavesymmetry = value5
        self.waverepeat = value6
        self.wavegap = value7
        self.waveheight = value8
        self.wavebaseline = value9
        self.wavestep = value10
        self.wavecycles = value11

    def generate(self):
        self.offsetsamples_number_ramp = int(
            1 + (self.waveoffset / 1000) * self.Daq_sample_rate
        )  # By default one 0 is added
        self.offsetsamples_ramp = self.wavebaseline * np.ones(
            self.offsetsamples_number_ramp
        )  # Be default offsetsamples_number is an integer.

        t = np.linspace(
            0,
            (self.waveperiod / 1000),
            int(self.Daq_sample_rate * (self.waveperiod / 1000)),
        )
        triangle_in_1s = (
            self.waveheight
            / 2
            * (
                signal.sawtooth(
                    2 * np.pi * self.wavefrequency * t, self.wavesymmetry
                )
            )
        )
        self.sample_singlecycle_ramp = (
            triangle_in_1s + self.waveheight / 2 + self.wavebaseline
        )

        if self.wavegap != 0:
            self.gapsample = self.wavebaseline * np.ones(self.wavegap)
            self.waveallcyclewithgap_ramp = np.append(
                self.sample_singlecycle_ramp, self.gapsample
            )
        else:
            self.waveallcyclewithgap_ramp = self.sample_singlecycle_ramp

        self.waverepeated = np.tile(
            self.waveallcyclewithgap_ramp, self.waverepeat
        )

        self.finalwave_ramp = np.append(
            self.offsetsamples_ramp, self.waverepeated
        )
        self.finalwave_ramp = np.append(self.finalwave_ramp, 0)
        return self.finalwave_ramp

    # %%


class generate_AO:
    def __init__(
        self,
        Daq_sample_rate,
        wavefrequency_2,
        waveoffset_2,
        waveperiod_2,
        waveDC_2,
        waverepeat_2,
        wavegap_2,
        wavestartamplitude_2,
        wavebaseline_2,
        wavestep_2,
        wavecycles_2,
        start_time_2,
        control_amp2,
    ):
        self.Daq_sample_rate = Daq_sample_rate
        self.wavefrequency_2 = wavefrequency_2
        self.waveoffset_2 = waveoffset_2
        self.waveperiod_2 = waveperiod_2
        self.waveDC_2 = waveDC_2
        self.waverepeat_2 = waverepeat_2
        self.wavegap_2 = wavegap_2
        self.wavestartamplitude_2 = wavestartamplitude_2
        self.wavebaseline_2 = wavebaseline_2
        self.wavestep_2 = wavestep_2
        self.wavecycles_2 = wavecycles_2
        self.start_time_2 = start_time_2
        self.controlamp_2 = control_amp2

    def generate(self):
        def rect(T):
            """create a centered rectangular pulse of width $T"""
            return lambda t: (-T / 2 <= t) & (t < T / 2)

        def pulse_train(t, at, shape):
            """create a train of pulses over $t at times $at and shape $shape"""
            return np.sum(shape(t - at[:, np.newaxis]), axis=0)

        self.shape = 0.5 * (1 / self.wavefrequency_2)
        sig = pulse_train(
            t=np.arange(
                (self.waveperiod_2 / 1000) * self.Daq_sample_rate
            ),  # time domain
            at=np.array(
                [self.start_time_2 * self.Daq_sample_rate]
            ),  # times of pulses
            shape=rect(self.shape * self.Daq_sample_rate),  # shape of pulse
        )

        sig = self.wavestep_2 * sig
        number = 1
        sigdouble = []
        # print(sig)
        while number <= self.waverepeat_2:
            number = number + 1
            # print(number)
            sigdouble = (
                number
                * sig[
                    0 : (int(self.waveperiod_2 / 1000) * self.Daq_sample_rate)
                ]
            )
            sig = np.append(sig, sigdouble)
            # print(sig)

        # define the control signal
        self.time_control = self.waveperiod_2 / 1000
        at_control = []
        while self.time_control < (
            (self.waveperiod_2 / 1000) * self.waverepeat_2
        ):
            self.time_control = self.time_control + (self.waveperiod_2 / 1000)
            self.time_control_2 = self.time_control * self.Daq_sample_rate
            at_control.append(self.time_control_2)

        sig2 = pulse_train(
            t=np.arange(
                (self.waverepeat_2 + 1)
                * (self.waveperiod_2 / 1000)
                * self.Daq_sample_rate
            ),  # time domain
            at=np.array(
                [
                    1 * self.Daq_sample_rate,
                    2 * self.Daq_sample_rate,
                    3 * self.Daq_sample_rate,
                    4 * self.Daq_sample_rate,
                    5 * self.Daq_sample_rate,
                    6 * self.Daq_sample_rate,
                    7 * self.Daq_sample_rate,
                    8 * self.Daq_sample_rate,
                    9 * self.Daq_sample_rate,
                    10 * self.Daq_sample_rate,
                    11 * self.Daq_sample_rate,
                    12 * self.Daq_sample_rate,
                    13 * self.Daq_sample_rate,
                    14 * self.Daq_sample_rate,
                    15 * self.Daq_sample_rate,
                ]
            ),  # times of pulses
            shape=rect(self.shape * self.Daq_sample_rate),  # shape of pulse
        )

        sig2 = self.controlamp_2 * sig2

        self.finalwave_ = sig + sig2
        return self.finalwave_


#
# Dark probe code
#


def dark_probe(
    samplingrate,
    cameraframerate,
    dt_min,
    dt_max,
    dt_step,
    flash=True,
    t_start=1.0,
    t_flash=4.0,
    t_dark=5.0,
    t_on=15.0,
    t_probe=0.5,
):
    savedirectory = "M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/Marco/phd/2021-04-08 NovArch dark probe/"  # TODO hardcoded path

    n = int((dt_max - dt_min) / dt_step) + 1
    Dt = np.linspace(dt_min, dt_max, n)

    T = (
        t_start
        + flash * (t_flash + t_dark)
        + n * (t_on + t_dark + t_probe)
        + np.sum(Dt)
    )

    samples = int(samplingrate * T)
    sig = np.zeros([samples])
    camera_signal = np.zeros([samples])

    t_0 = t_start
    if flash:
        a = int(t_start * samplingrate)
        b = int((t_start + t_flash) * samplingrate)
        sig[a:b] = 1
        t_0 = t_0 + t_flash + t_dark

    for dt in Dt:
        a = int(t_0 * samplingrate)
        b = int((t_0 + t_on) * samplingrate)
        camera_start = int((t_0 + t_on) * samplingrate - 5)
        sig[a:b] = 1
        t_0 = t_0 + t_on + dt

        c = int(t_0 * samplingrate)
        d = int((t_0 + t_probe) * samplingrate)
        camera_stop = int((t_0 + t_probe) * samplingrate + 5)
        sig[c:d] = 1
        t_0 = t_0 + t_probe + t_dark

        camera_signal[camera_start:camera_stop] = 1

    sig = 5 * sig

    x = np.linspace(0, T, len(sig))
    squarewave = 0.5 * (signal.square(2 * np.pi * cameraframerate * x) + 1)
    camera_signal = camera_signal * squarewave

    # count=0
    # for i in range(samples-1):
    # if camera_signal[i+1]>camera_signal[i]:
    # count=count+1○
    # print(count)

    # plt.plot(x,sig)
    # plt.plot(x,camera_signal)
    # plt.show()

    dataType_analog = waveform_specification.make_dtype(len(sig), float)
    dataType_digital = waveform_specification.make_dtype(
        len(camera_signal), bool
    )
    analog_array = np.zeros(1, dtype=dataType_analog)
    digital_array = np.zeros(1, dtype=dataType_digital)
    analog_array[0] = np.array([(sig, "640AO")], dtype=dataType_analog)
    digital_array[0] = np.array(
        [(camera_signal, "cameratrigger")], dtype=dataType_digital
    )

    ciao = []  # Variable name 'ciao' was defined by Nicolo Ceffa.
    ciao.append(analog_array[0])
    ciao.append(digital_array[0])

    np.save(
        savedirectory
        + "DarkProbe_min"
        + format(dt_min).replace(".", "")
        + "-max"
        + format(dt_max).replace(".", "")
        + "-step"
        + format(dt_step).replace(".", "")
        + "_cr_"
        + format(cameraframerate)
        + "_sr_"
        + format(samplingrate),
        ciao,
    )

    return True
