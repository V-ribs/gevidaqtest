#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:14:53 2020

@author: xinmeng
"""
import ctypes
import importlib.resources
import logging
import sys
import threading
import time

import numpy as np
import tifffile as skimtiff

from . import HamamatsuDCAM

# Script based Hamamatsu camera operations


class CamActuator:
    """
    This is a script based operation class for the HamamatsuDCAM which is a ctype based dll wrapper.

    Frequent used parameters:
        params = ["internal_frame_rate",
                  "timing_readout_time",
                  "exposure_time",
                  "subarray_hsize",
                  "subarray_hpos",
                  "subarray_vsize",
                  "subarray_vpos",
                  "subarray_mode",
                  "image_framebytes",
                  "buffer_framebytes",
                  "trigger_source",
                  "trigger_active"]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        # Initialization of the camera.
        # Load dcamapi.dll version: 19.12.641.5901
        """
        self.isLiving = False
        self.isStreaming = False
        self.isSaving = False
        self.metaData = "Hamamatsu C13440-20CU "

    def initializeCamera(self):
        # Initialize the camera
        # Set default camera properties.
        files = importlib.resources.files(sys.modules[__package__])
        traversable = files.joinpath("19_12/dcamapi.dll")
        with importlib.resources.as_file(traversable) as path:
            self.dcam = ctypes.WinDLL(str(path))

        paraminit = HamamatsuDCAM.DCAMAPI_INIT(0, 0, 0, 0, None, None)
        paraminit.size = ctypes.sizeof(paraminit)
        error_code = self.dcam.dcamapi_init(
            ctypes.byref(paraminit)
        )  # TODO unused
        # if (error_code != DCAMERR_NOERROR):
        # raise DCAMException("DCAM initialization failed with error code " + str(error_code))

        n_cameras = paraminit.iDeviceCount
        logging.info(f"found: {n_cameras} cameras")

        if n_cameras > 0:
            # === Initialization ===
            self.hcam = HamamatsuDCAM.HamamatsuCameraMR(camera_id=0)

            # Enable defect correction
            self.hcam.setPropertyValue("defect_correct_mode", 2)
            # Set the readout speed to fast.
            self.hcam.setPropertyValue("readout_speed", 2)
            # Set the binning to 1.
            self.hcam.setPropertyValue("binning", "1x1")

            self.GetKeyCameraProperties()

    def GetKeyCameraProperties(self):
        params = [
            "internal_frame_rate",
            "timing_readout_time",
            "exposure_time",
            "subarray_hsize",
            "subarray_hpos",
            "subarray_vsize",
            "subarray_vpos",
            "subarray_mode",
            "image_framebytes",
            "buffer_framebytes",
            "trigger_source",
            "trigger_active",
        ]

        self.metaData = "Hamamatsu C13440-20CU "

        for param in params:
            if param == "exposure_time":
                self.exposure_time = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_exposure_time" + str(self.exposure_time)
            if param == "subarray_hsize":
                self.subarray_hsize = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_subarray_hsize" + str(self.subarray_hsize)
            if param == "subarray_hpos":
                self.subarray_hpos = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_subarray_hpos" + str(self.subarray_hpos)
            if param == "subarray_vsize":
                self.subarray_vsize = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_subarray_vsize" + str(self.subarray_vsize)
            if param == "subarray_vpos":
                self.subarray_vpos = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_subarray_vpos" + str(self.subarray_vpos)
            if param == "internal_frame_rate":
                self.internal_frame_rate = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_internal_frame_rate" + str(
                    self.internal_frame_rate
                )
            if param == "image_framebytes":
                self.image_framebytes = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_image_framebytes" + str(
                    self.image_framebytes
                )
            if param == "buffer_framebytes":
                self.buffer_framebytes = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_buffer_framebytes" + str(
                    self.buffer_framebytes
                )
            if param == "timing_readout_time":
                self.timing_readout_time = self.hcam.getPropertyValue(param)[0]
                self.metaData += "_timing_readout_time" + str(
                    self.timing_readout_time
                )

    def setROI(self, ROI_vpos, ROI_hpos, ROI_vsize, ROI_hsize):
        # Set the roi of caamera, first the roi poitions and then the size.
        if ROI_hsize == 2048 and ROI_vsize == 2048:
            self.hcam.setPropertyValue("subarray_mode", "OFF")

        else:
            # set subarray mode off. This setting is not mandatory, but you have to control the setting order of offset and size when mode is on.
            self.hcam.setPropertyValue("subarray_mode", "OFF")
            self.hcam.setPropertyValue("subarray_hsize", ROI_hsize)
            self.hcam.setPropertyValue("subarray_vsize", ROI_vsize)
            self.hcam.setPropertyValue("subarray_hpos", ROI_hpos)
            self.hcam.setPropertyValue("subarray_vpos", ROI_vpos)
            self.hcam.setPropertyValue("subarray_mode", "ON")

    def SnapImage(self, exposure_time):
        # Snap and return captured image.
        # - exposure_time: Exposure time of the camera.
        self.hcam.setPropertyValue("trigger_source", "INTERNAL")
        self.hcam.setPropertyValue("exposure_time", exposure_time)

        self.hcam.setACQMode("fixed_length", number_frames=1)
        # Get propreties and stored as metadata
        self.GetKeyCameraProperties()
        self.hcam.startAcquisition()

        # Wait a little while, otherwise for large exposure times there is not
        # yet anything in the buffer.

        time.sleep(exposure_time)

        # Start pulling out frames from buffer
        video_list = []
        imageCount = 0  # The actual frame number that gets recorded.
        for _ in range(1):  # Record for range() number of images.
            [
                frames,
                dims,
            ] = (
                self.hcam.getFrames()
            )  # frames is a list with HCamData type, with np_array being the image.
            for aframe in frames:
                video_list.append(aframe.np_array)
                imageCount += 1

        if len(video_list) > 1:
            ImageSnapped = np.resize(video_list[-1], (dims[1], dims[0]))
        else:
            ImageSnapped = np.resize(video_list[0], (dims[1], dims[0]))

        self.hcam.stopAcquisition()

        return ImageSnapped

    def LIVE(self):
        # Start the continuous stream
        self.isLiving = True
        self.hcam.acquisition_mode = "run_till_abort"
        self.hcam.startAcquisition()

        while self.isLiving is True:
            [
                frames,
                dims,
            ] = (
                self.hcam.getFrames()
            )  # frames is a list with HCamData type, with np_array being the image.
            self.Live_image = np.resize(
                frames[-1].np_array, (dims[1], dims[0])
            )

            self.subarray_vsize = dims[1]
            self.subarray_hsize = dims[0]

    def StopLIVE(self):
        self.isLiving = False
        # Stop the acquisition
        self.hcam.stopAcquisition()

    def StartStreaming(self, BufferNumber, **kwargs):
        # Start the camera video streaming.
        # - trigger_source: specify the camera trigger mode.
        # - BufferNumber: number of frames assigned for video.
        # - **kwargs can be set as camera property name and desired value pairs,
        # like: trigger_active = "SYNCREADOUT"

        # Set extra input settings
        for camProName, value in kwargs.items():
            self.hcam.setPropertyValue(camProName, value)
            logging.info(
                "setProperty {} to Value {}".format(camProName, value)
            )

        # Start the acquisition
        self.hcam.setACQMode("fixed_length", number_frames=BufferNumber)
        # Get propreties and stored as metadata
        self.GetKeyCameraProperties()
        self.hcam.startAcquisition()
        self.isStreaming = True

        self.getFrames_Thread = threading.Thread(
            target=self.pullFrames, args=(BufferNumber,)
        )
        self.getFrames_Thread.start()

    def pullFrames(self, BufferNumber):
        # Start pulling out frames from buffer
        self.video_list = []
        self.imageCount = 0  # The actual frame number that gets recorded.
        while self.isStreaming is True:  # Record for range() number of images.
            [
                frames,
                self.dims,
            ] = (
                self.hcam.getFrames()
            )  # frames is a list with HCamData type, with np_array being the image.
            for aframe in frames:
                self.video_list.append(aframe.np_array)
                self.imageCount += 1

            if self.imageCount >= BufferNumber:
                self.isStreaming = False

    def StopStreaming(self, saving_dir=None):
        # Stop the streaming and save the file.
        # - saving_dir: directory in which the video is saved.
        self.isStreaming = False
        # Stop the acquisition
        self.hcam.stopAcquisition()

        if saving_dir is not None:
            self.isSaving = True
            # Save the file.
            with skimtiff.TiffWriter(saving_dir, append=True) as tif:
                for eachframe in range(self.imageCount):
                    image = np.resize(
                        self.video_list[eachframe],
                        (self.dims[1], self.dims[0]),
                    )
                    tif.save(image, compress=0, description=self.metaData)
        self.isSaving = False

    def Exit(self):
        self.dcam.dcamapi_uninit()


if __name__ == "__main__":
    #
    # Initialization
    # Load dcamapi.dll version 19.12.641.5901
    cam = CamActuator()
    cam.initializeCamera()

    cam.StartStreaming(
        BufferNumber=10, trigger_source="INTERNAL", exposure_time=0.0015
    )
    logging.info("main thread continues")
    # Make sure that the camera is prepared before waveform execution.
    time.sleep(3.5)
    cam.isSaving = True
    tif_name = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\test.tif"  # TODO hardcoded path
    tif_name = r"M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/Patch clamp/2024_05_07 Ian Test/test_123.tif"  # TODO hardcoded path
    cam.StopStreaming(saving_dir=tif_name)
    # Make sure that the saving process is finished.
    while cam.isSaving is True:
        logging.info("Camera saving...")
        time.sleep(0.5)

    cam.Exit()
