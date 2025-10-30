# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:54:51 2020

@author: xinmeng
"""

# === Auto focus workflow ===
# --1. Find local minima range. (one direction until value getting smaller)
# --2. Bisection to find the optimal.

import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from ..GalvoWidget.GalvoScan_backend import RasterScan
from ..HamamatsuCam.HamamatsuActuator import CamActuator
from ..ImageAnalysis.ImageProcessing import ProcessImage
from ..NIDAQ.DAQoperator import DAQmission
from .focuser import PIMotor


class FocusFinder:
    def __init__(
        self,
        source_of_image="PMT",
        init_search_range=0.010,
        total_step_number=5,
        imaging_conditions={"edge_volt": 5},
        motor_handle=None,
        camera_handle=None,
        twophoton_handle=None,
        *args,
        **kwargs,
    ):
        """


        Parameters
        source_of_image : string, optional
            The input source of image. The default is PMT.
        init_search_range : int, optional
            The step size when first doing coarse searching.
            The default is 0.010.
        total_step_number : int, optional
            Number of steps in total to find optimal focus. The default is 5.
        imaging_conditions : list
            Parameters for imaging.
            For PMT, it specifies the scanning voltage.
            For camera, it specifies the AOTF voltage and exposure time.
        motor_handle : TYPE, optional
            Handle to control PI motor. The default is None.
        twophoton_handle : TYPE, optional
            Handle to control Insight X3. The default is None.

        Returns
        None.

        """
        super().__init__(*args, **kwargs)

        # The step size when first doing coarse searching.
        self.init_search_range = init_search_range

        # Number of steps in total to find optimal focus.
        self.total_step_number = total_step_number

        # Parameters for imaging.
        self.imaging_conditions = imaging_conditions

        if motor_handle is None:
            # Connect the objective if the handle is not provided.
            self.pi_device_instance = PIMotor()
        else:
            self.pi_device_instance = motor_handle

        # Current position of the focus.
        self.current_pos = self.pi_device_instance.GetCurrentPos()

        # Number of steps already tried.
        self.steps_taken = 0
        # The focus degree of previous position.
        self.previous_degree_of_focus = 0
        # Number of going backwards.
        self.turning_point = 0
        # The input source of image.
        self.source_of_image = source_of_image
        if source_of_image == "PMT":
            self.galvo = RasterScan(
                Daq_sample_rate=250000,
                edge_volt=self.imaging_conditions["edge_volt"],
            )
        elif source_of_image == "Camera":
            if camera_handle is None:
                # If no camera instance fed in, initialize camera.
                self.HamamatsuCam_ins = CamActuator()
                self.HamamatsuCam_ins.initializeCamera()
            else:
                self.HamamatsuCam_ins = camera_handle

    def gaussian_fit(self, move_to_focus=True):
        # The upper edge.
        upper_position = self.current_pos + self.init_search_range
        # The lower edge.
        lower_position = self.current_pos - self.init_search_range

        # Generate the sampling positions.
        sample_positions = np.linspace(
            lower_position, upper_position, self.total_step_number
        )

        degree_of_focus_list = []
        for each_pos in sample_positions:
            # Go through each position and write down the focus degree.
            degree_of_focus = self.evaluate_focus(round(each_pos, 6))
            degree_of_focus_list.append(degree_of_focus)
        logging.info(degree_of_focus_list)

        logging.info("Fitting failed. Find max in the list.")

        max_focus_pos = sample_positions[
            degree_of_focus_list.index(max(degree_of_focus_list))
        ]
        logging.info(max_focus_pos)

        if move_to_focus is True:
            self.pi_device_instance.move(max_focus_pos)

        return max_focus_pos

    def bisection(self):
        """
        Bisection way of finding focus.

        Returns
        mid_position : float
            DESCRIPTION.

        """
        # The upper edge in which we run bisection.
        upper_position = self.current_pos + self.init_search_range
        # The lower edge in which we run bisection.
        lower_position = self.current_pos - self.init_search_range

        for step_index in range(1, self.total_step_number + 1):
            # In each step of bisection finding.

            # In the first round, get degree of focus at three positions.
            if step_index == 1:
                # Get degree of focus in the mid.
                mid_position = (upper_position + lower_position) / 2
                degree_of_focus_mid = self.evaluate_focus(mid_position)
                logging.info(
                    "mid focus degree: {}".format(
                        round(degree_of_focus_mid, 5)
                    )
                )

                # Break the loop if focus degree is below threshold which means
                # that there's no cell in image.
                if not ProcessImage.if_theres_cell(
                    self.galvo_image.astype("float32")
                ):
                    logging.info("no cell")
                    mid_position = False
                    break

                # Move to top and evaluate.
                degree_of_focus_up = self.evaluate_focus(
                    obj_position=upper_position
                )
                logging.info(
                    "top focus degree: {}".format(round(degree_of_focus_up, 5))
                )
                # Move to bottom and evaluate.
                degree_of_focus_low = self.evaluate_focus(
                    obj_position=lower_position
                )
                logging.info(
                    "bot focus degree: {}".format(
                        round(degree_of_focus_low, 5)
                    )
                )
                # Sorting dicitonary of degrees in ascending.
                biesection_range_dic = {
                    "top": [upper_position, degree_of_focus_up],
                    "bot": [lower_position, degree_of_focus_low],
                }

            # In the next rounds, only need to go to center and update
            # boundaries.
            elif step_index > 1:
                # The upper edge in which we run bisection.
                upper_position = biesection_range_dic["top"][0]
                # The lower edge in which we run bisection.
                lower_position = biesection_range_dic["bot"][0]

                # Get degree of focus in the mid.
                mid_position = (upper_position + lower_position) / 2
                degree_of_focus_mid = self.evaluate_focus(mid_position)

                logging.info(
                    "Current focus degree: {}".format(
                        round(degree_of_focus_mid, 5)
                    )
                )

            # If sits in upper half, make the middle values new bottom.
            if biesection_range_dic["top"][1] > biesection_range_dic["bot"][1]:
                biesection_range_dic["bot"] = [
                    mid_position,
                    degree_of_focus_mid,
                ]
            else:
                biesection_range_dic["top"] = [
                    mid_position,
                    degree_of_focus_mid,
                ]

            logging.info(
                "The upper pos: {}; The lower: {}".format(
                    biesection_range_dic["top"][0],
                    biesection_range_dic["bot"][0],
                )
            )

        return mid_position

    def evaluate_focus(self, obj_position=None):
        """
        Evaluate the focus degree of certain objective position.

        Parameters
        obj_position : float, optional
            The target objective position. The default is None.

        Returns
        degree_of_focus : float
            Degree of focus.

        """

        if obj_position is not None:
            self.pi_device_instance.move(obj_position)

        # Get the image.
        if self.source_of_image == "PMT":
            self.galvo_image = self.galvo.run()
            plt.figure()
            plt.imshow(self.galvo_image)
            plt.show()

            degree_of_focus = ProcessImage.local_entropy(
                self.galvo_image.astype("float32")
            )

        elif self.source_of_image == "Camera":
            # First configure the AOTF.
            self.AOTF_runner = DAQmission()
            # Find the AOTF channel key
            for key in self.imaging_conditions:
                if "AO" in key:
                    # like '488AO'
                    AOTF_channel_key = key

            # Set the AOTF first.
            self.AOTF_runner.sendSingleDigital("blankingall", True)
            self.AOTF_runner.sendSingleAnalog(
                AOTF_channel_key, self.imaging_conditions[AOTF_channel_key]
            )

            # Snap an image from camera
            self.camera_image = self.HamamatsuCam_ins.SnapImage(
                self.imaging_conditions["exposure_time"]
            )
            time.sleep(0.5)

            # Set back AOTF
            self.AOTF_runner.sendSingleDigital("blankingall", False)
            self.AOTF_runner.sendSingleAnalog(AOTF_channel_key, 0)

            plt.figure()
            plt.imshow(self.camera_image)
            plt.show()

            degree_of_focus = ProcessImage.variance_of_laplacian(
                self.camera_image.astype("float32")
            )

        time.sleep(0.2)

        return degree_of_focus


if __name__ == "__main__":
    ins = FocusFinder()
    ins.total_step_number = 7
    ins.init_search_range = 0.012
    ins.imaging_conditions = ({"edge_volt": 3},)
    ins.bisection()  # will return false if there's no cell in view.
    ins.pi_device_instance.CloseMotorConnection()
