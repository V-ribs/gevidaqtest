# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:30:41 2020

@author: Izak de Heer
"""
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.draw

from ..HamamatsuCam.HamamatsuActuator import CamActuator
from ..NIDAQ.DAQoperator import DAQmission
from . import CoordinateTransformations
from .backend import readRegistrationImages


class GalvoRegistrator:
    def __init__(self, *args, **kwargs):
        self.cam = CamActuator()
        self.cam.initializeCamera()

    def registration(self, grid_points_x=3, grid_points_y=3):
        """
        By default, generate 9 galvo voltage coordinates from (-5,-5) to (5,5),
        take the camera images of these points, return a function matrix that
        transforms camera_coordinates into galvo_coordinates using polynomial transform.

        Parameters
        grid_points_x : TYPE, optional
            DESCRIPTION. The default is 3.
        grid_points_y : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        transformation : TYPE
            DESCRIPTION.

        """
        galvothread = DAQmission()

        x_coords = np.linspace(-10, 10, grid_points_x + 2)[1:-1]
        y_coords = np.linspace(-10, 10, grid_points_y + 2)[1:-1]

        xy_mesh = np.reshape(
            np.meshgrid(x_coords, y_coords), (2, -1), order="F"
        ).transpose()

        galvo_coordinates = xy_mesh
        camera_coordinates = np.zeros((galvo_coordinates.shape))

        for i in range(galvo_coordinates.shape[0]):
            galvothread.sendSingleAnalog("galvosx", galvo_coordinates[i, 0])
            galvothread.sendSingleAnalog("galvosy", galvo_coordinates[i, 1])
            time.sleep(1)

            image = self.cam.SnapImage(0.06)
            plt.imsave(
                os.getcwd()  # TODO fix path
                + "/CoordinatesManager/Registration_Images/2P/image_"
                + str(i)
                + ".png",
                image,
            )

            camera_coordinates[i, :] = readRegistrationImages.gaussian_fitting(
                image
            )

        logging.info("Galvo Coordinate")
        logging.info(galvo_coordinates)
        logging.info("Camera coordinates")
        logging.info(camera_coordinates)
        del galvothread
        self.cam.Exit()

        transformation_cam2galvo = CoordinateTransformations.polynomial2DFit(
            camera_coordinates, galvo_coordinates, order=1
        )

        transformation_galvo2cam = CoordinateTransformations.polynomial2DFit(
            galvo_coordinates, camera_coordinates, order=1
        )

        logging.info("Transformation found for x:")
        logging.info(transformation_cam2galvo[:, :, 0])
        logging.info("Transformation found for y:")
        logging.info(transformation_cam2galvo[:, :, 1])

        logging.info("galvo2cam found for x:")
        logging.info(transformation_galvo2cam[:, :, 0])
        logging.info("galvo2cam found for y:")
        logging.info(transformation_galvo2cam[:, :, 1])

        return transformation_cam2galvo


class DMDRegistator:
    def __init__(self, DMD, *args, **kwargs):
        self.DMD = DMD
        self.cam = CamActuator()
        self.cam.initializeCamera()

    def registration(
        self,
        laser="640",
        grid_points_x=2,
        grid_points_y=3,
        registration_pattern="circles",
    ):
        x_coords = np.linspace(0, 768, grid_points_x + 2)[1:-1]
        y_coords = np.linspace(0, 1024, grid_points_y + 2)[1:-1]

        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        x_coords = np.ravel(x_mesh)
        y_coords = np.ravel(y_mesh)

        dmd_coordinates = np.stack((x_coords, y_coords), axis=1)

        camera_coordinates = np.zeros(dmd_coordinates.shape)

        for i in range(dmd_coordinates.shape[0]):
            x = int(dmd_coordinates[i, 0])
            y = int(dmd_coordinates[i, 1])

            if registration_pattern == "squares":
                mask = (
                    DMDRegistator.create_registration_image_touching_squares(
                        x, y
                    )
                )
            else:
                mask = DMDRegistator.create_registration_image_circle(x, y)

            self.DMD.send_data_to_DMD(mask)
            self.DMD.start_projection()

            image = self.cam.SnapImage(0.01)
            plt.imsave(
                os.getcwd()  # TODO fix path
                + "/CoordinatesManager/Registration_Images/TouchingSquares/image_"
                + str(i)
                + ".png",
                image,
            )
            camera_coordinates[i, :] = (
                readRegistrationImages.touchingCoordinateFinder(
                    image, method="curvefit"
                )
            )

            self.DMD.stop_projection()

        logging.info("DMD coordinates:")
        logging.info(dmd_coordinates)
        logging.info("Found camera coordinates:")
        logging.info(camera_coordinates)

        self.DMD.free_memory()
        self.cam.Exit()

        transformation = CoordinateTransformations.polynomial2DFit(
            camera_coordinates, dmd_coordinates, order=1
        )
        logging.info("Transformation found for x:")
        logging.info(transformation[:, :, 0])
        logging.info("Transformation found for y:")
        logging.info(transformation[:, :, 1])
        return transformation

    def create_registration_image_touching_squares(x, y, sigma=75):
        array = np.zeros((768, 1024))
        array[skimage.draw.rectangle((x - sigma, y - sigma), (x, y))] = 255
        array[skimage.draw.rectangle((x + sigma, y + sigma), (x, y))] = 255
        return array

    def create_registration_image_circle(x, y, sigma=75):
        array = np.zeros((768, 1024))
        array[skimage.draw.circle(x, y, sigma)] = 255
        return array
