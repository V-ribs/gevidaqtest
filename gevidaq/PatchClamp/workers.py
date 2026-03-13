# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:15:30 2021

@author: TvdrBurgt
"""

import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .ImageProcessing_patchclamp import PatchClampImageProcessing as ia


class Worker(QObject):
    draw = pyqtSignal(list)
    graph1 = pyqtSignal(np.ndarray)
    graph2 = pyqtSignal(np.ndarray)
    status = pyqtSignal(str)
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.STOP = False

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def STOP(self):
        return self._STOP

    @STOP.setter
    def STOP(self, state):
        self._STOP = state

    @pyqtSlot()
    def target2center(self):
        """Target to center moves the XY stage so that the user-selected
        target ends up in the center of the camera field-of-view.

        The camera field-of-view is turned 90 degrees counter-clockwise
        compared to the ludl XY sample stage.
        """
        self.status.emit("Centering target...")

        # get all relevant parent attributes
        stage = self._parent.XYstage
        pixelsize = self._parent.pixel_size
        width, height = self._parent.image_size
        xtarget, ytarget, ztarget = self._parent.target_coordinates

        # Calculate image center pixels
        xcenter, ycenter = int(width / 2), int(height / 2)

        # Caculate path to travel from target to center
        dx_pi = xtarget - xcenter  # in pixels
        dy_pi = ytarget - ycenter  # in pixels

        # Convert pixels to ludl indices (1577 ludl index = 340 um)
        dx = int(dx_pi * pixelsize / 1000 * 1577 / 340)
        dy = int(dy_pi * pixelsize / 1000 * 1577 / 340)

        # Move XY stage a distance (-dx,-dy), note that stage axis are rotated
        ismoving = True
        stage.moveRel(xRel=dy, yRel=-dx)
        while ismoving:
            ismoving = not stage.motorsStopped()
            time.sleep(0.1)

        # Update target coordinates in the backend
        self.parent.target_coordinates = np.array([xcenter, ycenter, ztarget])
        self.draw.emit(["target", dx_pi, dy_pi])

        self.finished.emit()

    @pyqtSlot()
    def hardcalibration(self):
        """Hardcalibration aligns the coordinate system of the micromanipulator
        with that of the camera field-of-view (FOV) by constructing a rotation
        matrix that maps coordinates between the coordinate systems. It can
        also estimate the pixelsize.

        We can choose three modes of operation:
            XY          for aligning only the x- and y-axes,
            XYZ         for aligning the x-, y- and z-axes,
            pixelsize   for estimating the pixel size.
        We can set the variable:
            stepsize    should be set small enough to keep the pipette visible

        The rotation angles are calculated as passive rotation angles (counter-
        clockwise + acting on coordinate systems) in the order: gamma, alpha,
        beta.
        The matrix E contains the micromanipulator axes in unit vectors as:
        E = [Exx Exy Exz
             Eyx Eyy Eyz
             Ezx Ezy Ezz],
        if perfectly aligned with the FOV, E should be the 3x3 identity matrix.

        outputs:
            gamma       (rotation angle of the micromanipulator z-axis w.r.t.
                         camera z-axis)
            alpha       (rotation angle of the micromanipulator x-axis w.r.t.
                         camera x-axis)
            beta        (rotation angle of the micromanipulator y-axis w.r.t.
                         camera y-axis)
            pixelsize   (pixel size in nanometers)
        """
        self.status.emit("Calibrating...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        micromanipulator = self._parent.micromanipulator
        camera = self._parent.camerathread
        account4rotation = self._parent.account4rotation
        mode = self._parent.operation_mode
        diameter = self._parent.pipette_diameter
        orientation = self._parent.pipette_orientation

        # algorithm variables
        stepsize = 25  # in microns

        # modes of operation
        if mode == "XY":
            dimension = 2
            del self._parent.rotation_angles
        elif mode == "XYZ":
            dimension = 3
            del self._parent.rotation_angles
        elif mode == "pixelsize":
            dimension = 2
        else:
            self.progress.emit("Hardcalibration mode not recognized")

        positions = np.linspace(-3 * stepsize, 3 * stepsize, num=7)
        directions = np.eye(3)
        tipcoords = np.tile(np.nan, (3, len(positions), 3))
        reference = micromanipulator.getPos()

        # move hardware to retrieve tip coordinates
        # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))                   #FLAG: relevant for MSc thesis
        for i in range(dimension):
            for j, pos in enumerate(positions):
                # snap images for pipettet tip detection
                x, y, z = account4rotation(
                    origin=reference, target=reference + directions[i] * pos
                )
                micromanipulator.moveAbs(x, y, z)
                image_left = camera.snap()
                micromanipulator.moveRel(dx=5)
                image_right = camera.snap()

                # pipette tip detection algorithm
                x1, y1 = ia.detectPipettetip(
                    image_left,
                    image_right,
                    diameter=diameter,
                    orientation=orientation,
                )
                self.draw.emit(["cross", x1, y1])
                # W = ia.makeGaussian(size=image_left.shape, mu=(x1,y1), sigma=(image_left.shape[0]//12,image_left.shape[1]//12))
                # self.draw.emit(['image',np.multiply(image_right,W)])
                # x2, y2 = ia.detectPipettetip(np.multiply(image_left,W), np.multiply(image_right,W), diameter=(5/4)*diameter, orientation=orientation)
                # self.draw.emit(['cross',x2,y2])

                # save tip coordinates
                tipcoords[i, j, :] = np.array([x1, y1, np.nan])
                # np.save(save_directory+'hardcalibration_'+timestamp, tipcoords)         #FLAG: relevant for MSc thesis

                # (emergency) stop
                if self.STOP:
                    break

            # (emergency) stop
            if self.STOP:
                break

        # move hardware back to start position
        x, y, z = reference
        micromanipulator.moveAbs(x, y, z)

        # reduce tip coordinates to a unit step mean deviation
        tipcoords_diff = np.diff(tipcoords, axis=1)
        E = np.mean(tipcoords_diff, axis=1) / stepsize

        # calculate rotation angles
        if E[0, 0] > 0:
            gamma = np.arctan(-E[0, 1] / E[0, 0])
        else:
            gamma = np.arctan(-E[0, 1] / E[0, 0]) - np.pi
        alpha = np.arcsin(E[2, 0] * np.sin(gamma) + E[2, 1] * np.cos(gamma))
        beta = np.arcsin(
            (-E[2, 0] * np.cos(gamma) + E[2, 1] * np.sin(gamma))
            / np.cos(alpha)
        )

        # set rotation angles or calculate pixelsize
        if mode == "XY" and not self.STOP:
            self._parent.rotation_angles = (0, 0, -gamma)
            self.draw.emit(
                [
                    "calibrationline",
                    tipcoords[0, 3, 0],
                    tipcoords[0, 3, 1],
                    -np.rad2deg(gamma),
                ]
            )
            self.draw.emit(
                [
                    "calibrationline",
                    tipcoords[0, 3, 0],
                    tipcoords[0, 3, 1],
                    -np.rad2deg(gamma - np.pi / 2),
                ]
            )
        elif mode == "XYZ" and not self.STOP:
            self._parent.rotation_angles = (-alpha, -beta, -gamma)
            self.draw.emit(
                [
                    "calibrationline",
                    tipcoords[0, 3, 0],
                    tipcoords[0, 3, 1],
                    -np.rad2deg(gamma),
                ]
            )
            self.draw.emit(
                [
                    "calibrationline",
                    tipcoords[0, 3, 0],
                    tipcoords[0, 3, 1],
                    -np.rad2deg(gamma - np.pi / 2),
                ]
            )
        elif mode == "pixelsize" and not self.STOP:
            # get all micromanipulator-pixel pairs in the XY plane
            pixcoords = tipcoords[0:2, :, 0:2]
            realcoords = np.tile(np.nan, (2, len(positions), 2))
            for i in range(dimension):
                for j, pos in enumerate(positions):
                    x, y, z = account4rotation(
                        origin=reference,
                        target=reference + directions[i] * pos,
                    )
                    realcoords[i, j] = np.array([x, y])

            # couple micrometer distance with pixeldistance and take the mean
            diff_realcoords = np.abs(np.diff(realcoords, axis=1))  # in microns
            diff_pixcoords = np.abs(np.diff(pixcoords, axis=1))  # in pixels
            samples = diff_realcoords / diff_pixcoords  # microns per pixel
            sample = np.array([samples[0, :, 0], samples[1, :, 1]])
            sample_mean = np.mean(sample) * 1000  # in nanometers
            sample_var = np.var(sample) * 1000  # in nanometers

            # set pixelsize to backend
            self._parent.pixel_size = sample_mean
            self.progress.emit(
                "pixelsize estimation: mean +/- s.d. = "
                + "{:.2f}".format(sample_mean)
                + " +/- "
                + "{:.2f}".format(np.sqrt(sample_var))
            )

        self.status.emit("Calibration finished")
        self.finished.emit()

    @pyqtSlot()
    def prechecks(self):
        """Pre-checks make sure that the patch preparation is successful.
        We make sure that over pressure is applied before entering the sample
        medium, but we scale it to the right value. We also check if the
        pipette is suited for whole-cell patching by checking its resistance.

        I) set pressure,
        II) check if resistance is within 3 to 11 MΩ.
        """
        self.status.emit("Pre-checks...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        pressurecontroller = self._parent.pressurethread

        # Algorithm variables
        P_PRECHECK_CONDITION = 20  # mBar
        R_PRECHECK_CONDITION = [3, 11]  # MΩ
        SUCCESS_1 = True
        SUCCESS_2 = True

        # I) set pressure to prevent pipette contamination
        pressure = np.zeros(10)
        for i in range(0, 10):
            pressure[i] = np.nanmean(self._parent.pressure[0][-10::])
            time.sleep(0.2)

        if all(pressure >= P_PRECHECK_CONDITION):
            self.progress.emit("Pressure check passed")
            pressurecontroller.set_pressure_stop_waveform(50)
        else:
            self.progress.emit("Pressure check failed")
            SUCCESS_1 = False

        # II) measure pipette resistance and check if it is consistent
        del self._parent.resistance_reference
        resistance = np.zeros(10)
        for i in range(0, 10):
            resistance[i] = np.nanmax(self._parent.resistance[-10::])
            time.sleep(0.2)

        if all(resistance >= R_PRECHECK_CONDITION[0] * 1e6) and all(
            resistance <= R_PRECHECK_CONDITION[1] * 1e6
        ):
            self._parent.resistance_reference = np.nanmean(resistance)
            self.progress.emit("Resistance check passed")
        else:
            self.progress.emit("Resistance check passed")
            SUCCESS_2 = False

        # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))                   #FLAG: relevant for MSc thesis
        # np.save(save_directory+'precheckresistance_'+timestamp, resistance)             #FLAG: relevant for MSc thesis

        if SUCCESS_1 and SUCCESS_2:
            self.progress.emit("Pre-checks successful")
        elif SUCCESS_1 and not SUCCESS_2:
            self.progress.emit("Resistance too high/low")
        elif not SUCCESS_1 and SUCCESS_2:
            self.progress.emit("No pipette pressure")
        else:
            self.progress.emit(
                "No pipette pressure \nand resistance too high/low"
            )

        self.status.emit("Pre-checks finished")
        self.finished.emit()

    @pyqtSlot()
    def autopatch(self):
        """Autopatch is where the full pipeline of automatic patch algorithms
        is supposed to start IF the tip detection step would be done by an AI.
        Image analysis is not precise enough and the user has to correct the
        localization offset.

        I) Register focal height (where in-focus cells should be)).
        II) Activate the autofocus algorithm.
        III) Activate the softcalibration algorithm.
        IV) Return focal plane.
        """
        self.status.emit("Autopatching...")

        # get all relevant parent attributes
        objective = self._parent.objectivemotor
        xtarget, ytarget, ztarget = self._parent.target_coordinates

        # I) Register target cell height
        ztarget = objective.getPos()
        self._parent.target_coordinates = np.array([xtarget, ytarget, ztarget])

        # II) Focus pipette tip
        self.autofocus_tip()

        # III) Detect pipette tip
        self.softcalibration()

        # IV) Return focal plane to target cell height
        self.progress.emit("Return focal plane to cell height")
        objective.moveAbs(ztarget)

        self.status.emit("Autopatch finished")
        self.progress.emit("Warning: CHECK TIP LOCALIZATION!")
        self.finished.emit()

    @pyqtSlot()
    def autofocus_tip(self):
        """Autofocus pipette tip iteratively moves the pipette up or down
        untill the tip is in focus.

        The algorithm uses the variance of the Laplacian as its sharpness
        function. The sharpness values are high for images with sharp edges and
        low for images with little sharp edges. We use this phenomenon to find
        the maximum of the sharpness function, the image contains sharp lines
        when the pipette is in-focus. Note that the sharpness values go up when
        the pipette moves through the focal plane, thereby creating more sharp
        edges as a result of pipette geometry and light bending. However, the
        sharpness function dips a little when the pipette tip is in focus. The
        autofocus algorithm here is designed to stop at the local maximum
        before the dip. Another advantage is that this dip is present at a
        constant height above the focal plance, this allows us to move past the
        the peak without pressing the pipette down into the sample/coverslip.

        The *focusbias* is around 10-30 micrometers!!!
        """
        self.status.emit("Autofocus pipette...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        micromanipulator = self._parent.micromanipulator
        objective = self._parent.objectivemotor
        camera = self._parent.camerathread
        focus_offset = self._parent.focus_offset
        xtarget, ytarget, ztarget = self._parent.target_coordinates

        # algorithm variables
        STEPSIZE = 7  # micron
        MIN_TAILLENGTH = 13  # datapoints (=X*STEPSIZE in micron)

        reference = micromanipulator.getPos()
        penaltyhistory = np.array([])
        positionhistory = np.array([])

        # I) move objective up to not penatrate cells when focussing
        objective.moveAbs(z=ztarget + focus_offset / 1000)

        # II) fill first three sharpness scores towards the tail of the graph
        pen = np.zeros(3)
        pos = np.zeros(3)
        for i in range(0, 3):
            micromanipulator.moveRel(dz=+STEPSIZE)
            img = camera.snap()
            pen[i] = ia.comp_variance_of_Laplacian(img)
            pos[i] = reference[2] + (i + 1) * STEPSIZE
        penaltyhistory = np.append(penaltyhistory, pen)
        positionhistory = np.append(positionhistory, pos)

        going_up = True
        going_down = not going_up
        lookingforpeak = True
        while lookingforpeak and not self.STOP:
            # emit graph
            self.graph1.emit(np.vstack([positionhistory, penaltyhistory]))

            # IIIa) check which side of the sharpness graph to extend
            move = None
            if going_up:
                pen = penaltyhistory[-3::]
            else:
                pen = penaltyhistory[0:3]

            # IIIb) check where maximum penalty score is: left, middle, right
            if np.argmax(pen) == 0:
                maximum = "left"
            elif np.argmax(pen) == 1:
                maximum = "middle"
            elif np.argmax(pen) == 2:
                maximum = "right"

            # IVa) possible actions to undertake while going up
            if maximum == "right" and going_up:
                move = "step up"
            elif maximum == "left" and going_up:
                going_up = False
                going_down = True
            elif maximum == "middle" and going_up:
                if pen[1] == np.max(penaltyhistory):
                    pos = positionhistory[-1]
                    micromanipulator.moveAbs(
                        x=reference[0], y=reference[1], z=pos
                    )
                    penaltytail = pen[1::]
                    monotonicity_condition = True
                    for i in range(2, MIN_TAILLENGTH):
                        if monotonicity_condition:
                            micromanipulator.moveRel(dz=+STEPSIZE)
                            img = camera.snap()
                            penalty = ia.comp_variance_of_Laplacian(img)
                            penaltytail = np.append(penaltytail, penalty)
                            monotonicity_condition = np.all(
                                np.diff(penaltytail) <= 0
                            )
                        else:
                            break
                    if monotonicity_condition:
                        self.progress.emit("Maximum is a sharpness peak!")
                        lookingforpeak = False
                        move = None
                        foundfocus = positionhistory[-2]
                    else:
                        self.progress.emit("Maximum is noise")
                        going_up = False
                        going_down = True
                else:
                    going_up = False
                    going_down = True

            # IVb) possible actions to undertake while going down
            elif maximum == "left" and going_down:
                move = "step down"
            elif maximum == "right" and going_down:
                if pen[2] == np.max(penaltyhistory):
                    pos = positionhistory[2]
                    micromanipulator.moveAbs(
                        x=reference[0], y=reference[1], z=pos
                    )
                    penaltytail = pen[2]
                    monotonicity_condition = True
                    for i in range(2, MIN_TAILLENGTH):
                        if monotonicity_condition:
                            micromanipulator.moveRel(dz=+STEPSIZE)
                            img = camera.snap()
                            penalty = ia.comp_variance_of_Laplacian(img)
                            penaltytail = np.append(penaltytail, penalty)
                            monotonicity_condition = np.all(
                                np.diff(penaltytail) <= 0
                            )
                        else:
                            break
                    if monotonicity_condition:
                        self.progress.emit("Maximum is a sharpness peak!")
                        lookingforpeak = False
                        move = None
                        foundfocus = positionhistory[2]
                    else:
                        self.progress.emit("Maximum is noise")
                        move = "step down"
                else:
                    move = "step down"
            elif maximum == "middle" and going_down:
                penaltytail = penaltyhistory[1::]
                taillength = len(penaltytail)
                if taillength < MIN_TAILLENGTH:
                    pos = positionhistory[-1]
                    micromanipulator.moveAbs(
                        x=reference[0], y=reference[1], z=pos
                    )
                    monotonicity_condition = True
                    for i in range(taillength, MIN_TAILLENGTH):
                        if monotonicity_condition:
                            micromanipulator.moveRel(dz=+STEPSIZE)
                            img = camera.snap()
                            penalty = ia.comp_variance_of_Laplacian(img)
                            penaltytail = np.append(penaltytail, penalty)
                            monotonicity_condition = np.all(
                                np.diff(penaltytail) <= 0
                            )
                        else:
                            break
                else:
                    penaltytail = penaltytail[0:MIN_TAILLENGTH]
                    monotonicity_condition = np.all(np.diff(penaltytail) <= 0)
                if monotonicity_condition:
                    self.progress.emit("Maximum is a sharpness peak!")
                    lookingforpeak = False
                    move = None
                    foundfocus = positionhistory[1]
                else:
                    self.progress.emit("Maximum is noise")
                    move = "step down"

            # V) extend the sharpness function on either side
            if move == "step up":
                pos = positionhistory[-1] + STEPSIZE
                micromanipulator.moveAbs(x=reference[0], y=reference[1], z=pos)
                img = camera.snap()
                pen = ia.comp_variance_of_Laplacian(img)
                penaltyhistory = np.append(penaltyhistory, pen)
                positionhistory = np.append(positionhistory, pos)
            elif move == "step down":
                pos = positionhistory[0] - STEPSIZE
                micromanipulator.moveAbs(x=reference[0], y=reference[1], z=pos)
                img = camera.snap()
                pen = ia.comp_variance_of_Laplacian(img)
                penaltyhistory = np.append(pen, penaltyhistory)
                positionhistory = np.append(pos, positionhistory)

        # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))                   #FLAG: relevant for MSc thesis
        # np.save(save_directory+'autofocus_positionhistory_'+timestamp, positionhistory) #FLAG: relevant for MSc thesis
        # np.save(save_directory+'autofocus_penaltyhistory_'+timestamp, penaltyhistory)   #FLAG: relevant for MSc thesis

        # VI) continue with finding the fine focus position
        self.progress.emit("Sampling the sharpness peak")
        for step in [STEPSIZE, STEPSIZE / 3]:
            # (emergency) stop
            if self.STOP:
                break
            # Sample six points between the last three penalty scores
            penalties = np.zeros(7)
            positions = np.linspace(foundfocus - step, foundfocus + step, 7)
            for idx, pos in enumerate(positions):
                # (emergency) stop
                if self.STOP:
                    break
                micromanipulator.moveAbs(x=reference[0], y=reference[1], z=pos)
                img = camera.snap()
                penalties[idx] = ia.comp_variance_of_Laplacian(img)
                positionhistory = np.append(positionhistory, pos)
                penaltyhistory = np.append(penaltyhistory, penalties[idx])

            # Locate maximum penalty value
            foundfocus = positions[np.argmax(penalties)]

            # emit graph
            self.graph1.emit(np.vstack([positions, penalties]))

        # emit graph
        self.graph1.emit(np.vstack([positionhistory, penaltyhistory]))

        # VIIa) move pipette into focus
        micromanipulator.moveAbs(x=reference[0], y=reference[1], z=foundfocus)
        self.progress.emit("Pipette in focus")

        # img = camera.snap()                                                                   #FLAG: relevant for MSc thesis
        # io.imsave(save_directory+'autofocus_'+timestamp+'.tif', img, check_contrast=False)    #FLAG: relevant for MSc thesis
        # np.save(save_directory+'autofocus_positionhistory_'+timestamp, positionhistory)     #FLAG: relevant for MSc thesis
        # np.save(save_directory+'autofocus_penaltyhistory_'+timestamp, penaltyhistory)       #FLAG: relevant for MSc thesis

        self.status.emit("Autofocus finished")
        self.finished.emit()

    @pyqtSlot()
    def softcalibration(self):
        """Softcalibration couples the micromanipulator coordinates with the
        pixelcoordinates of a pipette tip.

        The user can select to correct for bias.

        The positions matrix contains the calibration points. We can change,
        add, or remove calibration points if necessary.

        outputs:
            pipette_coordinates_pair    np.array([reference (in microns);
                                                  tip coordinates (in pixels)])
        """
        self.status.emit("Detecting tip...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        micromanipulator = self._parent.micromanipulator
        objective = self._parent.objectivemotor
        camera = self._parent.camerathread
        focus_offset = self._parent.focus_offset
        account4rotation = self._parent.account4rotation
        _, _, ztarget = self._parent.target_coordinates
        diameter = self._parent.pipette_diameter
        orientation = self._parent.pipette_orientation

        # algorithm variables
        CALIBRATION_HEIGHT = (
            focus_offset + 10
        )  # microns above coverslip (+term = bias + height above focus for tip detection)
        POSITIONS = np.array(
            [[-25, -25, 0], [25, -25, 0], [25, 25, 0], [-25, 25, 0]]
        )

        # bring focal plane beyond its offset where pipette tip is in focus
        objective.moveAbs(z=ztarget + CALIBRATION_HEIGHT / 1000)

        tipcoords1 = POSITIONS[:, 0:2] * 0
        tipcoords2 = POSITIONS[:, 0:2] * 0
        reference = micromanipulator.getPos()
        for i in range(0, POSITIONS.shape[0]):
            # snap images for pipettet tip detection
            x, y, z = account4rotation(
                origin=reference, target=reference + POSITIONS[i]
            )
            micromanipulator.moveAbs(x, y, z)
            image_left = camera.snap()
            micromanipulator.moveRel(dx=5)
            image_right = camera.snap()

            # pipette tip detection algorithm
            x1, y1 = ia.detectPipettetip(
                image_left,
                image_right,
                diameter=diameter,
                orientation=orientation,
            )
            self.draw.emit(["cross", x1, y1])
            W = ia.makeGaussian(
                size=image_left.shape,
                mu=(x1, y1),
                sigma=(image_left.shape[0] // 6, image_left.shape[1] // 6),
            )
            camera.snapsignal.emit(np.multiply(image_right, W))
            x2, y2 = ia.detectPipettetip(
                np.multiply(image_left, W),
                np.multiply(image_right, W),
                diameter=(5 / 4) * diameter,
                orientation=orientation,
            )
            self.draw.emit(["cross", x2, y2])

            # save tip coordinates in an array
            tipcoords1[i, :] = x1, y1
            tipcoords2[i, :] = x2, y2

            # (emergency) stop
            if self.STOP:
                break

        # outlier detection, replace outlier with mean-interpolation
        norm_dist = np.sqrt(np.sum((tipcoords1 - tipcoords2) ** 2, axis=1))
        idx_possible_outlier = np.argmax(norm_dist)
        if norm_dist[idx_possible_outlier] > 50:
            positions = (
                POSITIONS * 1.0
            )  # converts array of ints to array floas (just in case)
            posx, posy, _ = positions[idx_possible_outlier]
            positions[idx_possible_outlier, :] = np.array(
                [np.nan, np.nan, np.nan]
            )
            idx_x_interpolation = np.where(positions[:, 0] == posx)
            idx_y_interpolation = np.where(positions[:, 1] == posy)
            tipcoords2[idx_possible_outlier, 0] = np.mean(
                tipcoords2[idx_x_interpolation, 0]
            )
            tipcoords2[idx_possible_outlier, 1] = np.mean(
                tipcoords2[idx_y_interpolation, 1]
            )
            self.progress.emit(
                "Emitting tip-detection outlier: "
                + "[{:.1f}".format(posx)
                + ", {:.1f})".format(posy)
            )

        # return pipette to starting position
        x, y, z = reference
        micromanipulator.moveAbs(x, y, z)

        if not self.STOP:
            # calculate final tip coordinates
            tipcoord = np.mean(tipcoords2, axis=0)
            self.draw.emit(["cross", tipcoord[0], tipcoord[1]])
            img = camera.snap()
            # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))                                                                       #FLAG: relevant for MSc thesis
            # io.imsave(save_directory+'softcalibration_X'+str(tipcoord[0])+'_Y'+str(tipcoord[1])+'_'+timestamp+'.tif', img, check_contrast=False)  #FLAG: relevant for MSc thesis
            # np.save(save_directory+'softcalibration_'+timestamp, tipcoords2)

            # set micromanipulator and camera coordinate pair of pipette tip
            tipcoord_objective_height = ztarget + CALIBRATION_HEIGHT / 1000
            self._parent.pipette_coordinates_pair = np.vstack(
                [
                    reference,
                    np.array(
                        [tipcoord[0], tipcoord[1], tipcoord_objective_height]
                    ),
                ]
            )

        self.status.emit("Tip-detection finished")
        self.finished.emit()

    @pyqtSlot()
    def pipette2target(self):
        """Pipette tip to target manoeuvres the micromanipulator to a target
        in the camera field-of-view

        I) Apply small overpressure.
        II) Calculate trajectory and bring pipette tip above the target cell.
        III) Pipette tip descent until resistance increases slightly.
        IV) Release pressure.

        Safety measures in place:
            <!>     Pipette descent range: <50 microns so pipette does not
                    penatrate the petridish.
            <!>     Resistance check: Resistance rises when pipette tip is
                    blocked by a cell or the petridish, a broken tip results
                    in an abrupt drop in resistance after which we stop
                    pipette descent immediately.
        """
        self.status.emit("Approaching target...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        micromanipulator = self._parent.micromanipulator
        pressurecontroller = self._parent.pressurethread
        focus_offset = self._parent.focus_offset
        account4rotation = self._parent.account4rotation
        pixelsize = self._parent.pixel_size
        tipcoords_manip, tipcoords_cam = self._parent.pipette_coordinates_pair
        xtarget, ytarget, _ = self._parent.target_coordinates

        # Algorithm variables
        R_CRITICAL = 0.15e6  # ohm
        PIPETTE_DESCENT_RANGE = 50  # microns
        STEPSIZE = 0.2  # microns
        PIPETTE_PRESSURE = 30  # mBar

        # I) make sure pressure is set at the right value
        pressurecontroller.set_pressure_stop_waveform(PIPETTE_PRESSURE)

        # IIa) calculate shortest trajectory to target and apply coordinate transformation
        dx = xtarget - tipcoords_cam[0]  # x trajectory (in pixels)
        dy = ytarget - tipcoords_cam[1]  # y trajectory (in pixels)
        trajectory = (
            np.array([dx, dy, 0]) * pixelsize / 1000
        )  # trajectory (in microns)
        trajectory = account4rotation(origin=np.zeros(3), target=trajectory)

        # IIb) manoeuvre pipette above target and descent the focal offset
        micromanipulator.moveAbs(
            x=tipcoords_manip[0] + trajectory[0],
            y=tipcoords_manip[1] + trajectory[1],
            z=tipcoords_manip[2] + trajectory[2],
        )
        micromanipulator.moveRel(dz=20 - focus_offset)

        # IIIa) measure resistance and set graph thresholds
        time.sleep(0.5)
        resistance_ref = np.nanmean(self._parent.resistance)
        self.progress.emit(
            "R reference: "
            + "{:.2f}".format(resistance_ref * 1e-6)
            + " MΩ. Contact R: "
            + "{:.2f}".format(R_CRITICAL * 1e-6)
            + "MΩ"
        )

        # IIIb) descent pipette until R increases by R_CRITICAL
        resistance = resistance_ref
        position = micromanipulator.getPos()[2]
        resistancehistory = np.array([resistance])
        positionhistory = np.array([position])
        while resistance < resistance_ref + R_CRITICAL and not self.STOP:
            # step down and measure the resistance
            micromanipulator.moveRel(dx=0, dy=0, dz=-STEPSIZE)
            position = micromanipulator.getPos()[2]
            resistance = np.nanmean(self._parent.resistance[-10::])
            self.graph2.emit(resistancehistory)

            # safety checks on maximum descent range and resistance drop
            if resistance < resistance_ref - R_CRITICAL:
                self.progress.emit("Tip broke")
                break
            else:
                resistancehistory = np.append(resistancehistory, resistance)
            if (
                positionhistory[-1] - positionhistory[0]
                >= PIPETTE_DESCENT_RANGE
            ):
                self.progress.emit("Maximum descent range achieved")
                break
            else:
                positionhistory = np.append(positionhistory, position)

        # IV) set pressure to ATM
        if not self.STOP:
            pressurecontroller.set_pressure_stop_waveform(0)

        # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # np.save(save_directory+'approach_positionhistory_'+timestamp, positionhistory)      #FLAG: relevant for MSc thesis
        # np.save(save_directory+'approach_resistancehistory_'+timestamp, resistancehistory)  #FLAG: relevant for MSc thesis

        self.status.emit("Approach finished")
        self.finished.emit()

    @pyqtSlot()
    def gigaseal(self):
        """Gigaseal applies suction to get seal a patch of cell membrane.

        Ia) Apply low suction waves.
        Ib) Apply high suction waves.
        II) Set pressure to atmoshpere.
        """
        self.status.emit("Gigasealing...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        pressurecontroller = self._parent.pressurethread
        tipcoords_manip, tipcoords_cam = self._parent.pipette_coordinates_pair
        xtarget, ytarget, _ = self._parent.target_coordinates

        # Algorithm variables
        TIMEOUT = 30  # seconds

        # Ia) wait for Gigaseal with suction pulses
        resistance = np.nanmean(self._parent.resistance[-10::])
        resistancehistory = np.array([resistance])
        if resistance < 1e9 and not self.STOP:
            self.progress.emit("Applying waves of light suction")
            start = time.time()
            pressurecontroller.set_waveform(high=0, low=-10, high_T=1, low_T=1)
            while (
                resistance < 1e9
                and time.time() - start < TIMEOUT
                and not self.STOP
            ):
                resistance = np.nanmax(self._parent.resistance[-10::])
                self.graph2.emit(resistancehistory)
                resistancehistory = np.append(resistancehistory, resistance)
                time.sleep(0.1)

        # Ib) wait for Gigaseal with increased suction pulses
        if resistance < 1e9 and not self.STOP:
            self.progress.emit("Applying waves of stronger suction")
            start = time.time()
            pressurecontroller.set_waveform(
                high=-10, low=-30, high_T=1, low_T=1
            )
            while (
                resistance < 1e9
                and time.time() - start < TIMEOUT
                and not self.STOP
            ):
                resistance = np.nanmax(self._parent.resistance[-10::])
                self.graph2.emit(resistancehistory)
                resistancehistory = np.append(resistancehistory, resistance)
                time.sleep(0.1)

        # II) Release pressure
        pressurecontroller.set_pressure_stop_waveform(0)

        # Evaluate seal formation by fast sampling
        if resistance > 1e9:
            self.progress.emit("Gigaseal formed!")
            start = time.time()
            while time.time() - start < 5:
                resistance = np.nanmax(self._parent.resistance[-10::])
                self.graph2.emit(resistancehistory)
                resistancehistory = np.append(resistancehistory, resistance)
                time.sleep(0.1)
        else:
            self.progress.emit("Gigaseal failed")

        # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # np.save(save_directory+'gigaseal_resistancehistory_'+timestamp, resistancehistory)  #FLAG: relevant for MSc thesis

        self.status.emit("Gigaseal finished")
        self.finished.emit()

    @pyqtSlot()
    def break_in(self):
        """Break-in applies pressure pulses to rupture the membrane patch.

        I) Apply suction pulses for 30 seconds with increasing vacuum pressure.
        II) Apply long suction while zapping the membrane as a last resort.

        The break-in is successful if the resistance drops below 300MΩ and the
        current is in the range [-300, 300]pA.
        """
        self.status.emit("Break-in...")

        # get all relevant parent attributes
        save_directory = self._parent.save_directory  # TODO unused
        pressurecontroller = self._parent.pressurethread
        sealtestthread = self._parent.sealtestthread

        # Algorithm variables
        TIMEOUT = 60  # seconds
        I_BREAKIN_CONDITION = 1e-9  # ampere absolute valued
        R_BREAKIN_CONDITION = 300 * 1e6  # ohm
        C_BREAKIN_CONDITION = [
            50,
            200,
        ]  # units? farad? farad per surface unit?
        PULSES = np.linspace(-100, -300, 15)
        SUCCESS = False

        # I) attempt breaking-in by increasing suction pulses
        self.progress.emit("Attempt 1/3: increasing suction pulses")
        resistancehistory = np.array([])
        currenthistory = np.array([[], []])
        start = time.time()
        i = 0
        while time.time() - start < TIMEOUT and not SUCCESS and not self.STOP:
            i += 1
            Imax = np.max(self._parent.current)
            Imin = np.min(self._parent.current)
            resistance = np.nanmax(self._parent.resistance[-10::])
            capacitance = np.nanmean(self._parent.capacitance[-10::])
            if (
                Imax <= I_BREAKIN_CONDITION
                and Imin >= -I_BREAKIN_CONDITION
                and capacitance > C_BREAKIN_CONDITION[0]
                and capacitance < C_BREAKIN_CONDITION[1]
                and resistance <= R_BREAKIN_CONDITION
            ):
                SUCCESS = True
                break
            else:
                pressurecontroller.set_pulse_stop_waveform(
                    PULSES[i % len(PULSES)]
                )
                time.sleep(1)
            currenthistory = np.append(
                currenthistory, np.array([[Imax], [Imin]]), axis=1
            )
            resistancehistory = np.append(resistancehistory, resistance)
            self.graph2.emit(resistancehistory)

        # II) second attempt but with zap
        self.progress.emit("Attempt 2/3: - increasing suction pulses with ZAP")
        start = time.time()
        i = 0
        while (
            time.time() - start < TIMEOUT / 6 and not SUCCESS and not self.STOP
        ):
            i += 1
            Imax = np.max(self._parent.current)
            Imin = np.min(self._parent.current)
            resistance = np.nanmax(self._parent.resistance[-10::])
            capacitance = np.nanmean(self._parent.capacitance[-10::])
            if (
                Imax <= I_BREAKIN_CONDITION
                and Imin >= -I_BREAKIN_CONDITION
                and capacitance > C_BREAKIN_CONDITION[0]
                and capacitance < C_BREAKIN_CONDITION[1]
                and resistance <= R_BREAKIN_CONDITION
            ):
                SUCCESS = True
                break
            else:
                sealtestthread.zap()
                pressurecontroller.set_pulse_stop_waveform(
                    PULSES[i % len(PULSES)]
                )
                time.sleep(2)
            currenthistory = np.append(
                currenthistory, np.array([[Imax], [Imin]]), axis=1
            )
            resistancehistory = np.append(resistancehistory, resistance)
            self.graph2.emit(resistancehistory)

        # III) third attempt but with zap and longer suction
        self.progress.emit("Attempt 3/3 - ZAP and strong suction pulses")
        start = time.time()
        while (
            time.time() - start < TIMEOUT / 2 and not SUCCESS and not self.STOP
        ):
            Imax = np.max(self._parent.current)
            Imin = np.min(self._parent.current)
            resistance = np.nanmax(self._parent.resistance[-10::])
            capacitance = np.nanmean(self._parent.capacitance[-10::])
            if (
                Imax <= I_BREAKIN_CONDITION
                and Imin >= -I_BREAKIN_CONDITION
                and capacitance > C_BREAKIN_CONDITION[0]
                and capacitance < C_BREAKIN_CONDITION[1]
                and resistance <= R_BREAKIN_CONDITION
            ):
                SUCCESS = True
                break
            else:
                sealtestthread.zap()
                pressurecontroller.set_pulse_stop_waveform(-250)
                pressurecontroller.set_pressure_stop_waveform(-200)
                time.sleep(0.5)
                pressurecontroller.set_pulse_stop_waveform(-300)
                time.sleep(2)
            currenthistory = np.append(
                currenthistory, np.array([[Imax], [Imin]]), axis=1
            )
            resistancehistory = np.append(resistancehistory, resistance)
            self.graph2.emit(resistancehistory)

        if SUCCESS:
            self.progress.emit("Break-in successful")
        else:
            self.progress.emit("Failed to break in")

        # IV) measure and average sliding windows for saving
        slidingwindow_current = self._parent.current
        for i in range(0, 10):
            slidingwindow_current += self._parent.current
            time.sleep(0.1)
        slidingwindow_current = slidingwindow_current / 10

        # timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # np.save(save_directory+'breakin_currenthistory_'+timestamp, currenthistory)  #FLAG: relevant for MSc thesis
        # np.save(save_directory+'breakin_resistancehistory_'+timestamp, resistancehistory)  #FLAG: relevant for MSc thesis
        # np.save(save_directory+'breakin_slidingwindowcurrent_'+timestamp, slidingwindow_current)  #FLAG: relevant for MSc thesis

        self.status.emit("Break-in finished")
        self.finished.emit()
