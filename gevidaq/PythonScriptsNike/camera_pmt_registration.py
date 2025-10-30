import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import curve_fit


class CameraPmtRegistration:

    def generateData(
        self, size=2048, amplitude=50, sigma_x=50, sigma_y=50, noise_level=0.2
    ):
        """Generate a 2D Gaussian with noise."""
        x = np.linspace(0, size - 1, size)
        y = np.linspace(0, size - 1, size)
        x, y = np.meshgrid(x, y)
        xo = np.random.randint(size / 4, 3 * size / 4)
        yo = np.random.randint(size / 4, 3 * size / 4)
        gauss = amplitude * np.exp(
            -(
                ((x - xo) ** 2) / (2 * sigma_x**2)
                + ((y - yo) ** 2) / (2 * sigma_y**2)
            )
        )
        noise = noise_level * np.random.normal(size=gauss.shape)
        data = gauss + noise
        return data

    def gaussian(self, x, y, amplitude, xo, yo, sigma_x, sigma_y, offset):
        """2D Gaussian function."""
        return offset + amplitude * np.exp(
            -(
                ((x - xo) ** 2) / (2 * sigma_x**2)
                + ((y - yo) ** 2) / (2 * sigma_y**2)
            )
        )

    def fitDoubleGaussian(self, data):
        """Fit a 2D Gaussian to the data and retrieve the coordinates with the highest intensity."""

        def gaussian_fit(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
            x, y = coords
            return self.gaussian(
                x, y, amplitude, xo, yo, sigma_x, sigma_y, offset
            ).ravel()

        # Downsample the data to speed up the fitting process
        downsample_factor = 0.25
        data_downsampled = zoom(data, downsample_factor)
        x = np.linspace(
            0, data_downsampled.shape[1] - 1, data_downsampled.shape[1]
        )
        y = np.linspace(
            0, data_downsampled.shape[0] - 1, data_downsampled.shape[0]
        )
        x, y = np.meshgrid(x, y)

        max_intensity = np.max(data_downsampled)
        xy_tup = np.where(data_downsampled == max_intensity)
        xmax, ymax = xy_tup[1][0], xy_tup[0][0]
        offset = np.mean(data_downsampled)
        initial_guess = (max_intensity, xmax, ymax, 10, 10, offset)

        popt, _ = curve_fit(
            gaussian_fit, (x, y), data_downsampled.ravel(), p0=initial_guess
        )
        amplitude, xo, yo, sigma_x, sigma_y, offset = popt

        # Scale the coordinates back to the original data size
        xo = int(xo / downsample_factor)
        yo = int(yo / downsample_factor)
        sigma_x = sigma_x / downsample_factor
        sigma_y = sigma_y / downsample_factor

        logging.info(
            f"Amplitude: {amplitude:.2f}, xo: {xo}, yo: {yo}, sigma_x: {sigma_x:.2f}, sigma_y: {sigma_y:.2f}, offset: {offset:.2f}"
        )
        return amplitude, xo, yo, sigma_x, sigma_y, offset

    def _plot_registration_gaussian(self, registration_params, data_shape):
        amplitude, xo, yo, sigma_x, sigma_y, offset = registration_params

        # Generate a grid of x and y values
        x = np.linspace(0, data_shape[1] - 1, data_shape[1])
        y = np.linspace(0, data_shape[0] - 1, data_shape[0])
        x, y = np.meshgrid(x, y)

        # Define the 2D Gaussian function
        def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, offset):
            return offset + amplitude * np.exp(
                -(
                    ((x - xo) ** 2) / (2 * sigma_x**2)
                    + ((y - yo) ** 2) / (2 * sigma_y**2)
                )
            )

        # Generate the Gaussian data
        gaussian_data = gaussian(
            x, y, amplitude, xo, yo, sigma_x, sigma_y, offset
        )

        # Plot the Gaussian data
        plt.imshow(
            gaussian_data,
            extent=(0, data_shape[1], 0, data_shape[0]),
            origin="lower",
        )
        plt.title("Fitted 2D Gaussian")
        plt.colorbar()

        # Plot the contour
        plt.contour(
            x, y, gaussian_data, levels=[amplitude / 2], colors="r", alpha=0.5
        )

        # Plot a cross at the center of the Gaussian
        plt.scatter([xo], [yo], color="r", marker="+", s=50)

        plt.show()


def main():
    yahh = CameraPmtRegistration()
    data = yahh.generateData()

    amp, xo, yo, sx, sy, offset = yahh.fitDoubleGaussian(data)
    logging.info(f"Coordinates with the highest intensity: ({xo}, {yo})")

    # plt.imshow(data, extent=(0, 2048, 0, 2048), origin='lower')
    # plt.title("Generated 2D Gaussian with Noise")
    # plt.colorbar()
    # plt.show()

    # Plot the fitted Gaussian
    yahh._plot_registration_gaussian((amp, xo, yo, sx, sy, offset), data.shape)


if __name__ == "__main__":
    main()
