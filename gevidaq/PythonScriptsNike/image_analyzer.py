import matplotlib.pyplot as plt
import numpy as np
import tifffile as skimtiff
from matplotlib.widgets import Slider
from pyqtgraph import ImageView
from scipy.interpolate import splev, splprep
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from skimage.measure import find_contours


class ImageAnalyzer:
    def __init__(self, file_path=None, intensity_threshold=3000):

        # Load the TIFF image
        if file_path:
            image = skimtiff.imread(file_path)
            self.tiff_image = image
            self.image_dimension = (0, self.tiff_image.shape[1])

        self.intensity_threshold = intensity_threshold

    def update_threshold(self, val):
        self.ui_contour = self.find_contour(self.tiff_image, val)
        self.slider_scaling.set_val(1)
        self.update_plots(self.ui_contour, self.slider_window_size.val, 1)

    def update_window_size(self, val):
        contour = (
            self.ui_contour
            if hasattr(self, "ui_contour")
            else self.base_contour
        )
        self.update_plots(contour, val, self.slider_scaling.val)

    def update_scaling(self, val):
        contour = (
            self.ui_contour
            if hasattr(self, "ui_contour")
            else self.base_contour
        )
        self.update_plots(contour, self.slider_window_size.val, val)

    def update_plots(self, contour, window_size, scaling):
        smoothed_contour = self.smoothen_contour(contour, window_size)
        scaled_contour = self.resize_contour(smoothed_contour, scaling)
        self.varying_intensity_plot.set_data(contour[:, 1], contour[:, 0])
        self.smoothened_curve_plot.set_data(
            scaled_contour[:, 1], scaled_contour[:, 0]
        )
        plt.draw()

    def resize_contour(self, contour, scaling_factor):
        """Resize the contour without crossing existing contours and unfold when increasing size."""
        # Create a polygon from the contour
        polygon = Polygon(contour)

        # Calculate the centroid of the polygon
        centroid = polygon.centroid

        # Scale the polygon
        scaled_polygon = scale(
            polygon,
            xfact=scaling_factor,
            yfact=scaling_factor,
            origin=centroid,
        )

        # Ensure the scaled polygon does not cross existing contours
        if scaling_factor < 1:
            # Buffer with a very small value to clean up the geometry
            scaled_polygon = scaled_polygon.buffer(0)
            polygon = polygon.buffer(0)
            scaled_polygon = scaled_polygon.intersection(polygon)

        # Handle MultiPolygon case
        if isinstance(scaled_polygon, MultiPolygon):
            # Extract the largest polygon by area
            scaled_polygon = max(scaled_polygon.geoms, key=lambda p: p.area)

        # Convert the scaled polygon back to a contour
        scaled_contour = np.array(scaled_polygon.exterior.coords)

        return scaled_contour

    def find_contour(self, image, threshold, num_points=100):
        """Find the largest contour in the image based on the given intensity threshold."""
        contours = find_contours(image, threshold)
        largest_contour = max(contours, key=len)

        # Interpolate the contour to get evenly distributed points
        tck, u = splprep([largest_contour[:, 1], largest_contour[:, 0]], s=0)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck)

        # Combine x and y coordinates
        evenly_distributed_contour = np.vstack((y_new, x_new)).T

        return evenly_distributed_contour

    def display_multiple_contours(
        self,
        ax,
        image,
        minimum_intensity,
        maximum_intensity,
        number_of_contours,
    ):
        """Display multiple contours in the image based on the given intensity thresholds."""
        colors = plt.colormaps.get_cmap("Blues")(
            np.linspace(0.3, 1, number_of_contours + 1)
        )

        for i in range(number_of_contours + 1):
            threshold = int(
                minimum_intensity
                + i
                * (maximum_intensity - minimum_intensity)
                / number_of_contours
            )
            largest_contour = self.find_contour(image, threshold)
            ax.plot(
                largest_contour[:, 1],
                largest_contour[:, 0],
                color=colors[i],
                linewidth=1.5,
                label="Intensity = " + str(threshold),
            )

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.8))

    def smoothen_contour(self, contour, window_size):

        if window_size <= 1 or len(contour) < window_size:
            return contour  # No smoothing needed

        x = contour[:, 1]
        y = contour[:, 0]

        # Make sure the contour is cyclic before smoothing (avoid edge effects)
        x_extended = np.concatenate([x[-window_size:], x, x[:window_size]])
        y_extended = np.concatenate([y[-window_size:], y, y[:window_size]])

        # Apply moving average with 'same' mode to maintain same number of points
        kernel = np.ones(window_size) / window_size
        x_smooth = np.convolve(x_extended, kernel, mode="same")
        y_smooth = np.convolve(y_extended, kernel, mode="same")

        # Extract the middle portion to remove padding effects
        x_smooth = x_smooth[window_size:-window_size]
        y_smooth = y_smooth[window_size:-window_size]

        # Stack back into (y, x) format
        smoothened_contour = np.column_stack((y_smooth, x_smooth))

        return smoothened_contour

    def zoom_in_on_roi(self, ax, contour, factor=1.25):
        """Zoom in on the region of interest (ROI) based on the contour."""
        x_left, x_right = int(np.min(contour[:, 1])), int(
            np.max(contour[:, 1])
        )
        x_diff = x_right - x_left
        y_top, y_bottom = int(np.min(contour[:, 0])), int(
            np.max(contour[:, 0])
        )
        y_diff = y_bottom - y_top
        x_centroid, y_centroid = (x_left + x_right) // 2, (
            y_top + y_bottom
        ) // 2

        x_roi = (x_centroid - factor * x_diff, x_centroid + factor * x_diff)
        y_roi = (y_centroid - factor * y_diff, y_centroid + factor * y_diff)
        roi = (x_roi, y_roi)

        if ax is not None:
            ax.set_xlim(roi[0])
            ax.set_ylim(roi[1][::-1])

        return roi

    def determine_thresholds(self, image):
        """Determine intensity thresholds using IQR and ImageView method."""
        q1, q3 = np.percentile(image.flatten(), [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        imv = ImageView()
        imv.setImage(image)
        min_max = imv.quickMinMax(image)

        return max(lower_bound, 0), max(upper_bound, min_max[0][1])

    def auto_level_image(self, image, lower_bound, upper_bound):
        """Auto-level the image based on the given intensity range."""
        image = np.clip(image, lower_bound, upper_bound)  # Clip outliers
        return (
            (image - lower_bound) / (upper_bound - lower_bound) * upper_bound
        )

    def display_tiff_image(self, ax, lower_bound, upper_bound, alpha=1.0):
        """Display the TIFF image with auto-leveling."""
        leveled_image = self.auto_level_image(
            self.tiff_image, lower_bound, upper_bound
        )
        ax.imshow(leveled_image, cmap="gray", aspect="equal", alpha=alpha)


if __name__ == "__main__":
    image_analyzer = ImageAnalyzer(
        r"M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/People/Xin Meng/Code/Python_test_TF2/cell0000.TIF"  # TODO hardcoded path
    )

    # Analyze and display the image
    fig, axes = plt.subplots(2, 2)

    # Determine intensity thresholds
    lower_bound, upper_bound = image_analyzer.determine_thresholds(
        image_analyzer.tiff_image
    )
    image_analyzer.base_contour = image_analyzer.find_contour(
        image_analyzer.tiff_image, image_analyzer.intensity_threshold
    )
    image_analyzer.base_smoothened_curve = image_analyzer.smoothen_contour(
        image_analyzer.base_contour, 10
    )

    # Plot original, zoomed-in image
    axes[0, 0].set_title("Original image (zoomed in)")
    image_analyzer.display_tiff_image(axes[0, 0], lower_bound, upper_bound)
    image_analyzer.zoom_in_on_roi(axes[0, 0], image_analyzer.base_contour)

    # Plot contours at different intensities
    axes[0, 1].set_title("Contours at different intensities")
    image_analyzer.display_tiff_image(axes[0, 1], lower_bound, upper_bound)
    image_analyzer.display_multiple_contours(
        axes[0, 1], image_analyzer.tiff_image, 2000, 6000, 4
    )
    image_analyzer.zoom_in_on_roi(axes[0, 1], image_analyzer.base_contour)

    # Plot contour with UI for intensity threshold
    axes[1, 0].set_title("Contour with varying intensity")
    image_analyzer.display_tiff_image(axes[1, 0], lower_bound, upper_bound)
    (image_analyzer.varying_intensity_plot,) = axes[1, 0].plot(
        image_analyzer.base_contour[:, 1],
        image_analyzer.base_contour[:, 0],
        color="#AFEEEE",
        linewidth=1,
    )
    image_analyzer.zoom_in_on_roi(axes[1, 0], image_analyzer.base_contour)

    # Create a horizontal slider for threshold adjustment
    ax_threshold = plt.axes(
        [0.12, 0.02, 0.2, 0.03], facecolor="lightgoldenrodyellow"
    )
    image_analyzer.slider_threshold = Slider(
        ax_threshold,
        "Intensity",
        lower_bound,
        upper_bound // 2,
        valinit=image_analyzer.intensity_threshold,
        valstep=50,
    )

    # Plot smoothened curve with UI for window size (smoothness)
    axes[1, 1].set_title("Smoothened contour")
    image_analyzer.display_tiff_image(axes[1, 1], lower_bound, upper_bound)
    (image_analyzer.smoothened_curve_plot,) = axes[1, 1].plot(
        image_analyzer.base_smoothened_curve[:, 1],
        image_analyzer.base_smoothened_curve[:, 0],
        color="#AFEEEE",
        linewidth=1,
    )
    image_analyzer.zoom_in_on_roi(axes[1, 1], image_analyzer.base_contour)

    # Create a horizontal slider for the smoothing window size
    ax_window_size = plt.axes(
        [0.48, 0.02, 0.2, 0.03], facecolor="lightgoldenrodyellow"
    )
    image_analyzer.slider_window_size = Slider(
        ax_window_size, "Smoothness", 1, 20, valinit=10, valstep=1
    )

    # Create a vertical slider for scaling the contour
    ax_scaling = plt.axes(
        [0.70, 0.1, 0.02, 0.3], facecolor="lightgoldenrodyellow"
    )
    image_analyzer.slider_scaling = Slider(
        ax_scaling,
        "Scaling",
        0.5,
        1.5,
        valinit=1,
        valstep=0.05,
        orientation="vertical",
    )

    # Update function for the sliders
    image_analyzer.slider_threshold.on_changed(image_analyzer.update_threshold)
    image_analyzer.slider_window_size.on_changed(
        image_analyzer.update_window_size
    )
    image_analyzer.slider_scaling.on_changed(image_analyzer.update_scaling)

    fig.tight_layout()
    plt.show()
