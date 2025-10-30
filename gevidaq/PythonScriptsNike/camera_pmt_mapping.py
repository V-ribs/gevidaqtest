import logging
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep


class CameraPmtMapping:

    def create_affine_transformation_matrix(
        self, camera_vertices, pmt_vertices
    ):

        # Check that both sets have the same shape
        camera_vertices, pmt_vertices = np.array(camera_vertices), np.array(
            pmt_vertices
        )
        if pmt_vertices.shape != camera_vertices.shape:
            raise ValueError(
                "The input sets must have the same number of points and dimensions."
            )

        # Solve for the affine transformation parameters for x and y separately
        X = np.vstack(
            [camera_vertices.T, np.ones((1, camera_vertices.shape[0]))]
        ).T
        params_x = np.linalg.solve(X, pmt_vertices[:, 0])
        params_y = np.linalg.solve(X, pmt_vertices[:, 1])

        # Construct the 2x3 affine transformation matrix:
        affine_matrix = np.vstack([params_x, params_y])
        logging.info("Affine Transformation Matrix (2x3):")
        logging.info(affine_matrix)

        camera_transformed_affine = self.transform_contour(
            camera_vertices, affine_matrix
        )

        # Verify the mapping on the three vertices
        if np.allclose(camera_transformed_affine, pmt_vertices, atol=1e-8):
            logging.info(
                "Affine transformation exactly maps the camera vertices to the PMT vertices.\n"
            )
        else:
            logging.info(
                "Warning: Affine transformation did not perfectly map the vertices.\n"
            )

        return affine_matrix

    def transform_contour(self, camera_contour, affine_matrix):

        def affine_transform(point, affine_matrix):
            point_aug = np.hstack([point, 1])
            return affine_matrix @ point_aug

        return np.array(
            [affine_transform(pt, affine_matrix) for pt in camera_contour]
        )

    def _generate_correlated_triangles(self, pmt_dimension, camera_dimension):
        sl_cam = random.randint(camera_dimension // 20, camera_dimension // 4)
        sl_pmt = random.randint(pmt_dimension // 20, pmt_dimension // 4)

        def random_center(pixel_dim, side_length):
            border_patrol = 1.2
            x_center = random.randint(
                int(border_patrol * side_length),
                pixel_dim - int(border_patrol * side_length),
            )
            y_center = random.randint(
                int(border_patrol * side_length),
                pixel_dim - int(border_patrol * side_length),
            )
            return (x_center, y_center)

        center_cam = random_center(camera_dimension, sl_cam)
        center_pmt = random_center(pmt_dimension, sl_pmt)
        random_angle_cam = math.radians(random.randint(0, 360))
        random_angle_pmt = math.radians(random.randint(0, 360))

        angles = [
            random.uniform(i * 2 * math.pi / 3, (i + 1) * 2 * math.pi / 3)
            for i in range(3)
        ]

        def calculate_vertices(center, side_length, angles):
            return [
                (
                    center[0] + side_length * math.cos(angle),
                    center[1] + side_length * math.sin(angle),
                )
                for angle in angles
            ]

        vertices_cam = calculate_vertices(center_cam, sl_cam, angles)
        vertices_pmt = calculate_vertices(center_pmt, sl_pmt, angles)

        new_center_cam = np.mean(vertices_cam, axis=0, dtype=int)
        new_center_pmt = np.mean(vertices_pmt, axis=0, dtype=int)

        def rotate_vertices(vertices, center, angle):
            return [self._rotate_point(v, center, angle) for v in vertices]

        rot_vertices_cam = rotate_vertices(
            vertices_cam, new_center_cam, random_angle_cam
        )
        rot_vertices_pmt = rotate_vertices(
            vertices_pmt, new_center_pmt, random_angle_pmt
        )

        return rot_vertices_cam, rot_vertices_pmt

    def _rotate_point(self, point, center, angle):
        px, py = point
        cx, cy = center
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (
            cos_a * (px - cx) - sin_a * (py - cy) + cx,
            sin_a * (px - cx) + cos_a * (py - cy) + cy,
        )

    def _generate_random_contour(
        self, center, min_radius, max_radius, num_points
    ):
        x_center, y_center = center
        random_angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
        radii = np.random.uniform(min_radius, max_radius, num_points)
        x_coords = x_center + radii * np.cos(random_angles)
        y_coords = y_center + radii * np.sin(random_angles)
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        tck, _ = splprep([x_coords, y_coords], s=0, per=True)
        u_fine = np.linspace(0, 1, num_points * 10)
        smooth_x, smooth_y = splev(u_fine, tck)
        smooth_contour = list(zip(smooth_x, smooth_y))
        return smooth_contour

    def _generate_random_center(self, camera_contour):
        camera_x_center, camera_y_center = np.mean(camera_contour, axis=0)
        random_x_center = np.random.uniform(
            camera_x_center * 0.8, camera_x_center * 1.2
        )
        random_y_center = np.random.uniform(
            camera_y_center * 0.8, camera_y_center * 1.2
        )
        return (random_x_center, random_y_center)

    def _plot_vertices(self, ax, vertices, dimension):
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        x_coords.append(vertices[0][0])
        y_coords.append(vertices[0][1])

        colors = plt.colormaps.get_cmap("Blues")(
            np.linspace(0.3, 1, len(vertices))
        )

        for i, (x, y) in enumerate(zip(x_coords[:-1], y_coords[:-1])):
            ax.scatter(x, y, color=colors[i])

        ax.plot(x_coords, y_coords, linestyle="--", color="gray", alpha=0.1)
        center = np.mean(vertices, axis=0)
        ax.scatter(*center, color="red", s=30, marker="x")

        ax.set_xlim(-dimension * 0.05, dimension * 1.05)
        ax.set_ylim(-dimension * 0.05, dimension * 1.05)
        ax.set_aspect("equal", "box")

        ax.grid(True, alpha=0.5)

    def _plot_contour(self, ax, contour, name, dimension):
        x_coords = [c[0] for c in contour]
        y_coords = [c[1] for c in contour]

        light_to_dark_blue = plt.colormaps.get_cmap("Blues")(
            np.linspace(0.3, 1, len(contour))
        )

        ax.set_title(name)

        for i in range(len(contour) - 1):
            ax.plot(
                [x_coords[i], x_coords[i + 1]],
                [y_coords[i], y_coords[i + 1]],
                color=light_to_dark_blue[i],
                linewidth=2,
            )

        ax.set_title(f"{name} contour")
        ax.set_xlim(-dimension * 0.05, dimension * 1.05)
        ax.set_ylim(-dimension * 0.05, dimension * 1.05)
        ax.set_aspect("equal", "box")

        ax.grid(True, alpha=0.5)


if __name__ == "__main__":
    cpm = CameraPmtMapping()
    pmt_vertices = [(0, 0), (250, 500), (500, 100)]
    camera_vertices = [(1422, 1643), (570, 1210), (1256, 769)]
    pmt_dimension, camera_dimension = 500, 2048

    plot_calibration_points = True

    if plot_calibration_points:
        # Calculate the affine matrix and map the camera vertices
        affine_matrix = cpm.create_affine_transformation_matrix(
            camera_vertices, pmt_vertices
        )
        mapped_camera_vertices = cpm.transform_contour(
            camera_vertices, affine_matrix
        )

        # Plot the examples
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        # fig.suptitle("Camera and PMT images")

        axes[0, 0].set_title("Camera registration points")
        cpm._plot_vertices(axes[0, 0], camera_vertices, camera_dimension)

        axes[0, 1].set_title("PMT registration points")
        cpm._plot_vertices(axes[0, 1], pmt_vertices, pmt_dimension)
        # cpm._plot_vertices(axes[0, 1], mapped_camera_vertices, pmt_dimension)

        # Generate a random contour and apply the same transformation
        random_center = cpm._generate_random_center(camera_vertices)
        random_contour = cpm._generate_random_contour(
            random_center, 100, 200, 20
        )
        mapped_random_contour = cpm.transform_contour(
            random_contour, affine_matrix
        )

        cpm._plot_contour(
            axes[1, 0], random_contour, "Randomly generated", camera_dimension
        )
        cpm._plot_contour(
            axes[1, 1], mapped_random_contour, "Mapped", pmt_dimension
        )

        fig.tight_layout()
        plt.show()

    else:
        # Generate random vertices and their affine matrix
        random_camera_vertices, random_pmt_vertices = (
            cpm._generate_correlated_triangles(pmt_dimension, camera_dimension)
        )
        random_affine_matrix = cpm.create_affine_transformation_matrix(
            random_camera_vertices, random_pmt_vertices
        )
        mapped_random_camera_vertices = cpm.transform_contour(
            random_camera_vertices, random_affine_matrix
        )

        # Plot the examples
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle("Camera and PMT images (random)")

        axes[0, 0].set_title("Random camera image")
        cpm._plot_vertices(
            axes[0, 0], random_camera_vertices, camera_dimension
        )

        axes[0, 1].set_title("PMT and mapped images")
        cpm._plot_vertices(axes[0, 1], random_pmt_vertices, pmt_dimension)
        cpm._plot_vertices(
            axes[0, 1], mapped_random_camera_vertices, pmt_dimension
        )

        # Generate a random contour and apply the same transformation
        random_center = cpm._generate_random_center(random_camera_vertices)

        random_contour = cpm._generate_random_contour(
            random_center, 100, 200, 20
        )
        mapped_random_contour = cpm.transform_contour(
            random_contour, random_affine_matrix
        )

        cpm._plot_contour(
            axes[1, 0], random_contour, "Randomly generated", camera_dimension
        )
        cpm._plot_contour(
            axes[1, 1], mapped_random_contour, "Mapped", pmt_dimension
        )

        fig.tight_layout()
        plt.show()
