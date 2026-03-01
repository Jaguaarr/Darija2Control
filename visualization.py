"""Visualization utilities for N-dimensional systems."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict
import colorsys


class Visualizer:
    """Visualizer for symbolic control framework."""

    def __init__(self, projection_dims: Tuple[int, int] = (0, 1)):
        """
        Args:
            projection_dims: Which dimensions to plot (for N>2)
        """
        self.projection_dims = projection_dims
        self.fig = None
        self.ax = None

    def plot_regions(self, region_definitions: Dict[str, List[Tuple[float, float]]],
                     workspace_bounds: List[Tuple[float, float]]):
        """Plot workspace regions."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Generate colors for regions
        colors = []
        for i in range(len(region_definitions)):
            hue = i / len(region_definitions)
            colors.append(colorsys.hsv_to_rgb(hue, 0.7, 0.9))

        for (region_name, bounds_list), color in zip(region_definitions.items(), colors):
            if len(bounds_list) >= 2:
                # 2D plot
                x_bounds = bounds_list[self.projection_dims[0]]
                y_bounds = bounds_list[self.projection_dims[1]]

                rect = Rectangle(
                    (x_bounds[0], y_bounds[0]),
                    x_bounds[1] - x_bounds[0],
                    y_bounds[1] - y_bounds[0],
                    alpha=0.3, color=color, label=region_name
                )
                self.ax.add_patch(rect)

        self.ax.set_xlim(workspace_bounds[self.projection_dims[0]])
        self.ax.set_ylim(workspace_bounds[self.projection_dims[1]])
        self.ax.set_xlabel(f'x{self.projection_dims[0] + 1}')
        self.ax.set_ylabel(f'x{self.projection_dims[1] + 1}')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def plot_trajectory(self, trajectory: List[np.ndarray],
                        color: str = 'blue', label: str = 'Trajectory',
                        show_points: bool = True):
        """Plot trajectory in projected dimensions."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        traj_proj = np.array([[s[d] for d in self.projection_dims]
                              for s in trajectory])

        if show_points:
            self.ax.scatter(traj_proj[:, 0], traj_proj[:, 1],
                            s=20, color=color, alpha=0.6, zorder=3)

        self.ax.plot(traj_proj[:, 0], traj_proj[:, 1],
                     '-', color=color, linewidth=2, label=label, zorder=2)

    def plot_cells(self, cells, color: str = 'gray', alpha: float = 0.1):
        """Plot partition cells."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        for cell in cells:
            bounds = cell.bounds
            x_bounds = bounds[self.projection_dims[0]]
            y_bounds = bounds[self.projection_dims[1]]

            rect = Rectangle(
                (x_bounds[0], y_bounds[0]),
                x_bounds[1] - x_bounds[0],
                y_bounds[1] - y_bounds[0],
                alpha=alpha, color=color, linewidth=0.5, edgecolor='black'
            )
            self.ax.add_patch(rect)

    def plot_3d(self, trajectory: List[np.ndarray],
                dims: Tuple[int, int, int] = (0, 1, 2)):
        """3D plot of trajectory."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        traj_3d = np.array([[s[d] for d in dims] for s in trajectory])

        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                'b-', linewidth=2, label='Trajectory')
        ax.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2],
                   color='green', s=100, label='Start')
        ax.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2],
                   color='red', s=100, label='End')

        ax.set_xlabel(f'x{dims[0] + 1}')
        ax.set_ylabel(f'x{dims[1] + 1}')
        ax.set_zlabel(f'x{dims[2] + 1}')
        ax.legend()
        ax.grid(True)

        return fig, ax

    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()

    def save(self, filename: str):
        """Save plot to file."""
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')