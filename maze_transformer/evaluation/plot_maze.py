from __future__ import annotations  # for type hinting self as return value

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from muutils.tensor_utils import NDArray

from maze_transformer.generation.latticemaze import CoordArray, LatticeMaze


@dataclass
class PathFormat:
    """formatting options for path plot
    (path_true, "true", "-", "red")
    """

    path: CoordArray
    label: str | None = None
    fmt: str | None = None
    color: str | None = None
    line_width: float | None = None
    quiver_kwargs: dict | None = (None,)


class MazePlot:
    """Class for displaying mazes and paths"""
    DEFAULT_PREDICTED_PATH_COLORS = [
        "tab:blue",
        "tab:green",
        "tab:purple",
        "tab:orange",
        "tab:olive",
        "tab:cyan",
    ]

    def __init__(self, maze: LatticeMaze) -> None:
        """
        UNIT_LENGTH: Set ratio between node size and wall thickness in image.
        Wall thickness is fixed to 1px
        A "unit" consists of a single node and the right and lower connection/wall.
        Example: ul = 14 yields 13:1 ratio between node size and wall thickness
        """
        self.unit_length: int = 14
        self.maze: LatticeMaze = maze
        self.true_path: PathFormat = None
        self.predicted_paths: list = []
        self.node_values: NDArray(self.maze.grid_shape) = None

    def add_true_path(
        self,
        path: CoordArray,
        label: str = "true path",
        fmt: str = "--",
        color: str = "red",
        line_width: float = 2.5,
        quiver_kwargs: dict | None = None,
    ) -> MazePlot:
        """
        Recieve true path and formatting preferences from input and save as dict in true_path variable.
        """
        self.true_path = PathFormat(
            path=path,
            label=label,
            fmt=fmt,
            color=color,
            line_width=line_width,
            quiver_kwargs=quiver_kwargs,
        )
        return self

    def add_predicted_path(
        self,
        path: CoordArray,
        label: str | None = None,
        fmt: str = ":",
        color: str | None = None,
        line_width: float = 2,
        quiver_kwargs: dict | None = {"width": 0.015},
    ) -> MazePlot:
        """
        Recieve predicted path and formatting preferences from input and save in predicted_path list.
        Default formatting depends on nuber of paths already saved in predicted path list.
        """
        if label is None:
            label = "predicted path {path_num}".format(
                path_num=len(self.predicted_paths) + 1
            )
        if color is None:
            color_num = len(self.predicted_paths) % len(
                self.DEFAULT_PREDICTED_PATH_COLORS
            )
            color = self.DEFAULT_PREDICTED_PATH_COLORS[color_num]

        self.predicted_paths.append(
            PathFormat(
                path=path,
                label=label,
                fmt=fmt,
                color=color,
                line_width=line_width,
                quiver_kwargs=quiver_kwargs,
            )
        )
        return self

    def add_multiple_paths(self, path_list: list):
        """
        Function for adding multiple paths to MazePlot at once. This can be done in two ways:
        1. Passing a list of
        """
        for path in path_list:
            if type(path) == PathFormat:
                self.predicted_paths.append(path)
            else:
                self.add_predicted_path(path=path)
        return self

    def add_node_values(
            self,
            node_values: NDArray
    ) -> MazePlot:
        # Always true warning??
        # assert(node_values.shape == self.maze.grid_shape,
        #            'Passed node values have to have the same shape as the maze.')
            #TODO normalize node values
        self.node_values = node_values
        return self
    
    def show(self) -> None:
        """Plot the maze and paths."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        self._plot_maze()

        if self.true_path is not None:
            self._plot_path(self.true_path)
        for path in self.predicted_paths:
            self._plot_path(path)

        # Plot labels
        tick_arr = np.arange(self.maze.grid_shape[0])
        self.ax.set_xticks(self.UNIT_LENGTH * (tick_arr + 0.5), tick_arr)
        self.ax.set_yticks(self.UNIT_LENGTH * (tick_arr + 0.5), tick_arr)
        self.ax.set_xlabel("col")
        self.ax.set_ylabel("row")

        plt.show()

    def _rowcol_to_coord(self, points: CoordArray) -> NDArray:
        """Transform Points from MazeTransformer (row, column) notation to matplotlib default (x, y) notation where x is the horizontal axis."""
        points = np.array([(x, y) for (y, x) in points])
        return self.UNIT_LENGTH * (points + 0.5)

    def _plot_maze(self) -> None:
        img = self._latticemaze_to_img()
        self.ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    def _latticemaze_to_img(self) -> NDArray["row col", bool]:
        """
        Build an image to visualise the maze.
        Each "unit" consists of a node and the right and lower adjacent wall/connection. Its area is ul * ul.
        - Nodes have area: (ul-1) * (ul-1) and value 1 by default
            - take node_value if passed via .add_node_values()
        - Walls have area: 1 * (ul-1) and value -1
        - Connections have area: 1 * (ul-1); color and value 0.93 by default
            - take node_value if passed via .add_node_values()

        Axes definition:
        (0,0)     col
        ----|----------->
            |
        row |
            |
            v

        Returns a matrix of side length (ul) * n + 1 where n is the number of nodes.
        """
        

        # Set node and connection values
        if self.node_values is None:
            self.node_values = np.ones(self.maze.grid_shape)
            connection_values = np.ones(self.maze.grid_shape) * 0.93
        else:
            connection_values = self.node_values

        # Create background image (all pixels set to -1, walls everywhere)
        img: NDArray["row col", int] = -np.ones(
            (
                self.maze.grid_shape[0] * self.unit_length + 1,
                self.maze.grid_shape[1] * self.unit_length + 1,
            ),
            dtype=float,
        )

        # Draw nodes and connections by iterating through lattice
        for row in range(self.maze.grid_shape[0]):
            for col in range(self.maze.grid_shape[1]):
                # Draw node
                img[
                    row * self.unit_length + 1 : (row + 1) * self.unit_length,
                    col * self.unit_length + 1 : (col + 1) * self.unit_length,
                ] = self.node_values[row, col]

                # Down connection
                if self.maze.connection_list[0, row, col]:
                    img[
                        (row + 1) * self.unit_length,
                        col * self.unit_length + 1 : (col + 1) * self.unit_length,
                    ] = connection_values[row, col]

                # Right connection
                if self.maze.connection_list[1, row, col]:
                    img[
                        row * self.unit_length + 1 : (row + 1) * self.unit_length,
                        (col + 1) * self.unit_length,
                    ] = connection_values[row, col]

        return img

    def _plot_path(self, path_format: PathFormat) -> None:
        p_transformed = self._rowcol_to_coord(path_format.path)
        if path_format.quiver_kwargs is not None:
            x: NDArray = p_transformed[:, 0]
            y: NDArray = p_transformed[:, 1]
            self.ax.quiver(
                x[:-1],
                y[:-1],
                x[1:] - x[:-1],
                y[1:] - y[:-1],
                scale_units="xy",
                angles="xy",
                scale=1,
                color=path_format.color,
                **path_format.quiver_kwargs,
            )
        else:
            self.ax.plot(
                *zip(*p_transformed),
                path_format.fmt,
                lw=path_format.line_width,
                color=path_format.color,
                label=path_format.label,
            )
        # mark endpoints
        self.ax.plot([p_transformed[0][0]], [p_transformed[0][1]], "o", color=path_format.color)
        self.ax.plot([p_transformed[-1][0]], [p_transformed[-1][1]], "x", color=path_format.color)

    def as_ascii(self, start=None, end=None):
        """
        Returns an ASCII visualization of the maze.
        Courtesy of ChatGPT
        """
        wall_char = "#"
        path_char = " "
        self.unit_length = 2

        # Determine the size of the maze
        maze = self._latticemaze_to_img()
        n_rows, n_cols = maze.shape
        maze_str = ""

        # Iterate through each element of the maze and print the appropriate symbol
        for i in range(n_rows):
            for j in range(n_cols):
                if start is not None and start[0] == i - 1 and start[1] == j - 1:
                    maze_str += "S"
                elif end is not None and end[0] == i - 1 and end[1] == j - 1:
                    maze_str += "E"
                elif maze[i, j]:
                    maze_str += path_char
                else:
                    maze_str += wall_char
            maze_str += "\n"  # Start a new line after each row
        return maze_str
