import random
import typing
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Bool, Int, Shaped
from muutils.json_serialize.serializable_dataclass import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.misc import list_split
from muutils.tensor_utils import NDArray

from maze_transformer.generation.constants import (
    NEIGHBORS_MASK,
    SPECIAL_TOKENS,
    Coord,
    CoordArray,
    CoordTup,
)
from maze_transformer.utils.token_utils import (
    get_adj_list_tokens,
    get_path_tokens,
    tokens_to_coords,
)

RGB = tuple[int, int, int]

PixelGrid = Int[np.ndarray, "x y rgb"]
BinaryPixelGrid = Bool[np.ndarray, "x y"]
ConnectionList = Bool[np.ndarray, "lattice_dim x y"]


@dataclass(frozen=True)
class PixelColors:
    WALL: RGB = (0, 0, 0)
    OPEN: RGB = (255, 255, 255)
    START: RGB = (0, 255, 0)
    END: RGB = (255, 0, 0)
    PATH: RGB = (0, 0, 255)


@dataclass(frozen=True)
class AsciiChars:
    WALL: str = "#"
    OPEN: str = " "
    START: str = "S"
    END: str = "E"
    PATH: str = "X"


ASCII_PIXEL_PAIRINGS: dict[str, RGB] = {
    AsciiChars.WALL: PixelColors.WALL,
    AsciiChars.OPEN: PixelColors.OPEN,
    AsciiChars.START: PixelColors.START,
    AsciiChars.END: PixelColors.END,
    AsciiChars.PATH: PixelColors.PATH,
}


def coord_str_to_tuple(coord_str: str) -> CoordTup:
    """convert a coordinate string to a tuple"""

    stripped: str = coord_str.lstrip("(").rstrip(")")
    return tuple(int(x) for x in stripped.split(","))


@serializable_dataclass(frozen=True, kw_only=True)
class LatticeMaze(SerializableDataclass):
    """lattice maze (nodes on a lattice, connections only to neighboring nodes)

    Connection List represents which nodes (N) are connected in each direction.

    First and second elements represent rightward and downward connections,
    respectively.

    Example:
      Connection list:
        [
          [ # down
            [F T],
            [F F]
          ],
          [ # right
            [T F],
            [T F]
          ]
        ]

      Nodes with connections
        N T N F
        F   T
        N T N F
        F   F

      Graph:
        N - N
            |
        N - N

    Note: the bottom row connections going down, and the
    right-hand connections going right, will always be False.
    """

    connection_list: ConnectionList
    generation_meta: dict | None = serializable_field(default=None, compare=False)
    lattice_dim: int = serializable_field(default=2)

    grid_shape = property(lambda self: self.connection_list.shape[1:])

    n_connections = property(lambda self: self.connection_list.sum())

    # ============================================================
    # basic methods
    # ============================================================
    @staticmethod
    def heuristic(a: CoordTup, b: CoordTup) -> float:
        """return manhattan distance between two points"""
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    def nodes_connected(self, a: Coord, b: Coord, /) -> bool:
        """returns whether two nodes are connected"""
        delta: Coord = b - a
        if np.abs(delta).sum() != 1:
            # return false if not even adjacent
            return False
        else:
            # test for wall
            dim: int = np.argmax(np.abs(delta))
            clist_node: Coord = a if (delta.sum() > 0) else b
            return self.connection_list[dim, clist_node[0], clist_node[1]]

    def get_coord_neighbors(self, c: Coord) -> CoordArray:
        neighbors: list[Coord] = [
            neighbor
            for neighbor in (c + NEIGHBORS_MASK)
            if (
                (0 <= neighbor[0] < self.grid_shape[0])  # in x bounds
                and (0 <= neighbor[1] < self.grid_shape[1])  # in y bounds
                and self.nodes_connected(c, neighbor)  # connected
            )
        ]

        output: CoordArray = np.array(neighbors)
        if len(neighbors) > 0:
            assert output.shape == (
                len(neighbors),
                2,
            ), f"invalid shape: {output.shape}, expected ({len(neighbors)}, 2))\n{c = }\n{neighbors = }\n{self.as_ascii()}"
        return output

    def find_shortest_path(
        self,
        c_start: CoordTup,
        c_end: CoordTup,
    ) -> list[CoordTup]:
        """find the shortest path between two coordinates, using A*"""
        c_start = tuple(c_start)
        c_end = tuple(c_end)

        # cheapest currently known path to node from start
        g_score: dict[CoordTup, float] = {c_start: 0.0}
        # estimated total cost of path through a node: f_score[c] = g_score[c] + heuristic(c, c_end)
        f_score: dict[CoordTup, float] = {c_start: self.heuristic(c_start, c_end)}

        nodes_evaluated: set[CoordTup] = set()
        nodes_to_evaluate: set[CoordTup] = {c_start}

        # node immediately preceding each node in the path (currently known shortest path)
        source: dict[CoordTup, CoordTup] = dict()

        while nodes_to_evaluate:
            # get lowest f_score node
            current_node: CoordTup = min(nodes_to_evaluate, key=lambda c: f_score[c])

            # check if goal is reached
            if current_node == c_end:
                path: list[CoordTup] = [current_node]
                path_current: CoordTup = current_node
                while path_current in source:
                    path_current = source[path_current]
                    path.append(path_current)
                return path[::-1]

            # close current node
            nodes_evaluated.add(current_node)
            nodes_to_evaluate.remove(current_node)

            neighbor: CoordTup
            for neighbor in map(tuple, self.get_coord_neighbors(current_node)):
                if neighbor in nodes_evaluated:
                    continue

                neighbor_g_score: float = g_score[current_node] + 1

                if neighbor not in nodes_to_evaluate:
                    nodes_to_evaluate.add(neighbor)
                elif neighbor_g_score >= g_score[neighbor]:
                    # if already knew about this one, but current g_score is worse, skip
                    continue

                # store g_score and source
                source[neighbor] = current_node
                g_score[neighbor] = neighbor_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, c_end)

    def get_nodes(self) -> list[CoordTup]:
        """return a list of all nodes in the maze"""

        return [
            (row, col)
            for row in range(self.grid_shape[0])
            for col in range(self.grid_shape[1])
        ]

    def generate_random_path(self) -> list[CoordTup]:
        """ "return a path between randomly chosen start and end nodes"""

        # we can't create a "path" in a single-node maze
        assert self.grid_shape[0] > 1 and self.grid_shape[1] > 1

        start, end = random.sample(self.get_nodes(), 2)
        return self.find_shortest_path(start, end)

    # ============================================================
    # to and from adjacency list
    # ============================================================
    def as_adj_list(
        self, shuffle_d0: bool = True, shuffle_d1: bool = True
    ) -> NDArray["conn start_end coord", np.int8]:
        adj_list: NDArray["conn start_end coord", np.int8] = np.full(
            (self.n_connections, 2, 2),
            -1,
        )

        if shuffle_d1:
            flip_d1: NDArray["conn", np.float16] = np.random.rand(self.n_connections)

        # loop over all nonzero elements of the connection list
        i: int = 0
        for d, x, y in np.ndindex(self.connection_list.shape):
            if self.connection_list[d, x, y]:
                c_start: CoordTup = (x, y)
                c_end: CoordTup = (
                    x + (1 if d == 0 else 0),
                    y + (1 if d == 1 else 0),
                )
                adj_list[i, 0] = np.array(c_start)
                adj_list[i, 1] = np.array(c_end)

                # flip if shuffling
                if shuffle_d1 and (flip_d1[i] > 0.5):
                    c_s, c_e = adj_list[i, 0].copy(), adj_list[i, 1].copy()
                    adj_list[i, 0] = c_e
                    adj_list[i, 1] = c_s

                i += 1

        if shuffle_d0:
            np.random.shuffle(adj_list)

        return adj_list

    @classmethod
    def from_adj_list(
        cls,
        adj_list: NDArray["conn start_end coord", np.int8],
    ) -> "LatticeMaze":
        """create a LatticeMaze from a list of connections"""

        # Note: This has only been tested for square mazes. Might need to change some things if rectangular mazes are needed.
        grid_n: int = adj_list.max() + 1

        connection_list: NDArray["lattice_dim x y", bool] = np.zeros(
            (2, grid_n, grid_n),
            dtype=bool,
        )

        for c_start, c_end in adj_list:
            # check that exactly 1 coordinate matches
            if (c_start == c_end).sum() != 1:
                raise ValueError("invalid connection")

            # get the direction
            d: int = (c_start != c_end).argmax()

            x: int
            y: int
            # pick whichever has the lesser value in the direction `d`
            if c_start[d] < c_end[d]:
                x, y = c_start
            else:
                x, y = c_end

            connection_list[d, x, y] = True

        return LatticeMaze(
            connection_list=connection_list,
        )

    # ============================================================
    # TODO: write a to_tokens method?
    # from tokens
    # ============================================================
    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "LatticeMaze":
        """create a LatticeMaze from a list of tokens"""
        if tokens[0] == SPECIAL_TOKENS["adj_list_start"]:
            adj_list_tokens = get_adj_list_tokens(tokens)
        else:
            # If we're not getting a "complete" tokenized maze, assume it's a list of coord tokens already
            adj_list_tokens = tokens

        edges: list[str] = list_split(
            adj_list_tokens,
            SPECIAL_TOKENS["adjacency_endline"],
        )

        coordinates: list[tuple[str, str]] = list()
        # what we're doing here:
        # for each edge, convert to a coord tuple and add it to the list
        # check that for each edge we have a connector, and a single token on each side
        for e in edges:
            # skip last endline
            if len(e) != 0:
                # split into start and end
                e_split: list[list] = list_split(e, SPECIAL_TOKENS["connector"])
                # print(f"{e = } {e_split = }")
                assert len(e_split) == 2, f"invalid edge: {e = } {e_split = }"
                assert all(
                    len(c) == 1 for c in e_split
                ), f"invalid edge: {e = } {e_split = }"
                coordinates.append(tuple([c[0] for c in e_split]))

        assert all(
            len(c) == 2 for c in coordinates
        ), f"invalid coordinates: {coordinates = }"

        adj_list: NDArray["conn start_end coord", np.int8] = np.full(
            (len(coordinates), 2, 2),
            -1,
        )

        for i, (c_start, c_end) in enumerate(coordinates):
            adj_list[i, 0] = np.array(coord_str_to_tuple(c_start))
            adj_list[i, 1] = np.array(coord_str_to_tuple(c_end))

        return cls.from_adj_list(adj_list)

    # ============================================================
    # to and from pixels
    # ============================================================
    def _as_pixels_bw(self) -> BinaryPixelGrid:
        assert self.lattice_dim == 2, "only 2D mazes are supported"
        # Create an empty pixel grid with walls
        pixel_grid: Int[np.ndarray, "x y"] = np.full(
            (self.grid_shape[0] * 2 + 1, self.grid_shape[1] * 2 + 1),
            False,
            dtype=np.bool_,
        )

        # Set white nodes
        pixel_grid[1::2, 1::2] = True

        # Set white connections (downward)
        for i, row in enumerate(self.connection_list[0]):
            for j, connected in enumerate(row):
                if connected:
                    pixel_grid[i * 2 + 2, j * 2 + 1] = True

        # Set white connections (rightward)
        for i, row in enumerate(self.connection_list[1]):
            for j, connected in enumerate(row):
                if connected:
                    pixel_grid[i * 2 + 1, j * 2 + 2] = True

        return pixel_grid

    def as_pixels(
        self,
        show_endpoints: bool = True,
        show_solution: bool = True,
    ) -> PixelGrid:
        if show_solution and not show_endpoints:
            raise ValueError("show_solution=True requires show_endpoints=True")
        # convert original bool pixel grid to RGB
        pixel_grid_bw: BinaryPixelGrid = self._as_pixels_bw()
        pixel_grid: PixelGrid = np.full(
            (*pixel_grid_bw.shape, 3), PixelColors.WALL, dtype=np.uint8
        )
        pixel_grid[pixel_grid_bw == True] = PixelColors.OPEN

        if self.__class__ == LatticeMaze:
            return pixel_grid

        # set endpoints for TargetedLatticeMaze
        if self.__class__ == TargetedLatticeMaze:
            if show_endpoints:
                pixel_grid[
                    self.start_pos[0] * 2 + 1, self.start_pos[1] * 2 + 1
                ] = PixelColors.START
                pixel_grid[
                    self.end_pos[0] * 2 + 1, self.end_pos[1] * 2 + 1
                ] = PixelColors.END
            return pixel_grid

        # set solution
        if show_solution:
            for coord in self.solution:
                pixel_grid[coord[0] * 2 + 1, coord[1] * 2 + 1] = PixelColors.PATH

            # set pixels between coords
            for index, coord in enumerate(self.solution[:-1]):
                next_coord = self.solution[index + 1]
                # check they are adjacent using norm
                assert (
                    np.linalg.norm(np.array(coord) - np.array(next_coord)) == 1
                ), f"Coords {coord} and {next_coord} are not adjacent"
                # set pixel between them
                pixel_grid[
                    coord[0] * 2 + 1 + next_coord[0] - coord[0],
                    coord[1] * 2 + 1 + next_coord[1] - coord[1],
                ] = PixelColors.PATH

            # set endpoints (again, since path would overwrite them)
            pixel_grid[
                self.start_pos[0] * 2 + 1, self.start_pos[1] * 2 + 1
            ] = PixelColors.START
            pixel_grid[
                self.end_pos[0] * 2 + 1, self.end_pos[1] * 2 + 1
            ] = PixelColors.END

        return pixel_grid

    @classmethod
    def _from_pixel_grid_bw(
        cls, pixel_grid: BinaryPixelGrid
    ) -> tuple[ConnectionList, tuple[int, int]]:
        grid_shape: tuple[int, int] = (
            pixel_grid.shape[0] // 2,
            pixel_grid.shape[1] // 2,
        )
        connection_list: ConnectionList = np.zeros((2, *grid_shape), dtype=np.bool_)

        # Extract downward connections
        connection_list[0] = pixel_grid[2::2, 1::2]

        # Extract rightward connections
        connection_list[1] = pixel_grid[1::2, 2::2]

        return connection_list, grid_shape

    @classmethod
    def _from_pixel_grid_with_positions(
        cls,
        pixel_grid: PixelGrid | BinaryPixelGrid,
        marked_positions: dict[str, RGB],
    ) -> tuple[ConnectionList, tuple[int, int], dict[str, CoordArray]]:
        # Convert RGB pixel grid to Bool pixel grid
        pixel_grid_bw: BinaryPixelGrid = ~np.all(
            pixel_grid == PixelColors.WALL, axis=-1
        )
        connection_list: ConnectionList
        grid_shape: tuple[int, int]
        connection_list, grid_shape = cls._from_pixel_grid_bw(pixel_grid_bw)

        # Find any marked positions
        out_positions: dict[str, CoordArray] = dict()
        for key, color in marked_positions.items():
            pos_temp: Int[np.ndarray, "x y"] = np.argwhere(
                np.all(pixel_grid == color, axis=-1)
            )
            pos_save: list[CoordTup] = list()
            for pos in pos_temp:
                # if it is a coordinate and not connection (transform position, %2==1)
                if pos[0] % 2 == 1 and pos[1] % 2 == 1:
                    pos_save.append((pos[0] // 2, pos[1] // 2))

            out_positions[key] = np.array(pos_save)

        return connection_list, grid_shape, out_positions

    @classmethod
    def from_pixels(
        cls,
        pixel_grid: PixelGrid,
    ) -> "LatticeMaze":
        connection_list: ConnectionList
        grid_shape: tuple[int, int]

        # if a binary pixel grid, return regular LaticeMaze
        if len(pixel_grid.shape) == 2:
            connection_list, grid_shape = cls._from_pixel_grid_bw(pixel_grid)
            return LatticeMaze(connection_list=connection_list)

        # otherwise, detect and check it's valid
        cls_detected: typing.Type[LatticeMaze] = detect_pixels_type(pixel_grid)
        if not cls in cls_detected.__mro__:
            raise ValueError(
                f"Pixel grid cannot be cast to {cls.__name__}, detected type {cls_detected.__name__}"
            )

        (
            connection_list,
            grid_shape,
            marked_pos,
        ) = cls._from_pixel_grid_with_positions(
            pixel_grid=pixel_grid,
            marked_positions=dict(
                start=PixelColors.START, end=PixelColors.END, solution=PixelColors.PATH
            ),
        )
        # if we wanted a LatticeMaze, return it
        if cls == LatticeMaze:
            return LatticeMaze(connection_list=connection_list)

        # otherwise, keep going
        temp_maze: LatticeMaze = LatticeMaze(connection_list=connection_list)

        # start and end pos
        start_pos_arr, end_pos_arr = marked_pos["start"], marked_pos["end"]
        assert start_pos_arr.shape == (
            1,
            2,
        ), f"start_pos_arr {start_pos_arr} has shape {start_pos_arr.shape}, expected shape (1, 2) -- a single coordinate"
        assert end_pos_arr.shape == (
            1,
            2,
        ), f"end_pos_arr {end_pos_arr} has shape {end_pos_arr.shape}, expected shape (1, 2) -- a single coordinate"

        start_pos: Coord = start_pos_arr[0]
        end_pos: Coord = end_pos_arr[0]

        # return a TargetedLatticeMaze if that's what we wanted
        if cls == TargetedLatticeMaze:
            return TargetedLatticeMaze(
                connection_list=connection_list,
                start_pos=start_pos,
                end_pos=end_pos,
            )

        # solution
        solution_raw: CoordArray = marked_pos["solution"]
        assert (
            solution_raw.shape[1] == 2
        ), f"solution {solution_raw} has shape {solution_raw.shape}, expected shape (n, 2)"
        assert (
            start_pos in solution_raw
        ), f"start_pos {start_pos} not in solution {solution_raw}"
        assert (
            end_pos in solution_raw
        ), f"end_pos {end_pos} not in solution {solution_raw}"
        # order the solution, by creating a list from the start to the end
        solution_raw_list: list[CoordTup] = [tuple(c) for c in solution_raw] + [
            tuple(end_pos)
        ]
        solution: list[CoordTup] = [tuple(start_pos)]
        while solution[-1] != tuple(end_pos):
            # use `get_coord_neighbors` to find connected neighbors
            neighbors: CoordArray = temp_maze.get_coord_neighbors(solution[-1])
            # TODO: make this less ugly
            assert (len(neighbors.shape) == 2) and (
                neighbors.shape[1] == 2
            ), f"neighbors {neighbors} has shape {neighbors.shape}, expected shape (n, 2)\n{neighbors = }\n{solution = }\n{solution_raw = }\n{temp_maze.as_ascii()}"
            # neighbors = neighbors[:, [1, 0]]
            # filter out neighbors that are not in the raw solution
            neighbors_filtered: CoordArray = np.array(
                [
                    coord
                    for coord in neighbors
                    if (
                        tuple(coord) in solution_raw_list
                        and not tuple(coord) in solution
                    )
                ]
            )
            # assert only one element is left, and then add it to the solution
            assert neighbors_filtered.shape == (
                1,
                2,
            ), f"neighbors_filtered has shape {neighbors_filtered.shape}, expected shape (1, 2)\n{neighbors = }\n{neighbors_filtered = }\n{solution = }\n{solution_raw_list = }\n{temp_maze.as_ascii()}"
            solution.append(tuple(neighbors_filtered[0]))

        # assert the solution is complete
        assert solution[0] == tuple(
            start_pos
        ), f"solution {solution} does not start at start_pos {start_pos}"
        assert solution[-1] == tuple(
            end_pos
        ), f"solution {solution} does not end at end_pos {end_pos}"

        return cls(
            connection_list=np.array(connection_list),
            solution=np.array(solution),
        )

    # ============================================================
    # to and from ASCII
    # ============================================================
    def _as_ascii_grid(self) -> Shaped[np.ndarray, "x y"]:
        # Get the pixel grid using to_pixels().
        pixel_grid: Bool[np.ndarray, "x y"] = self._as_pixels_bw()

        # Replace pixel values with ASCII characters.
        ascii_grid: Shaped[np.ndarray, "x y"] = np.full(
            pixel_grid.shape, AsciiChars.WALL, dtype=str
        )
        ascii_grid[pixel_grid == True] = AsciiChars.OPEN

        return ascii_grid

    def as_ascii(
        self,
        show_endpoints: bool = True,
        show_solution: bool = True,
    ) -> str:
        """return an ASCII grid of the maze"""
        ascii_grid: Shaped[np.ndarray, "x y"] = self._as_ascii_grid()
        pixel_grid: PixelGrid = self.as_pixels(
            show_endpoints=show_endpoints, show_solution=show_solution
        )

        chars_replace: tuple = tuple()
        if show_endpoints:
            chars_replace += (AsciiChars.START, AsciiChars.END)
        if show_solution:
            chars_replace += (AsciiChars.PATH,)

        for ascii_char, pixel_color in ASCII_PIXEL_PAIRINGS.items():
            if ascii_char in chars_replace:
                ascii_grid[(pixel_grid == pixel_color).all(axis=-1)] = ascii_char

        return "\n".join("".join(row) for row in ascii_grid)

    @classmethod
    def from_ascii(cls, ascii_str: str) -> "LatticeMaze":
        lines: list[str] = ascii_str.strip().split("\n")
        ascii_grid: Shaped[np.ndarray, "x y"] = np.array(
            [list(line) for line in lines], dtype=str
        )
        pixel_grid: PixelGrid = np.zeros((*ascii_grid.shape, 3), dtype=np.uint8)

        for ascii_char, pixel_color in ASCII_PIXEL_PAIRINGS.items():
            pixel_grid[ascii_grid == ascii_char] = pixel_color

        return cls.from_pixels(pixel_grid)


@serializable_dataclass(frozen=True, kw_only=True)
class TargetedLatticeMaze(LatticeMaze):
    """A LatticeMaze with a start and end position"""

    # this jank is so that SolvedMaze can inherit from this class without needing arguments for start_pos and end_pos
    start_pos: Coord
    end_pos: Coord

    def __post_init__(self) -> None:
        # make things numpy arrays (very jank to override frozen dataclass)
        self.__dict__["start_pos"] = np.array(self.start_pos)
        self.__dict__["end_pos"] = np.array(self.end_pos)
        assert self.start_pos is not None
        assert self.end_pos is not None
        # check that start and end are in bounds
        if (
            self.start_pos[0] >= self.grid_shape[0]
            or self.start_pos[1] >= self.grid_shape[1]
        ):
            raise ValueError(
                f"start_pos {self.start_pos} is out of bounds for grid shape {self.grid_shape}"
            )
        if (
            self.end_pos[0] >= self.grid_shape[0]
            or self.end_pos[1] >= self.grid_shape[1]
        ):
            raise ValueError(
                f"end_pos {self.end_pos} is out of bounds for grid shape {self.grid_shape}"
            )

    @classmethod
    def from_lattice_maze(
        cls,
        lattice_maze: LatticeMaze,
        start_pos: Coord,
        end_pos: Coord,
    ) -> "TargetedLatticeMaze":
        return cls(
            connection_list=lattice_maze.connection_list,
            start_pos=start_pos,
            end_pos=end_pos,
        )


@serializable_dataclass(frozen=True, kw_only=True)
class SolvedMaze(TargetedLatticeMaze):
    """Stores a maze and a solution"""

    solution: CoordArray

    def __init__(
        self,
        connection_list: ConnectionList,
        solution: CoordArray,
        generation_meta: dict | None = None,
        lattice_dim: int = 2,
    ) -> None:
        super().__init__(
            connection_list=connection_list,
            generation_meta=generation_meta,
            start_pos=np.array(solution[0]),
            end_pos=np.array(solution[-1]),
            lattice_dim=lattice_dim,
        )
        self.__dict__["solution"] = solution

    # for backwards compatibility
    @property
    def maze(self) -> LatticeMaze:
        warnings.warn(
            "maze is deprecated, SolvedMaze now inherits from LatticeMaze.",
            DeprecationWarning,
        )
        return LatticeMaze(connection_list=self.connection_list)

    @classmethod
    def from_lattice_maze(
        cls, lattice_maze: LatticeMaze, solution: list[CoordTup]
    ) -> "SolvedMaze":
        return cls(
            connection_list=lattice_maze.connection_list,
            solution=solution,
        )

    @classmethod
    def from_targeted_lattice_maze(
        cls, targeted_lattice_maze: TargetedLatticeMaze
    ) -> "SolvedMaze":
        """solves the given targeted lattice maze and returns a SolvedMaze"""
        solution: list[CoordTup] = targeted_lattice_maze.find_shortest_path(
            targeted_lattice_maze.start_pos,
            targeted_lattice_maze.end_pos,
        )
        return cls(
            connection_list=targeted_lattice_maze.connection_list,
            solution=np.array(solution),
        )

    @classmethod
    def from_tokens(cls, tokens: list[str], data_cfg) -> "SolvedMaze":
        maze: LatticeMaze = LatticeMaze.from_tokens(tokens)
        path_tokens: list[str] = get_path_tokens(tokens)
        solution: list[str | tuple[int, int]] = tokens_to_coords(path_tokens, data_cfg)

        assert len(solution) > 0, f"No solution found: {solution = }"

        try:
            solution_cast: list[CoordTup] = cast(list[CoordTup], solution)
            solution_np: CoordArray = np.array(solution_cast)
        except ValueError as e:
            raise ValueError(f"Invalid solution: {solution = }") from e

        return cls.from_lattice_maze(lattice_maze=maze, solution=solution_np)


def detect_pixels_type(data: PixelGrid) -> typing.Type[LatticeMaze]:
    """Detects the type of pixels data by checking for the presence of start and end pixels"""
    if PixelColors.START in data or PixelColors.END in data:
        if PixelColors.PATH in data:
            return SolvedMaze
        else:
            return TargetedLatticeMaze
    else:
        return LatticeMaze
