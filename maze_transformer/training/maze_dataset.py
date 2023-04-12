import enum
import json
import multiprocessing
import os
from pathlib import Path
import typing
import warnings
from functools import cached_property, partial
from typing import Callable

import numpy as np
import torch
from jaxtyping import Int, Shaped
from muutils.json_serialize import (
    JSONitem,
    json_serialize,
    serializable_dataclass,
    serializable_field,
)
from muutils.json_serialize.util import safe_getsource, string_as_lines
from muutils.misc import freeze, sanitize_fname
from muutils.statcounter import StatCounter
from muutils.tensor_utils import ATensor, NDArray
from tqdm import tqdm

from maze_transformer.generation.constants import SPECIAL_TOKENS, CoordArray, CoordTup
from maze_transformer.generation.generators import GENERATORS_MAP, LatticeMazeGenerators
from maze_transformer.generation.lattice_maze import LatticeMaze, SolvedMaze, TargetedLatticeMaze
from maze_transformer.training.dataset import GPTDataset, GPTDatasetConfig, IndexedArray, SaveFormats
from maze_transformer.training.tokenizer import maze_to_tokens

_MAZEDATASET_PROPERTIES_TO_SERIALIZE: list[str] = [
    "padding_token_index",
    "token_arr",
    "tokenizer_map",
    "grid_shape",
    # "node_token_map", # doesn't work by default due to keys being tuples
    "token_node_map",
    "n_tokens",
]

# TODO: re-add later, depends on a feature coming in muutils 0.3.2
__MAZEDATASET_PROPERTIES_TO_VALIDATE: list[str] = [
    "token_arr",
    "padding_token_index",
    "tokenizer_map",
    "grid_shape",
    "token_node_map",
    "n_tokens",
]


def _load_maze_ctor(maze_ctor_serialized: str | dict) -> Callable:
    if isinstance(maze_ctor_serialized, dict):
        # this is both the new and old version of the serialization
        return GENERATORS_MAP[maze_ctor_serialized["__name__"]]
    elif isinstance(maze_ctor_serialized, str):
        # this is a version I switched to for a while but now we are switching back
        warnings.warn(
            f"you are loading an old model/config!!! this should not be happening, please report to miv@knc.ai"
        )
        return GENERATORS_MAP[maze_ctor_serialized]
    else:
        raise ValueError(
            f"maze_ctor_serialized is of type {type(maze_ctor_serialized)}, expected str or dict"
        )


@serializable_dataclass(
    kw_only=True, properties_to_serialize=_MAZEDATASET_PROPERTIES_TO_SERIALIZE
)
class MazeDatasetConfig(GPTDatasetConfig):
    """maze dataset configuration, including tokenizers"""

    grid_n: int
    n_mazes: int
    maze_ctor: Callable = serializable_field(
        default_factory=lambda: LatticeMazeGenerators.gen_dfs,
        serialization_fn=lambda gen_func: {
            "__name__": gen_func.__name__,
            "__module__": gen_func.__module__,
            "__doc__": string_as_lines(gen_func.__doc__),
            "source_code": safe_getsource(gen_func),
        },
        loading_fn=lambda data: _load_maze_ctor(data["maze_ctor"]),
    )

    # paths_per_maze: int = 5,
    # p_min_tgt_dist: float = 0.2,

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_n, self.grid_n)

    @cached_property
    def node_token_map(self) -> dict[CoordTup, str]:
        """map from node to token"""
        return {tuple(c): f"({c[0]},{c[1]})" for c in np.ndindex(self.grid_shape)}

    @cached_property
    def token_node_map(self) -> dict[str, CoordTup]:
        """map from token to node"""
        return {v: k for k, v in self.node_token_map.items()}

    @cached_property
    def token_arr(self) -> list[str]:
        """map from index to token"""
        return [
            *list(SPECIAL_TOKENS.values()),
            *list(self.node_token_map.values()),
        ]

    @property
    def n_tokens(self) -> int:
        return len(self.token_arr)

    @cached_property
    def padding_token_index(self) -> str:
        return self.tokenizer_map[SPECIAL_TOKENS["padding"]]

    def to_fname(self) -> str:
        self_json_str: str = json.dumps(self.serialize())
        self_json_hash: int = int(abs(hash(self_json_str))%1e5)
        return sanitize_fname(f"{self.name}-g{self.grid_n}-n{self.n_mazes}-h{self_json_hash}.zanj")


class MazeDataset(GPTDataset):
    """maze dataset"""

    def __init__(
            self, 
            cfg: MazeDatasetConfig, 
            mazes: typing.Sequence[SolvedMaze],
        ) -> None:
        super().__init__()
        self.cfg: MazeDatasetConfig = cfg
        self.mazes: list[SolvedMaze] = list(mazes)

    def get(self, index: int, fmt: SaveFormats = SaveFormats.OBJECTS) -> SolvedMaze|list[str]|np.ndarray:
        """get a single maze, as one of the formats"""
        if fmt == SaveFormats.OBJECTS:
            return self.mazes[index]
        elif fmt == SaveFormats.TOKENS:
            return maze_to_tokens(self.mazes[index], self.cfg.node_token_map)
        elif fmt == SaveFormats.ARRAY:
            raise NotImplementedError("getting as array not implemented yet")
        else:
            raise ValueError(f"unknown fmt {fmt}, expected an instance of `SaveFormats` enum")

    def __getitem__(self, index: int) -> list[str]:
        """index into mazes_array.arr, getting from the start of the correct sequence, padding if necessary"""

        return maze_to_tokens(self.mazes[index], self.cfg.node_token_map)

    mazes_objs: list[SolvedMaze] = property(lambda self: list(self.get_all(fmt=SaveFormats.OBJECTS)))
    mazes_tokens: list[list[str]] = property(lambda self: list(self.get_all(fmt=SaveFormats.TOKENS)))
    mazes_array: IndexedArray = property(lambda self: IndexedArray(self.get_all(fmt=SaveFormats.ARRAY)))
    

    def __len__(self) -> int:
        return len(self.mazes)

    def get_all_lengths(self) -> list[int]:
        raise NotImplementedError("not implemented yet")

    @classmethod
    def generate(cls, cfg: MazeDatasetConfig) -> "MazeDataset":
        mazes: list[SolvedMaze] = list()
        endpoint_nodes: Int[np.int8, "maze_index 2 2"] = np.random.randint(
            0, cfg.grid_shape, 
            (cfg.n_mazes, 2, 2),
        )
        # TODO: filter min distanced based on MazeDatasetConfig
        # TODO: parallelize

        for i, (c_start, c_end) in enumerate(endpoint_nodes):
            m: TargetedLatticeMaze = TargetedLatticeMaze.from_lattice_maze(
                lattice_maze = cfg.maze_ctor(cfg.grid_shape),
                start_pos=c_start,
                end_pos=c_end,
            )
            path: CoordArray = np.array(m.find_shortest_path(c_start, c_end))
            mazes.append(
                SolvedMaze.from_lattice_maze(lattice_maze=m, solution=path)
            )

        return cls(
            cfg=cfg,
            mazes=mazes,
        )
    
    @classmethod
    def download(cls, cfg: MazeDatasetConfig, **kwargs) -> "MazeDataset":
        raise NotImplementedError("not implemented yet")
        

    @classmethod
    def load(cls, data: JSONitem) -> "MazeDataset":
        """load from zanj/json"""
        assert data["__format__"] == "MazeDataset"
        return cls(
            cfg=MazeDatasetConfig.load(data["cfg"]),
            mazes=[SolvedMaze.load(m) for m in data["mazes"]],
        )

    def serialize(self) -> JSONitem:
        """serialize to zanj/json"""
        return {
            "__format__": "MazeDataset",
            "cfg": self.cfg.serialize(),
            "mazes": [m.serialize() for m in self.mazes],
        }

    @classmethod
    def disk_load(cls, path: str, **kwargs) -> "MazeDataset":
        """load from disk"""
        warnings.warn(
            "deprecated, use `MazeDataset.read(path)` or `MazeDataset.load(ZANJ().read(path)))` instead",
            DeprecationWarning,
        )
        if kwargs:
            warnings.warn(f"kwargs to disk_load dont do anything: {kwargs = }", DeprecationWarning)
        return cls.read(path)


MazeDatasetConfig._dataset_class = MazeDataset
