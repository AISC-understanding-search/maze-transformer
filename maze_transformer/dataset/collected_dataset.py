import itertools
import json
from functools import cached_property
import typing

import numpy as np
from jaxtyping import Int
from muutils.json_serialize import JSONitem, serializable_dataclass, serializable_field, json_serialize
from muutils.misc import sanitize_fname, stable_hash
from muutils.zanj.loading import register_loader_handler, LoaderHandler, load_item_recursive

from maze_transformer.dataset.dataset import GPTDataset, GPTDatasetConfig
from maze_transformer.dataset.maze_dataset import (
    _MAZEDATASET_PROPERTIES_TO_SERIALIZE,
    MazeDataset,
    MazeDatasetConfig,
    _coord_to_str,
)
from maze_transformer.generation.constants import SPECIAL_TOKENS, Coord, CoordTup
from maze_transformer.generation.lattice_maze import LatticeMaze


@serializable_dataclass(
    kw_only=True, properties_to_serialize=_MAZEDATASET_PROPERTIES_TO_SERIALIZE
)
class MazeDatasetCollectionConfig(GPTDatasetConfig):
    """maze dataset collection configuration, including tokenizers and shuffle"""

    maze_dataset_configs: list[MazeDatasetConfig] = serializable_field(
        serialization_fn=lambda configs: [config.serialize() for config in configs],
        loading_fn=lambda data: [
            MazeDatasetConfig.load(config) for config in data["maze_dataset_configs"]
        ],
    )

    @property
    def n_mazes(self) -> int:
        return sum(config.n_mazes for config in self.maze_dataset_configs)

    @property
    def max_grid_n(self) -> int:
        return max(config.grid_n for config in self.maze_dataset_configs)

    @property
    def max_grid_shape(self) -> CoordTup:
        return (self.max_grid_n, self.max_grid_n)

    @property
    def max_grid_shape_np(self) -> Coord:
        return np.array(self.max_grid_shape, dtype=np.int32)

    @cached_property
    def node_token_map(self) -> dict[CoordTup, str]:
        """map from node to token"""
        return {
            tuple(coord): _coord_to_str(coord)
            for coord in list(np.ndindex(self.max_grid_shape))
        }

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
    
    @cached_property
    def padding_token_index(self) -> int:
        return self.tokenizer_map[SPECIAL_TOKENS["padding"]]

    @property
    def n_tokens(self) -> int:
        return len(self.token_arr)

    def stable_hash_cfg(self) -> int:
        return stable_hash(json.dumps(self.serialize()))

    def to_fname(self) -> str:
        """convert config to a filename"""
        return sanitize_fname(f"collected-{self.name}_{self.stable_hash_cfg()%10**5}")


class MazeDatasetCollection(GPTDataset):
    """a collection of maze datasets"""

    def __init__(
        self,
        cfg: MazeDatasetCollectionConfig,
        maze_datasets: list[MazeDataset],
        generation_metadata_collected: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg: MazeDatasetCollectionConfig = cfg
        self.maze_datasets: list[MazeDataset] = list(maze_datasets)
        self.generation_metadata_collected: dict | None = generation_metadata_collected

    @property
    def dataset_lengths(self) -> list[int]:
        return [len(dataset) for dataset in self.maze_datasets]

    @property
    def dataset_cum_lengths(self) -> Int[np.ndarray, "indices"]:
        return np.array(list(itertools.accumulate(self.dataset_lengths)))

    @cached_property
    def mazes(self) -> list[LatticeMaze]:
        return list(
            itertools.chain.from_iterable(
                dataset.mazes for dataset in self.maze_datasets
            )
        )

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.maze_datasets)

    def __getitem__(self, index: int):
        # find which dataset the index belongs to
        dataset_idx: int = np.searchsorted(self.dataset_cum_lengths, index)
        index_adjusted: int = index
        if dataset_idx > 0:
            # if the index is 0, `dataset_idx - 1` will be -1. 
            # We just want to use the base index 
            index_adjusted -= self.dataset_cum_lengths[dataset_idx - 1]
        return self.maze_datasets[dataset_idx][index_adjusted]

    @classmethod
    def generate(
        cls, cfg: MazeDatasetCollectionConfig, **kwargs
    ) -> "MazeDatasetCollection":
        datasets = [
            MazeDataset.generate(config, **kwargs)
            for config in cfg.maze_dataset_configs
        ]
        return cls(cfg, datasets)

    @classmethod
    def download(
        cls, cfg: MazeDatasetCollectionConfig, **kwargs
    ) -> "MazeDatasetCollection":
        datasets = [
            MazeDataset.download(config, **kwargs)
            for config in cfg.maze_dataset_configs
        ]
        return cls(cfg, datasets)

    def serialize(self) -> JSONitem:
        return dict(
            __format__ = "MazeDatasetCollection",
            cfg=self.cfg.serialize(),
            maze_datasets=[dataset.serialize() for dataset in self.maze_datasets],
            generation_metadata_collected = json_serialize(
                self.generation_metadata_collected
            ),
        )

    @classmethod
    def load(cls, data: JSONitem) -> "MazeDatasetCollection":
        return cls(
            cfg=MazeDatasetCollectionConfig.load(data["cfg"]),
            maze_datasets=[
                MazeDataset.load(dataset_data) for dataset_data in data["maze_datasets"]
            ],
        )

    def update_self_config(self) -> None:
        # TODO: why cant we set this directly? its not frozen, and it seems to work in a regular MazeDataset
        self.cfg.__dict__["n_mazes"] = len(self)
        for dataset in self.maze_datasets:
            dataset.update_self_config()


MazeDatasetCollectionConfig._dataset_class = MazeDatasetCollection
register_loader_handler(LoaderHandler(
    check= lambda json_item, path=None, z=None: (
        isinstance(json_item, typing.Mapping)
        and "__format__" in json_item
        and json_item["__format__"].startswith("MazeDatasetCollection")
    ),
    load = lambda json_item, path=None, z=None: load_item_recursive(json_item, path, z),
    uid = "MazeDatasetCollection",
    source_pckg = "maze_transformer.generation.maze_dataset_collection",
    desc = "MazeDatasetCollection"
))