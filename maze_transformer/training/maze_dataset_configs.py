from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.maze_dataset import MazeDatasetConfig

MAZE_DATASET_CONFIGS: dict[str, MazeDatasetConfig] = {
    cfg.to_fname(): cfg
    for cfg in [
        MazeDatasetConfig(
            name="test",
            grid_n=3,
            n_mazes=5,
            maze_ctor=LatticeMazeGenerators.gen_dfs,
        ),
    ]
}
