from pathlib import Path

from muutils.zanj import ZANJ

from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig

temp_dir: Path = Path("tests/_temp/dataset")

dataset_cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test_create_dataset", grid_n=3, n_mazes=5
)


def test_generate_mazedataset():
    m: MazeDataset = MazeDataset.from_config(
        dataset_cfg,
        load_local=False,
        do_download=False,
        save_local=False,
    )

    assert len(m.mazes) == 5


def test_load_save_mazedataset_auto():
    m: MazeDataset = MazeDataset.from_config(
        dataset_cfg,
        load_local=False,
        do_download=False,
        save_local=True,
        local_base_path=temp_dir,
    )

    m2: MazeDataset = MazeDataset.from_config(
        dataset_cfg,
        load_local=True,
        do_download=False,
        do_generate=False,
        save_local=False,
        local_base_path=temp_dir,
    )

    assert len(m.mazes) == 5
    assert len(m2.mazes) == 5

    assert m.cfg == m2.cfg
    assert all([m1 == m2 for m1, m2 in zip(m.mazes, m2.mazes)])


def test_load_save_mazedataset_manual():
    m: MazeDataset = MazeDataset.from_config(
        dataset_cfg,
        load_local=False,
        do_download=False,
        save_local=True,
        local_base_path=temp_dir,
    )

    m_fname: Path = temp_dir / (m.cfg.to_fname() + ".zanj")

    m2: MazeDataset = MazeDataset.load(ZANJ().read(m_fname))
    m3: MazeDataset = MazeDataset.read(m_fname)

    assert len(m.mazes) == 5
    assert len(m2.mazes) == 5
    assert len(m3.mazes) == 5

    assert m.cfg == m2.cfg
    assert all([m1 == m2 for m1, m2 in zip(m.mazes, m2.mazes)])

    assert m.cfg == m3.cfg
    assert all([m1 == m3 for m1, m3 in zip(m.mazes, m3.mazes)])
