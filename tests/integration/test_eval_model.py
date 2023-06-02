"""
Test that loading the model and configuration works

* I.e. the outputs of the model are identical when loading into
    a HookedTransformer with folding etc., as they would be from
    just applying the model to the input
"""
from pathlib import Path

import numpy as np
import pytest
from zanj import ZANJ
from zanj.torchutil import assert_model_cfg_equality

from maze_transformer.dataset.maze_dataset import MazeDataset
from maze_transformer.evaluation.eval_model import evaluate_model, predict_maze_paths
from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.generation.constants import CoordTup
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.train_model import TrainingResult, train_model
from maze_transformer.training.training import TRAIN_SAVE_FILES
from maze_transformer.training.wandb_logger import WandbProject
from maze_transformer.utils.test_helpers.assertions import assert_model_output_equality

temp_dir: Path = Path("tests/_temp/test_eval_model")


def test_model_loading():
    zanj: ZANJ = ZANJ(
        custom_settings={
            "_load_state_dict_wrapper": {"recover_exact": True, "fold_ln": False}
        }
    )
    # get config
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs-h90179", "nano-v1", "test-v1"),
    )
    # train model
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )
    model_ret: ZanjHookedTransformer = result.model

    # load model
    model_load_auto: ZanjHookedTransformer = zanj.read(
        result.output_path / TRAIN_SAVE_FILES.model_final_zanj
    )

    # Load model manually without folding
    assert cfg == model_ret.zanj_model_config
    assert_model_cfg_equality(model_ret, model_load_auto)

    assert_model_output_equality(model_ret, model_load_auto)
    # assert_model_exact_equality(model_ret, model_load_auto)


def test_predict_maze_paths():
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs-h90179", "nano-v1", "test-v1"),
    )
    # train model
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )
    model: ZanjHookedTransformer = result.model

    dataset: MazeDataset = MazeDataset.from_config(
        cfg=cfg.dataset_cfg,
        verbose=False,
    )

    max_new_tokens = 2
    paths = predict_maze_paths(
        tokens_batch=dataset.as_tokens(),
        data_cfg=cfg.dataset_cfg,
        model=model,
        max_new_tokens=max_new_tokens,
    )

    all_coordinates: list[CoordTup] = [coord for path in paths for coord in path]

    assert len(paths) == 5
    # check the coords are actually coords
    assert all(
        isinstance(c, tuple) for c in all_coordinates
    ), f"expected tuples of ints in all_coordinates: {all_coordinates}"
    assert all(
        len(c) == 2 for c in all_coordinates
    ), f"expected tuples of ints in all_coordinates: {all_coordinates}"
    assert all(
        isinstance(c[0], int) and isinstance(c[1], int) for c in all_coordinates
    ), f"expected tuples of ints in all_coordinates: {all_coordinates}"

    assert max([len(path) for path in paths]) <= max_new_tokens + 1

    all_coordinates_np: np.ndarray = np.array(all_coordinates)
    assert np.max(all_coordinates_np) < cfg.dataset_cfg.grid_n


@pytest.mark.usefixtures("temp_dir")
def test_evaluate_model(temp_dir):
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs-h90179", "nano-v1", "test-v1"),
    )
    # train model
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )
    model: ZanjHookedTransformer = result.model

    dataset: MazeDataset = MazeDataset.from_config(
        cfg=cfg.dataset_cfg,
        verbose=False,
    )

    path_evals = PathEvals.fast
    eval_names = [name for name in path_evals.keys()]
    scores = evaluate_model(dataset=dataset, model=model)

    assert path_evals.keys() == scores.keys()
    assert scores[eval_names[0]].summary()["total_items"] == cfg.dataset_cfg.n_mazes
