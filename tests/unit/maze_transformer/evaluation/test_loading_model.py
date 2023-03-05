"""
Test that loading the model and configuration works
* I.e. the outputs of the model are identical when loading into
a HookedTransformer with folding etc., as they would be from
just applying the model to the input
"""
# TODO
import tempfile
from pathlib import Path

import pytest
import torch

from maze_transformer.evaluation.eval_model import load_model_with_configs
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.fixture()
def temp_dir() -> Path:
    data_dir = tempfile.TemporaryDirectory()
    yield Path(data_dir.name)
    data_dir.cleanup()


def test_model_loading(temp_dir):
    # First create a dataset and train a model
    #! Awaiting change of all paths to Path for training scripts
    if not Path.exists(temp_dir / "g3-n5-test"):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")
        train_model(
            basepath=str(temp_dir / "g3-n5-test"),
            training_cfg="tiny-v1",
            model_cfg="tiny-v1",
        )

    # Now load the model and compare the outputs
    # Get directory of the training run
    run_folder_path = Path(temp_dir / "g3-n5-test")
    run_folder_path = [x for x in run_folder_path.glob("*") if x.is_dir()][0]

    # Load model using our function (with layernorm folding etc.)
    model, cfg = load_model_with_configs(run_folder_path / "model.final.pt")

    # Load model manually without folding
    model_state_dict = torch.load(run_folder_path / "model.final.pt")
    model_basic = cfg.create_model()
    model_basic.load_state_dict(model_state_dict)

    # Random input tokens
    input_sequence = torch.randint(
        low=0,
        high=len(cfg.dataset_cfg.token_arr),
        size=(1, min(cfg.dataset_cfg.seq_len_max, 10)),
    )

    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    # Alternatively could apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    assert torch.all(
        model(input_sequence.clone()).argsort()
        == model_basic(input_sequence.clone()).argsort()
    )
