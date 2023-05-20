from pathlib import Path

import numpy as np
import pytest

from maze_transformer.utils.utils import (
    corner_first_ndindex,
    get_checkpoint_paths_for_run,
    register_method,
)


@pytest.mark.usefixtures("temp_dir")
def test_get_checkpoint_paths_for_run(temp_dir):
    run_path = Path(temp_dir)
    checkpoints_path = run_path / "checkpoints"
    checkpoint1_path = checkpoints_path / "model.iter_123.pt"
    checkpoint2_path = checkpoints_path / "model.iter_456.pt"
    other_path = checkpoints_path / "other_file.txt"

    checkpoints_path.mkdir()
    checkpoint1_path.touch()
    checkpoint2_path.touch()
    other_path.touch()

    checkpoint_paths = get_checkpoint_paths_for_run(run_path)

    assert checkpoint_paths == [(123, checkpoint1_path), (456, checkpoint2_path)]


def test_register_method():
    class TestEvalsA:
        evals = {}

        @register_method(evals)
        @staticmethod
        def eval_function():
            pass

        @staticmethod
        def other_function():
            pass

    class TestEvalsB:
        evals = {}

        @register_method(evals)
        @staticmethod
        def other_eval_function():
            pass

    evalsA = TestEvalsA.evals
    evalsB = TestEvalsB.evals
    assert list(evalsA.keys()) == ["eval_function"]
    assert list(evalsB.keys()) == ["other_eval_function"]


def test_corner_first_ndindex():
    for n in range(1, 11):
        a_n = corner_first_ndindex(n)
        a_n_plus_1 = corner_first_ndindex(n + 1)
        assert np.all(a_n == a_n_plus_1[: n**2])
