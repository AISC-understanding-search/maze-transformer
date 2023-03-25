import os
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch

DEFAULT_SEED = 42


def get_device():
    """Get the torch.device instance on which torch.Tensors should be allocated."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_reproducibility(seed=DEFAULT_SEED):
    """
    Improve model reproducibility. See https://github.com/NVIDIA/framework-determinism for more information.

    Deterministic operations tend to have worse performance than nondeterministic operations, so this method trades
    off performance for reproducibility. Set use_deterministic_algorithms to True to improve performance.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    # Ensure reproducibility for concurrent CUDA streams
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def chunks(it, chunk_size):
    """Yield successive chunks from an iterator."""
    # https://stackoverflow.com/a/61435714
    iterator = iter(it)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


def get_checkpoint_paths_for_run(run_path: Path) -> list[tuple[int, Path]]:
    assert (
        run_path.is_dir()
    ), f"Model path {run_path} is not a directory (expect run directory, not model files)"

    return [
        (int(checkpoint_path.stem.split("_")[-1].split(".")[0]), checkpoint_path)
        for checkpoint_path in sorted(
            Path(run_path).glob("checkpoints/model.iter_*.pt")
        )
    ]
