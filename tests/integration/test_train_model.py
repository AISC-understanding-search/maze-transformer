from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.wandb_logger import WandbProject
from scripts.train_model import TrainingResult, train_model


def test_train_model():
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs", "nano-v1", "integration-v1"),
    )
    result: TrainingResult = train_model(
        base_path="tests/_temp/test_train_model",
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg_names=("test-g3-n5-a_dfs", "nano-v1", "integration-v1"),
        do_generate_dataset=True,
    )

    assert isinstance(result.model, ZanjHookedTransformer)
    assert result.model.zanj_model_config == cfg
