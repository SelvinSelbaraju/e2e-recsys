from e2e_recsys.utils.io import load_json


def test_load_json(model_config_path):
    config = load_json(model_config_path)
    assert sorted(list(config.keys())) == [
        "architecture_config",
        "features",
        "hyperparam_config",
        "training_config",
    ]
