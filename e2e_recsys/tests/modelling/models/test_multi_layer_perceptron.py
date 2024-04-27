import torch
from e2e_recsys.modelling.models.MultiLayerPerceptron import (
    MultiLayerPerceptron,
)


def test_multi_layer_perceptron_init(model_config):
    model = MultiLayerPerceptron(
        hyperparam_config=model_config["hyperparam_config"],
        numeric_feature_names=set(model_config["features"]["quantitative"]),
        categorical_feature_names=set(model_config["features"]["categorical"]),
    )

    # Check the units in each layer
    actual_units = [layer.out_features for layer in model.hidden_layers]
    assert actual_units == [4, 2]
    assert model.output_layer.out_features == 1
    # Check the functions
    assert isinstance(model.activation, torch.nn.ReLU)
    assert isinstance(model.output_transform, torch.nn.Sigmoid)
    # Check the features
    assert model.numeric_feature_names == set(["num1", "num2", "num3"])
    assert model.categorical_feature_names == set(["cat1", "cat2"])
