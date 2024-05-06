import json
import pytest
import torch
from e2e_recsys.modelling.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
)


@pytest.fixture
def test_model(model_config, mock_vocab_path) -> MultiLayerPerceptron:
    with open(mock_vocab_path, "r") as f:
        vocab = json.load(f)
    model = MultiLayerPerceptron(
        architecture_config=model_config["architecture_config"],
        numeric_feature_names=set(model_config["features"]["quantitative"]),
        categorical_feature_names=set(model_config["features"]["categorical"]),
        vocab=vocab,
    )
    return model


def test_multi_layer_perceptron_init(test_model):

    # Check the units in each layer
    actual_units = [layer.out_features for layer in test_model.hidden_layers]
    assert actual_units == [4, 2]
    assert test_model.output_layer.out_features == 1
    # Check the functions
    assert isinstance(test_model.activation, torch.nn.ReLU)
    assert isinstance(test_model.output_transform, torch.nn.Sigmoid)
    # Check the features
    assert test_model.numeric_feature_names == ["num1", "num2", "num3"]
    assert test_model.categorical_feature_names == ["cat1", "cat2"]
    # Check the vocab sizes
    assert test_model.vocab_sizes["cat1"] == 4
    assert test_model.vocab_sizes["cat2"] == 3
    # Check input size
    assert test_model.input_size == 10


def test_multi_layer_perceptron_preprocessing(test_model, test_data_dict):
    expected_output = torch.tensor(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [-1, -1, -1, 0, 0, 1, 0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    actual_output = test_model._preprocessing(test_data_dict)

    torch.testing.assert_close(actual_output, expected_output)


def test_init_vocab_sizes(test_model):
    # Check the vocab sizes are correct
    assert test_model.vocab_sizes["cat1"] == 4
    assert test_model.vocab_sizes["cat2"] == 3


def test_one_hot_encode_feature(test_model):
    test_data = torch.tensor([[1], [2], [1]], dtype=torch.float32)
    expected = torch.tensor(
        [[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32
    )
    vocab_size = 3
    result = test_model._one_hot_encode_feature(test_data, vocab_size)
    torch.testing.assert_close(result, expected)


def test_multi_layer_perceptron_forward(test_model, test_data_dict):
    output = test_model.forward(test_data_dict)
    # Test call-like behaviour
    torch.testing.assert_close(output, test_model(test_data_dict))
    assert output.shape == (3, 1)
    # Ensure we have valid probabilities
    for score in output:
        assert score > 0 and score < 1
