import json
from typing import Dict
import pytest
import os
import pandas as pd
import torch
from e2e_recsys.data_generation.file_converter import FileConverter
from e2e_recsys.features.csv_vocab_builder import CSVVocabBuilder

LOCAL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
COLUMNS_TO_USE = ["cat1", "num2"]
TARGET_COL = "target"


@pytest.fixture
def mock_data_path(tmpdir) -> str:
    data_path = os.path.join(LOCAL_DIRECTORY, "data", "mock_data.csv")
    output_path = os.path.join(tmpdir, "data", "mock_data.csv")
    os.makedirs(os.path.dirname(output_path))
    mock_df = pd.read_csv(data_path)
    mock_df.to_csv(output_path, index=False)

    return output_path


# For testing on  preprocessed data
@pytest.fixture
def mock_preprocessed_data_path(tmpdir) -> str:
    data_path = os.path.join(LOCAL_DIRECTORY, "data", "mock_preprocessed.csv")
    output_path = os.path.join(
        tmpdir, "preprocessed_data", "mock_preprocessed.csv"
    )
    os.makedirs(os.path.dirname(output_path))
    mock_df = pd.read_csv(data_path)
    mock_df.to_csv(output_path, index=False)

    return output_path


@pytest.fixture
def mock_converted_data_dir(mock_preprocessed_data_path, tmpdir) -> str:
    output_dir = os.path.join(tmpdir, "converted_data")

    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_preprocessed_data_path,
        output_dir=output_dir,
    )
    file_converter.convert_rows()
    return output_dir


@pytest.fixture
def mock_vocab_path(tmpdir, mock_data_path):
    output_path = os.path.join(tmpdir, "vocab", "vocab.json")
    df = pd.read_csv(mock_data_path)
    vb = CSVVocabBuilder(features=set(["cat1", "cat2"]), data=df)
    vb.build_vocab()
    vb.save_vocab(output_path)
    return output_path


@pytest.fixture()
def model_config_path() -> str:
    return os.path.join(LOCAL_DIRECTORY, "configs", "test_model.json")


@pytest.fixture
def model_config(model_config_path) -> Dict[str, Dict[str, int]]:
    with open(model_config_path, "r") as f:
        config = json.load(f)
    return config


@pytest.fixture
def test_data_dict() -> Dict[str, torch.Tensor]:
    test_data = {
        "num1": torch.tensor([[0], [1], [-1]], dtype=torch.float32),
        "num2": torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float32),
        "num3": torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float32),
        "cat1": torch.tensor([[0], [1], [2]], dtype=torch.int32),
        "cat2": torch.tensor([[1], [2], [0]], dtype=torch.int32),
    }
    return test_data
