import pytest
import os
import pandas as pd
from e2e_recsys.data_generation.file_converter import FileConverter

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

@pytest.fixture
def mock_converted_data_dir(mock_data_path, tmpdir) -> str:
    output_dir = os.path.join(tmpdir, "converted_data")

    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_data_path,
        output_dir=output_dir,
    )
    file_converter.convert_rows()
    return output_dir


