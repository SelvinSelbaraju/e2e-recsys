import pytest
import os
import pandas as pd

LOCAL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def mock_data_path(tmpdir) -> str:
    data_path = os.path.join(LOCAL_DIRECTORY, "data", "mock_data.csv")
    output_path = os.path.join(tmpdir, "data", "mock_data.csv")
    os.makedirs(os.path.dirname(output_path))
    mock_df = pd.read_csv(data_path)
    mock_df.to_csv(output_path, index=False)

    return output_path


