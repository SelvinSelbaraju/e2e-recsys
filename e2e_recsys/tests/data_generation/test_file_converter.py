import os
import pytest
import pandas as pd

from e2e_recsys.data_generation.file_converter import FileConverter

COLUMNS_TO_USE = ["cat1", "num2"]
TARGET_COL = "target"

def test_file_converter_init(mock_data_path, tmpdir) -> None:
    output_dir = os.path.join(tmpdir, "converted_data")
    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_data_path,
        output_dir=output_dir,
    )
    # Check the constructor attributes
    assert file_converter.columns == COLUMNS_TO_USE
    assert file_converter.target_col == TARGET_COL
    assert file_converter.output_file_prefix == "data_row"
    assert file_converter.file_extension == "pt"
    assert file_converter.current_row_idx == 0

    # Check the data loader returns first row
    expected_row = pd.DataFrame({
        "cat1": "a",
        "num2": 1.0,
        "target": 0
    }, index=[0])
    test_row = next(file_converter.data_loader)
    pd.testing.assert_frame_equal(expected_row, test_row)


def test_load_row(mock_data_path, tmpdir):
    output_dir = os.path.join(tmpdir, "converted_data")
    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_data_path,
        output_dir=output_dir,
    )
    # Fetch a different row from the first one
    # Need to yield the first two rows as well
    file_converter.current_row_idx = 2
    for _ in range(file_converter.current_row_idx):
        next(file_converter.data_loader)

    expected_row = (
        {
            "cat1": "c",
            "num2": 14.0
        },
        0
    )
    file_converter._load_row()
    test_row = file_converter.row
    # Check features
    assert expected_row[0].keys() == test_row[0].keys()
    assert list(expected_row[0].values()) == list(test_row[0].values())
    # Check target
    assert expected_row[1] == test_row[1]
    

