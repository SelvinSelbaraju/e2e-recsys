import os
import pytest
import pandas as pd
import torch

from e2e_recsys.data_generation.file_converter import FileConverter

COLUMNS_TO_USE = ["cat1", "num2"]
TARGET_COL = "target"

def test_file_converter_init(mock_data_path, tmpdir):
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


def test_save_row(mock_data_path, tmpdir):
    expected_row = (
        {
            "cat1": "a",
            "num2": 1.0
        },
        0
    )
    output_dir = os.path.join(tmpdir, "converted_data")
    
    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_data_path,
        output_dir=output_dir,
    )
    file_converter._load_row()
    file_converter._save_row()

    # Check a file was output and that it has the correct information
    expected_output_path = os.path.join(output_dir, "data_row_0.pt")
    assert os.path.exists(expected_output_path)
    assert torch.load(expected_output_path) == expected_row


def test_convert_row(mock_data_path, tmpdir):
    expected_row = (
        {
            "cat1": "a",
            "num2": 1.0
        },
        0
    )
    
    output_dir = os.path.join(tmpdir, "converted_data")
    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_data_path,
        output_dir=output_dir,
    )

    file_converter._convert_row()
    # Check a file was output and that it has the correct information
    expected_output_path = os.path.join(output_dir, "data_row_0.pt")
    assert os.path.exists(expected_output_path)
    assert torch.load(expected_output_path) == expected_row

    # Check the index was incremented
    assert file_converter.current_row_idx == 1


@pytest.mark.parametrize(
    ["max_rows", "expected_row"],
    [
        (
            1,
            (
                {
                    "cat1": "a",
                    "num2": 1.0
                },
                0
            )
        ),
        (
            3,
            (
                {
                    "cat1": "c",
                    "num2": 14.0
                },
                0
            )
        ),
    ]
)
def test_convert_rows(mock_data_path, tmpdir, max_rows, expected_row):
    output_dir = os.path.join(tmpdir, "converted_data")
    file_converter = FileConverter(
        columns=COLUMNS_TO_USE,
        target_col=TARGET_COL,
        input_filepath=mock_data_path,
        output_dir=output_dir,
    )
    file_converter.convert_rows(max_rows)

    # Check a file was output and that it has the correct information
    # We -1 from max_rows as the files are named by row index
    expected_output_path = os.path.join(output_dir, f"data_row_{max_rows-1}.pt")
    assert os.path.exists(expected_output_path)
    assert torch.load(expected_output_path) == expected_row

    

