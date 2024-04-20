import os
import pytest
from e2e_recsys.data_generation.disk_dataset import DiskDataset

def test_disk_dataset_init(mock_converted_data_dir):
    ds = DiskDataset(mock_converted_data_dir)

    for i,file in enumerate(ds.files):
        assert file == os.path.join(mock_converted_data_dir, f"data_row_{i}.pt")


def test_disk_dataset_len(mock_converted_data_dir):
    ds = DiskDataset(mock_converted_data_dir)
    assert len(ds) == 5


def test_disk_dataset_get_item(mock_converted_data_dir):
    ds = DiskDataset(mock_converted_data_dir)

    # Check first two items
    expected_row_1 = ({
        "cat1": "a",
        "num2": 1.0
    }, 0)
    expected_row_2 = ({
        "cat1": "b",
        "num2": 2.0
    }, 1) 
    assert ds[0] == expected_row_1
    assert ds[1] == expected_row_2
