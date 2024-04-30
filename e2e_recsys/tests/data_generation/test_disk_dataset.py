import os

import numpy as np
from e2e_recsys.data_generation.disk_dataset import DiskDataset


def test_disk_dataset_init(mock_converted_data_dir):
    ds = DiskDataset(mock_converted_data_dir)

    for i, file in enumerate(ds.files):
        assert file == os.path.join(
            mock_converted_data_dir, f"data_row_{i}.pt"
        )


def test_disk_dataset_len(mock_converted_data_dir):
    ds = DiskDataset(mock_converted_data_dir)
    assert len(ds) == 5


def test_disk_dataset_get_item(mock_converted_data_dir):
    ds = DiskDataset(mock_converted_data_dir)

    # Check first two items
    expected_row_1 = (
        {
            "cat1": np.array(
                [
                    1.0,
                ],
                dtype=np.float32,
            ),
            "num2": np.array(
                [
                    1.0,
                ],
                dtype=np.float32,
            ),
        },
        np.array([0.0], dtype=np.float32),
    )
    expected_row_2 = (
        {
            "cat1": np.array(
                [
                    2.0,
                ],
                dtype=np.float32,
            ),
            "num2": np.array(
                [
                    2.0,
                ],
                dtype=np.float32,
            ),
        },
        np.array([1.0], dtype=np.float32),
    )
    np.testing.assert_equal(ds[0][0], expected_row_1[0])
    np.testing.assert_equal(ds[0][1], expected_row_1[1])
    np.testing.assert_equal(ds[1][0], expected_row_2[0])
    np.testing.assert_equal(ds[1][1], expected_row_2[1])
