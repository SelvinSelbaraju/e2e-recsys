import os
from typing import Any, Dict, Tuple
import torch
from torch.utils.data.dataset import Dataset


class DiskDataset(Dataset):
    """
    A Custom PyTorch Dataset class for loading serialized data tuples
    Each file contains one row of data
    To fetch a specific item, we index the file array and load
    PyTorch DataLoader handles optimal batching, shuffling, parallelisation etc
    """

    def __init__(self, data_dir: str, extension: str = "pt") -> None:
        # Keep only files with the extension
        self.files = [
            os.path.join(data_dir, file)
            for file in os.listdir(data_dir)
            if file.endswith(extension)
        ]
        self.files.sort()

    # Used by the PyTorch DataLoader class
    # Helps it understand the max index
    def __len__(self) -> int:
        return len(self.files)

    # Used by the PyTorch DataLoader class
    # Helps with batching, shuffling etc
    def __getitem__(self, idx) -> Tuple[Dict[str, Any], int]:
        return torch.load(self.files[idx])
