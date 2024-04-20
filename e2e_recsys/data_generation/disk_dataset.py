import os
from typing import List
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from e2e_recsys.data_generation.file_converter import FileConverter

class DiskDataset(Dataset):
    def __init__(self, data_dir: str, extension: str = "pt"):
        # Keep only files with the extension
        self.files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(extension)]
        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.load(self.files[idx])



converter = FileConverter(
    ["prod_name", "age"],
    "purchased",
    "/Users/selvino/e2e-recsys/data/val_data.csv",
    "converted_data",
)

converter.convert_rows(10)
    
ds = DiskDataset(data_dir="/Users/selvino/e2e-recsys/converted_data")

loader = DataLoader(ds, batch_size=5, num_workers=0)

for i,batch in enumerate(loader):
    print(batch)

    if i > 0:
        break

