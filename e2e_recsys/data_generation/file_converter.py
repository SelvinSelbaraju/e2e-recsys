import os
from typing import List, Optional
import pandas as pd
import torch

class FileConverter:
    def __init__(self, columns: List[str], target_col: str, input_filepath: str, output_dir: str, output_file_prefix: str = "data_row", file_extension: str = "pt"):
        self.columns = columns
        self.target_col = target_col
        self.input_filepath = input_filepath
        self.output_dir = output_dir
        self.output_file_prefix = output_file_prefix
        self.file_extension = file_extension

        self.data_loader = pd.read_csv(self.input_filepath, usecols=self.columns + [self.target_col], chunksize=1)
        self.row = None
        
        # Keep track of which row has been loaded 
        self.current_row_idx = 0

        # Make output dir if does not exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_row(self) -> None:
        row_dict = {}
        row_data =  next(self.data_loader)
        for col in self.columns:
            row_dict[col] = row_data[col].to_numpy()[0]
        self.row = (row_dict, row_data[self.target_col].to_numpy()[0])
    
    def _save_row(self):
        output_filename = f"{self.output_file_prefix}_{self.current_row_idx}.{self.file_extension}"
        output_path = os.path.join(self.output_dir, output_filename)
        torch.save(self.row, output_path)
    
    def _convert_row(self):
        self._load_row()
        self._save_row()
        # Move to the next row
        self.current_row_idx += 1
    
    def convert_rows(self, max_rows: Optional[int] = None):
        if max_rows:
            for _ in range(max_rows):
                try:
                    self._convert_row()
                except StopIteration:
                    print(f"Reached max idx of {self.current_row_idx}")
                    break
        else:
            while True:
                try:
                    self._convert_row()
                except StopIteration:
                    print(f"Reached max idx of {self.current_row_idx}")
                    break


