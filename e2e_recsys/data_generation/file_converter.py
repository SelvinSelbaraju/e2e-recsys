import os
from typing import List, Optional
import numpy as np
import pandas as pd
import torch


class FileConverter:
    # FIXME: Can we save tensors with explicit type casting based on a schema?
    """
    This class is responsible for converting CSV data to serialised data
    Each row of the CSV is converted to its own file, to enable easy batching
    The serialized data is a tuple
    The first element is a dictionary of features, key=fname, value=fvalue
    The second element is the target
    """

    def __init__(
        self,
        columns: List[str],
        target_col: str,
        input_filepath: str,
        output_dir: str,
        output_file_prefix: str = "data_row",
        file_extension: str = "pt",
    ) -> None:
        self.columns = columns
        self.target_col = target_col
        self.input_filepath = input_filepath
        self.output_dir = output_dir
        self.output_file_prefix = output_file_prefix
        self.file_extension = file_extension

        self.data_loader = pd.read_csv(
            self.input_filepath,
            usecols=self.columns + [self.target_col],
            chunksize=1,
        )
        self.row = None

        # Keep track of which row has been loaded
        self.current_row_idx = 0

        # Make output dir if does not exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_row(self) -> None:
        row_dict = {}
        row_data = next(self.data_loader)
        for col in self.columns:
            # Reshape into 1D arrays
            # The PyTorch Dataloader will batch these into 2D arrays
            data = row_data[col].to_numpy(dtype=np.float32)[0]
            target = row_data[self.target_col].to_numpy(dtype=np.float32)[0]
            if np.isnan(data):
                # FIXME: Don't just impute to zero
                data = np.array([0], dtype=np.float32)
            row_dict[col] = np.reshape(data, (1,))
        # For binary classification, set target col to float32
        # So can be used with loss function
        self.row = (
            row_dict,
            np.reshape(target, (1,)),
        )

    def _save_row(self) -> None:
        output_filename = f"{self.output_file_prefix}_{self.current_row_idx}.{self.file_extension}"  # noqa: E501
        output_path = os.path.join(self.output_dir, output_filename)
        torch.save(self.row, output_path)

    def _convert_row(self) -> None:
        self._load_row()
        self._save_row()
        # Move to the next row
        self.current_row_idx += 1

    def convert_rows(self, max_rows: Optional[int] = None) -> None:
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


train_fc = FileConverter(
    columns=[
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "department_name",
        "club_member_status",
        "price",
        "age",
    ],
    target_col="purchased",
    input_filepath="/Users/selvino/e2e-recsys/data/train_data.csv",
    output_dir="./converted_train_data",
)

train_fc.convert_rows()

val_fc = FileConverter(
    columns=[
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "department_name",
        "club_member_status",
        "price",
        "age",
    ],
    target_col="purchased",
    input_filepath="/Users/selvino/e2e-recsys/data/val_data.csv",
    output_dir="./converted_val_data",
)

val_fc.convert_rows()
