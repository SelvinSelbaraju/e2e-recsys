# This file is a proxy for large-scale feature engineering that would be done in Spark or a Data Warehouse
from typing import Tuple
import pandas as pd

class DataSplitter:
    def __init__(self, train_dates: Tuple[str, str], val_dates: Tuple[str, str], date_col: str = "t_dat"):
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.date_col = date_col

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = self._split_by_time(data, self.train_dates)
        val = self._split_by_time(data, self.val_dates)
        return train, val
    
    def _split_by_time(self, data: pd.DataFrame, dates: Tuple[str, str]):
        return data[(data[self.date_col] >= dates[0]) & (data[self.date_col] <= dates[1])]


