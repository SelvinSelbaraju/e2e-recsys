import json
import os
from typing import Dict, Set
import numpy as np
import pandas as pd


class CSVVocabBuilder:
    """
    Map categorical features to integers for Torch
    Create from Dataframe and features
    Ideally would be created from a data warehouse

    Output a JSON so works cross-lang
    Just in case backend is not Python
    """

    # Init data loader here to prevent OOM issues
    def __init__(self, features: Set[str], data_path: str):
        self.features = features
        self.data_loader = pd.read_csv(
            data_path, chunksize=1, usecols=features
        )
        self.vocab = {}
        # Keeps track of current index for each feature
        self.feature_indices = {}

        # Create empty vocab
        for feature in features:
            self.vocab[feature] = {}
            # Index 0 maintained for oov items
            self.feature_indices[feature] = 1

    def _add_to_vocab(self, feature, value) -> None:
        # For a specific feature, check if the value is in the vocab
        if not self.vocab[feature].get(value, None):
            # If not in vocab, get the current index to use
            converted_value = self._convert_value(value)
            self.vocab[feature][converted_value] = self.feature_indices[
                feature
            ]
            self.feature_indices[feature] += 1

    def _convert_value(self, value) -> str:
        INTABLE_TYPES = [int, float, np.float64, np.float32]
        if type(value) in INTABLE_TYPES:
            value = int(value)
        return str(value)

    def build_vocab(self) -> Dict[str, Dict[str, int]]:
        while True:
            try:
                chunk = next(self.data_loader)
                for feature in chunk:
                    value = chunk[feature].values[0]
                    self._add_to_vocab(feature, value)
            # Have reached the end of the data
            except StopIteration:
                break

    def save_vocab(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving vocab JSON to {output_path}")
        with open(output_path, "w") as f:
            json.dump(self.vocab, f)
