import json
import os
from typing import Dict, Set
import pandas as pd


class CSVVocabBuilder:
    # Map any unknown to 0
    default_key = "<DEFAULT_VALUE>"
    default_value = 0
    """
    Map categorical features to integers for Torch
    Create from Dataframe and features, assumes data fits in memory
    Ideally would be created from a data warehouse

    Output a JSON so works cross-lang
    Just in case backend is not Python
    """

    # Init data loader here to prevent OOM issues
    def __init__(self, features: Set[str], data: pd.DataFrame):
        self.features = features
        self.data = data
        self.vocab = {}

    def build_vocab(self) -> Dict[str, Dict[str, int]]:
        for feature in self.features:
            feature_vocab = {}
            unique_vals = self.data[feature].unique()
            current_idx = self.default_value + 1
            for val in unique_vals:
                # Add into vocab if not there
                if not feature_vocab.get(val):
                    feature_vocab[val] = current_idx
                    current_idx += 1
            self.vocab[feature] = feature_vocab
        self.vocab[self.default_key] = self.default_value

    def save_vocab(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving vocab JSON to {output_path}")
        with open(output_path, "w") as f:
            json.dump(self.vocab, f)
