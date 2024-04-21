import json
import pandas as pd


def preprocess_data(vocab_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Given a vocab, map the corresponding features to their lookup values
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        features = set(vocab.keys())
    print(f"Preprocessing features: {features}")
    for feature in features:
        data[feature] = data[feature].apply(
            lambda x: vocab[feature].get(x, vocab["<DEFAULT_VALUE>"])
        )
    return data
