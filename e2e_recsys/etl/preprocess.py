import json
import pandas as pd

from e2e_recsys.features.csv_vocab_builder import CSVVocabBuilder


def preprocess_data(vocab_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Given a vocab, map the corresponding features to their lookup values
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        features = [key for key in vocab.keys() if not key.startswith("<")]
    print(f"Preprocessing features: {features}")
    for feature in features:
        data[feature] = data[feature].apply(
            lambda x: vocab[feature].get(x, vocab[CSVVocabBuilder.default_key])
        )
    return data
