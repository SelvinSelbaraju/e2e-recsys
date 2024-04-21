import os
import pandas as pd
from e2e_recsys.features.csv_vocab_builder import CSVVocabBuilder

FEATURES = ["cat1", "num2"]


def test_csv_vocab_builder_init(mock_data_path):
    data = pd.read_csv(mock_data_path)
    vb = CSVVocabBuilder(features=set(FEATURES), data=data)

    assert vb.features == set(FEATURES)
    assert vb.vocab == {}


def test_csv_vocab_builder_build_to_vocab(mock_data_path):
    data = pd.read_csv(mock_data_path)
    vb = CSVVocabBuilder(features=set(FEATURES), data=data)
    vb.build_vocab()

    expected_vocab = {
        "cat1": {"a": 1, "b": 2, "c": 3},
        "num2": {1.0: 1, 2.0: 2, 14.0: 3, 15.0: 4, 0.0: 5},
        "<DEFAULT_VALUE>": 0,
    }

    for feature in FEATURES:
        for feature_value in vb.vocab[feature]:
            assert (
                vb.vocab[feature][feature_value]
                == expected_vocab[feature][feature_value]
            )


def test_csv_vocab_builder_save_vocab(mock_data_path, tmpdir):
    output_path = os.path.join(tmpdir, "vocab", "vocab.json")
    data = pd.read_csv(mock_data_path)
    vb = CSVVocabBuilder(features=set(FEATURES), data=data)

    vb.build_vocab()
    vb.save_vocab(output_path)

    assert os.path.exists(output_path)
