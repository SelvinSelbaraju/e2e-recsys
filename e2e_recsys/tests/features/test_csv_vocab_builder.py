import os
import pandas as pd
from e2e_recsys.features.csv_vocab_builder import CSVVocabBuilder


def test_csv_vocab_builder_init(mock_data_path):
    vb = CSVVocabBuilder(
        features=set(["cat1", "num2"]), data_path=mock_data_path
    )

    test_row = next(vb.data_loader)
    expected_row = pd.DataFrame({"cat1": "a", "num2": 1.0}, index=[0])

    assert vb.features == set(["cat1", "num2"])
    pd.testing.assert_frame_equal(test_row, expected_row)
    assert vb.vocab == {"cat1": {}, "num2": {}}
    assert vb.feature_indices == {"cat1": 1, "num2": 1}


def test_csv_vocab_builder_add_to_vocab(mock_data_path):
    vb = CSVVocabBuilder(
        features=set(["cat1", "num2"]), data_path=mock_data_path
    )

    row1 = next(vb.data_loader)
    row2 = next(vb.data_loader)

    # Check the vocab and indices have updated
    vb._add_to_vocab("cat1", row1["cat1"].values[0])
    vb._add_to_vocab("num2", row1["num2"].values[0])
    assert vb.vocab["cat1"] == {"a": 1}
    assert vb.feature_indices["cat1"] == 2
    assert vb.vocab["num2"] == {"1": 1}
    assert vb.feature_indices["num2"] == 2

    vb._add_to_vocab("cat1", row2["cat1"].values[0])
    vb._add_to_vocab("num2", row2["num2"].values[0])
    assert vb.vocab["cat1"] == {"a": 1, "b": 2}
    assert vb.feature_indices["cat1"] == 3
    assert vb.vocab["num2"] == {"1": 1, "2": 2}
    assert vb.feature_indices["num2"] == 3


def test_csv_vocab_builder_build_to_vocab(mock_data_path):
    vb = CSVVocabBuilder(
        features=set(["cat1", "num2"]), data_path=mock_data_path
    )
    vb.build_vocab()

    assert vb.vocab == {
        "cat1": {"a": 1, "b": 2, "c": 3},
        "num2": {"1": 1, "2": 2, "14": 3, "15": 4, "0": 5},
        "<DEFAULT_VALUE>": 0,
    }


def test_csv_vocab_builder_save_vocab(mock_data_path, tmpdir):
    output_path = os.path.join(tmpdir, "vocab", "vocab.json")
    vb = CSVVocabBuilder(
        features=set(["cat1", "num2"]), data_path=mock_data_path
    )

    vb.build_vocab()
    vb.save_vocab(output_path)

    assert os.path.exists(output_path)
