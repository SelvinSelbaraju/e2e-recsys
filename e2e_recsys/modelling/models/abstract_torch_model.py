from abc import ABC, abstractmethod
from typing import Any, Dict, Set
import importlib
import torch
from e2e_recsys.features.csv_vocab_builder import CSVVocabBuilder


class AbstractTorchModel(ABC, torch.nn.Module):
    default_activation = "ReLU"
    default_output_transform = "Sigmoid"

    def __init__(
        self,
        architecture_config: Dict[str, Any],
        numeric_feature_names: Set[str],
        categorical_feature_names: Set[str],
        vocab: Dict[str, Dict[str, int]],
    ):
        super().__init__()

        self.architecture_config = architecture_config
        self.numeric_feature_names = sorted(numeric_feature_names)
        self.categorical_feature_names = sorted(categorical_feature_names)

        self.vocab = vocab
        # Get rid of the <DEFAULT_VALUE> key as not useful here
        self.vocab.pop(CSVVocabBuilder.default_key)
        self._init_vocab_sizes()
        self._get_input_size()
        self._init_layers()
        self._init_functions()

    # Set activation functions and output layer transformation
    def _init_functions(self) -> None:
        module = importlib.import_module("torch.nn")
        self.activation = getattr(
            module,
            self.architecture_config.get(
                "activation", self.default_activation
            ),
        )()
        self.output_transform = getattr(
            module,
            self.architecture_config.get(
                "output_transform", self.default_output_transform
            ),
        )()

    # For each feature, say what the number of classes in the training data is
    # Its the max lookup value + 1, as there is the zero class
    def _init_vocab_sizes(self) -> None:
        vocab_sizes = {}
        for categorical_feature in self.vocab:
            num_classes = max(self.vocab[categorical_feature].values()) + 1
            vocab_sizes[categorical_feature] = num_classes
        self.vocab_sizes = vocab_sizes

    # X is a B x 1 matrix
    # Output is a B x Num Classes Matrix
    def _one_hot_encode_feature(
        self, categorical_feature: torch.Tensor, vocab_size: int
    ) -> torch.Tensor:
        result = torch.zeros(
            (categorical_feature.shape[0], vocab_size), dtype=torch.int64
        )
        return result.scatter(1, categorical_feature.type(torch.int64), 1)

    def _get_input_size(self) -> None:
        input_size = 0
        input_size += len(self.numeric_feature_names)
        # Need to multiply by num classes for each categorical feature
        for categorical_feature in self.categorical_feature_names:
            vocab_size = self.vocab_sizes[categorical_feature]
            input_size += vocab_size
        self.input_size = input_size

    @abstractmethod
    def _init_layers(self) -> None:
        pass

    @abstractmethod
    def _preprocessing(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
