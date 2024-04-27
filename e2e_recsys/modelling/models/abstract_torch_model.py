from abc import ABC, abstractmethod
from typing import Dict, Set
import torch


class AbstractTorchModel(ABC, torch.nn.Module):
    default_activation = "Relu"
    default_output_transform = "Sigmoid"

    def __init__(
        self,
        hyperparam_config: Dict,
        numeric_feature_names: Set[str],
        categorical_feature_names: Set[str],
    ):
        self.hyperparam_config = hyperparam_config
        self.numeric_feature_names = numeric_feature_names
        self.categorical_feature_names = categorical_feature_names

        self._init_functions()
        self._init_layers()

    # Set activation functions and output layer transformation
    def _init_functions(self) -> None:
        module = torch.nn
        self.activation = getattr(
            module,
            self.hyperparam_config.get("activation", self.default_activation),
        )()
        self.output_transform = getattr(
            module,
            self.hyperparam_config.get(
                "output_transform", self.default_output_transform
            ),
        )()

    @abstractmethod
    def _init_layers(self) -> None:
        pass

    # This should be called in _init_layers
    # Categorical layers are assigned to integers during data preprocessing
    # This is because PyTorch does not handle strings well
    @abstractmethod
    def _init_categorical_layers(self) -> None:
        pass

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass
