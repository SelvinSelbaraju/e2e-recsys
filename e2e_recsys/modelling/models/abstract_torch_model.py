from abc import ABC, abstractmethod
from typing import Dict
import torch


class AbstractTorchModel(ABC, torch.nn.Module):
    default_activation = "Relu"
    default_output_transform = "Sigmoid"

    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self._init_functions()
        self._init_layers()

    # Set activation functions and output layer transformation
    def _init_functions(self) -> None:
        module = torch.nn
        self.activation = getattr(
            module,
            self.model_config.get("activation", self.default_activation),
        )()
        self.output_transform = getattr(
            module,
            self.model_config.get(
                "output_transform", self.default_output_transform
            ),
        )()

    @abstractmethod
    def _init_layers(self) -> None:
        pass

    # This should be called in _init_layers
    @abstractmethod
    def _init_categorical_layers(self) -> None:
        pass

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass

    @abstractmethod
    def _preprocessing(self, x) -> Dict[str, torch.Tensor]:
        pass
