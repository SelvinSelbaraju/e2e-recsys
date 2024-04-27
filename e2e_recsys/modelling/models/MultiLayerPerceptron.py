from typing import Dict
import torch
from e2e_recsys.modelling.models.abstract_torch_model import AbstractTorchModel


class MultiLayerPerceptron(AbstractTorchModel):
    def _init_layers(self) -> None:
        self._init_hidden_layers()

    def _init_hidden_layers(self):
        hidden_units = self.architecture_config["hidden_units"]
        self.hidden_layers = []
        input_size = self.input_size
        for units in hidden_units:
            # Create a layer with desired output units
            # Then that number of units is the input for the next layer
            self.hidden_layers.append(torch.nn.Linear(input_size, units))
            input_size = units
        # Append the output layer
        self.output_layer = torch.nn.Linear(input_size, 1)

    # FIXME: We should be able to support embeddings too
    def _preprocessing(self, x: torch.Dict[str, torch.Tensor]) -> torch.Tensor:
        numerical_inputs = []
        categorical_inputs = []
        for numeric_feature in self.numeric_feature_names:
            numerical_inputs.append(x[numeric_feature])
        for categorical_feature in self.categorical_feature_names:
            # -1 automatically infers the number of classes for OHE
            # Each elem is a (B, N, C) matrix, where C is the number of classes
            categorical_inputs.append(
                self._one_hot_encode_feature(
                    x[categorical_feature],
                    self.vocab_sizes[categorical_feature],
                )
            )

        numerical_inputs = torch.cat(numerical_inputs, -1)
        categorical_inputs = torch.cat(categorical_inputs, -1)
        return torch.cat([numerical_inputs, categorical_inputs], -1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_input = self._preprocessing(x)
        for layer in self.hidden_layers:
            x_input = self.activation(layer(x_input))
        x_output = self.output_layer(x_input)
        return self.output_transform(x_output)
