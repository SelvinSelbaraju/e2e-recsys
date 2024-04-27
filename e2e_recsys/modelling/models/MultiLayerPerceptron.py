import torch
from e2e_recsys.modelling.models.abstract_torch_model import AbstractTorchModel


class MultiLayerPerceptron(AbstractTorchModel):
    # Need to set this when initialising OHE layers
    input_size = 1

    def _init_hidden_layers(self):
        hidden_units = self.hyperparam_config["hidden_units"]
        self.hidden_layers = []
        input_size = self.input_size
        for units in hidden_units:
            # Create a layer with desired output units
            # Then that number of units is the input for the next layer
            self.hidden_layers.append(torch.nn.Linear(input_size, units))
            input_size = units
        # Append the output layer
        self.hidden_layers.append(torch.nn.Linear(input_size, 1))

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
                torch.nn.functional.one_hot(x[categorical_feature], -1)
            )

        numerical_inputs = torch.cat(numerical_inputs)
        categorical_inputs = torch.cat(categorical_inputs, -1)
        return torch.cat(numerical_inputs, categorical_inputs, -1)
