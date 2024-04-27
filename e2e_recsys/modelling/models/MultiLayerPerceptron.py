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
