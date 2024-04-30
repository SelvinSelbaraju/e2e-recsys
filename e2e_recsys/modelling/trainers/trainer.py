import json
from typing import Dict, Union
import torch
import importlib
from e2e_recsys.data_generation.disk_dataset import DiskDataset
from e2e_recsys.modelling.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
)
from e2e_recsys.modelling.models.abstract_torch_model import AbstractTorchModel


class Trainer:
    """
    Class for training a model from data and configs
    """

    def __init__(
        self,
        model: AbstractTorchModel,
        vocab_path: str,
        train_data_dir: str,
        validation_data_dir,
        config_path: str,
    ):
        self.model = model
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self._init_configs(config_path)
        self.train_ds = self._create_data_loader(
            train_data_dir,
            self.hyperparam_config["train_batch_size"],
            self.hyperparam_config.get("shuffle", None),
        )
        self.validation_ds = self._create_data_loader(
            train_data_dir,
            self.hyperparam_config["validation_batch_size"],
            self.hyperparam_config.get("shuffle", None),
        )

    def train(self, epochs: int):
        for i in range(epochs):
            self._train_one_epoch()

    def _init_configs(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.features = config["features"]
        self.hyperparam_config = config["hyperparam_config"]
        self.optimizer = self._get_optimizer(
            self.hyperparam_config["optimizer"]
        )
        self.loss_function = self._get_loss_function(
            self.hyperparam_config["loss_function"]
        )

    def _create_data_loader(
        self, data_dir: str, batch_size: int, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        ds = DiskDataset(data_dir)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle
        )
        return loader

    def _get_optimizer(
        self, optimizer_config: Dict[str, Union[str, Dict[str, float]]]
    ) -> torch.optim.Optimizer:
        optimizer_class = getattr(
            importlib.import_module("torch.optim"), optimizer_config["type"]
        )
        optimizer_kwargs = optimizer_config.get("kwargs", {})
        return optimizer_class(self.model.parameters(), **optimizer_kwargs)

    # Every loss function is a Module
    def _get_loss_function(self, loss_function_name: str) -> torch.nn.Module:
        return getattr(
            importlib.import_module("torch.nn"), loss_function_name
        )()

    def _train_one_epoch(self):
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_ds):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_function(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            if i % 1000 == 0:
                print("  batch {} loss: {}".format(i + 1, loss))

        return loss


VOCAB_PATH = "/Users/selvino/e2e-recsys/vocab.json"
TRAIN_DATA_DIR = "/Users/selvino/e2e-recsys/converted_data"
VAL_DATA_DIR = "/Users/selvino/e2e-recsys/converted_data"
CONFIG_PATH = "/Users/selvino/e2e-recsys/configs/baseline_model.json"

architecture_config = {
    "hidden_units": [4, 2],
    "activation": "ReLU",
    "output_transform": "Sigmoid",
}
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
model = MultiLayerPerceptron(
    architecture_config=architecture_config,
    numeric_feature_names=set(["price", "age"]),
    categorical_feature_names=set(
        [
            "product_type_name",
            "product_group_name",
            "colour_group_name",
            "department_name",
            "club_member_status",
        ]
    ),
    vocab=vocab,
)

trainer = Trainer(
    model,
    VOCAB_PATH,
    train_data_dir=TRAIN_DATA_DIR,
    validation_data_dir=VAL_DATA_DIR,
    config_path=CONFIG_PATH,
)

trainer.train(1)
