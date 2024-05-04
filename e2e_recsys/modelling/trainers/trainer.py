import json
from tqdm import tqdm
import importlib
from typing import Dict, List, Optional, Union, Callable
import torch
from torch.utils.tensorboard import SummaryWriter
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
        logs_dir: Optional[str] = "./runs/",
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
            validation_data_dir,
            self.hyperparam_config["validation_batch_size"],
            self.hyperparam_config.get("shuffle", None),
        )
        self.writer = SummaryWriter(log_dir=logs_dir)

    def _init_configs(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.features = config["features"]
        self.hyperparam_config = config["hyperparam_config"]
        self.training_config = config["training_config"]
        self.metrics_functions = self._get_metrics(
            self.training_config["metrics"]
        )
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

    # Loss metrics are calculated by default
    def _get_metrics(self, metrics: List[str]) -> Dict[str, Callable]:
        self.metrics = metrics
        return {
            metric_name: getattr(
                importlib.import_module("torcheval.metrics.functional"),
                metric_name,
            )
            for metric_name in metrics
        }

    def train(self, epochs: int):
        self.train_metrics = {}
        self.val_metrics = {}
        # Used for plotting metrics
        self._current_batch = 0
        for i in range(epochs):
            self._current_epoch = i + 1
            self.train_data_progress = tqdm(
                enumerate(self.train_ds),
                unit="batch",
                total=len(self.train_ds),
            )
            self.train_data_progress.set_description(
                f"Epoch {self._current_epoch}"
            )
            self._train_one_epoch()
        # Save events to disk and close logging for Tensorboard
        self.writer.flush()
        self.writer.close()

    def _train_one_epoch(self):
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        running_metrics = {metric_name: 0.0 for metric_name in self.metrics}
        running_metrics["loss"] = 0.0
        for i, data in self.train_data_progress:
            self._current_batch += 1
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
            running_loss = running_metrics["loss"] + loss.item()
            running_metrics = {
                metric_name: metric_func(
                    torch.squeeze(outputs), torch.squeeze(labels)
                ).item()
                + running_metrics[metric_name]
                for metric_name, metric_func in self.metrics_functions.items()
            }
            running_metrics["loss"] = running_loss
            self.train_metrics = {
                # squeeze as func API needs 1d tensors
                metric_name: running_metric / (i + 1)
                for metric_name, running_metric in running_metrics.items()
            }
            self.train_data_progress.set_postfix(
                {**self.train_metrics, **self.val_metrics}
            )
            self._log_to_tensorboard(self.train_metrics)
        # Validate at the end of every epoch
        self._validate()

    def _log_to_tensorboard(self, metrics: Dict[str, float]) -> None:
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(
                tag=metric_name,
                scalar_value=metric_value,
                global_step=self._current_batch,
            )

    def _validate(self) -> None:
        # Turn off dropout and switch batch norm mode
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_metrics = {
            f"val_{metric_name}": 0.0 for metric_name in self.metrics
        }
        running_metrics["val_loss"] = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.validation_ds):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                running_loss = running_metrics["val_loss"] + loss.item()
                running_metrics = {
                    f"val_{metric_name}": metric_func(
                        torch.squeeze(outputs), torch.squeeze(labels)
                    ).item()
                    + running_metrics[f"val_{metric_name}"]
                    for metric_name, metric_func in self.metrics_functions.items()
                }
                running_metrics["val_loss"] = running_loss
                self.val_metrics = {
                    # squeeze as func API needs 1d tensors
                    metric_name: running_metric / (i + 1)
                    for metric_name, running_metric in running_metrics.items()
                }
        # Switch back to training mode
        self.model.train(True)
        self._log_to_tensorboard(self.val_metrics)


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

trainer.train(10)
