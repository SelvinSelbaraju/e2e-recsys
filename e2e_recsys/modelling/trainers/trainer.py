import json
from tqdm import tqdm
import importlib
from typing import Dict, Optional, Tuple, Union, Callable
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
        val_data_dir,
        config_path: str,
        logs_dir: Optional[str] = "./runs/",
    ):
        self.model = model
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self._init_configs(config_path)
        self._init_data_loaders(train_data_dir, val_data_dir)
        self._init_metrics_state()
        self.writer = SummaryWriter(log_dir=logs_dir)

    def _init_configs(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # Update attributes using config
        self.__dict__.update(config)
        # Update specific attributes explicitly
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

    def _init_data_loaders(
        self,
        train_data_dir: str,
        val_data_dir: str,
    ) -> None:
        self.train_ds = self._create_data_loader(
            train_data_dir,
            self.hyperparam_config["train_batch_size"],
            self.hyperparam_config.get("shuffle", False),
        )
        self.validation_ds = self._create_data_loader(
            val_data_dir,
            self.hyperparam_config["validation_batch_size"],
            self.hyperparam_config.get("shuffle", False),
        )

    def _reset_progress_bar(self) -> None:
        # Init a training progress bar
        self.train_progress = tqdm(
            enumerate(self.train_ds),
            unit="batch",
            total=len(self.train_ds),
        )

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

    # Loss metrics are always calculated within the training loop
    # Therefore are excluded here
    def _get_metrics(self, metrics: Dict[str, str]) -> Dict[str, Callable]:
        self.metrics = metrics.values()
        return {
            metric_alias: getattr(
                importlib.import_module("torcheval.metrics.functional"),
                metric_name,
            )
            for metric_name, metric_alias in metrics.items()
        }

    # Return the metrics state dictionary
    # The structure is used for running and recent metric states
    def _get_metrics_state(self) -> Dict[str, Dict[str, float]]:
        metrics = {
            metric_type: {metric_name: 0.0 for metric_name in self.metrics}
            for metric_type in ["train", "val"]
        }
        for metric_type in ["train", "val"]:
            metrics[metric_type]["loss"] = 0.0
        return metrics

    # Initialise metrics state
    def _init_metrics_state(self) -> None:
        self.running_metrics = self._get_metrics_state()
        self.recent_metrics = self._get_metrics_state()

    # Calculate metrics for output / labels and add to existing state
    # Class-based TorchEval metrics accumulate data, not metric values
    # This is not suitable for disk-based training
    # Instead, we accumulate loss metrics on a running basis, and average them
    def _update_metric_state(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        loss: float,
        metric_type: str,
    ) -> None:
        self.running_metrics[metric_type]["loss"] += loss
        for metric_name, metric_func in self.metrics_functions.items():
            self.running_metrics[metric_type][metric_name] += metric_func(
                torch.squeeze(outputs), torch.squeeze(labels)
            ).item()

    # Divide the running metric state by the num batches
    # Use this to update the recent metrics
    def _compute_metric_state(
        self, metric_type: str, num_batches: int
    ) -> None:
        for metric_name in self.metrics_functions.keys():
            self.recent_metrics[metric_type][metric_name] = (
                self.running_metrics[metric_type][metric_name] / (num_batches)
            )
        self.recent_metrics[metric_type]["loss"] = self.running_metrics[
            metric_type
        ]["loss"] / (num_batches)

    def _reset_metric_state(self, metric_type: str) -> None:
        for metric_name in self.metrics_functions.keys():
            self.running_metrics[metric_type][metric_name] = 0.0
            self.recent_metrics[metric_type][metric_name] = 0.0
        self.recent_metrics[metric_type]["loss"] = 0.0
        self.running_metrics[metric_type]["loss"] = 0.0

    def _log_to_tensorboard(self, metric_type: str) -> None:
        for metric_name, metric_value in self.recent_metrics[
            metric_type
        ].items():
            self.writer.add_scalar(
                tag=metric_name,
                scalar_value=metric_value,
                global_step=self._global_batch,
            )

    def _train_one_batch(
        self, data: Tuple[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        inputs, labels = data
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        # Compute the loss and its gradients
        loss = self.loss_function(outputs, labels)
        loss.backward()
        # Adjust learning weights
        self.optimizer.step()
        # Return the model output, target and loss
        return outputs, labels, loss.item()

    def _train_one_epoch(self):
        epoch_batch = 0
        for i, batch in self.train_progress:
            epoch_batch += 1
            self._global_batch += 1
            outputs, labels, loss = self._train_one_batch(batch)

            # Gather data and report
            with torch.no_grad():
                self._update_metric_state(outputs, labels, loss, "train")
                self._compute_metric_state("train", epoch_batch)
            self.train_progress.set_postfix({**self.recent_metrics["train"]})
            self._log_to_tensorboard("train")
        # Validate at the end of every epoch
        self._validate()
        self._reset_metric_state("train")
        self._reset_metric_state("val")

    def _validate(self) -> None:
        # Turn off dropout and switch batch norm mode
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, data in enumerate(self.validation_ds):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels).item()
                self._update_metric_state(outputs, labels, loss, "val")
                self._compute_metric_state("val", i + 1)
        # Switch back to training mode
        self.model.train(True)
        self._log_to_tensorboard("val")

    def train(self, epochs: int):
        self.train_metrics = {}
        self.val_metrics = {}
        # Used for plotting metrics
        self._global_batch = 0
        for i in range(epochs):
            # Reset the progress bar to iterate over the data
            self._reset_progress_bar()
            self._current_epoch = i + 1
            self.train_progress.set_description(f"Epoch {self._current_epoch}")
            self._train_one_epoch()
        # Save events to disk and close logging for Tensorboard
        self.writer.flush()
        self.writer.close()


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
    val_data_dir=VAL_DATA_DIR,
    config_path=CONFIG_PATH,
)

trainer.train(2)
