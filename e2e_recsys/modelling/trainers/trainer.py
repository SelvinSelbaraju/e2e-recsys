import logging
import os
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
from e2e_recsys.modelling.utils.metric_store import MetricsStore

FORMAT = "%(name)s::%(levelname)s::%(asctime)s::%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger(__name__)


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
        model_output_path: str,
        logs_dir: str = "./runs/",
        # How often to reset the training metrics state during an epoch
        # Eg if 1, only use the metrics for the current batch
        # If None, never reset during an epoch
        reset_metric_freq: Optional[int] = None,
    ):
        self.model = model
        self.model_output_path = model_output_path
        self.logs_dir = logs_dir
        self.writer = SummaryWriter(log_dir=self.logs_dir)
        self.reset_metric_freq = reset_metric_freq
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self._init_configs(config_path)
        self._init_data_loaders(train_data_dir, val_data_dir)
        self._init_metrics_state()

    def _init_configs(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Initialising configs using {config}")
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
        ds = DiskDataset(data_dir, max_files=100000)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle
        )
        return loader

    def _init_data_loaders(
        self,
        train_data_dir: str,
        val_data_dir: str,
    ) -> None:
        logger.info("Initialising train and val data loaders")
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
        logger.info("Finished train and val data loaders")

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

    # Initialise metrics state
    def _init_metrics_state(self) -> None:
        logger.info("Initialising train and val metrics stores")
        self.train_metrics = MetricsStore(self.metrics_functions, self.writer)
        self.val_metrics = MetricsStore(
            self.metrics_functions, self.writer, prefix="val_"
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
        # Step num is zero indexed
        for step_num, batch in self.train_progress:
            self._global_batch += 1
            outputs, labels, loss = self._train_one_batch(batch)

            # Gather data and report
            with torch.no_grad():
                self.train_metrics.update_metric_state(outputs, labels, loss)
                if (
                    not self.reset_metric_freq
                    or (
                        self.reset_metric_freq
                        and (step_num + 1) % self.reset_metric_freq == 0
                    )
                    or (step_num + 1) == len(self.train_ds)
                ):
                    self.train_metrics.compute_metric_state()
                    self.train_metrics.log_to_tensorboard(self._global_batch)
                    # Check if at the reset interval or the last batch
                    if self.reset_metric_freq and (
                        (step_num + 1) % self.reset_metric_freq == 0
                        or (step_num + 1) == len(self.train_ds)
                    ):
                        self.train_metrics.reset_metric_state()
                self.train_progress.set_postfix(
                    {
                        **self.train_metrics.metrics["recent"],
                        **self.val_metrics.metrics["recent"],
                    }
                )
        # Compute train and val metrics at end of epoch
        self._validate()

    def _validate(self) -> None:
        logger.info(f"Validating at the end of epoch {self._current_epoch}")
        self.val_metrics.reset_metric_state()
        # Turn off dropout and switch batch norm mode
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for _, data in enumerate(self.validation_ds):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels).item()
                self.val_metrics.update_metric_state(outputs, labels, loss)
            self.val_metrics.compute_metric_state()
        # Switch back to training mode
        self.model.train(True)
        self.train_progress.set_postfix(
            {
                **self.train_metrics.metrics["recent"],
                **self.val_metrics.metrics["recent"],
            }
        )
        self.val_metrics.log_to_tensorboard(self._global_batch)
        logger.info(
            f"Finished validating at the end of epoch {self._current_epoch}"
        )

    def train(self, epochs: int):
        logger.info("Beginning model training")
        # Used for plotting metrics
        self._global_batch = 0
        for i in range(epochs):
            # Reset the progress bar each epoch to iterate over the data
            self._reset_progress_bar()
            self._current_epoch = i + 1
            self.train_progress.set_description(f"Epoch {self._current_epoch}")
            self._train_one_epoch()
        # Save events to disk and close logging for Tensorboard
        self.writer.flush()
        self.writer.close()
        logger.info(f"Saving model to: {self.model_output_path}")
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        torch.save(self.model, self.model_output_path)


VOCAB_PATH = "/Users/selvino/e2e-recsys/vocab.json"
TRAIN_DATA_DIR = "/Users/selvino/e2e-recsys-data/converted_train_data"
VAL_DATA_DIR = "/Users/selvino/e2e-recsys-data/converted_val_data"
CONFIG_PATH = "/Users/selvino/e2e-recsys/configs/baseline_model.json"
MODEL_OUTPUT_PATH = "/Users/selvino/e2e-recsys/trained_models/model.pt"

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
    model_output_path=MODEL_OUTPUT_PATH,
)

trainer.train(2)
