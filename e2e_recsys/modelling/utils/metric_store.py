from typing import Callable, Dict
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricsStore:
    """
    Store, update, compute and reset metrics for model training
    """

    def __init__(
        self,
        metrics_functions: Dict[str, Callable],
        summary_writer: SummaryWriter,
        prefix: str = "",
    ):
        self.prefix = prefix
        self.metrics_functions = {
            f"{self.prefix}{metric_name}": metric_func
            for metric_name, metric_func in metrics_functions.items()
        }
        metrics = {}
        for metric_type in ["running", "recent"]:
            metrics[metric_type] = {
                metric_name: 0.0 for metric_name in self.metrics_functions
            }
            metrics[metric_type][f"{self.prefix}loss"] = 0.0
        self.metrics = metrics
        self.writer = summary_writer

    # Calculate metrics for output / labels and add to existing state
    # Class-based TorchEval metrics accumulate data, not metric values
    # This is not suitable for disk-based training
    # Instead, we accumulate loss metrics on a running basis, and average them
    def update_metric_state(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        loss: float,
    ) -> None:
        self.metrics["running"][f"{self.prefix}loss"] += loss
        for metric_name, metric_func in self.metrics_functions.items():
            self.metrics["running"][metric_name] += metric_func(
                torch.squeeze(outputs), torch.squeeze(labels)
            ).item()

    # Divide the running metric state by the num batches
    # Use this to update the recent metrics
    def compute_metric_state(self, num_batches: int) -> None:
        for metric_name in self.metrics["running"].keys():
            self.metrics["recent"][metric_name] = self.metrics["running"][
                metric_name
            ] / (num_batches)

    def reset_metric_state(self) -> None:
        for metric_name in self.metrics["running"].keys():
            self.metrics["running"][metric_name] = 0.0
            self.metrics["recent"][metric_name] = 0.0

    def log_to_tensorboard(self, batch_num: int) -> None:
        for metric_name, metric_value in self.metrics["recent"].items():
            self.writer.add_scalar(
                tag=metric_name,
                scalar_value=metric_value,
                global_step=batch_num,
            )
