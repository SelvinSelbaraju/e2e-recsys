from abc import ABC, abstractmethod
import json
from e2e_recsys.data_generation.disk_dataset import DiskDataset


class BaseTrainer(ABC):
    def __init__(self, vocab_path: str, data_dir: str, config_path: str):
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.ds = DiskDataset(data_dir)

    @abstractmethod
    def train(self):
        pass
