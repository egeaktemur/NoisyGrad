import hashlib
import json
from dataclasses import asdict, dataclass

import torch


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    dataset: str = "MNIST"  # "MNIST" or "CIFAR10"
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 25
    val_size: int = 5000  # Held out from the training set for model selection
    gradient_noise_std: float = 0.0  # Standard deviation for gradient noise
    gradient_noise_mode: str = "addition"  # "multiplication" or "addition"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def name(self) -> str:
        if self.gradient_noise_std == 0:
            return f"{self.dataset}_0_s{self.seed}"
        return f"{self.dataset}_{self.gradient_noise_std}_{self.gradient_noise_mode}_s{self.seed}"

    def key(self) -> str:
        """Hash of every setting that affects results (device excluded)"""
        settings = asdict(self)
        del settings["device"]
        if self.gradient_noise_std == 0:
            del settings["gradient_noise_mode"]
        blob = json.dumps(settings, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]
