from dataclasses import dataclass

import torch


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    dataset: str = "MNIST"  # "MNIST" or "CIFAR10"
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 10
    gradient_noise_std: float = 0.0  # Standard deviation for gradient noise
    gradient_noise_prob: float = 1.0  # Probability of applying noise to each gradient
    gradient_noise_mode: str = "addition"  # "multiplication" or "addition"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
