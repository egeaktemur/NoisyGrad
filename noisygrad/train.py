import time
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import load_dataset
from .models import CIFAR10Net, MNISTNet
from .noise import GradientNoiseHook

# Offsets give init, data order and noise independent RNG streams for the same seed
DATA_SEED_OFFSET = 10_000
NOISE_SEED_OFFSET = 20_000


def create_model(dataset: str) -> nn.Module:
    if dataset == "MNIST":
        return MNISTNet()
    elif dataset == "CIFAR10":
        return CIFAR10Net()
    raise ValueError(f"Unsupported dataset: {dataset}")


def train_epoch(model, loader: DataLoader, criterion, optimizer, device) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()

    return {'loss': running_loss / len(loader), 'accuracy': 100. * correct / total}


def evaluate(model, loader: DataLoader, criterion, device) -> Dict[str, float]:
    """Evaluate the model on a loader"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            total_loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()

    return {'loss': total_loss / len(loader), 'accuracy': 100. * correct / total}


def train(config: ExperimentConfig) -> Dict:
    """Run a complete training experiment"""
    print(f"\nStarting experiment: {config.name}")
    device = torch.device(config.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_gen = torch.Generator().manual_seed(config.seed + DATA_SEED_OFFSET)
    noise_gen = torch.Generator(device=device).manual_seed(config.seed + NOISE_SEED_OFFSET)

    train_loader, val_loader, test_loader = load_dataset(
        config.dataset, config.batch_size, config.val_size, data_gen
    )
    torch.manual_seed(config.seed)
    model = create_model(config.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    noise_hook = GradientNoiseHook(config.gradient_noise_std, config.gradient_noise_mode, noise_gen)
    if config.gradient_noise_std > 0:
        noise_hook.register_hooks(model)

    history = {f"{split}_{metric}": [] for split in ("train", "val", "test") for metric in ("losses", "accuracies")}
    noise_stds = []
    start_time = time.time()

    for epoch in range(config.epochs):
        # Anneal gradient noise std
        noise_hook.std = config.gradient_noise_std * ((1 - (epoch / config.epochs)) ** 2)
        noise_stds.append(noise_hook.std)

        metrics = {
            'train': train_epoch(model, train_loader, criterion, optimizer, device),
            'val': evaluate(model, val_loader, criterion, device),
            'test': evaluate(model, test_loader, criterion, device),
        }
        for split, m in metrics.items():
            history[f"{split}_losses"].append(m['loss'])
            history[f"{split}_accuracies"].append(m['accuracy'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.epochs}: "
                  f"Noise std: {noise_hook.std:.6f}, "
                  f"Train Loss: {metrics['train']['loss']:.4f}, "
                  f"Train Acc: {metrics['train']['accuracy']:.2f}%, "
                  f"Val Acc: {metrics['val']['accuracy']:.2f}%, "
                  f"Test Acc: {metrics['test']['accuracy']:.2f}%")

    noise_hook.remove_hooks()
    training_time = time.time() - start_time

    # Pick the epoch on val, report test at that epoch
    best_epoch = max(range(config.epochs), key=lambda e: history['val_accuracies'][e])
    results = {
        'experiment_name': config.name,
        'config': asdict(config),
        'config_key': config.key(),
        **history,
        'noise_stds': noise_stds,
        'final_test_accuracy': history['test_accuracies'][-1],
        'best_val_epoch': best_epoch + 1,
        'best_val_accuracy': history['val_accuracies'][best_epoch],
        'test_accuracy_at_best_val': history['test_accuracies'][best_epoch],
        'training_time': training_time,
    }

    print(f"Experiment completed in {training_time:.2f}s")
    print(f"Final test accuracy: {results['final_test_accuracy']:.2f}%")
    print(f"Test accuracy at best val epoch ({best_epoch + 1}): {results['test_accuracy_at_best_val']:.2f}%")
    return results
