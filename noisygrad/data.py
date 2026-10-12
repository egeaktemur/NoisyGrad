from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# The train/val split is fixed across seeds so every run is selected on the same val set
SPLIT_SEED = 0

# Dataset class and per-channel normalization mean/std
CIFAR_DATASETS = {
    "CIFAR10": (torchvision.datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": (torchvision.datasets.CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def load_dataset(dataset: str, batch_size: int, val_size: int,
                 generator: torch.Generator) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test loaders. `generator` drives shuffling and augmentation."""
    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_full = train_full
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset in CIFAR_DATASETS:
        dataset_cls, mean, std = CIFAR_DATASETS[dataset]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_full = dataset_cls(root='./data', train=True, download=True, transform=transform_train)
        # Separate instance so the val split is not augmented
        val_full = dataset_cls(root='./data', train=True, download=True, transform=transform_test)
        test_dataset = dataset_cls(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == "SVHN":
        normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        # No horizontal flip: mirrored digits are different digits
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        train_full = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        # Separate instance so the val split is not augmented
        val_full = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_test)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    train_dataset = Subset(train_full, perm[val_size:])
    val_dataset = Subset(val_full, perm[:val_size])

    # Worker seeds (and therefore augmentation) are drawn from `generator`, not the global RNG
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
