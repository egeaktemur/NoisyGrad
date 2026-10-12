import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .models import CIFAR10Net, MNISTNet
from .noise import GradientNoiseHook


class CNNTestFramework:
    """Main framework for CNN testing with gradient noise"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = []
        self.seed = config.seed

        # Initialize gradient noise hook
        self.grad_noise_hook = GradientNoiseHook(
            std=config.gradient_noise_std,
            prob=config.gradient_noise_prob,
            mode=config.gradient_noise_mode
        )

        # Load dataset and model
        self.train_loader, self.test_loader = self._load_dataset()
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare dataset"""
        if self.config.dataset == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )

        elif self.config.dataset == "CIFAR10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2
        )

        return train_loader, test_loader

    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset"""
        if self.config.dataset == "MNIST":
            model = MNISTNet()
        elif self.config.dataset == "CIFAR10":
            model = CIFAR10Net()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        return model.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def test(self) -> Dict[str, float]:
        """Test the model"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        return {'loss': test_loss, 'accuracy': test_acc}

    def run_experiment(self, experiment_name: str = "default") -> Dict:
        """Run a complete training experiment"""
        print(f"\nStarting experiment: {experiment_name}")
        print(f"Gradient noise std: {self.config.gradient_noise_std} ({self.config.gradient_noise_mode})")

        # Register gradient noise hooks
        if self.config.gradient_noise_std > 0:
            self.grad_noise_hook.register_hooks(self.model)

        # Training loop
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []

        start_time = time.time()

        for epoch in range(self.config.epochs):
            annealed_std = self.config.gradient_noise_std
            if self.config.gradient_noise_std > 0:
                # Anneal gradient noise std
                annealed_std = self.config.gradient_noise_std * ((1 - (epoch / self.config.epochs)) ** 2)

                # Update gradient noise params (keep prob/mode same)
                self.grad_noise_hook.update_noise_params(
                    std=annealed_std,
                    prob=self.config.gradient_noise_prob,
                    mode=self.grad_noise_hook.mode
                )

            # Train
            train_metrics = self.train_epoch()
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])

            # Test
            test_metrics = self.test()
            test_losses.append(test_metrics['loss'])
            test_accs.append(test_metrics['accuracy'])

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Noise std: {annealed_std:.6f}, "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Test Loss: {test_metrics['loss']:.4f}, "
                    f"Test Acc: {test_metrics['accuracy']:.2f}%")

        training_time = time.time() - start_time

        # Remove hooks
        self.grad_noise_hook.remove_hooks()

        # Store results
        experiment_results = {
            'experiment_name': experiment_name,
            'config': self.config.__dict__,
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'test_losses': test_losses,
            'test_accuracies': test_accs,
            'final_test_accuracy': test_accs[-1],
            'best_test_accuracy': max(test_accs),
            'training_time': training_time
        }

        self.results.append(experiment_results)

        print(f"Experiment completed in {training_time:.2f}s")
        print(f"Final test accuracy: {test_accs[-1]:.2f}%")
        print(f"Best test accuracy: {max(test_accs):.2f}%")

        return experiment_results

    def compare(self, noise_stds: List[float], modes: List[str], experiment_prefix: str = "noise_mode") -> Dict:
        """Compare different gradient noise modes"""
        print(f"\n{'='*50}")
        print("GRADIENT NOISE MODE COMPARISON")
        print(f"{'='*50}")
        comparison_results = {}
        trained_noise_level_0 = False

        for noise_level in noise_stds:
            # Train noise level 0 only once
            for mode in modes:
                experiment_name = f"{self.config.dataset}_{noise_level}_{mode}"

                if noise_level == 0:
                    experiment_name = f"{self.config.dataset}_0"

                    if not trained_noise_level_0:
                        trained_noise_level_0 = True
                    else:
                        print(f"Skipping noise level 0 training for mode {mode}")
                        continue

                if not os.path.exists(os.path.join("results", f"results_{experiment_name}.json")):
                    # Set seed for reproducibility
                    self.set_seed(self.seed)
                    # Update noise parameters
                    self.config.gradient_noise_std = noise_level
                    self.config.gradient_noise_mode = mode

                    # Reinitialize model for fair comparison
                    self.model = self._create_model()
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

                    # Run experiment
                    results = self.run_experiment(experiment_name)
                    if comparison_results.get(noise_level) is None:
                        comparison_results[noise_level] = {}
                    comparison_results[noise_level][mode] = results

                    self.save_results(f"results_{experiment_name}.json", results)
                else:
                    results = self.load_results(f"results_{experiment_name}.json")
                    if isinstance(results, list):
                        results = results[0]
                        self.save_results(f"results_{experiment_name}.json", results)

                    if comparison_results.get(noise_level) is None:
                        comparison_results[noise_level] = {}
                    comparison_results[noise_level][mode] = results
                    print(results)

                    print(f"Loaded results for {experiment_name}")
                    print(f"Gradient noise std: {self.config.gradient_noise_std} ({self.config.gradient_noise_mode})")
                    for epoch in range(self.config.epochs):
                        if (epoch + 1) % 5 == 0 or epoch == 0:
                            annealed_std = self.config.gradient_noise_std
                            if self.config.gradient_noise_std > 0:
                                # Anneal gradient noise std
                                annealed_std = self.config.gradient_noise_std * ((1 - (epoch / self.config.epochs)) ** 2)
                            print(f"Epoch {epoch+1}/{self.config.epochs}: "
                                f"Noise std: {annealed_std:.6f}, "
                                f"Train Loss: {results['train_losses'][epoch]:.4f}, "
                                f"Train Acc: {results['train_accuracies'][epoch]:.2f}%, "
                                f"Test Loss: {results['test_losses'][epoch]:.4f}, "
                                f"Test Acc: {results['test_accuracies'][epoch]:.2f}%")

                    print(f"Final test accuracy: {results['final_test_accuracy']:.2f}%")
                    print(f"Best test accuracy: {results['best_test_accuracy']:.2f}%")

                self.plot_results(results, save_path=f"results_{experiment_name}.png")

        # Print comparison summary
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(f"{'Noise Std':<12} {'Final Acc':<12} {'Best Acc':<12} {'Time (s)':<12}")
        print("-" * 50)

        for std in noise_stds:
            for mode in modes:
                # Noise level 0 is trained only once, under the first mode
                if mode not in comparison_results[std]:
                    continue
                results = comparison_results[std][mode]
                print(f"{std:<12} {results['final_test_accuracy']:.2f}% {results['best_test_accuracy']:.2f}% {results['training_time']:.2f}s")

        print(f"{'='*50}")
        return comparison_results

    def plot_results(self, results: Optional[Dict] = None, save_path: Optional[str] = None):
        """Plot training vs test results on the same plots"""
        if results is None:
            if not self.results:
                print("No results to plot")
                return
            results = self.results[-1]  # Use latest results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(results['train_losses']) + 1)

        # Loss (train vs test)
        ax1.plot(epochs, results['train_losses'], 'b-', label='Train Loss')
        ax1.plot(epochs, results['test_losses'], 'r-', label='Test Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()

        # Accuracy (train vs test)
        ax2.plot(epochs, results['train_accuracies'], 'b-', label='Train Accuracy')
        ax2.plot(epochs, results['test_accuracies'], 'r-', label='Test Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.suptitle(
            f"Results: {results['experiment_name']} (Noise std: {results['config']['gradient_noise_std']})",
            y=1.05
        )

        if save_path:
            plots_dir = "plots"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)

            plt.savefig(os.path.join(plots_dir, save_path))

        plt.close(fig)

    def save_results(self, filepath: str, results = None):
        """Save all results to JSON file"""
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(os.path.join(results_dir, filepath), 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """Load results from JSON file"""
        results_dir = "results"

        if not os.path.exists(os.path.join(results_dir, filepath)):
            raise FileNotFoundError(f"File {filepath} not found in {results_dir}")

        with open(os.path.join(results_dir, filepath), 'r') as f:
            return json.load(f)
