import json
import os
import statistics
from dataclasses import replace
from typing import Dict, List

import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .train import train

RESULTS_DIR = "results"
PLOTS_DIR = "plots"


def run_cached(config: ExperimentConfig) -> Dict:
    """Train `config`, or load its results if this exact config was already run"""
    path = os.path.join(RESULTS_DIR, f"{config.name}_{config.key()}.json")
    if os.path.exists(path):
        print(f"Loaded cached results for {config.name} from {path}")
        with open(path) as f:
            return json.load(f)

    results = train(config)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {path}")
    plot_results(results, os.path.join(PLOTS_DIR, f"{config.name}.png"))
    return results


def run_sweep(base: ExperimentConfig, noise_stds: List[float], modes: List[str],
              seeds: List[int]) -> Dict[tuple, List[Dict]]:
    """Run every (std, mode) for every seed. The noise-free baseline is run once per seed."""
    grid = [(0.0, "none")] + [(std, mode) for std in noise_stds if std > 0 for mode in modes]
    all_results = {}
    for std, mode in grid:
        all_results[(std, mode)] = [
            run_cached(replace(base, gradient_noise_std=std,
                               gradient_noise_mode=base.gradient_noise_mode if std == 0 else mode,
                               seed=seed))
            for seed in seeds
        ]
    print_summary(all_results)
    return all_results


def _mean_std(values: List[float]) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {sd:.2f}"


def print_summary(all_results: Dict[tuple, List[Dict]]):
    print(f"\n{'='*70}")
    print("SUMMARY (mean ± std over seeds, test accuracy %)")
    print(f"{'='*70}")
    print(f"{'Noise std':<10} {'Mode':<15} {'Seeds':<6} {'@ best val':<16} {'Final':<16}")
    print("-" * 70)
    for (std, mode), runs in all_results.items():
        print(f"{std:<10} {mode:<15} {len(runs):<6} "
              f"{_mean_std([r['test_accuracy_at_best_val'] for r in runs]):<16} "
              f"{_mean_std([r['final_test_accuracy'] for r in runs]):<16}")
    print(f"{'='*70}")


def plot_results(results: Dict, save_path: str):
    """Plot train/val/test curves for one run"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(results['train_losses']) + 1)

    for ax, metric, ylabel in ((ax1, 'losses', 'Loss'), (ax2, 'accuracies', 'Accuracy (%)')):
        for split, style in (('train', 'b-'), ('val', 'g-'), ('test', 'r-')):
            ax.plot(epochs, results[f'{split}_{metric}'], style, label=split.capitalize())
        ax.set_title(ylabel.split()[0])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.suptitle(
        f"Results: {results['experiment_name']} (Noise std: {results['config']['gradient_noise_std']})",
        y=1.05
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
