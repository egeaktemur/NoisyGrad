import argparse

import torch

from noisygrad import CNNTestFramework, ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Compare gradient noise levels and modes")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset,
        batch_size=128,
        learning_rate=0.001,
        epochs=args.epochs,
        gradient_noise_std=0.0,
        gradient_noise_prob=1.0,
        gradient_noise_mode="addition",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create framework
    framework = CNNTestFramework(config)

    noise_levels = [0.0, 0.01, 1, 0.001, 0.5, 0.1, 0.05]
    mode_types = ["addition", "multiplication"]
    print(f"\nComparing gradient noise levels: {noise_levels}")
    print(f"Comparing gradient noise modes: {mode_types}")

    framework.compare(noise_levels, mode_types)

    # Plot results for the last experiment
    framework.plot_results()

    # Save all results
    framework.save_results(f"{args.dataset}experiment_results.json", framework.results)


if __name__ == "__main__":
    main()
