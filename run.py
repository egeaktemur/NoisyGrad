import argparse

from noisygrad import ExperimentConfig, run_sweep


def main():
    parser = argparse.ArgumentParser(description="Compare gradient noise levels and modes")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=None, help="Default: 0.01 for sgd, 0.001 for adam")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stds", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1, 3],
                        help="Noise-to-signal ratios (noise RMS / gradient RMS)")
    parser.add_argument("--modes", nargs="+", default=["addition", "multiplication"])
    args = parser.parse_args()

    lr = args.lr if args.lr is not None else {"sgd": 0.01, "adam": 0.001}[args.optimizer]
    config = ExperimentConfig(dataset=args.dataset, epochs=args.epochs, optimizer=args.optimizer, learning_rate=lr)
    print(f"\nComparing gradient noise levels: {args.stds}")
    print(f"Comparing gradient noise modes: {args.modes}")
    print(f"Seeds: {args.seeds}")

    run_sweep(config, args.stds, args.modes, args.seeds)


if __name__ == "__main__":
    main()
