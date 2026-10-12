# NoisyGrad

Human learning is noisy. Can neural networks learn better if we add noise to their gradients?

This repo trains small CNNs on MNIST / CIFAR-10 with Gaussian noise injected into every parameter's gradient via backward hooks, and compares noise levels and noise types.

## Method

- **Noise modes** ([noisygrad/noise.py](noisygrad/noise.py)):
  - `addition`: `grad + N(0, std)`
  - `multiplication`: `grad * N(1, std)`
- **Annealing**: noise std decays per epoch as `std * (1 - epoch / epochs) ** 2`.
- **Optimizer**: Adam, lr 1e-3, batch size 128, 25 epochs, seed 42. Every run starts from the same initialization.
- **Sweep**: std ∈ {0, 0.01, 1, 0.001, 0.5, 0.1, 0.05} × {addition, multiplication}. The noise-free baseline is trained only once.

## Usage

```bash
pip install -r requirements.txt
python run.py --dataset MNIST      # or CIFAR10
```

Datasets download to `data/`. Per-run results go to `results/results_<name>.json` and plots to `plots/results_<name>.png`. Any run whose JSON already exists is loaded from disk instead of being retrained.

## MNIST results (25 epochs)

| Noise std | Mode           | Final test acc | Best test acc |
|-----------|----------------|----------------|---------------|
| 0         | none           | 99.35%         | 99.38%        |
| 0.001     | addition       | 99.11%         | 99.38%        |
| 0.001     | multiplication | 99.06%         | 99.36%        |
| 0.01      | addition       | 99.15%         | 99.29%        |
| 0.01      | multiplication | 99.18%         | 99.24%        |
| 0.05      | addition       | 98.75%         | 99.11%        |
| 0.05      | multiplication | 98.88%         | 99.23%        |
| 0.1       | addition       | 98.92%         | 98.92%        |
| 0.1       | multiplication | 98.73%         | 98.87%        |
| 0.5       | addition       | 98.23%         | 98.51%        |
| 0.5       | multiplication | 98.47%         | 98.67%        |
| 1         | addition       | 98.59%         | 98.59%        |
| 1         | multiplication | 98.48%         | 98.48%        |

So far no noise setting beats the noise-free baseline on MNIST. Training curves for each run are in [plots/](plots/). CIFAR-10 has not been run yet.
