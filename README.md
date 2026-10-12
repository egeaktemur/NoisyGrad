# NoisyGrad

Human learning is noisy. Can neural networks learn better if we add noise to their gradients?

This repo trains small CNNs on MNIST / CIFAR-10 with Gaussian noise added to every parameter's gradient just before each optimizer step, and compares noise levels and noise types.

## Method

- **Noise modes** ([noisygrad/noise.py](noisygrad/noise.py)), with `eps ~ N(0, 1)` per element:
  - `addition`: `grad + std * rms(grad) * eps`, where `rms` is taken over each parameter tensor
  - `multiplication`: `grad * (1 + std * eps)`

  Noise is scaled relative to each tensor's gradient, so `std` is a noise-to-signal ratio (noise RMS / gradient RMS) and means the same thing for every layer and for both modes. Each run logs the realized ratio `||noise|| / ||grad||` and the gradient norm per epoch.
- **Annealing**: noise std decays per epoch as `std * (1 - epoch / epochs) ** 2`.
- **Optimizer**: SGD with momentum 0.9, lr 0.01, batch size 128, 25 epochs. SGD is the default because Adam rescales each update by its running gradient magnitude, which largely cancels injected noise. Adam is available with `--optimizer adam` (default lr 1e-3).
- **Seeds**: each config runs over seeds {0, 1, 2}. Initialization, data order/augmentation and noise use separate RNG streams, so for a given seed every noise setting sees the same init and the same batches as the baseline.
- **Model selection**: 5,000 training images are held out as a fixed validation split. The reported number is test accuracy at the epoch with the best validation accuracy.
- **Sweep**: std ∈ {0, 0.1, 0.3, 1, 3} × {addition, multiplication}. The noise-free baseline is trained once per seed.

## Usage

```bash
pip install -r requirements.txt
python run.py --dataset MNIST      # or CIFAR10
python -m pytest tests             # noise unit tests
```

Options: `--optimizer`, `--lr`, `--epochs`, `--seeds 0 1 2`, `--stds ...`, `--modes ...`. Datasets download to `data/`. Per-run results go to `results/<name>_<config hash>.json` and plots to `plots/<name>.png`. A run is loaded from disk instead of retrained only if a result with the same config hash exists, so changing any setting triggers a rerun.

## MNIST results (25 epochs, earlier setup: Adam, absolute noise `grad + N(0, std)` / `grad * N(1, std)`, single seed 42, no validation split)

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

Under that setup no noise setting beat the noise-free baseline on MNIST. Its training curves are the `plots/results_*.png` files. These numbers are not comparable with the current setup: Adam largely cancels absolute gradient noise, the noise scale ignored gradient magnitude, and a single seed cannot resolve differences of this size.

Results for the current setup (SGD, relative noise, 3 seeds) are pending. CIFAR-10 has not been run yet.
