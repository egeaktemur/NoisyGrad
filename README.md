# NoisyGrad

Human learning is noisy. Can neural networks learn better if we add noise to their gradients?

This repo trains small CNNs on MNIST / CIFAR-10 / CIFAR-100 / SVHN with Gaussian noise added to every parameter's gradient just before each optimizer step, and compares noise levels and noise types.

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

Dependencies are managed with [uv](https://docs.astral.sh/uv/). On Linux, torch comes from the CUDA 12.4 wheel index (see `pyproject.toml`), which works with NVIDIA drivers that are too old for the default PyPI build.

```bash
uv sync                            # create .venv and install the locked dependencies

# Download datasets to data/ (run.py also downloads on first use, but parallel runs would race on the download)
uv run python -c "import torchvision; torchvision.datasets.MNIST('data', download=True)"
uv run python -c "import torchvision; torchvision.datasets.CIFAR10('data', download=True)"   # ~170 MB, the server can be slow
uv run python -c "import torchvision; torchvision.datasets.CIFAR100('data', download=True)"  # ~170 MB, same server
uv run python -c "import torchvision; [torchvision.datasets.SVHN('data', split=s, download=True) for s in ('train', 'test')]"  # ~250 MB

uv run python run.py --dataset MNIST      # or CIFAR10, CIFAR100, SVHN
uv run pytest                             # noise unit tests
```

To add a dependency, use `uv add <package>` (or `uv add --dev <package>` for dev-only tools) so `pyproject.toml` and `uv.lock` stay in sync.

Options: `--optimizer`, `--lr`, `--lr-schedule cosine`, `--subdir <name>` (save under `results/<name>/` and `plots/<name>/`), `--epochs`, `--seeds 0 1 2`, `--stds ...`, `--modes ...`. Datasets download to `data/`. Per-run results go to `results/<dataset>/<name>_<config hash>.json` and training curves to `plots/<dataset>/<name>_<config hash>.png`. A run is loaded from disk instead of retrained only if a result with the same config hash exists, so changing any setting triggers a rerun.

## Results

Pending for all datasets. CIFAR-100 uses the CIFAR-10 CNN with a 100-way output layer; SVHN (street-view house-number digits, 73k train / 26k test) uses the CIFAR-10 CNN as is.
