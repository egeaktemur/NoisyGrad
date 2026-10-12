from typing import Dict

import torch

MODES = ("addition", "multiplication")


class GradientNoise:
    """Adds zero-mean noise to every gradient right before optimizer.step().

    Noise is scaled per tensor relative to its gradient, so `std` is the noise-to-signal ratio:
      addition:       g + std * rms(g) * eps
      multiplication: g * (1 + std * eps)
    In both modes the expected noise RMS is std * rms(g), so the two modes are comparable at equal std.
    Also tracks the gradient norm and the realized noise-to-signal ratio ||noise|| / ||g||.
    """

    def __init__(self, std: float, mode: str, generator: torch.Generator):
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.std = std
        self.mode = mode
        self.generator = generator  # Dedicated RNG so noise draws don't shift data order or init
        self._reset_stats()

    def attach(self, optimizer: torch.optim.Optimizer):
        """Register on `optimizer`; returns a handle with .remove()"""
        return optimizer.register_step_pre_hook(self._step_pre_hook)

    def sample(self, grad: torch.Tensor) -> torch.Tensor:
        """Noise to add to `grad`"""
        eps = torch.randn(grad.shape, generator=self.generator, device=grad.device, dtype=grad.dtype)
        if self.mode == "multiplication":
            return grad * eps * self.std
        rms = grad.norm() / grad.numel() ** 0.5
        return eps * rms * self.std

    def _step_pre_hook(self, optimizer, args, kwargs):
        grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        grad_norm = torch.stack([g.norm() for g in grads]).norm()
        self._grad_norm_sum += grad_norm
        self._steps += 1
        if self.std == 0:
            return

        noises = [self.sample(g) for g in grads]
        self._snr_sum += torch.stack([n.norm() for n in noises]).norm() / grad_norm
        for g, n in zip(grads, noises):
            g.add_(n)

    def _reset_stats(self):
        self._grad_norm_sum = 0.0
        self._snr_sum = 0.0
        self._steps = 0

    def pop_stats(self) -> Dict[str, float]:
        """Mean gradient norm (before noise) and realized noise-to-signal ratio since the last call"""
        steps = max(self._steps, 1)
        stats = {'grad_norm': float(self._grad_norm_sum) / steps, 'snr': float(self._snr_sum) / steps}
        self._reset_stats()
        return stats
