import torch


class GradientNoiseHook:
    """Hook to add zero-centered noise to gradients during backpropagation"""

    def __init__(self, std: float, mode: str, generator: torch.Generator):
        self.std = std
        self.mode = mode
        self.generator = generator  # Dedicated RNG so noise draws don't shift data order or init
        self.hooks = []

    def apply_noise_to_grad(self, grad):
        """Apply zero-centered noise to gradients via multiplication or addition"""
        if self.std == 0:
            return grad
        noise = torch.randn(grad.shape, generator=self.generator, device=grad.device, dtype=grad.dtype) * self.std
        if self.mode == "multiplication":
            return grad * (1 + noise)
        elif self.mode == "addition":
            return grad + noise
        raise ValueError(f"Unsupported mode: {self.mode}")

    def register_hooks(self, model):
        """Register hooks to all parameters in the model"""
        self.remove_hooks()  # Remove existing hooks first

        for param in model.parameters():
            if param.requires_grad:
                hook = param.register_hook(self.apply_noise_to_grad)
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
