import torch


class GradientNoiseHook:
    """Hook to multiply gradients by zero-centered noise during backpropagation"""

    def __init__(self, std: float = 0.01, prob: float = 1.0, mode: str = "addition"):
        self.std = std
        self.prob = prob
        self.mode = mode
        self.hooks = []
        print(f"GradientNoiseHook initialized with std={std}, prob={prob}, mode={mode}")

    def apply_noise_to_grad(self, grad):
        """Apply zero-centered noise to gradients via multiplication or addition"""
        if self.std > 0 and torch.rand(1).item() < self.prob:
            if self.mode == "multiplication":
                noise_multiplier = torch.normal(mean=1, std=self.std, size=grad.size()).to(grad.device)
                return grad * noise_multiplier
            elif self.mode == "addition":
                noise = torch.normal(mean=0, std=self.std, size=grad.size()).to(grad.device)
                return grad + noise
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        return grad

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

    def update_noise_params(self, std: float, prob: float = 1.0, mode: str = "addition"):
        """Update noise parameters"""
        self.std = std
        self.prob = prob
        self.mode = mode
