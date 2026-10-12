import pytest
import torch

from noisygrad import GradientNoise


def _setup(std, mode, seed=0, shapes=((256, 512), (512,))):
    """Params with fixed grads and an lr=0 SGD, so step() only runs the noise hook"""
    torch.manual_seed(123)
    params = [torch.nn.Parameter(torch.zeros(s)) for s in shapes]
    for i, p in enumerate(params):
        p.grad = torch.randn(p.shape) * 10 ** -i  # Tensors at different scales
    optimizer = torch.optim.SGD(params, lr=0.0)
    noise = GradientNoise(std, mode, torch.Generator().manual_seed(seed))
    noise.attach(optimizer)
    return params, optimizer, noise


@pytest.mark.parametrize("mode", ["addition", "multiplication"])
def test_zero_std_is_identity(mode):
    params, optimizer, _ = _setup(0.0, mode)
    before = [p.grad.clone() for p in params]
    optimizer.step()
    for p, g in zip(params, before):
        assert torch.equal(p.grad, g)


@pytest.mark.parametrize("mode", ["addition", "multiplication"])
@pytest.mark.parametrize("std", [0.1, 1.0, 3.0])
def test_noise_is_relative_per_tensor(mode, std):
    params, optimizer, noise = _setup(std, mode)
    before = [p.grad.clone() for p in params]
    optimizer.step()
    for p, g in zip(params, before):
        ratio = (p.grad - g).norm() / g.norm()
        assert ratio.item() == pytest.approx(std, rel=0.1)
    assert noise.pop_stats()['snr'] == pytest.approx(std, rel=0.05)


@pytest.mark.parametrize("mode", ["addition", "multiplication"])
def test_noise_is_unbiased(mode):
    std = 1.0
    params, optimizer, _ = _setup(std, mode, shapes=((1000,),))
    g = params[0].grad.clone()
    draws = 2000
    total = torch.zeros_like(g)
    for _ in range(draws):
        params[0].grad = g.clone()
        optimizer.step()
        total += params[0].grad - g
    noise_scale = g.norm() / g.numel() ** 0.5 if mode == "addition" else g.abs()
    # Mean of `draws` zero-mean draws, normalized to unit variance, should be ~N(0, 1)
    z = total / draws / (std * noise_scale / draws ** 0.5)
    assert z.mean().abs() < 0.2
    assert z.std().item() == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize("mode", ["addition", "multiplication"])
def test_same_seed_same_noise(mode):
    results = []
    for seed in (0, 0, 1):
        params, optimizer, _ = _setup(1.0, mode, seed=seed)
        optimizer.step()
        results.append(params[0].grad.clone())
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])


def test_stats_reset_and_grad_norm():
    params, optimizer, noise = _setup(0.0, "addition")
    expected = torch.stack([p.grad.norm() for p in params]).norm().item()
    optimizer.step()
    optimizer.step()
    stats = noise.pop_stats()
    assert stats['grad_norm'] == pytest.approx(expected, rel=1e-5)
    assert stats['snr'] == 0.0
    assert noise.pop_stats() == {'grad_norm': 0.0, 'snr': 0.0}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        GradientNoise(1.0, "subtraction", torch.Generator())
