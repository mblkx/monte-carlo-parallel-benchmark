"""Testy jąder GPU: wzory funkcji, siatka, powtarzalność i poprawność statystyczna.

Testy wymagające karty NVIDIA pomijają się same. Walidacja argumentów i rejestr
funkcji są sprawdzane zawsze.
"""

import numpy as np
import pytest

from functions import INTEGRANDS, SINE, SQUARE, Integrand
from gpu_kernels import (
    DEVICE_FUNCS, cuda_available, evaluate_on_gpu, launch_config,
    sample_values, sample_values_device,
)
from stats import Stats, integral_from_stats

ALL = list(INTEGRANDS.values())
needs_gpu = pytest.mark.skipif(not cuda_available(), reason="CUDA device not available")


# Rejestr funkcji urządzenia

def test_every_integrand_has_device_version():
    assert set(INTEGRANDS) <= set(DEVICE_FUNCS)


def test_cuda_available_returns_bool():
    assert isinstance(cuda_available(), bool)


# Konfiguracja siatki

@pytest.mark.parametrize("n, expected_blocks", [(1, 1), (256, 1), (257, 2), (1_000, 4)])
def test_launch_config_covers_n(n, expected_blocks):
    assert launch_config(n, 256, 2048) == (expected_blocks, 256)


def test_launch_config_is_capped():
    assert launch_config(10**9, 256, 100) == (100, 256)


def test_launch_config_rejects_non_positive_n():
    with pytest.raises(ValueError):
        launch_config(0)


# Walidacja argumentów (bez karty)

@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_n_raises(n):
    with pytest.raises(ValueError):
        sample_values_device(SQUARE, n, seed=1)


def test_negative_seed_raises():
    with pytest.raises(ValueError):
        sample_values_device(SQUARE, 10, seed=-1)


@pytest.mark.parametrize("tpb", [0, 16, 100, 2048])
def test_invalid_threads_per_block_raises(tpb):
    with pytest.raises(ValueError):
        sample_values_device(SQUARE, 10, seed=1, threads_per_block=tpb)


def test_invalid_max_blocks_raises():
    with pytest.raises(ValueError):
        sample_values_device(SQUARE, 10, seed=1, max_blocks=0)


def test_integrand_without_gpu_version_raises():
    custom = Integrand("custom", lambda x: x, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        sample_values_device(custom, 10, seed=1)
    with pytest.raises(ValueError):
        evaluate_on_gpu(custom, np.array([0.5]))


def test_evaluate_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate_on_gpu(SQUARE, np.array([]))
    with pytest.raises(ValueError):
        evaluate_on_gpu(SQUARE, np.zeros((2, 2)))


# Wzory funkcji na GPU kontra numpy (te same u)

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_apply_matches_cpu_on_random_u(integrand):
    u = np.random.default_rng(1).random(100_000)
    gpu = evaluate_on_gpu(integrand, u)
    assert gpu.dtype == np.float64
    assert np.allclose(gpu, integrand.from_unit(u), rtol=1e-12, atol=1e-14)


@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_apply_matches_cpu_at_interval_ends(integrand):
    u = np.array([0.0, np.nextafter(1.0, 0.0), 0.5])
    gpu = evaluate_on_gpu(integrand, u)
    assert np.allclose(gpu, integrand.from_unit(u), rtol=1e-12, atol=1e-14)


@needs_gpu
@pytest.mark.parametrize("n", [1, 31, 32, 255, 256, 257, 10_000])
def test_apply_handles_sizes_around_block_boundaries(n):
    u = np.random.default_rng(n).random(n)
    assert np.allclose(evaluate_on_gpu(SINE, u), SINE.from_unit(u), rtol=1e-12, atol=1e-14)


@needs_gpu
def test_apply_with_small_grid_uses_grid_stride():
    # Siatka dużo mniejsza niż liczba próbek: wszystkie elementy muszą zostać policzone
    u = np.random.default_rng(2).random(50_000)
    gpu = evaluate_on_gpu(SQUARE, u, threads_per_block=32, max_blocks=2)
    assert np.allclose(gpu, SQUARE.from_unit(u), rtol=1e-12, atol=1e-14)


# Generowanie próbek na GPU

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_sample_values_in_range_and_finite(integrand):
    y = sample_values(integrand, 200_000, seed=3)
    assert y.shape == (200_000,)
    assert np.all(np.isfinite(y))
    # Obie funkcje przyjmują na swoich przedziałach wartości z [0, 1]
    assert y.min() >= 0.0 and y.max() <= 1.0


@needs_gpu
@pytest.mark.parametrize("n", [1, 2, 255, 257, 10_000])
def test_sample_values_sizes(n):
    y = sample_values(SQUARE, n, seed=1)
    assert y.size == n
    assert np.all(np.isfinite(y))


@needs_gpu
def test_same_seed_and_grid_gives_identical_values():
    a = sample_values(SINE, 50_000, seed=7)
    b = sample_values(SINE, 50_000, seed=7)
    assert np.array_equal(a, b)


@needs_gpu
def test_different_seeds_give_different_values():
    a = sample_values(SINE, 50_000, seed=1)
    b = sample_values(SINE, 50_000, seed=2)
    assert not np.array_equal(a, b)


@needs_gpu
def test_grid_stride_covers_all_samples():
    # Mała siatka: statystyka musi nadal odpowiadać całce (niezapisane elementy zepsułyby średnią)
    y = sample_values(SQUARE, 400_000, seed=5, threads_per_block=32, max_blocks=4)
    r = integral_from_stats(Stats.from_array(y), SQUARE.a, SQUARE.b)
    assert abs(r.estimate - SQUARE.exact) < 5 * r.se


@needs_gpu
def test_device_array_has_requested_size():
    d = sample_values_device(SQUARE, 1_000, seed=1)
    assert d.size == 1_000


# Poprawność statystyczna estymatora GPU

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_estimate_close_to_exact(integrand):
    y = sample_values(integrand, 2_000_000, seed=123)
    r = integral_from_stats(Stats.from_array(y), integrand.a, integrand.b)
    # Tolerancja 5 SE, żeby test nie był losowo niestabilny
    assert abs(r.estimate - integrand.exact) < 5 * r.se


@needs_gpu
@pytest.mark.slow
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_confidence_interval_coverage(integrand):
    seeds = range(100)
    hits = 0
    for s in seeds:
        y = sample_values(integrand, 20_000, seed=s)
        r = integral_from_stats(Stats.from_array(y), integrand.a, integrand.b)
        hits += r.ci_low <= integrand.exact <= r.ci_high
    coverage = hits / len(seeds)
    print(f"\n  gpu {integrand.name}: pokrycie = {coverage:.3f} (oczekiwane ~0.95)")
    assert 0.88 <= coverage <= 0.995


@needs_gpu
@pytest.mark.slow
def test_standardized_error_is_roughly_standard_normal():
    z = []
    for s in range(200):
        y = sample_values(SINE, 20_000, seed=s)
        r = integral_from_stats(Stats.from_array(y), SINE.a, SINE.b)
        z.append((r.estimate - SINE.exact) / r.se)
    z = np.array(z)
    print(f"\n  gpu: średnia z = {z.mean():.3f}, odchylenie z = {z.std(ddof=1):.3f}")
    assert abs(z.mean()) < 0.3
    assert 0.8 < z.std(ddof=1) < 1.2