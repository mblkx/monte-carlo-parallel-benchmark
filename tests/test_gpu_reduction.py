"""Testy redukcji statystyk na GPU: zgodność ze ścieżką z pełną tablicą, bloki, brzegi.

Testy wymagające karty NVIDIA pomijają się same. Walidacja argumentów jest
sprawdzana zawsze.
"""

import math

import numpy as np
import pytest

from functions import INTEGRANDS, SINE, SQUARE, Integrand
from gpu_kernels import (
    FINALS, block_partials, cuda_available, gpu_integral, gpu_stats, sample_values,
)
from stats import Stats, merge_all

ALL = list(INTEGRANDS.values())
needs_gpu = pytest.mark.skipif(not cuda_available(), reason="CUDA device not available")


# Walidacja argumentów (bez karty)

@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_n_raises(n):
    with pytest.raises(ValueError):
        gpu_stats(SQUARE, n, seed=1)


def test_negative_seed_raises():
    with pytest.raises(ValueError):
        gpu_stats(SQUARE, 10, seed=-1)


@pytest.mark.parametrize("tpb", [0, 48, 96, 100, 2048])
def test_invalid_threads_per_block_raises(tpb):
    with pytest.raises(ValueError):
        gpu_stats(SQUARE, 10, seed=1, threads_per_block=tpb)


def test_unknown_final_raises():
    with pytest.raises(ValueError):
        gpu_stats(SQUARE, 10, seed=1, final="unknown")


def test_invalid_max_blocks_raises():
    with pytest.raises(ValueError):
        gpu_stats(SQUARE, 10, seed=1, max_blocks=0)


def test_integrand_without_gpu_version_raises():
    custom = Integrand("custom", lambda x: x, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        gpu_stats(custom, 10, seed=1)


def test_finals_constant():
    assert set(FINALS) == {"device", "host"}


# Zgodność ze ścieżką z pełną tablicą (te same wartości, inna kolejność sumowania)

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("final", FINALS)
def test_matches_full_array_statistics(integrand, final):
    s = gpu_stats(integrand, 1_000_000, seed=11, final=final)
    ref = Stats.from_array(sample_values(integrand, 1_000_000, seed=11))
    assert s.n == ref.n
    assert s.mean == pytest.approx(ref.mean, rel=1e-11)
    assert s.m2 == pytest.approx(ref.m2, rel=1e-9)


@needs_gpu
@pytest.mark.parametrize("n", [1, 2, 31, 32, 255, 256, 257, 10_000, 262_145])
def test_matches_full_array_for_sizes_around_boundaries(n):
    s = gpu_stats(SINE, n, seed=3)
    ref = Stats.from_array(sample_values(SINE, n, seed=3))
    assert s.n == n
    assert s.mean == pytest.approx(ref.mean, rel=1e-11, abs=1e-14)
    if n > 1:
        assert s.m2 == pytest.approx(ref.m2, rel=1e-9, abs=1e-12)


@needs_gpu
@pytest.mark.parametrize("tpb", [32, 64, 128, 512, 1024])
def test_matches_full_array_for_block_sizes(tpb):
    s = gpu_stats(SQUARE, 300_000, seed=5, threads_per_block=tpb)
    ref = Stats.from_array(sample_values(SQUARE, 300_000, seed=5, threads_per_block=tpb))
    assert s.n == ref.n
    assert s.mean == pytest.approx(ref.mean, rel=1e-11)
    assert s.m2 == pytest.approx(ref.m2, rel=1e-9)


@needs_gpu
def test_small_grid_uses_grid_stride():
    # Siatka dużo mniejsza niż liczba próbek: każda próbka musi zostać policzona
    s = gpu_stats(SQUARE, 400_000, seed=5, threads_per_block=32, max_blocks=4)
    ref = Stats.from_array(sample_values(SQUARE, 400_000, seed=5,
                                         threads_per_block=32, max_blocks=4))
    assert s.n == 400_000
    assert s.mean == pytest.approx(ref.mean, rel=1e-11)
    assert s.m2 == pytest.approx(ref.m2, rel=1e-9)


# Poziomy redukcji

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_device_and_host_final_agree(integrand):
    d = gpu_stats(integrand, 500_000, seed=2, final="device")
    h = gpu_stats(integrand, 500_000, seed=2, final="host")
    assert d.n == h.n
    assert d.mean == pytest.approx(h.mean, rel=1e-12)
    assert d.m2 == pytest.approx(h.m2, rel=1e-10)


@needs_gpu
def test_block_partials_cover_all_samples():
    parts = block_partials(SINE, 100_000, seed=4)
    assert parts.shape[1] == 3
    assert parts[:, 0].sum() == 100_000
    assert np.all(parts[:, 0] > 0)


@needs_gpu
def test_block_partials_merge_to_total():
    parts = block_partials(SQUARE, 100_000, seed=4)
    merged = merge_all(Stats(int(round(n)), m, s) for n, m, s in parts)
    total = gpu_stats(SQUARE, 100_000, seed=4, final="host")
    assert merged == total


@needs_gpu
def test_partials_balanced_between_blocks():
    parts = block_partials(SQUARE, 1_000_000, seed=1, threads_per_block=256, max_blocks=64)
    counts = parts[:, 0]
    # Pętla grid-stride rozdziela próbki prawie po równo
    assert counts.max() - counts.min() <= 256


@needs_gpu
def test_more_threads_than_samples():
    # Większość wątków i bloków jest pusta i nie może zaburzyć wyniku
    s = gpu_stats(SQUARE, 5, seed=1, threads_per_block=1024)
    ref = Stats.from_array(sample_values(SQUARE, 5, seed=1, threads_per_block=1024))
    assert s.n == 5
    assert s.mean == pytest.approx(ref.mean, rel=1e-12)
    assert s.m2 == pytest.approx(ref.m2, rel=1e-9)


# Przypadki brzegowe

@needs_gpu
def test_single_sample_has_undefined_variance():
    s = gpu_stats(SQUARE, 1, seed=1)
    assert s.n == 1
    assert 0.0 <= s.mean <= 1.0
    assert math.isnan(s.variance())


@needs_gpu
def test_two_samples_have_finite_variance():
    s = gpu_stats(SQUARE, 2, seed=1)
    assert s.n == 2
    assert math.isfinite(s.variance())


# Powtarzalność

@needs_gpu
@pytest.mark.parametrize("final", FINALS)
def test_same_seed_and_grid_is_bitwise_repeatable(final):
    a = gpu_stats(SINE, 300_000, seed=7, final=final)
    b = gpu_stats(SINE, 300_000, seed=7, final=final)
    assert a == b


@needs_gpu
def test_different_seeds_differ():
    assert gpu_stats(SINE, 100_000, seed=1) != gpu_stats(SINE, 100_000, seed=2)


# Poprawność statystyczna estymatora

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_estimate_close_to_exact(integrand):
    r = gpu_integral(integrand, 2_000_000, seed=123)
    # Tolerancja 5 SE, żeby test nie był losowo niestabilny
    assert abs(r.estimate - integrand.exact) < 5 * r.se
    assert r.n == 2_000_000


@needs_gpu
def test_large_sample_count_without_large_memory():
    # 2e8 próbek: tablica n-elementowa zajęłaby 1,6 GB, redukcja potrzebuje kilku kilobajtów
    r = gpu_integral(SQUARE, 200_000_000, seed=9)
    assert r.n == 200_000_000
    assert abs(r.estimate - SQUARE.exact) < 5 * r.se


@needs_gpu
@pytest.mark.slow
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_confidence_interval_coverage(integrand):
    seeds = range(200)
    hits = 0
    for s in seeds:
        r = gpu_integral(integrand, 20_000, seed=s)
        hits += r.ci_low <= integrand.exact <= r.ci_high
    coverage = hits / len(seeds)
    print(f"\n  gpu redukcja {integrand.name}: pokrycie = {coverage:.3f} (oczekiwane ~0.95)")
    assert 0.90 <= coverage <= 0.99


@needs_gpu
@pytest.mark.slow
def test_standardized_error_is_roughly_standard_normal():
    z = []
    for s in range(300):
        r = gpu_integral(SINE, 20_000, seed=s)
        z.append((r.estimate - SINE.exact) / r.se)
    z = np.array(z)
    print(f"\n  gpu redukcja: średnia z = {z.mean():.3f}, odchylenie z = {z.std(ddof=1):.3f}")
    assert abs(z.mean()) < 0.25
    assert 0.85 < z.std(ddof=1) < 1.15