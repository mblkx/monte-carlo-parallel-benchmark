"""Testy kosztu generatora i funkcji na GPU: warianty jądra, rozkład kosztu, CSV.

Testy wymagające karty NVIDIA pomijają się same. Rozkład kosztu, walidacja
i zapis CSV są sprawdzane zawsze.
"""

import csv
import math
import statistics

import numpy as np
import pytest

from functions import INTEGRANDS, SINE, SQUARE, Integrand
from gpu_costs import (
    COST_COLUMNS, VARIANTS, VariantTimes, decompose, merge_partials,
    time_variants, variant_partials, write_cost_rows,
)
from gpu_kernels import block_partials, cuda_available
from stats import integral_from_stats

ALL = list(INTEGRANDS.values())
needs_gpu = pytest.mark.skipif(not cuda_available(), reason="CUDA device not available")


# Rozkład kosztu (bez karty)

def test_decompose_additive_costs():
    d = decompose(full=10.0, no_func=6.0, no_rng=7.0, baseline=3.0)
    assert d.rng == 3.0 and d.func == 4.0
    assert d.rng_alone == 3.0 and d.func_alone == 4.0
    assert d.base == 3.0
    assert d.interaction == 0.0
    assert d.rng_share == pytest.approx(0.3)
    assert d.func_share == pytest.approx(0.4)
    assert d.base_share == pytest.approx(0.3)


def test_decompose_shares_sum_to_one_when_additive():
    d = decompose(full=12.0, no_func=8.0, no_rng=6.0, baseline=2.0)
    assert d.interaction == 0.0
    assert d.rng_share + d.func_share + d.base_share == pytest.approx(1.0)


def test_decompose_detects_overlap():
    # Karta nakłada część pracy: pełne jądro jest szybsze niż suma składników
    d = decompose(full=8.0, no_func=6.0, no_rng=7.0, baseline=3.0)
    assert d.interaction == pytest.approx(-2.0)


def test_decompose_zero_total_gives_nan_shares():
    d = decompose(0.0, 0.0, 0.0, 0.0)
    assert math.isnan(d.rng_share) and math.isnan(d.func_share) and math.isnan(d.base_share)


# Walidacja argumentów (bez karty)

def test_variants_constant():
    assert set(VARIANTS) == {"full", "no_func", "no_rng", "baseline"}


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        variant_partials(SQUARE, "unknown", 100)


@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_n_raises(n):
    with pytest.raises(ValueError):
        variant_partials(SQUARE, "full", n)
    with pytest.raises(ValueError):
        time_variants(SQUARE, n)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        variant_partials(SQUARE, "full", 100, threads_per_block=96)
    with pytest.raises(ValueError):
        time_variants(SQUARE, 100, repeats=4)
    with pytest.raises(ValueError):
        time_variants(SQUARE, 100, warmup=-1)
    with pytest.raises(ValueError):
        time_variants(SQUARE, 100, seed=-1)


def test_integrand_without_gpu_version_raises():
    custom = Integrand("custom", lambda x: x, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        variant_partials(custom, "full", 100)


# Zapis CSV (bez karty)

def make_times(k):
    return [VariantTimes(full=1.0 + i, no_func=0.6, no_rng=0.7, baseline=0.3) for i in range(k)]


def test_csv_header_written_once_and_rows_appended(tmp_path):
    path = tmp_path / "raw" / "costs.csv"
    write_cost_rows(path, "sine", 1000, 1, 256, 4, make_times(5))
    write_cost_rows(path, "square", 2000, 1, 256, 8, make_times(5))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == COST_COLUMNS
    assert sum(1 for line in lines if line.startswith("func,")) == 1
    assert len(lines) == 1 + 10


def test_csv_roundtrip_values(tmp_path):
    path = tmp_path / "costs.csv"
    write_cost_rows(path, "sine", 1000, 7, 128, 4, make_times(3))
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["repeat"] for r in rows] == ["0", "1", "2"]
    assert rows[1]["func"] == "sine"
    assert int(rows[1]["threads_per_block"]) == 128
    assert float(rows[1]["t_full"]) == pytest.approx(2.0)
    assert float(rows[1]["t_baseline"]) == pytest.approx(0.3)


# Poprawność wariantów jądra

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_full_variant_identical_to_production_kernel(integrand):
    # Ta sama arytmetyka i ta sama siatka: statystyki bloków muszą być identyczne
    variant = variant_partials(integrand, "full", 300_000, seed=5)
    production = block_partials(integrand, 300_000, seed=5)
    assert np.array_equal(variant, production)


@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_no_rng_variant_is_midpoint_rule(integrand):
    # u = (j + 0.5) / n pokrywa wszystkie próbki dokładnie raz: reguła punktu środkowego
    s = merge_partials(variant_partials(integrand, "no_rng", 1_000_000))
    r = integral_from_stats(s, integrand.a, integrand.b)
    assert s.n == 1_000_000
    assert r.estimate == pytest.approx(integrand.exact, abs=1e-8)


@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_baseline_variant_is_mean_of_interval(integrand):
    # Tożsamość zamiast funkcji, deterministyczne u: średnia to środek przedziału
    n = 1_000_000
    s = merge_partials(variant_partials(integrand, "baseline", n))
    assert s.n == n
    assert s.mean == pytest.approx((integrand.a + integrand.b) / 2, abs=1e-8)
    width = integrand.b - integrand.a
    # Punkty (j + 0.5) / n: wariancja próbkowa wynosi dokładnie width^2 (n + 1) / (12 n)
    expected = width ** 2 * (n + 1) / (12 * n)
    assert s.variance() == pytest.approx(expected, rel=1e-9)

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_no_func_variant_is_uniform_on_interval(integrand):
    # Generator + tożsamość: próbki jednostajne na [a, b]
    s = merge_partials(variant_partials(integrand, "no_func", 1_000_000, seed=3))
    width = integrand.b - integrand.a
    se = width / math.sqrt(12) / math.sqrt(s.n)
    assert s.n == 1_000_000
    assert abs(s.mean - (integrand.a + integrand.b) / 2) < 5 * se


@needs_gpu
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n", [1, 31, 257, 10_000])
def test_every_variant_covers_all_samples(variant, n):
    parts = variant_partials(SQUARE, variant, n)
    assert parts[:, 0].sum() == n


@needs_gpu
@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_are_repeatable(variant):
    a = variant_partials(SINE, variant, 200_000, seed=9)
    b = variant_partials(SINE, variant, 200_000, seed=9)
    assert np.array_equal(a, b)


# Pomiary czasu

@needs_gpu
def test_time_variants_returns_requested_repeats():
    times = time_variants(SQUARE, 1_000_000, repeats=5)
    assert len(times) == 5
    for t in times:
        assert all(getattr(t, v) > 0 for v in VARIANTS)


@needs_gpu
def test_full_kernel_is_slower_than_baseline_for_many_samples():
    # Generator i funkcja trygonometryczna muszą kosztować więcej niż sama pętla z redukcją
    times = time_variants(SINE, 100_000_000, repeats=5)
    full = statistics.median(t.full for t in times)
    baseline = statistics.median(t.baseline for t in times)
    assert full > baseline


@needs_gpu
def test_decomposition_of_measured_times_is_consistent():
    times = time_variants(SINE, 20_000_000, repeats=5)
    med = {v: statistics.median(getattr(t, v) for t in times) for v in VARIANTS}
    d = decompose(med["full"], med["no_func"], med["no_rng"], med["baseline"])
    # Tożsamości algebraiczne: rozkład musi się domykać do czasu pełnego jądra
    assert d.rng + d.func_alone + d.base == pytest.approx(d.t_full)
    assert d.func + d.rng_alone + d.base == pytest.approx(d.t_full)
    # Interakcja to różnica między dwiema ścieżkami rozkładu
    assert d.interaction == pytest.approx(
        med["full"] - med["no_func"] - med["no_rng"] + med["baseline"])