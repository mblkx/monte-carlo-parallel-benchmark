"""Testy modułu stats: statystyki częściowe, scalanie i wynik całki."""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stats import (  # noqa: E402
    Stats, merge_all, integral_from_stats, abs_error, rel_error, Z_95,
)


def test_from_array_matches_numpy():
    x = np.random.default_rng(1).random(10_000)
    s = Stats.from_array(x)
    assert s.n == x.size
    assert s.mean == pytest.approx(np.mean(x), rel=1e-12)
    assert s.variance() == pytest.approx(np.var(x, ddof=1), rel=1e-10)


# Scalanie

def test_merge_two_parts_equals_whole():
    x = np.random.default_rng(2).normal(5.0, 3.0, 5000)
    whole = Stats.from_array(x)
    merged = Stats.from_array(x[:1234]).merge(Stats.from_array(x[1234:]))
    assert merged.n == whole.n
    assert merged.mean == pytest.approx(whole.mean, rel=1e-12)
    assert merged.m2 == pytest.approx(whole.m2, rel=1e-10)


def test_merge_uneven_parts_in_any_order():
    x = np.random.default_rng(3).random(1 + 7 + 1000 + 3)
    chunks = [x[:1], x[1:8], x[8:1008], x[1008:]]
    expected = Stats.from_array(x)
    parts = [Stats.from_array(c) for c in chunks]
    for _ in range(10):
        random.shuffle(parts)
        result = merge_all(parts)
        assert result.n == expected.n
        assert result.mean == pytest.approx(expected.mean, rel=1e-12)
        assert result.m2 == pytest.approx(expected.m2, rel=1e-9)


def test_merge_many_small_parts():
    x = np.random.default_rng(4).random(10_000)
    parts = [Stats.from_array(c) for c in np.array_split(x, 137)]
    result = merge_all(parts)
    expected = Stats.from_array(x)
    assert result.mean == pytest.approx(expected.mean, rel=1e-12)
    assert result.variance() == pytest.approx(expected.variance(), rel=1e-9)


# Przypadki brzegowe

def test_empty_part():
    empty = Stats.from_array(np.array([]))
    assert empty == Stats()
    s = Stats.from_array(np.array([1.0, 2.0, 3.0]))
    assert empty.merge(s) == s
    assert s.merge(empty) == s


def test_single_sample_variance_undefined():
    s = Stats.from_array(np.array([4.0]))
    assert s.n == 1
    assert s.mean == 4.0
    assert math.isnan(s.variance())


def test_constant_data_zero_m2():
    s = Stats.from_array(np.full(100, 7.5))
    assert s.mean == 7.5
    assert s.m2 == 0.0
    assert s.variance() == 0.0


def test_merge_two_single_samples():
    s = Stats.from_array(np.array([1.0])).merge(Stats.from_array(np.array([3.0])))
    assert s.n == 2
    assert s.mean == 2.0
    assert s.variance() == pytest.approx(2.0)


# Stabilność numeryczna

def test_large_mean_small_variance():
    """Naiwna metoda sum(x^2) - n*mean^2 tu zawodzi, a Stats ma działać."""
    rng = np.random.default_rng(5)
    x = 1e9 + rng.normal(0.0, 1.0, 100_000)
    s = merge_all(Stats.from_array(c) for c in np.array_split(x, 16))
    assert s.mean == pytest.approx(1e9, abs=0.1)
    assert s.variance() == pytest.approx(1.0, rel=0.05)

    # Dla porównania: metoda naiwna daje błędny wynik
    naive = (np.sum(x * x) - x.size * np.mean(x) ** 2) / (x.size - 1)
    print(f"\n  wariancja Stats: {s.variance():.6f}, naiwna: {naive:.6f}")


# Wynik całki

def test_integral_from_stats():
    # f(x) = 2 na [0, 3]: całka 6, wariancja 0
    s = Stats.from_array(np.full(50, 2.0))
    r = integral_from_stats(s, 0.0, 3.0)
    assert r.estimate == pytest.approx(6.0)
    assert r.se == 0.0
    assert r.ci_low == r.ci_high == pytest.approx(6.0)
    assert r.n == 50


def test_confidence_interval_symmetric():
    x = np.random.default_rng(6).random(1000)
    r = integral_from_stats(Stats.from_array(x), 0.0, 1.0)
    assert r.estimate - r.ci_low == pytest.approx(r.ci_high - r.estimate)
    assert r.ci_high - r.estimate == pytest.approx(Z_95 * r.se)


def test_integral_single_sample_nan():
    r = integral_from_stats(Stats.from_array(np.array([1.0])), 0.0, 1.0)
    assert math.isnan(r.se)
    assert math.isnan(r.ci_low)


# Błędy

def test_abs_and_rel_error():
    assert abs_error(1.1, 1.0) == pytest.approx(0.1)
    assert rel_error(2.2, 2.0) == pytest.approx(0.1)


def test_rel_error_raises_for_zero_exact():
    with pytest.raises(ValueError):
        rel_error(0.1, 0.0)