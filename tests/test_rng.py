"""Testy modułu rng: powtarzalność, rozłączność strumieni i podział na partie."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rng import (  # noqa: E402
    DEFAULT_BATCH_SIZE, batch_generator, batch_uniform, split_into_batches,
)


# Powtarzalność

def test_same_seed_and_batch_gives_identical_samples():
    a = batch_uniform(42, 3, 1000)
    b = batch_uniform(42, 3, 1000)
    assert np.array_equal(a, b)


def test_result_independent_of_call_order():
    # Kolejność tworzenia strumieni nie może wpływać na ich zawartość
    first = [batch_uniform(7, k, 100) for k in range(5)]
    second = [batch_uniform(7, k, 100) for k in reversed(range(5))][::-1]
    for x, y in zip(first, second):
        assert np.array_equal(x, y)


def test_prefix_consistency():
    # Krótsza partia jest początkiem dłuższej (przydatne przy ostatniej partii)
    short = batch_uniform(5, 2, 100)
    long = batch_uniform(5, 2, 1000)
    assert np.array_equal(short, long[:100])


# Rozłączność strumieni

def test_different_batches_differ():
    a = batch_uniform(42, 0, 1000)
    b = batch_uniform(42, 1, 1000)
    assert not np.array_equal(a, b)


def test_different_seeds_differ():
    a = batch_uniform(1, 0, 1000)
    b = batch_uniform(2, 0, 1000)
    assert not np.array_equal(a, b)


def test_streams_are_uncorrelated():
    a = batch_uniform(42, 0, 100_000)
    b = batch_uniform(42, 1, 100_000)
    corr = np.corrcoef(a, b)[0, 1]
    # Dla niezależnych strumieni korelacja ~ N(0, 1/sqrt(n)) ~ 0.003
    assert abs(corr) < 0.02


def test_no_overlapping_values_between_many_batches():
    # Żadne dwie partie nie powinny zawierać tych samych wartości
    # (kolizje 53-bitowych liczb są praktycznie niemożliwe)
    chunks = [batch_uniform(11, k, 10_000) for k in range(20)]
    all_values = np.concatenate(chunks)
    assert np.unique(all_values).size == all_values.size


# Własności statystyczne 

def test_range_dtype_and_mean():
    x = batch_uniform(123, 0, 1_000_000)
    assert x.dtype == np.float64
    assert x.min() >= 0.0
    assert x.max() < 1.0
    # Średnia ~ N(0.5, 0.2887/sqrt(n)) ~ 0.0003; tolerancja 5 sigma
    assert abs(x.mean() - 0.5) < 5 * 0.2887 / np.sqrt(x.size)


def test_generator_type_is_philox():
    g = batch_generator(1, 0)
    assert isinstance(g.bit_generator, np.random.Philox)


# Walidacja argumentów 

def test_negative_arguments_raise():
    with pytest.raises(ValueError):
        batch_generator(-1, 0)
    with pytest.raises(ValueError):
        batch_generator(0, -1)
    with pytest.raises(ValueError):
        batch_uniform(0, 0, -5)


def test_zero_size_returns_empty():
    assert batch_uniform(1, 0, 0).size == 0


# Podział na partie 

def test_split_exact_multiple():
    assert split_into_batches(30, 10) == [(0, 10), (1, 10), (2, 10)]


def test_split_with_remainder():
    assert split_into_batches(25, 10) == [(0, 10), (1, 10), (2, 5)]


def test_split_smaller_than_batch():
    assert split_into_batches(3, 10) == [(0, 3)]


def test_split_zero_samples():
    assert split_into_batches(0, 10) == []


def test_split_sizes_sum_to_total():
    for n in (1, 2, 999, 1_000_000, 2_500_001):
        batches = split_into_batches(n, 1000)
        assert sum(size for _, size in batches) == n


def test_split_invalid_arguments():
    with pytest.raises(ValueError):
        split_into_batches(-1, 10)
    with pytest.raises(ValueError):
        split_into_batches(10, 0)


def test_default_batch_size_value():
    assert DEFAULT_BATCH_SIZE == 1_000_000
    assert split_into_batches(DEFAULT_BATCH_SIZE + 1) == [
        (0, DEFAULT_BATCH_SIZE), (1, 1)
    ]