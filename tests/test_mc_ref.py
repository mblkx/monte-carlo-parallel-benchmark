"""Testy referencji: poprawność, powtarzalność, partie, czasy i CLI."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functions import INTEGRANDS, SINE, SQUARE, Integrand  # noqa: E402
from mc_ref import compute_batch, main, run_reference  # noqa: E402
from rng import batch_uniform, split_into_batches  # noqa: E402
from stats import Stats, merge_all  # noqa: E402

ALL = list(INTEGRANDS.values())


# Powtarzalność

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_same_seed_gives_identical_result(integrand):
    a = run_reference(integrand, 50_000, seed=7, batch_size=10_000)
    b = run_reference(integrand, 50_000, seed=7, batch_size=10_000)
    assert a.estimate == b.estimate
    assert a.se == b.se


def test_different_seeds_give_different_results():
    a = run_reference(SQUARE, 50_000, seed=1, batch_size=10_000)
    b = run_reference(SQUARE, 50_000, seed=2, batch_size=10_000)
    assert a.estimate != b.estimate


# Poprawność

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_estimate_close_to_exact(integrand):
    r = run_reference(integrand, 2_000_000, seed=123, batch_size=500_000)
    # Tolerancja 5 SE, żeby test nie był losowo niestabilny
    assert r.abs_err < 5 * r.se
    assert r.exact == integrand.exact


def test_errors_consistent_with_estimate():
    r = run_reference(SINE, 100_000, seed=5)
    assert r.abs_err == pytest.approx(abs(r.estimate - 2.0))
    assert r.rel_err == pytest.approx(r.abs_err / 2.0)
    assert r.ci_low < r.estimate < r.ci_high


# Zgodność z liczeniem "naraz" 

def test_compute_batch_matches_direct_computation():
    stats, _, _, _ = compute_batch(SINE, seed=9, batch_index=3, size=1000)
    expected = Stats.from_array(SINE.from_unit(batch_uniform(9, 3, 1000)))
    assert stats == expected


def test_batched_result_matches_whole_array():
    n, batch = 25_000, 10_000
    r = run_reference(SQUARE, n, seed=11, batch_size=batch)
    # Te same próbki policzone w całości (bez scalania partii)
    u = np.concatenate([batch_uniform(11, k, s) for k, s in split_into_batches(n, batch)])
    whole = Stats.from_array(SQUARE.from_unit(u))
    assert r.n == whole.n
    assert r.estimate == pytest.approx(SQUARE.b * whole.mean, rel=1e-12)


# Przypadki brzegowe 

def test_sample_count_not_divisible_by_batch():
    r = run_reference(SQUARE, 25_001, seed=1, batch_size=10_000)
    assert r.n == 25_001


def test_single_batch_smaller_than_batch_size():
    r = run_reference(SQUARE, 100, seed=1, batch_size=10_000)
    assert r.n == 100


def test_one_sample_has_undefined_error():
    r = run_reference(SQUARE, 1, seed=1)
    assert r.n == 1
    assert math.isnan(r.se)
    assert not r.contains_exact


def test_two_samples():
    r = run_reference(SQUARE, 2, seed=1)
    assert r.n == 2
    assert math.isfinite(r.se)


def test_constant_function_has_zero_error():
    const = Integrand("const", lambda x: np.full_like(x, 2.0), 0.0, 3.0, 6.0)
    r = run_reference(const, 1000, seed=1, batch_size=300)
    assert r.estimate == pytest.approx(6.0)
    assert r.se == 0.0
    assert r.abs_err == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_sample_count_raises(n):
    with pytest.raises(ValueError):
        run_reference(SQUARE, n, seed=1)


# Czasy

def test_timings_are_consistent():
    r = run_reference(SINE, 200_000, seed=1, batch_size=50_000)
    assert r.t_rng >= 0 and r.t_func >= 0 and r.t_agg >= 0
    assert r.t_total > 0
    # Suma składników nie może przekraczać czasu całości (z małym zapasem)
    assert r.t_rng + r.t_func + r.t_agg <= r.t_total * 1.05 + 1e-3


# Interfejs wiersza poleceń 

def test_cli_prints_result(capsys):
    code = main(["--func", "sine", "--n", "1e4", "--seed", "3", "--batch", "5000"])
    out = capsys.readouterr().out
    assert code == 0
    assert "estimate" in out
    assert "sine" in out
    assert "95% CI" in out


def test_cli_rejects_unknown_function():
    with pytest.raises(SystemExit):
        main(["--func", "nope"])