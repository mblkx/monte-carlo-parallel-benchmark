"""Testy statystyczne referencji: zbieżność, pokrycie przedziałów, skalowanie SE.

Ziarna są stałe, więc testy są deterministyczne. Progi dobrano na podstawie
rozkładu dwumianowego (pokrycie) i błędu standardowego (zbieżność).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functions import INTEGRANDS, SINE, SQUARE, Integrand  # noqa: E402
from mc_ref import run_reference  # noqa: E402

ALL = list(INTEGRANDS.values())


# Zbieżność ~ 1/sqrt(N) 

@pytest.mark.slow
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_error_decreases_like_inverse_sqrt_n(integrand):
    """Nachylenie log(błąd) względem log(N) powinno wynosić około -0.5."""
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    seeds = range(60)
    mean_errors = []
    for n in sizes:
        errs = [run_reference(integrand, n, seed=s, batch_size=n).abs_err for s in seeds]
        # Średnia z wielu ziaren, bo błąd pojedynczego uruchomienia jest zbyt losowy
        mean_errors.append(np.mean(errs))

    slope = np.polyfit(np.log(sizes), np.log(mean_errors), 1)[0]
    print(f"\n  {integrand.name}: nachylenie = {slope:.3f} (oczekiwane -0.5)")
    assert -0.58 < slope < -0.42


@pytest.mark.slow
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_standard_error_scales_like_inverse_sqrt_n(integrand):
    """Czterokrotnie więcej próbek powinno dać około dwukrotnie mniejsze SE."""
    se_small = run_reference(integrand, 100_000, seed=1, batch_size=100_000).se
    se_large = run_reference(integrand, 400_000, seed=1, batch_size=100_000).se
    ratio = se_small / se_large
    assert ratio == pytest.approx(2.0, rel=0.05)


# Pokrycie przedziałów ufności 

@pytest.mark.slow
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("n", [1_000, 50_000])
def test_confidence_interval_coverage(integrand, n):
    """Około 95% przedziałów 95% powinno zawierać prawdziwą całkę."""
    seeds = range(200)
    hits = sum(
        run_reference(integrand, n, seed=s, batch_size=n).contains_exact for s in seeds
    )
    coverage = hits / len(seeds)
    print(f"\n  {integrand.name}, N={n}: pokrycie = {coverage:.3f} (oczekiwane ~0.95)")
    # sigma liczby trafień ~ 3.1 z 200; zakres 90-99% to więcej niż 3 sigma
    assert 0.90 <= coverage <= 0.99


@pytest.mark.slow
def test_coverage_does_not_depend_on_batching():
    """Podział na partie nie może zmieniać poprawności przedziału."""
    seeds = range(100)
    hits = sum(
        run_reference(SINE, 20_000, seed=s, batch_size=3_000).contains_exact
        for s in seeds
    )
    assert 0.88 <= hits / len(seeds) <= 0.995


# Niezależność od rozmiaru partii 

def test_batch_size_changes_samples_but_not_correctness():
    """Inny rozmiar partii daje inne próbki (inne strumienie), ale poprawny wynik."""
    a = run_reference(SQUARE, 200_000, seed=3, batch_size=50_000)
    b = run_reference(SQUARE, 200_000, seed=3, batch_size=20_000)
    assert a.estimate != b.estimate
    assert a.abs_err < 5 * a.se
    assert b.abs_err < 5 * b.se


# Rozkład standaryzowanego błędu

@pytest.mark.slow
def test_standardized_error_is_roughly_standard_normal():
    """(estymata - exact) / SE powinno mieć średnią ~0 i odchylenie ~1."""
    z = []
    for s in range(300):
        r = run_reference(SINE, 20_000, seed=s, batch_size=20_000)
        z.append((r.estimate - r.exact) / r.se)
    z = np.array(z)
    print(f"\n  średnia z = {z.mean():.3f}, odchylenie z = {z.std(ddof=1):.3f}")
    assert abs(z.mean()) < 0.25          # sigma średniej ~ 1/sqrt(300) ~ 0.058
    assert 0.85 < z.std(ddof=1) < 1.15   # SE nie jest zaniżone ani zawyżone


# Dodatkowe przypadki brzegowe 

def test_n_equals_batch_size_boundary():
    r = run_reference(SQUARE, 10_000, seed=1, batch_size=10_000)
    assert r.n == 10_000


def test_n_one_above_and_below_batch_boundary():
    below = run_reference(SQUARE, 9_999, seed=1, batch_size=10_000)
    above = run_reference(SQUARE, 10_001, seed=1, batch_size=10_000)
    assert below.n == 9_999
    assert above.n == 10_001


def test_batch_size_one():
    r = run_reference(SQUARE, 50, seed=1, batch_size=1)
    assert r.n == 50
    assert math.isfinite(r.se)


def test_batch_size_larger_than_n():
    r = run_reference(SQUARE, 10, seed=1, batch_size=1_000_000)
    assert r.n == 10


def test_seed_zero_is_valid():
    r = run_reference(SQUARE, 1_000, seed=0)
    assert r.n == 1_000


def test_large_seed_is_valid():
    r = run_reference(SQUARE, 1_000, seed=2**63)
    assert r.n == 1_000


def test_function_with_large_offset_keeps_accuracy():
    """Duża średnia wartości funkcji nie może psuć wariancji (stabilność)."""
    shifted = Integrand("shifted", lambda x: 1e9 + x, 0.0, 1.0, 1e9 + 0.5)
    r = run_reference(shifted, 500_000, seed=4, batch_size=100_000)
    # Odchylenie standardowe U(0,1) wynosi ~0.2887, więc SE ~ 0.2887/sqrt(N)
    expected_se = 0.2887 / math.sqrt(500_000)
    assert r.se == pytest.approx(expected_se, rel=0.05)


def test_negative_integrand_values():
    """Funkcja przyjmująca wartości ujemne: błąd względny liczony z |exact|."""
    neg = Integrand("neg", lambda x: -x, 0.0, 1.0, -0.5)
    r = run_reference(neg, 100_000, seed=2, batch_size=50_000)
    assert r.rel_err == pytest.approx(r.abs_err / 0.5)
    assert r.abs_err < 5 * r.se