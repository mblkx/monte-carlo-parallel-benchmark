"""Testy modułu functions: wartości, granice i zgodność z całką dokładną."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functions import INTEGRANDS, SINE, SQUARE, get_integrand  # noqa: E402
from stats import Stats, integral_from_stats  # noqa: E402


# Wartości w znanych punktach

def test_square_known_points():
    x = np.array([0.0, 0.5, 1.0])
    assert np.allclose(SQUARE(x), [0.0, 0.25, 1.0])


def test_sine_known_points():
    x = np.array([0.0, math.pi / 2, math.pi])
    assert np.allclose(SINE(x), [0.0, 1.0, 0.0], atol=1e-12)


# Granice i wartości dokładne

def test_bounds_and_exact_values():
    assert (SQUARE.a, SQUARE.b) == (0.0, 1.0)
    assert (SINE.a, SINE.b) == (0.0, math.pi)
    assert SQUARE.exact == pytest.approx(1.0 / 3.0)
    assert SINE.exact == 2.0


@pytest.mark.parametrize("integrand", list(INTEGRANDS.values()), ids=lambda f: f.name)
def test_exact_value_matches_numerical_quadrature(integrand):
    # Niezależna kontrola wartości dokładnej kwadraturą z scipy
    value, _ = integrate.quad(lambda t: float(integrand(np.array([t]))[0]),
                              integrand.a, integrand.b)
    assert integrand.exact == pytest.approx(value, rel=1e-10)


# Własności techniczne

@pytest.mark.parametrize("integrand", list(INTEGRANDS.values()), ids=lambda f: f.name)
def test_vectorized_shape_and_dtype(integrand):
    x = np.linspace(integrand.a, integrand.b, 1000)
    y = integrand(x)
    assert y.shape == x.shape
    assert y.dtype == np.float64


@pytest.mark.parametrize("integrand", list(INTEGRANDS.values()), ids=lambda f: f.name)
def test_from_unit_maps_interval_ends(integrand):
    u = np.array([0.0, 1.0])
    expected = integrand(np.array([integrand.a, integrand.b]))
    assert np.allclose(integrand.from_unit(u), expected)


def test_from_unit_midpoint():
    assert SQUARE.from_unit(np.array([0.5]))[0] == pytest.approx(0.25)


# Rejestr

def test_registry_lookup():
    assert get_integrand("square") is SQUARE
    assert get_integrand("sine") is SINE


def test_registry_unknown_name_raises():
    with pytest.raises(ValueError):
        get_integrand("does-not-exist")


# Sanity Test Monte Carlo - sprawdzenie czy działa poprawnie

@pytest.mark.parametrize("integrand", list(INTEGRANDS.values()), ids=lambda f: f.name)
def test_monte_carlo_estimate_close_to_exact(integrand):
    u = np.random.default_rng(123).random(1_000_000)
    r = integral_from_stats(Stats.from_array(integrand.from_unit(u)),
                            integrand.a, integrand.b)
    # Tolerancja 5 SE zamiast 1.96 SE, żeby test nie był losowo niestabilny
    assert abs(r.estimate - integrand.exact) < 5 * r.se