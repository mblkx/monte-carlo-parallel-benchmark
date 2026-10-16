"""Statystyki częściowe i obliczanie całki metodą Monte Carlo.

Każdy wykonawca (proces, wątek, proces MPI, blok GPU) zwraca trójkę
(n, mean, m2). Trójki te można scalać wzorem Chana, więc wszystkie wersje
programu (referencja, CPU, MPI, GPU) korzystają z tej samej matematyki.
"""

from dataclasses import dataclass
import math

import numpy as np

# Kwantyl rozkładu normalnego dla przedziału ufności 95%
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Stats:
    """Statystyki częściowe próbki.

    n    - liczba obserwacji
    mean - średnia arytmetyczna
    m2   - suma kwadratów odchyleń od średniej: sum((x - mean)^2)
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @staticmethod
    def from_array(x: np.ndarray) -> "Stats":
        """Liczy statystyki dla tablicy wartości (metoda dwuprzebiegowa)."""
        n = int(x.size)
        if n == 0:
            return Stats()
        mean = float(np.mean(x, dtype=np.float64))
        # Dwa przebiegi: najpierw średnia, potem odchylenia od niej.
        # Jest to stabilne numerycznie, w przeciwieństwie do sum(x^2) - n*mean^2.
        d = x.astype(np.float64, copy=False) - mean
        m2 = float(np.dot(d, d))
        return Stats(n, mean, m2)

    def merge(self, other: "Stats") -> "Stats":
        """Scala dwie statystyki częściowe wzorem Chana.

        Nie wolno uśredniać średnich bez uwzględnienia liczebności:
        m2 wymaga korekty o człon delta^2 * n1 * n2 / n.
        """
        if self.n == 0:
            return other
        if other.n == 0:
            return self
        n = self.n + other.n
        delta = other.mean - self.mean
        mean = self.mean + delta * other.n / n
        m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / n
        return Stats(n, mean, m2)

    def variance(self) -> float:
        """Wariancja próbkowa (dzielenie przez n - 1). NaN dla n < 2."""
        if self.n < 2:
            return math.nan
        return self.m2 / (self.n - 1)


def merge_all(parts) -> Stats:
    """Scala dowolną liczbę statystyk częściowych.

    Kolejność nie ma znaczenia z dokładnością do błędów zaokrągleń.
    """
    total = Stats()
    for p in parts:
        total = total.merge(p)
    return total


@dataclass(frozen=True)
class IntegralResult:
    """Wynik estymacji całki."""

    estimate: float  # estymata całki
    se: float        # błąd standardowy estymaty
    ci_low: float    # dolna granica przedziału ufności 95%
    ci_high: float   # górna granica przedziału ufności 95%
    n: int           # liczba próbek


def integral_from_stats(s: Stats, a: float, b: float) -> IntegralResult:
    """Zamienia statystyki wartości f(x_i) na wynik całki z przedziału [a, b].

    Estymata:              I  = (b - a) * mean(f)
    Błąd standardowy:      SE = (b - a) * sqrt(var / n)
    Przedział ufności 95%: I +/- 1.96 * SE
    """
    estimate = (b - a) * s.mean
    if s.n < 2:
        # Przy jednej próbce wariancja jest niezdefiniowana
        return IntegralResult(estimate, math.nan, math.nan, math.nan, s.n)
    se = (b - a) * math.sqrt(s.variance() / s.n)
    return IntegralResult(estimate, se, estimate - Z_95 * se, estimate + Z_95 * se, s.n)


def abs_error(estimate: float, exact: float) -> float:
    """Błąd bezwzględny |estimate - exact|."""
    return abs(estimate - exact)


def rel_error(estimate: float, exact: float) -> float:
    """Błąd względny. Wartość dokładna musi być różna od zera."""
    if exact == 0.0:
        raise ValueError("Błąd względny jest niezdefiniowany dla wartości dokładnej 0")
    return abs(estimate - exact) / abs(exact)