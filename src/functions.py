"""Funkcje podcałkowe ze znanymi wartościami dokładnymi.

Każda funkcja jest opisana przez obiekt Integrand (podcałka): nazwa, wektorowa
implementacja (numpy), granice całkowania i dokładna wartość całki.
Dokładne wartości służą do liczenia błędu estymacji w referencji
oraz w wersjach równoległych.
"""

from dataclasses import dataclass
import math
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class Integrand:
    """Funkcja podcałkowa wraz z przedziałem i wartością dokładną."""

    name: str
    func: Callable[[np.ndarray], np.ndarray]  # wersja wektorowa
    a: float                                  # dolna granica całkowania
    b: float                                  # górna granica całkowania
    exact: float                              # dokładna wartość całki
    description: str = ""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Oblicza f(x) dla tablicy punktów x."""
        return self.func(x)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Oblicza f(x), gdzie x = a + (b - a) * u, a u leży w [0, 1).

        Generatory zwracają liczby z [0, 1), więc wszystkie wersje programu
        (referencja, CPU, MPI, GPU) mogą losować to samo u i dopiero tu
        odwzorowywać je na przedział całkowania.
        """
        x = self.a + (self.b - self.a) * u
        return self.func(x)


def _square(x: np.ndarray) -> np.ndarray:
    return x * x


def _sine(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


# Funkcja 1 (prosta): całka z x^2 na [0, 1] wynosi 1/3
SQUARE = Integrand(
    name="square",
    func=_square,
    a=0.0,
    b=1.0,
    exact=1.0 / 3.0,
    description="f(x) = x^2 na [0, 1]",
)

# Funkcja 2 (trygonometryczna): całka z sin(x) na [0, pi] wynosi 2
SINE = Integrand(
    name="sine",
    func=_sine,
    a=0.0,
    b=math.pi,
    exact=2.0,
    description="f(x) = sin(x) na [0, pi]",
)

# Rejestr używany przez interfejs wiersza poleceń (--func)
INTEGRANDS: Dict[str, Integrand] = {SQUARE.name: SQUARE, SINE.name: SINE}


def get_integrand(name: str) -> Integrand:
    """Zwraca funkcję podcałkową o podanej nazwie."""
    try:
        return INTEGRANDS[name]
    except KeyError:
        available = ", ".join(sorted(INTEGRANDS))
        raise ValueError(f"Unknown integrand '{name}'. Available: {available}") from None