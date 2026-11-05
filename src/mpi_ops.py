"""Własna operacja MPI scalająca statystyki częściowe (n, mean, m2).

Statystyki są przesyłane jako tablica trzech liczb float64. Operacja stosuje
wzór Chana (Stats.merge), więc nie uśrednia średnich bez uwzględnienia
liczebności. Jest niekomutatywna, dzięki czemu MPI łączy wyniki w kolejności
rangów, a rezultat dla danej liczby procesów jest powtarzalny.
"""

import numpy as np
from mpi4py import MPI

from stats import Stats

# Liczba liczb opisujących jedne statystyki: n, mean, m2
STATS_WIDTH = 3


def stats_to_array(s: Stats) -> np.ndarray:
    """Zamienia Stats na tablicę [n, mean, m2] typu float64."""
    return np.array([float(s.n), s.mean, s.m2], dtype=np.float64)


def array_to_stats(a: np.ndarray) -> Stats:
    """Zamienia tablicę [n, mean, m2] na Stats."""
    return Stats(int(round(float(a[0]))), float(a[1]), float(a[2]))


def merge_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scala dwie tablice statystyk wzorem Chana (czysta funkcja, bez MPI)."""
    merged = array_to_stats(a).merge(array_to_stats(b))
    return stats_to_array(merged)


def _reduce_function(inbuf, inoutbuf, datatype) -> None:
    """Funkcja wywoływana przez MPI: inout = in (+) inout.

    Obsługuje wiele trójek naraz, gdyby implementacja MPI podzieliła dane.
    """
    a = np.frombuffer(inbuf, dtype=np.float64).reshape(-1, STATS_WIDTH)
    b = np.frombuffer(inoutbuf, dtype=np.float64).reshape(-1, STATS_WIDTH)
    for i in range(a.shape[0]):
        b[i, :] = merge_arrays(a[i], b[i])


def create_stats_op() -> "MPI.Op":
    """Tworzy operację MPI scalającą statystyki. Po użyciu trzeba wywołać Free()."""
    return MPI.Op.Create(_reduce_function, commute=False)