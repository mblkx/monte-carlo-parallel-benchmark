"""Generatory liczb losowych z jednoznacznym mapowaniem (seed, indeks partii).

Zasada: strumień partii k zależy wyłącznie od pary (seed, k). Dzięki temu
referencja, wersja wielordzeniowa i MPI losują dokładnie te same próbki dla tej
samej partii, niezależnie od liczby wykonawców i kolejności ich pracy.

Nie używamy SeedSequence.spawn(), bo zależy ono od licznika wewnętrznego
(kolejności wywołań), którego procesy nie współdzielą. Zamiast tego podajemy
spawn_key jawnie, więc strumień jest czystą funkcją argumentów.
"""

import numpy as np

# Domyślny rozmiar partii: stała w całym projekcie, bo od niej zależy mapowanie próbek na strumienie (zmiana rozmiaru partii zmienia próbki).
DEFAULT_BATCH_SIZE = 1_000_000


def batch_generator(seed: int, batch_index: int) -> np.random.Generator:
    """Zwraca generator Philox dla partii o numerze batch_index.

    Ten sam (seed, batch_index) zawsze daje ten sam strumień. Różne indeksy
    dają statystycznie niezależne strumienie.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(batch_index,))
    return np.random.Generator(np.random.Philox(ss))


def batch_uniform(seed: int, batch_index: int, size: int) -> np.ndarray:
    """Zwraca size liczb z rozkładu jednostajnego [0, 1) dla danej partii."""
    if size < 0:
        raise ValueError("size must be non-negative")
    return batch_generator(seed, batch_index).random(size, dtype=np.float64)


def split_into_batches(n_samples: int, batch_size: int = DEFAULT_BATCH_SIZE):
    """Dzieli n_samples na partie o rozmiarze batch_size (ostatnia może być krótsza).

    Zwraca listę par (batch_index, rozmiar). Suma rozmiarów równa się n_samples.
    Ta sama funkcja jest używana przez referencję, CPU i MPI, więc podział
    na partie jest wszędzie identyczny.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    full, rest = divmod(n_samples, batch_size)
    batches = [(i, batch_size) for i in range(full)]
    if rest:
        batches.append((full, rest))
    return batches