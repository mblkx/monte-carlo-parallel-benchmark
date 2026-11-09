"""Jądra Numba CUDA: generowanie próbek i obliczanie funkcji podcałkowej.

Dwa jądra:
- sample: każdy wątek generuje liczby u z [0, 1) (xoroshiro128p z Numby),
  odwzorowuje je na x = a + (b - a) * u i zapisuje f(x),
- apply: bierze gotową tablicę u z CPU i liczy f(x); służy do sprawdzenia
  wzorów funkcji na GPU względem numpy na identycznych danych.

Generator GPU różni się od generatora CPU (Philox), więc wynik nie jest bitowo
zgodny z referencją. Strumień zależy od (seed, liczba wątków), bo stan
generatora jest przypisany do wątku, a nie do indeksu próbki.
"""

import math
from typing import Dict, Tuple

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

from functions import INTEGRANDS, Integrand

# Domyślna konfiguracja siatki
DEFAULT_THREADS_PER_BLOCK = 256
DEFAULT_MAX_BLOCKS = 2048


def cuda_available() -> bool:
    """Czy Numba widzi działającą kartę NVIDIA."""
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


# Funkcje podcałkowe w wersji skalarnej dla urządzenia (odpowiedniki z functions.py)

@cuda.jit(device=True)
def _square_dev(x):
    return x * x


@cuda.jit(device=True)
def _sine_dev(x):
    return math.sin(x)


# Rejestr funkcji urządzenia, klucze zgodne z INTEGRANDS
DEVICE_FUNCS: Dict[str, object] = {"square": _square_dev, "sine": _sine_dev}


def _build_sample_kernel(dev_func):
    """Buduje jądro generujące próbki i liczące dev_func dla zadanej funkcji."""

    @cuda.jit
    def kernel(states, out, a, width):
        tid = cuda.grid(1)
        stride = cuda.gridsize(1)
        # Pętla grid-stride: wątek tid obsługuje próbki tid, tid + stride, ...
        for j in range(tid, out.size, stride):
            u = xoroshiro128p_uniform_float64(states, tid)
            out[j] = dev_func(a + width * u)

    return kernel


def _build_apply_kernel(dev_func):
    """Buduje jądro liczące dev_func dla gotowej tablicy u z przedziału [0, 1)."""

    @cuda.jit
    def kernel(u, out, a, width):
        for j in range(cuda.grid(1), u.size, cuda.gridsize(1)):
            out[j] = dev_func(a + width * u[j])

    return kernel


# Jądra kompilują się przy pierwszym użyciu, potem są trzymane w pamięci podręcznej
_SAMPLE_KERNELS: Dict[str, object] = {}
_APPLY_KERNELS: Dict[str, object] = {}


def _get_kernel(cache: Dict[str, object], builder, name: str):
    if name not in cache:
        cache[name] = builder(DEVICE_FUNCS[name])
    return cache[name]


def launch_config(
    n: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> Tuple[int, int]:
    """Zwraca (liczba bloków, wątki na blok): tyle bloków, by pokryć n, ale nie więcej niż max_blocks."""
    if n <= 0:
        raise ValueError("n must be positive")
    blocks = min(-(-n // threads_per_block), max_blocks)
    return blocks, threads_per_block


def _validate(integrand: Integrand, threads_per_block: int, max_blocks: int) -> None:
    """Sprawdza argumenty wspólne dla obu jąder (bez dostępu do karty)."""
    if INTEGRANDS.get(integrand.name) is not integrand or integrand.name not in DEVICE_FUNCS:
        raise ValueError(
            f"Integrand '{integrand.name}' has no GPU version; "
            f"available: {', '.join(sorted(DEVICE_FUNCS))}"
        )
    if not 32 <= threads_per_block <= 1024 or threads_per_block % 32 != 0:
        raise ValueError("threads_per_block must be a multiple of 32 in [32, 1024]")
    if max_blocks < 1:
        raise ValueError("max_blocks must be at least 1")


def _require_cuda() -> None:
    if not cuda_available():
        raise RuntimeError("CUDA device is not available")


def sample_values_device(
    integrand: Integrand,
    n: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
):
    """Generuje n próbek na GPU i zwraca tablicę f(x_i) w pamięci karty.

    Wywołanie jest asynchroniczne: przed pomiarem czasu trzeba wywołać
    cuda.synchronize().
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    _validate(integrand, threads_per_block, max_blocks)
    _require_cuda()

    blocks, tpb = launch_config(n, threads_per_block, max_blocks)
    states = create_xoroshiro128p_states(blocks * tpb, seed=seed)
    out = cuda.device_array(n, dtype=np.float64)
    kernel = _get_kernel(_SAMPLE_KERNELS, _build_sample_kernel, integrand.name)
    kernel[blocks, tpb](states, out, integrand.a, integrand.b - integrand.a)
    return out


def sample_values(
    integrand: Integrand,
    n: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> np.ndarray:
    """Jak sample_values_device, ale kopiuje wynik do pamięci hosta."""
    out = sample_values_device(integrand, n, seed, threads_per_block, max_blocks)
    cuda.synchronize()
    return out.copy_to_host()


def evaluate_on_gpu(
    integrand: Integrand,
    u: np.ndarray,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> np.ndarray:
    """Liczy f(a + (b - a) * u) na GPU dla tablicy u z CPU (odpowiednik Integrand.from_unit)."""
    u = np.ascontiguousarray(u, dtype=np.float64)
    if u.ndim != 1 or u.size == 0:
        raise ValueError("u must be a non-empty one-dimensional array")
    _validate(integrand, threads_per_block, max_blocks)
    _require_cuda()

    blocks, tpb = launch_config(u.size, threads_per_block, max_blocks)
    d_u = cuda.to_device(u)
    out = cuda.device_array(u.size, dtype=np.float64)
    kernel = _get_kernel(_APPLY_KERNELS, _build_apply_kernel, integrand.name)
    kernel[blocks, tpb](d_u, out, integrand.a, integrand.b - integrand.a)
    cuda.synchronize()
    return out.copy_to_host()