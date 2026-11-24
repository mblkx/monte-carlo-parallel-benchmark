"""Warianty jądra redukcji do oceny kosztu generatora i funkcji podcałkowej.

Cztery warianty jądra 1 (generowanie, funkcja, Welford w wątku, redukcja w bloku):
- full:     generator + funkcja,
- no_func:  generator, funkcja zastąpiona tożsamością,
- no_rng:   bez generatora (u = (j + 0.5) / n z indeksu próbki), funkcja zostaje,
- baseline: bez generatora i bez funkcji (pętla, odwzorowanie, Welford, redukcja).

Warianty bez generatora nie dają próbek losowych, więc służą wyłącznie do
pomiaru czasu. Różnice czasów rozdzielają koszt generatora od kosztu funkcji.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

from benchmark import MIN_REPEATS
from functions import Integrand
from gpu_kernels import (
    DEFAULT_MAX_BLOCKS, DEFAULT_THREADS_PER_BLOCK, DEVICE_FUNCS, _merge_dev,
    _require_cuda, _validate_reduction, launch_config,
)
from stats import Stats, merge_all

# Warianty: nazwa -> (czy używa generatora, czy używa funkcji podcałkowej)
VARIANT_SPECS: Dict[str, Tuple[bool, bool]] = {
    "full": (True, True),
    "no_func": (True, False),
    "no_rng": (False, True),
    "baseline": (False, False),
}
VARIANTS = tuple(VARIANT_SPECS)


@cuda.jit(device=True)
def _identity_dev(x):
    return x


def _build_variant_kernel(dev_func, tpb, use_rng):
    """Jądro 1 z opcjonalnym generatorem; reszta jak w produkcyjnej redukcji."""

    @cuda.jit
    def kernel(states, a, width, total, out_n, out_mean, out_m2):
        sh_n = cuda.shared.array(tpb, dtype=float64)
        sh_mean = cuda.shared.array(tpb, dtype=float64)
        sh_m2 = cuda.shared.array(tpb, dtype=float64)

        t = cuda.threadIdx.x
        tid = cuda.grid(1)
        stride = cuda.gridsize(1)
        inv = 1.0 / total

        # Poziom 1: statystyki wątku liczone na bieżąco (Welford)
        cnt = 0.0
        mean = 0.0
        m2 = 0.0
        for j in range(tid, total, stride):
            if use_rng:
                u = xoroshiro128p_uniform_float64(states, tid)
            else:
                # Deterministyczne u z indeksu próbki: ta sama praca arytmetyczna, bez generatora
                u = (j + 0.5) * inv
            y = dev_func(a + width * u)
            cnt += 1.0
            d = y - mean
            mean += d / cnt
            m2 += d * (y - mean)
        sh_n[t] = cnt
        sh_mean[t] = mean
        sh_m2[t] = m2
        cuda.syncthreads()

        # Poziom 2: drzewiasta redukcja w bloku
        s = tpb // 2
        while s > 0:
            if t < s:
                sh_n[t], sh_mean[t], sh_m2[t] = _merge_dev(
                    sh_n[t], sh_mean[t], sh_m2[t],
                    sh_n[t + s], sh_mean[t + s], sh_m2[t + s])
            cuda.syncthreads()
            s //= 2

        if t == 0:
            out_n[cuda.blockIdx.x] = sh_n[0]
            out_mean[cuda.blockIdx.x] = sh_mean[0]
            out_m2[cuda.blockIdx.x] = sh_m2[0]

    return kernel


# Cache jąder: klucz (nazwa funkcji, rozmiar bloku, wariant)
_VARIANT_KERNELS: Dict[Tuple[str, int, str], object] = {}


def _validate_variant(variant: str) -> None:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {', '.join(VARIANTS)}")


def _get_variant_kernel(integrand: Integrand, tpb: int, variant: str):
    key = (integrand.name, tpb, variant)
    if key not in _VARIANT_KERNELS:
        use_rng, use_func = VARIANT_SPECS[variant]
        dev_func = DEVICE_FUNCS[integrand.name] if use_func else _identity_dev
        _VARIANT_KERNELS[key] = _build_variant_kernel(dev_func, tpb, use_rng)
    return _VARIANT_KERNELS[key]


def variant_partials(
    integrand: Integrand,
    variant: str,
    n: int,
    seed: int = 12345,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> np.ndarray:
    """Uruchamia wariant jądra i zwraca statystyki bloków jako tablicę (blocks, 3)."""
    _validate_variant(variant)
    _validate_reduction(integrand, n, seed, threads_per_block, max_blocks, "device")
    _require_cuda()

    blocks, tpb = launch_config(n, threads_per_block, max_blocks)
    states = create_xoroshiro128p_states(blocks * tpb, seed=seed)
    d_n = cuda.device_array(blocks, dtype=np.float64)
    d_mean = cuda.device_array(blocks, dtype=np.float64)
    d_m2 = cuda.device_array(blocks, dtype=np.float64)
    kernel = _get_variant_kernel(integrand, tpb, variant)
    kernel[blocks, tpb](states, integrand.a, integrand.b - integrand.a, n,
                        d_n, d_mean, d_m2)
    cuda.synchronize()
    return np.column_stack([d_n.copy_to_host(), d_mean.copy_to_host(), d_m2.copy_to_host()])


def merge_partials(parts: np.ndarray) -> Stats:
    """Scala statystyki bloków (tablica (blocks, 3)) w kolejności bloków."""
    return merge_all(Stats(int(round(n)), m, s) for n, m, s in parts)


@dataclass(frozen=True)
class VariantTimes:
    """Czasy jądra 1 dla czterech wariantów w jednym powtórzeniu [s]."""

    full: float
    no_func: float
    no_rng: float
    baseline: float


def time_variants(
    integrand: Integrand,
    n: int,
    seed: int = 12345,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    repeats: int = MIN_REPEATS,
    warmup: int = 1,
) -> List[VariantTimes]:
    """Mierzy czas samego jądra 1 dla wszystkich wariantów.

    Stany generatora i tablice wyjściowe powstają raz, poza pomiarem. Przed
    każdym odczytem zegara wywoływane jest cuda.synchronize(). Kolejność
    wariantów jest obracana w kolejnych powtórzeniach, żeby dryf zegarów
    i temperatury nie faworyzował jednego z nich.
    """
    _validate_reduction(integrand, n, seed, threads_per_block, max_blocks, "device")
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be at least {MIN_REPEATS}")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    _require_cuda()

    blocks, tpb = launch_config(n, threads_per_block, max_blocks)
    states = create_xoroshiro128p_states(blocks * tpb, seed=seed)
    d_n = cuda.device_array(blocks, dtype=np.float64)
    d_mean = cuda.device_array(blocks, dtype=np.float64)
    d_m2 = cuda.device_array(blocks, dtype=np.float64)
    kernels = {v: _get_variant_kernel(integrand, tpb, v) for v in VARIANTS}
    width = integrand.b - integrand.a

    def launch(variant: str) -> None:
        kernels[variant][blocks, tpb](states, integrand.a, width, n, d_n, d_mean, d_m2)

    # Rozgrzewka: kompilacja i rozruch karty
    for _ in range(warmup):
        for v in VARIANTS:
            launch(v)
    cuda.synchronize()

    results: List[VariantTimes] = []
    for rep in range(repeats):
        shift = rep % len(VARIANTS)
        times: Dict[str, float] = {}
        for v in VARIANTS[shift:] + VARIANTS[:shift]:
            cuda.synchronize()
            t0 = perf_counter()
            launch(v)
            cuda.synchronize()
            times[v] = perf_counter() - t0
        results.append(VariantTimes(**times))
    return results


@dataclass(frozen=True)
class CostDecomposition:
    """Rozkład czasu jądra 1 na składniki [s]."""

    t_full: float
    rng: float             # koszt generatora w obecności funkcji: full - no_rng
    func: float            # koszt funkcji w obecności generatora: full - no_func
    rng_alone: float       # koszt generatora bez funkcji: no_func - baseline
    func_alone: float      # koszt funkcji bez generatora: no_rng - baseline
    base: float            # reszta (pętla, odwzorowanie, Welford, redukcja): baseline
    interaction: float     # full - no_func - no_rng + baseline (0 = koszty się sumują)

    @property
    def rng_share(self) -> float:
        """Udział generatora w czasie pełnego jądra."""
        return self.rng / self.t_full if self.t_full > 0 else float("nan")

    @property
    def func_share(self) -> float:
        """Udział funkcji w czasie pełnego jądra."""
        return self.func / self.t_full if self.t_full > 0 else float("nan")

    @property
    def base_share(self) -> float:
        """Udział reszty w czasie pełnego jądra."""
        return self.base / self.t_full if self.t_full > 0 else float("nan")


def decompose(full: float, no_func: float, no_rng: float, baseline: float) -> CostDecomposition:
    """Rozkłada czas pełnego jądra na koszt generatora, funkcji i reszty."""
    return CostDecomposition(
        t_full=full,
        rng=full - no_rng,
        func=full - no_func,
        rng_alone=no_func - baseline,
        func_alone=no_rng - baseline,
        base=baseline,
        interaction=full - no_func - no_rng + baseline,
    )


# Zapis wyników

COST_COLUMNS: List[str] = [
    "func", "n", "seed", "threads_per_block", "blocks", "repeat",
    "t_full", "t_no_func", "t_no_rng", "t_baseline",
]


def write_cost_rows(
    path: Path, func: str, n: int, seed: int, threads_per_block: int,
    blocks: int, times: Iterable[VariantTimes],
) -> None:
    """Dopisuje czasy wariantów do CSV (jeden wiersz na powtórzenie). Nagłówek raz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COST_COLUMNS)
        if write_header:
            writer.writeheader()
        for rep, t in enumerate(times):
            writer.writerow({
                "func": func, "n": n, "seed": seed,
                "threads_per_block": threads_per_block, "blocks": blocks,
                "repeat": rep, "t_full": t.full, "t_no_func": t.no_func,
                "t_no_rng": t.no_rng, "t_baseline": t.baseline,
            })