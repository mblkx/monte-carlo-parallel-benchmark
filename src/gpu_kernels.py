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
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from stats import IntegralResult, Stats, integral_from_stats, merge_all

from dataclasses import dataclass
from time import perf_counter

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

# Redukcja statystyk na GPU

# Dostępne sposoby scalania trójek między blokami
FINALS = ("device", "host")


@cuda.jit(device=True)
def _merge_dev(n1, mean1, m2_1, n2, mean2, m2_2):
    """Scala dwie trójki (n, mean, m2) wzorem Chana (odpowiednik Stats.merge).

    Pusta strona (n = 0) jest obsługiwana bez gałęzi: człony z jej liczebnością
    zerują się. Gałąź chroni tylko przed dzieleniem przez zero.
    """
    n = n1 + n2
    if n == 0.0:
        return 0.0, 0.0, 0.0
    delta = mean2 - mean1
    mean = mean1 + delta * n2 / n
    m2 = m2_1 + m2_2 + delta * delta * n1 * n2 / n
    return n, mean, m2


def _build_stats_kernel(dev_func, tpb):
    """Jądro 1: generowanie, funkcja, Welford na wątek i redukcja w bloku.

    Wynikiem są trzy tablice długości liczby bloków: n, mean, m2 każdego bloku.
    """

    @cuda.jit
    def kernel(states, a, width, total, out_n, out_mean, out_m2):
        sh_n = cuda.shared.array(tpb, dtype=float64)
        sh_mean = cuda.shared.array(tpb, dtype=float64)
        sh_m2 = cuda.shared.array(tpb, dtype=float64)

        t = cuda.threadIdx.x
        tid = cuda.grid(1)
        stride = cuda.gridsize(1)

        # Poziom 1: statystyki wątku liczone na bieżąco (Welford)
        cnt = 0.0
        mean = 0.0
        m2 = 0.0
        for j in range(tid, total, stride):
            u = xoroshiro128p_uniform_float64(states, tid)
            y = dev_func(a + width * u)
            cnt += 1.0
            d = y - mean
            mean += d / cnt
            m2 += d * (y - mean)
        sh_n[t] = cnt
        sh_mean[t] = mean
        sh_m2[t] = m2
        cuda.syncthreads()

        # Poziom 2: drzewiasta redukcja w bloku (stała kolejność, bez atomówek)
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


def _build_final_kernel(tpb):
    """Jądro 2 (jeden blok): scala trójki bloków w jedną trójkę."""

    @cuda.jit
    def kernel(in_n, in_mean, in_m2, nblocks, out):
        sh_n = cuda.shared.array(tpb, dtype=float64)
        sh_mean = cuda.shared.array(tpb, dtype=float64)
        sh_m2 = cuda.shared.array(tpb, dtype=float64)

        t = cuda.threadIdx.x
        cnt = 0.0
        mean = 0.0
        m2 = 0.0
        for i in range(t, nblocks, tpb):
            cnt, mean, m2 = _merge_dev(cnt, mean, m2, in_n[i], in_mean[i], in_m2[i])
        sh_n[t] = cnt
        sh_mean[t] = mean
        sh_m2[t] = m2
        cuda.syncthreads()

        s = tpb // 2
        while s > 0:
            if t < s:
                sh_n[t], sh_mean[t], sh_m2[t] = _merge_dev(
                    sh_n[t], sh_mean[t], sh_m2[t],
                    sh_n[t + s], sh_mean[t + s], sh_m2[t + s])
            cuda.syncthreads()
            s //= 2

        if t == 0:
            out[0] = sh_n[0]
            out[1] = sh_mean[0]
            out[2] = sh_m2[0]

    return kernel


# Cache jąder: klucze (nazwa funkcji, rozmiar bloku) i rozmiar bloku
_STATS_KERNELS: Dict[Tuple[str, int], object] = {}
_FINAL_KERNELS: Dict[int, object] = {}


def _validate_reduction(
    integrand: Integrand, n: int, seed: int,
    threads_per_block: int, max_blocks: int, final: str,
) -> None:
    """Walidacja argumentów redukcji (bez dostępu do karty)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    _validate(integrand, threads_per_block, max_blocks)
    if threads_per_block & (threads_per_block - 1) != 0:
        raise ValueError("threads_per_block must be a power of two for tree reduction")
    if final not in FINALS:
        raise ValueError(f"Unknown final '{final}'. Available: {', '.join(FINALS)}")


def _run_stage_one(integrand, n, seed, threads_per_block, max_blocks):
    """Uruchamia jądro 1 i zwraca (blocks, tpb, d_n, d_mean, d_m2) w pamięci karty."""
    blocks, tpb = launch_config(n, threads_per_block, max_blocks)
    states = create_xoroshiro128p_states(blocks * tpb, seed=seed)
    d_n = cuda.device_array(blocks, dtype=np.float64)
    d_mean = cuda.device_array(blocks, dtype=np.float64)
    d_m2 = cuda.device_array(blocks, dtype=np.float64)
    key = (integrand.name, tpb)
    if key not in _STATS_KERNELS:
        _STATS_KERNELS[key] = _build_stats_kernel(DEVICE_FUNCS[integrand.name], tpb)
    _STATS_KERNELS[key][blocks, tpb](
        states, integrand.a, integrand.b - integrand.a, n, d_n, d_mean, d_m2)
    return blocks, tpb, d_n, d_mean, d_m2


def block_partials(
    integrand: Integrand,
    n: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> np.ndarray:
    """Zwraca statystyki każdego bloku jako tablicę (blocks, 3): n, mean, m2.

    Służy do testów i diagnostyki (rozkład pracy między blokami).
    """
    _validate_reduction(integrand, n, seed, threads_per_block, max_blocks, "device")
    _require_cuda()
    _, _, d_n, d_mean, d_m2 = _run_stage_one(integrand, n, seed, threads_per_block, max_blocks)
    cuda.synchronize()
    return np.column_stack([d_n.copy_to_host(), d_mean.copy_to_host(), d_m2.copy_to_host()])


def gpu_stats_timed(
    integrand: Integrand,
    n: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    final: str = "device",
) -> Tuple[Stats, GpuTimings]:
    """Liczy statystyki f(x_i) na GPU i mierzy czas każdego etapu.

    Przed każdym odczytem zegara wywoływane jest cuda.synchronize(), bo jądra
    działają asynchronicznie. Kompilacja jąder następuje przy pierwszym
    uruchomieniu danej konfiguracji, więc przed pomiarem trzeba wykonać
    rozgrzewkę (patrz warm_up_gpu w mc_gpu).

    final:
      device - trójki bloków scala drugie jądro, na CPU wracają 3 liczby
      host   - na CPU wracają trójki wszystkich bloków i są scalane w kolejności bloków
    """
    _validate_reduction(integrand, n, seed, threads_per_block, max_blocks, final)
    _require_cuda()

    blocks, tpb = launch_config(n, threads_per_block, max_blocks)

    # Budowa obiektów jąder jest tania; kompilacja następuje przy pierwszym uruchomieniu
    key = (integrand.name, tpb)
    if key not in _STATS_KERNELS:
        _STATS_KERNELS[key] = _build_stats_kernel(DEVICE_FUNCS[integrand.name], tpb)
    if final == "device" and tpb not in _FINAL_KERNELS:
        _FINAL_KERNELS[tpb] = _build_final_kernel(tpb)
    stage1 = _STATS_KERNELS[key]

    # Zaczynamy od pustej kolejki karty
    cuda.synchronize()
    t0 = perf_counter()
    d_n = cuda.device_array(blocks, dtype=np.float64)
    d_mean = cuda.device_array(blocks, dtype=np.float64)
    d_m2 = cuda.device_array(blocks, dtype=np.float64)
    d_out = cuda.device_array(3, dtype=np.float64) if final == "device" else None
    cuda.synchronize()
    t1 = perf_counter()

    states = create_xoroshiro128p_states(blocks * tpb, seed=seed)
    cuda.synchronize()
    t2 = perf_counter()

    stage1[blocks, tpb](states, integrand.a, integrand.b - integrand.a, n,
                        d_n, d_mean, d_m2)
    cuda.synchronize()
    t3 = perf_counter()

    if final == "device":
        _FINAL_KERNELS[tpb][1, tpb](d_n, d_mean, d_m2, blocks, d_out)
        cuda.synchronize()
        t4 = perf_counter()
        out = d_out.copy_to_host()  # kopia jest synchroniczna
        t5 = perf_counter()
        stats = Stats(int(round(out[0])), float(out[1]), float(out[2]))
        t6 = perf_counter()
    else:
        t4 = t3
        host_n = d_n.copy_to_host()
        host_mean = d_mean.copy_to_host()
        host_m2 = d_m2.copy_to_host()
        t5 = perf_counter()
        stats = merge_all(Stats(int(round(cn)), cm, cs)
                          for cn, cm, cs in zip(host_n, host_mean, host_m2))
        t6 = perf_counter()

    timings = GpuTimings(
        t_alloc=t1 - t0,
        t_states=t2 - t1,
        t_stage1=t3 - t2,
        t_stage2=t4 - t3,
        t_transfer=t5 - t4,
        t_host_merge=t6 - t5,
        t_total=t6 - t0,
    )
    return stats, timings


def gpu_stats(
    integrand: Integrand,
    n: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    final: str = "device",
) -> Stats:
    """Liczy statystyki f(x_i) dla n próbek wygenerowanych na GPU.

    Wynik zależy od (seed, threads_per_block, max_blocks, n), bo stan generatora
    jest przypisany do wątku. Szczegóły parametru final: patrz gpu_stats_timed.
    """
    return gpu_stats_timed(integrand, n, seed, threads_per_block, max_blocks, final)[0]


def gpu_integral(
    integrand: Integrand,
    n: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    final: str = "device",
) -> IntegralResult:
    """Estymata całki, błąd standardowy i przedział ufności z redukcji na GPU."""
    s = gpu_stats(integrand, n, seed, threads_per_block, max_blocks, final)
    return integral_from_stats(s, integrand.a, integrand.b)


# Redukcja z pomiarem czasu

@dataclass(frozen=True)
class GpuTimings:
    """Składowe czasu jednego uruchomienia redukcji na GPU [s]."""

    t_alloc: float       # alokacja tablic wyjściowych w pamięci karty
    t_states: float      # inicjalizacja stanów generatora
    t_stage1: float      # jądro 1: generowanie, funkcja, redukcja w bloku
    t_stage2: float      # jądro 2: scalenie bloków (0 przy final="host")
    t_transfer: float    # kopia karta -> host
    t_host_merge: float  # scalenie wyników na CPU
    t_total: float       # całość, od alokacji do gotowych statystyk