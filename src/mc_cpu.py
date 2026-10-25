"""Wersja wielordzeniowa całkowania Monte Carlo (odpowiednik OpenMP).

Partie próbek są rozdzielane między procesy puli ProcessPoolExecutor. Każda
partia jest liczona funkcją compute_batch z referencji, więc strumienie losowe
zależą wyłącznie od (seed, indeks partii). Wyniki partii są scalane w kolejności
indeksów, dlatego wynik jest bitowo identyczny z referencją i niezależny od
liczby procesów.

Przykład uruchomienia:
    python src/mc_cpu.py --func sine --n 10000000 --seed 42 --workers 4
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from time import perf_counter
from typing import List, Optional

from functions import INTEGRANDS, Integrand, get_integrand
from mc_ref import compute_batch
from rng import DEFAULT_BATCH_SIZE, split_into_batches
from stats import Stats, abs_error, integral_from_stats, rel_error


@dataclass(frozen=True)
class CpuResult:
    """Wynik jednego uruchomienia wersji wielordzeniowej wraz z czasami."""

    func: str          # nazwa funkcji podcałkowej
    n: int             # liczba próbek
    seed: int          # ziarno generatora
    batch_size: int    # rozmiar partii
    workers: int       # liczba procesów p
    estimate: float    # estymata całki
    se: float          # błąd standardowy
    ci_low: float      # dolna granica przedziału ufności 95%
    ci_high: float     # górna granica przedziału ufności 95%
    exact: float       # wartość dokładna całki
    abs_err: float     # błąd bezwzględny
    rel_err: float     # błąd względny
    t_rng: float       # suma czasów pracy procesów: generator [s]
    t_func: float      # suma czasów pracy procesów: funkcja [s]
    t_agg: float       # suma czasów pracy procesów: statystyki [s]
    t_compute: float   # czas zegarowy części równoległej (zlecanie, obliczenia, scalanie) [s]
    t_total: float     # czas zegarowy całości, z tworzeniem puli, gdy pula jest własna [s]

    @property
    def contains_exact(self) -> bool:
        """Czy przedział ufności zawiera wartość dokładną."""
        return self.ci_low <= self.exact <= self.ci_high


def _batch_task(name: str, seed: int, batch_index: int, size: int):
    """Zadanie wykonywane w procesie roboczym.

    Przyjmuje nazwę funkcji (a nie obiekt), bo obiekt Integrand może zawierać
    funkcje nieserializowalne. Musi być zdefiniowane na poziomie modułu.
    """
    return compute_batch(get_integrand(name), seed, batch_index, size)


def _warmup_task(_: int) -> int:
    """Krótkie zadanie wymuszające uruchomienie kolejnych procesów puli."""
    time.sleep(0.05)
    return os.getpid()


def warm_up_pool(executor: ProcessPoolExecutor, workers: int) -> None:
    """Uruchamia wszystkie procesy puli przed pomiarem (rozgrzewka)."""
    list(executor.map(_warmup_task, range(workers)))


def _check_registered(integrand: Integrand) -> None:
    """Wersja CPU obsługuje tylko funkcje z rejestru (przekazujemy nazwę)."""
    if INTEGRANDS.get(integrand.name) is not integrand:
        raise ValueError(
            f"Integrand '{integrand.name}' is not registered in INTEGRANDS; "
            "the multiprocess version passes integrands to workers by name."
        )


def _run_on_pool(
    executor: ProcessPoolExecutor,
    integrand: Integrand,
    batches: List[tuple],
    seed: int,
    chunksize: int,
):
    """Zleca partie puli i scala wyniki w kolejności indeksów partii."""
    indices = [k for k, _ in batches]
    sizes = [s for _, s in batches]
    results = executor.map(
        _batch_task, repeat(integrand.name), repeat(seed), indices, sizes,
        chunksize=chunksize,
    )
    total = Stats()
    t_rng = t_func = t_agg = 0.0
    for stats, d_rng, d_func, d_agg in results:
        total = total.merge(stats)
        t_rng += d_rng
        t_func += d_func
        t_agg += d_agg
    return total, t_rng, t_func, t_agg


def run_cpu(
    integrand: Integrand,
    n_samples: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: Optional[int] = None,
    executor: Optional[ProcessPoolExecutor] = None,
    chunksize: int = 1,
) -> CpuResult:
    """Liczy całkę metodą Monte Carlo na wielu procesach.

    Bez executor tworzy własną pulę (jej koszt jest wliczony do t_total).
    Z podanym executor używa go ponownie; wtedy trzeba podać workers,
    które trafia do wyniku jako liczba procesów p.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")
    if executor is not None and workers is None:
        raise ValueError("workers must be given together with an executor")
    if executor is None:
        workers = workers if workers is not None else (os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    _check_registered(integrand)

    batches = split_into_batches(n_samples, batch_size)

    t_start = perf_counter()
    if executor is None:
        with ProcessPoolExecutor(max_workers=workers) as own:
            t_compute_start = perf_counter()
            total, t_rng, t_func, t_agg = _run_on_pool(
                own, integrand, batches, seed, chunksize)
            t_compute = perf_counter() - t_compute_start
    else:
        t_compute_start = perf_counter()
        total, t_rng, t_func, t_agg = _run_on_pool(
            executor, integrand, batches, seed, chunksize)
        t_compute = perf_counter() - t_compute_start
    t_total = perf_counter() - t_start

    res = integral_from_stats(total, integrand.a, integrand.b)
    return CpuResult(
        func=integrand.name,
        n=total.n,
        seed=seed,
        batch_size=batch_size,
        workers=workers,
        estimate=res.estimate,
        se=res.se,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        exact=integrand.exact,
        abs_err=abs_error(res.estimate, integrand.exact),
        rel_err=rel_error(res.estimate, integrand.exact),
        t_rng=t_rng,
        t_func=t_func,
        t_agg=t_agg,
        t_compute=t_compute,
        t_total=t_total,
    )


def cpu_core_time(result: CpuResult) -> float:
    """Czas rdzenia dla benchmark(): czas zegarowy części równoległej."""
    return result.t_compute


def _parse_count(text: str) -> int:
    """Pozwala podać liczbę np. jako 10000000 lub 1e7."""
    value = int(float(text))
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multiprocess Monte Carlo integration (OpenMP-like version)."
    )
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="square",
                        help="integrand name")
    parser.add_argument("--n", type=_parse_count, default=10_000_000,
                        help="number of samples (e.g. 1e7)")
    parser.add_argument("--seed", type=int, default=12345, help="generator seed")
    parser.add_argument("--batch", type=_parse_count, default=DEFAULT_BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--workers", type=_parse_count, default=os.cpu_count() or 1,
                        help="number of worker processes")
    parser.add_argument("--chunksize", type=_parse_count, default=1,
                        help="batches sent to a worker in one task")
    return parser


def format_result(r: CpuResult) -> str:
    """Czytelny wydruk wyniku dla terminala."""
    inside = "yes" if r.contains_exact else "no"
    return "\n".join([
        f"function     : {r.func}",
        f"samples      : {r.n} (batch {r.batch_size}, seed {r.seed}, workers {r.workers})",
        f"estimate     : {r.estimate:.10f}",
        f"exact        : {r.exact:.10f}",
        f"abs error    : {r.abs_err:.3e}",
        f"rel error    : {r.rel_err:.3e}",
        f"std error    : {r.se:.3e}",
        f"95% CI       : [{r.ci_low:.10f}, {r.ci_high:.10f}] (contains exact: {inside})",
        f"busy rng     : {r.t_rng:.4f} s (sum over workers)",
        f"busy func    : {r.t_func:.4f} s (sum over workers)",
        f"busy agg     : {r.t_agg:.4f} s (sum over workers)",
        f"time compute : {r.t_compute:.4f} s",
        f"time total   : {r.t_total:.4f} s",
        f"samples/s    : {r.n / r.t_total:.3e}" if r.t_total > 0 else "samples/s    : n/a",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_cpu(get_integrand(args.func), args.n, args.seed, args.batch,
                     workers=args.workers, chunksize=args.chunksize)
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())