"""Wersja GPU całkowania Monte Carlo: wynik z czasami składowymi.

Obliczenia wykonuje gpu_kernels.gpu_stats_timed (generowanie próbek, funkcja,
redukcja hierarchiczna na karcie). Ten moduł dodaje estymatę, błędy, przedział
ufności, przepustowość i zapis rozbicia czasu do CSV.

Generator GPU różni się od CPU, więc wynik jest zgodny z referencją
statystycznie, a nie bitowo, i zależy od (ziarno, konfiguracja siatki).

Przykład uruchomienia:
    python src/mc_gpu.py --func sine --n 1e8 --seed 42
"""

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, List, Optional

from functions import INTEGRANDS, Integrand, get_integrand
from gpu_kernels import (
    DEFAULT_MAX_BLOCKS, DEFAULT_THREADS_PER_BLOCK, FINALS, cuda_available,
    gpu_stats_timed, launch_config,
)
from mc_ref import _parse_count
from stats import abs_error, integral_from_stats, rel_error


@dataclass(frozen=True)
class GpuResult:
    """Wynik jednego uruchomienia wersji GPU wraz z czasami składowymi."""

    func: str                # nazwa funkcji podcałkowej
    n: int                   # liczba próbek
    seed: int                # ziarno generatora
    threads_per_block: int   # wątki na blok
    blocks: int              # liczba bloków
    final: str               # scalanie bloków: device lub host
    estimate: float          # estymata całki
    se: float                # błąd standardowy
    ci_low: float            # dolna granica przedziału ufności 95%
    ci_high: float           # górna granica przedziału ufności 95%
    exact: float             # wartość dokładna całki
    abs_err: float           # błąd bezwzględny
    rel_err: float           # błąd względny
    t_alloc: float           # alokacja w pamięci karty [s]
    t_states: float          # stany generatora [s]
    t_stage1: float          # jądro 1 [s]
    t_stage2: float          # jądro 2 [s] (0 przy final="host")
    t_transfer: float        # kopia karta -> host [s]
    t_host_merge: float      # scalenie na CPU [s]
    t_core: float            # rdzeń: stany + jądro 1 + jądro 2 [s]
    t_total: float           # pełny czas przetwarzania [s]

    @property
    def contains_exact(self) -> bool:
        """Czy przedział ufności zawiera wartość dokładną."""
        return self.ci_low <= self.exact <= self.ci_high

    @property
    def samples_per_second(self) -> float:
        """Przepustowość pełnego przetwarzania [próbki/s]."""
        return self.n / self.t_total if self.t_total > 0 else float("nan")

    @property
    def core_samples_per_second(self) -> float:
        """Przepustowość samego rdzenia obliczeń [próbki/s]."""
        return self.n / self.t_core if self.t_core > 0 else float("nan")


def run_gpu(
    integrand: Integrand,
    n_samples: int,
    seed: int,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    final: str = "device",
) -> GpuResult:
    """Liczy całkę metodą Monte Carlo na GPU i mierzy czasy składowe."""
    stats, t = gpu_stats_timed(integrand, n_samples, seed, threads_per_block,
                               max_blocks, final)
    blocks, tpb = launch_config(n_samples, threads_per_block, max_blocks)
    res = integral_from_stats(stats, integrand.a, integrand.b)
    return GpuResult(
        func=integrand.name,
        n=stats.n,
        seed=seed,
        threads_per_block=tpb,
        blocks=blocks,
        final=final,
        estimate=res.estimate,
        se=res.se,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        exact=integrand.exact,
        abs_err=abs_error(res.estimate, integrand.exact),
        rel_err=rel_error(res.estimate, integrand.exact),
        t_alloc=t.t_alloc,
        t_states=t.t_states,
        t_stage1=t.t_stage1,
        t_stage2=t.t_stage2,
        t_transfer=t.t_transfer,
        t_host_merge=t.t_host_merge,
        t_core=t.t_states + t.t_stage1 + t.t_stage2,
        t_total=t.t_total,
    )


def warm_up_gpu(
    integrand: Integrand,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    final: str = "device",
) -> None:
    """Kompiluje jądra dla danej konfiguracji przed pomiarem (rozgrzewka)."""
    run_gpu(integrand, 10_000, 0, threads_per_block, max_blocks, final)


def gpu_core_time(result: GpuResult) -> float:
    """Czas rdzenia dla benchmark(): stany generatora i oba jądra."""
    return result.t_core


# Zapis rozbicia czasu

BREAKDOWN_COLUMNS: List[str] = [
    "func", "n", "seed", "threads_per_block", "blocks", "final", "repeat",
    "t_alloc", "t_states", "t_stage1", "t_stage2", "t_transfer",
    "t_host_merge", "t_core", "t_total",
]


def write_breakdown(path: Path, results: Iterable[GpuResult]) -> None:
    """Dopisuje do CSV składowe czasu każdego pomiaru. Nagłówek jest zapisywany raz.

    Numer powtórzenia liczony jest osobno dla każdego ziarna, w kolejności
    podania wyników (tak samo jak w benchmark()).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    counters = {}
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BREAKDOWN_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            rep = counters.get(r.seed, 0)
            counters[r.seed] = rep + 1
            row = {c: getattr(r, c, None) for c in BREAKDOWN_COLUMNS if c != "repeat"}
            row["repeat"] = rep
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU Monte Carlo integration.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="square",
                        help="integrand name")
    parser.add_argument("--n", type=_parse_count, default=100_000_000,
                        help="number of samples (e.g. 1e8)")
    parser.add_argument("--seed", type=int, default=12345, help="generator seed")
    parser.add_argument("--tpb", type=int, default=DEFAULT_THREADS_PER_BLOCK,
                        help="threads per block (power of two)")
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--final", choices=FINALS, default="device",
                        help="where block results are merged")
    return parser


def format_result(r: GpuResult) -> str:
    """Czytelny wydruk wyniku dla terminala."""
    inside = "yes" if r.contains_exact else "no"
    return "\n".join([
        f"function     : {r.func}",
        f"samples      : {r.n} (grid {r.blocks} x {r.threads_per_block}, seed {r.seed}, final {r.final})",
        f"estimate     : {r.estimate:.10f}",
        f"exact        : {r.exact:.10f}",
        f"abs error    : {r.abs_err:.3e}",
        f"rel error    : {r.rel_err:.3e}",
        f"std error    : {r.se:.3e}",
        f"95% CI       : [{r.ci_low:.10f}, {r.ci_high:.10f}] (contains exact: {inside})",
        f"time alloc   : {r.t_alloc:.6f} s",
        f"time states  : {r.t_states:.6f} s",
        f"time stage 1 : {r.t_stage1:.6f} s",
        f"time stage 2 : {r.t_stage2:.6f} s",
        f"time transfer: {r.t_transfer:.6f} s",
        f"time merge   : {r.t_host_merge:.6f} s",
        f"time core    : {r.t_core:.6f} s ({r.core_samples_per_second:.3e} samples/s)",
        f"time total   : {r.t_total:.6f} s ({r.samples_per_second:.3e} samples/s)",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not cuda_available():
        print("CUDA device is not available")
        return 1
    integrand = get_integrand(args.func)
    warm_up_gpu(integrand, args.tpb, args.max_blocks, args.final)
    result = run_gpu(integrand, args.n, args.seed, args.tpb, args.max_blocks, args.final)
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())