"""Wpływ liczby próbek i konfiguracji siatki na czas GPU.

Pomiary wykonuje run_gpu (z synchronizacją karty). Konfiguracje są mierzone
przeplatane: powtórzenie to najbardziej zewnętrzna pętla, a kolejność
konfiguracji jest w nim obracana, żeby dryf temperatury i zegarów nie
faworyzował żadnej z nich.

Wynik zależy od siatki (stan generatora jest przypisany do wątku), więc
estymaty różnych konfiguracji różnią się i są zgodne tylko statystycznie.
"""

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from benchmark import MIN_REPEATS, summarize
from functions import Integrand
from gpu_kernels import _require_cuda, _validate_reduction, launch_config
from mc_gpu import run_gpu, warm_up_gpu

# Klucz konfiguracji: (liczba próbek, wątki na blok, maks. liczba bloków)
ConfigKey = Tuple[int, int, int]

# Dostępne metryki do wyboru najlepszej konfiguracji
METRICS = ("t_total", "t_core")


@dataclass(frozen=True)
class SweepRow:
    """Jeden pomiar konfiguracji (jeden wiersz CSV)."""

    func: str
    n: int
    seed: int
    threads_per_block: int
    max_blocks: int
    blocks: int          # faktyczna liczba bloków po zastosowaniu ograniczenia
    repeat: int
    t_states: float      # stany generatora [s]
    t_stage1: float      # jądro 1 [s]
    t_core: float        # rdzeń: stany + jądra [s]
    t_total: float       # pełny czas przetwarzania [s]
    estimate: float
    abs_err: float


SWEEP_COLUMNS: List[str] = list(SweepRow.__dataclass_fields__)


@dataclass(frozen=True)
class ConfigSummary:
    """Mediany pomiarów jednej konfiguracji."""

    n: int
    threads_per_block: int
    max_blocks: int
    blocks: int
    count: int
    t_states: float
    t_stage1: float
    t_core: float
    t_total: float
    iqr_total: float
    throughput: float        # n / mediana czasu pełnego [próbki/s]
    core_throughput: float   # n / mediana czasu rdzenia [próbki/s]


def _validate_sweep(
    integrand: Integrand, sizes: Sequence[int], tpbs: Sequence[int],
    max_blocks_list: Sequence[int], repeats: int,
) -> None:
    """Walidacja argumentów (bez dostępu do karty)."""
    if not sizes or not tpbs or not max_blocks_list:
        raise ValueError("sizes, tpbs and max_blocks_list must not be empty")
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be at least {MIN_REPEATS}")
    for n in sizes:
        for tpb in tpbs:
            for mb in max_blocks_list:
                _validate_reduction(integrand, n, 0, tpb, mb, "device")


def sweep(
    integrand: Integrand,
    sizes: Sequence[int],
    tpbs: Sequence[int],
    max_blocks_list: Sequence[int],
    seed: int = 12345,
    repeats: int = MIN_REPEATS,
) -> List[SweepRow]:
    """Mierzy wszystkie kombinacje (rozmiar, wątki na blok, maks. bloków)."""
    _validate_sweep(integrand, sizes, tpbs, max_blocks_list, repeats)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    _require_cuda()

    configs: List[ConfigKey] = [
        (n, tpb, mb) for n in sizes for tpb in tpbs for mb in max_blocks_list
    ]

    # Rozgrzewka: jądra kompilują się osobno dla każdego rozmiaru bloku
    for tpb in tpbs:
        warm_up_gpu(integrand, tpb, max(max_blocks_list))

    rows: List[SweepRow] = []
    for rep in range(repeats):
        shift = rep % len(configs)
        for n, tpb, mb in configs[shift:] + configs[:shift]:
            r = run_gpu(integrand, n, seed, tpb, mb)
            rows.append(SweepRow(
                func=integrand.name, n=n, seed=seed, threads_per_block=tpb,
                max_blocks=mb, blocks=launch_config(n, tpb, mb)[0], repeat=rep,
                t_states=r.t_states, t_stage1=r.t_stage1, t_core=r.t_core,
                t_total=r.t_total, estimate=r.estimate, abs_err=r.abs_err,
            ))
    return rows


def write_sweep_rows(path: Path, rows: Iterable[SweepRow]) -> None:
    """Dopisuje wiersze do CSV. Nagłówek jest zapisywany tylko raz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: getattr(row, c) for c in SWEEP_COLUMNS})


def summarize_configs(rows: Iterable[SweepRow]) -> Dict[ConfigKey, ConfigSummary]:
    """Liczy mediany czasów i przepustowość dla każdej konfiguracji."""
    groups: Dict[ConfigKey, List[SweepRow]] = {}
    for r in rows:
        groups.setdefault((r.n, r.threads_per_block, r.max_blocks), []).append(r)
    if not groups:
        raise ValueError("rows must not be empty")

    result: Dict[ConfigKey, ConfigSummary] = {}
    for key, items in groups.items():
        n, tpb, mb = key
        t_total = statistics.median(r.t_total for r in items)
        t_core = statistics.median(r.t_core for r in items)
        result[key] = ConfigSummary(
            n=n, threads_per_block=tpb, max_blocks=mb, blocks=items[0].blocks,
            count=len(items),
            t_states=statistics.median(r.t_states for r in items),
            t_stage1=statistics.median(r.t_stage1 for r in items),
            t_core=t_core, t_total=t_total,
            iqr_total=summarize([r.t_total for r in items])["iqr"],
            throughput=n / t_total if t_total > 0 else float("nan"),
            core_throughput=n / t_core if t_core > 0 else float("nan"),
        )
    return result


def best_config(
    summaries: Dict[ConfigKey, ConfigSummary], n: int, metric: str = "t_total",
) -> ConfigSummary:
    """Zwraca konfigurację o najkrótszym medianowym czasie dla danego n."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {', '.join(METRICS)}")
    candidates = [s for s in summaries.values() if s.n == n]
    if not candidates:
        raise ValueError(f"No measurements for n={n}")
    return min(candidates, key=lambda s: getattr(s, metric))


def saturation_size(points: Sequence[Tuple[int, float]], fraction: float = 0.9) -> int:
    """Najmniejszy rozmiar, dla którego przepustowość osiąga fraction maksimum.

    points to pary (liczba próbek, przepustowość). Pokazuje, od jakiego n
    narzut stały (stany, alokacja, uruchomienie jądra) przestaje dominować.
    """
    if not points:
        raise ValueError("points must not be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(points)
    peak = max(t for _, t in ordered)
    for n, t in ordered:
        if t >= fraction * peak:
            return n
    return ordered[-1][0]