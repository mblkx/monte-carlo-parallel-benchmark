"""Szkielet pomiarowy: rozgrzewka, powtórzenia, zapis surowych wyników do CSV.

Każdy pomiar to jeden wiersz CSV. Mediany i rozrzut liczy się przy analizie
(funkcja summarize), dzięki czemu surowe dane nie giną.

Ten sam szkielet obsługuje referencję, wersję wielordzeniową, MPI i GPU:
wystarczy podać funkcję run(seed), która zwraca obiekt z polami
estimate, se, abs_err, contains_exact, t_core i t_total.
"""

import csv
import statistics
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

# Minimalna liczba niezależnych pomiarów konfiguracji wymagana w projekcie
MIN_REPEATS = 5


@dataclass(frozen=True)
class Measurement:
    """Jeden pomiar (jeden wiersz CSV)."""

    method: str            # np. reference, cpu, mpi, gpu
    func: str              # nazwa funkcji podcałkowej
    n: int                 # liczba próbek
    seed: int              # ziarno
    batch_size: int        # rozmiar partii
    p: int                 # liczba wątków lub procesów (dla referencji 1)
    scope: str             # local lub cluster (dla MPI na jednym komputerze: local)
    repeat: int            # numer powtórzenia (od 0)
    t_core: float          # czas rdzenia obliczeń [s]
    t_total: float         # pełny czas przetwarzania [s]
    t_transfer: float      # czas transferów [s] (GPU, w pozostałych 0)
    estimate: float        # estymata całki
    se: float              # błąd standardowy
    abs_err: float         # błąd bezwzględny
    contains_exact: bool   # czy przedział ufności zawiera wartość dokładną


CSV_COLUMNS: List[str] = [f.name for f in fields(Measurement)]


def reference_core_time(result) -> float:
    """Czas rdzenia referencji: generator + funkcja + agregacja."""
    return result.t_rng + result.t_func + result.t_agg


def write_rows(path: Path, rows: Iterable[Measurement]) -> None:
    """Dopisuje wiersze do pliku CSV. Nagłówek jest zapisywany tylko raz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: getattr(row, c) for c in CSV_COLUMNS})


def read_rows(path: Path) -> List[dict]:
    """Wczytuje surowe pomiary z CSV (wartości jako tekst)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def benchmark(
    run: Callable[[int], object],
    *,
    method: str,
    func: str,
    n: int,
    batch_size: int,
    p: int = 1,
    scope: str = "local",
    seeds: Sequence[int] = (12345,),
    repeats: int = MIN_REPEATS,
    warmup: int = 1,
    core_time: Callable[[object], float] = reference_core_time,
    out_path: Optional[Path] = None,
) -> List[Measurement]:
    """Mierzy konfigurację: rozgrzewka, a potem repeats pomiarów dla każdego ziarna.

    run(seed) wykonuje jedno pełne obliczenie i zwraca obiekt wyniku.
    Rozgrzewka nie jest zapisywana. Zwraca listę pomiarów i opcjonalnie
    dopisuje je do pliku CSV.
    """
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be at least {MIN_REPEATS}")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not seeds:
        raise ValueError("at least one seed is required")

    # Rozgrzewka: kompilacja, cache, inicjalizacja pul procesów itp.
    for _ in range(warmup):
        run(seeds[0])

    rows: List[Measurement] = []
    for seed in seeds:
        for rep in range(repeats):
            r = run(seed)
            rows.append(Measurement(
                method=method,
                func=func,
                n=n,
                seed=seed,
                batch_size=batch_size,
                p=p,
                scope=scope,
                repeat=rep,
                t_core=core_time(r),
                t_total=r.t_total,
                t_transfer=getattr(r, "t_transfer", 0.0),
                estimate=r.estimate,
                se=r.se,
                abs_err=r.abs_err,
                contains_exact=bool(r.contains_exact),
            ))

    if out_path is not None:
        write_rows(out_path, rows)
    return rows


def summarize(values: Sequence[float]) -> dict:
    """Mediana i rozrzut (min, max, IQR) dla listy czasów."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) >= 2:
        q = statistics.quantiles(ordered, n=4, method="inclusive")
        iqr = q[2] - q[0]
    else:
        iqr = 0.0
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "iqr": iqr,
        "count": len(ordered),
    }