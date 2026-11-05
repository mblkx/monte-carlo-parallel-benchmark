"""Wersja MPI całkowania Monte Carlo.

Partie próbek są rozdzielane między procesy MPI. Każdy proces liczy swoje
partie funkcją compute_batch z referencji i wysyła statystyki każdej partii na
rank 0, który scala je w kolejności indeksów. Dzięki temu wynik jest bitowo
identyczny z referencją i niezależny od liczby procesów.

Przykład uruchomienia:
    mpiexec -n 4 python src/mc_mpi.py --func sine --n 10000000 --seed 42
"""

import argparse
import numpy as np

from mpi_ops import array_to_stats, create_stats_op, stats_to_array
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

from mpi4py import MPI

from functions import INTEGRANDS, Integrand, get_integrand
from mc_ref import _parse_count, compute_batch
from rng import DEFAULT_BATCH_SIZE, split_into_batches
from stats import abs_error, integral_from_stats, merge_all, rel_error

# Dostępne podziały partii między procesy
PARTITIONS = ("block", "cyclic")
# Dostępne sposoby łączenia wyników na ranku 0
REDUCTIONS = ("gather", "reduce")


@dataclass(frozen=True)
class MpiResult:
    """Wynik jednego uruchomienia wersji MPI (taki sam na wszystkich procesach)."""

    func: str            # nazwa funkcji podcałkowej
    n: int               # liczba próbek
    seed: int            # ziarno generatora
    batch_size: int      # rozmiar partii
    ranks: int           # liczba procesów MPI p
    partition: str       # podział partii: block lub cyclic
    reduction: str       # sposób łączenia wyników: gather lub reduce
    estimate: float      # estymata całki
    se: float            # błąd standardowy
    ci_low: float        # dolna granica przedziału ufności 95%
    ci_high: float       # górna granica przedziału ufności 95%
    exact: float         # wartość dokładna całki
    abs_err: float       # błąd bezwzględny
    rel_err: float       # błąd względny
    t_rng: float         # suma czasów pracy procesów: generator [s]
    t_func: float        # suma czasów pracy procesów: funkcja [s]
    t_agg: float         # suma czasów pracy procesów: statystyki [s]
    t_local_max: float   # najdłuższy czas obliczeń lokalnych wśród procesów [s]
    t_local_mean: float  # średni czas obliczeń lokalnych [s]
    t_gather: float      # czas kolektywu zbierającego, gather lub reduce, na ranku 0 [s]
    t_compute: float     # od startu do końca gather na ranku 0 [s]
    t_total: float       # od startu do gotowego wyniku na ranku 0 [s]

    @property
    def contains_exact(self) -> bool:
        """Czy przedział ufności zawiera wartość dokładną."""
        return self.ci_low <= self.exact <= self.ci_high

    @property
    def imbalance(self) -> float:
        """Nierównowaga obciążenia: max / średnia czasu lokalnego (1.0 = idealnie)."""
        if self.t_local_mean <= 0:
            return 1.0
        return self.t_local_max / self.t_local_mean


def assign_batches(n_batches: int, size: int, rank: int, partition: str = "block") -> List[int]:
    """Zwraca indeksy partii, które liczy proces o danym ranku.

    block  - ciągłe zakresy o niemal równej liczbie partii
    cyclic - proces r bierze partie k, dla których k % size == r
    Każda partia trafia dokładnie do jednego procesu.
    """
    if n_batches < 0:
        raise ValueError("n_batches must be non-negative")
    if size < 1:
        raise ValueError("size must be at least 1")
    if not 0 <= rank < size:
        raise ValueError("rank must be in [0, size)")
    if partition == "block":
        base, extra = divmod(n_batches, size)
        start = rank * base + min(rank, extra)
        count = base + (1 if rank < extra else 0)
        return list(range(start, start + count))
    if partition == "cyclic":
        return list(range(rank, n_batches, size))
    raise ValueError(f"Unknown partition '{partition}'. Available: {', '.join(PARTITIONS)}")


def run_mpi(
    integrand: Integrand,
    n_samples: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    comm=None,
    partition: str = "block",
    reduction: str = "gather",
) -> MpiResult:
    """Liczy całkę metodą Monte Carlo na procesach MPI.

    reduction:
      gather - rank 0 zbiera statystyki każdej partii i scala je w kolejności
               indeksów (wynik bitowo identyczny z referencją)
      reduce - każdy proces scala swoje partie lokalnie, a Reduce z własną
               operacją łączy trójki (n, mean, m2); wynik zgodny z referencją
               z dokładnością do zaokrągleń

    Funkcję muszą wywołać wszystkie procesy komunikatora z tymi samymi
    argumentami. Wynik jest zwracany na wszystkich procesach.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if reduction not in REDUCTIONS:
        raise ValueError(f"Unknown reduction '{reduction}'. Available: {', '.join(REDUCTIONS)}")
    comm = comm if comm is not None else MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    batches = split_into_batches(n_samples, batch_size)
    mine = assign_batches(len(batches), size, rank, partition)

    # Operacja powstaje przed pomiarem, żeby jej koszt nie wchodził do czasów
    op = create_stats_op() if reduction == "reduce" else None

    # Wszyscy startują razem, czas mierzymy od tego momentu
    comm.Barrier()
    t_start = perf_counter()

    local = []
    t_rng = t_func = t_agg = 0.0
    for k in mine:
        stats, d_rng, d_func, d_agg = compute_batch(integrand, seed, k, batches[k][1])
        local.append((k, stats))
        t_rng += d_rng
        t_func += d_func
        t_agg += d_agg
    # W trybie reduce lokalne scalanie (w kolejności indeksów) liczy się do pracy lokalnej
    local_total = merge_all(s for _, s in local) if reduction == "reduce" else None
    t_local = perf_counter() - t_start
    meta = (t_rng, t_func, t_agg, t_local)

    # Komunikacja: jedyny kolektyw wliczany do t_gather
    t_comm_start = perf_counter()
    if reduction == "gather":
        gathered = comm.gather((local, meta), root=0)
    else:
        recv = np.zeros(3, dtype=np.float64) if rank == 0 else None
        comm.Reduce(stats_to_array(local_total), recv, op=op, root=0)
    t_comm_end = perf_counter()
    t_gather = t_comm_end - t_comm_start
    t_compute = t_comm_end - t_start

    metas = None
    result = None
    if rank == 0:
        if reduction == "gather":
            items = sorted((item for part in gathered for item in part[0]),
                           key=lambda kv: kv[0])
            if [k for k, _ in items] != list(range(len(batches))):
                raise RuntimeError("Gathered batches do not cover all batch indices exactly once")
            # Scalanie w kolejności indeksów, jak w referencji
            total = merge_all(s for _, s in items)
            metas = [part[1] for part in gathered]
        else:
            total = array_to_stats(recv)
        res = integral_from_stats(total, integrand.a, integrand.b)
        t_total = perf_counter() - t_start

    # Diagnostyka w trybie reduce: dodatkowy gather już po zatrzymaniu zegarów
    if reduction == "reduce":
        collected = comm.gather(meta, root=0)
        if rank == 0:
            metas = collected

    if rank == 0:
        locals_ = [m[3] for m in metas]
        result = MpiResult(
            func=integrand.name,
            n=total.n,
            seed=seed,
            batch_size=batch_size,
            ranks=size,
            partition=partition,
            reduction=reduction,
            estimate=res.estimate,
            se=res.se,
            ci_low=res.ci_low,
            ci_high=res.ci_high,
            exact=integrand.exact,
            abs_err=abs_error(res.estimate, integrand.exact),
            rel_err=rel_error(res.estimate, integrand.exact),
            t_rng=sum(m[0] for m in metas),
            t_func=sum(m[1] for m in metas),
            t_agg=sum(m[2] for m in metas),
            t_local_max=max(locals_),
            t_local_mean=sum(locals_) / len(locals_),
            t_gather=t_gather,
            t_compute=t_compute,
            t_total=t_total,
        )

    if op is not None:
        op.Free()

    # Rozesłanie gotowego wyniku (poza pomiarem czasu), żeby benchmark
    # działał tak samo na każdym procesie
    return comm.bcast(result, root=0)


def mpi_core_time(result: MpiResult) -> float:
    """Czas rdzenia dla benchmark(): obliczenia i komunikacja do końca gather."""
    return result.t_compute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MPI Monte Carlo integration. Run with: mpiexec -n P python src/mc_mpi.py"
    )
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="square",
                        help="integrand name")
    parser.add_argument("--n", type=_parse_count, default=10_000_000,
                        help="number of samples (e.g. 1e7)")
    parser.add_argument("--seed", type=int, default=12345, help="generator seed")
    parser.add_argument("--batch", type=_parse_count, default=DEFAULT_BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--partition", choices=PARTITIONS, default="block",
                        help="how batches are split between ranks")
    parser.add_argument("--reduction", choices=REDUCTIONS, default="gather",
                        help="how results are combined on rank 0")
    return parser


def format_result(r: MpiResult) -> str:
    """Czytelny wydruk wyniku dla terminala."""
    inside = "yes" if r.contains_exact else "no"
    return "\n".join([
        f"function     : {r.func}",
        f"samples      : {r.n} (batch {r.batch_size}, seed {r.seed}, ranks {r.ranks}, {r.partition}, {r.reduction})",
        f"estimate     : {r.estimate:.10f}",
        f"exact        : {r.exact:.10f}",
        f"abs error    : {r.abs_err:.3e}",
        f"rel error    : {r.rel_err:.3e}",
        f"std error    : {r.se:.3e}",
        f"95% CI       : [{r.ci_low:.10f}, {r.ci_high:.10f}] (contains exact: {inside})",
        f"busy rng     : {r.t_rng:.4f} s (sum over ranks)",
        f"busy func    : {r.t_func:.4f} s (sum over ranks)",
        f"busy agg     : {r.t_agg:.4f} s (sum over ranks)",
        f"local max    : {r.t_local_max:.4f} s (imbalance {r.imbalance:.3f})",
        f"time gather  : {r.t_gather:.4f} s",
        f"time compute : {r.t_compute:.4f} s",
        f"time total   : {r.t_total:.4f} s",
        f"samples/s    : {r.n / r.t_total:.3e}" if r.t_total > 0 else "samples/s    : n/a",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_mpi(get_integrand(args.func), args.n, args.seed, args.batch,
                     partition=args.partition, reduction=args.reduction)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())