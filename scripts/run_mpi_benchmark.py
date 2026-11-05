"""Pomiar wersji MPI: czas dla bieżącej liczby procesów i kilku rozmiarów próbek.

Skrypt uruchamia się osobno dla każdej liczby procesów p, a wyniki są dopisywane
do wspólnego pliku data/raw/mpi.csv:

    mpiexec -n 1 python scripts/run_mpi_benchmark.py --func sine --sizes 1e7
    mpiexec -n 2 python scripts/run_mpi_benchmark.py --func sine --sizes 1e7
    mpiexec -n 4 python scripts/run_mpi_benchmark.py --func sine --sizes 1e7

Na jednym komputerze wynik oznaczamy jako lokalny (--scope local); nie jest to
pomiar skalowalności klastra.
"""

import argparse
import sys
from pathlib import Path

from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark import benchmark, summarize  # noqa: E402
from functions import INTEGRANDS, get_integrand  # noqa: E402
from mc_mpi import PARTITIONS, REDUCTIONS, mpi_core_time, run_mpi  # noqa: E402
from rng import DEFAULT_BATCH_SIZE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark of the MPI version.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1e7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--partition", choices=PARTITIONS, default="block")
    parser.add_argument("--reduction", choices=REDUCTIONS, default="gather")
    parser.add_argument("--scope", choices=["local", "cluster"], default="local")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "mpi.csv")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()
    integrand = get_integrand(args.func)
    # Nazwa metody odróżnia warianty; podstawowy (block, gather) to po prostu mpi
    method = "mpi"
    if args.partition != "block":
        method += f"-{args.partition}"
    if args.reduction != "gather":
        method += f"-{args.reduction}"

    for size in args.sizes:
        n = int(size)
        # Benchmark działa na wszystkich procesach (run_mpi jest kolektywne),
        # ale do pliku zapisuje tylko rank 0
        rows = benchmark(
            lambda seed: run_mpi(integrand, n, seed, args.batch, comm=comm,
                                 partition=args.partition, reduction=args.reduction),
            method=method, func=args.func, n=n, batch_size=args.batch,
            p=p, scope=args.scope, seeds=args.seeds, repeats=args.repeats,
            core_time=mpi_core_time, out_path=args.out if rank == 0 else None,
        )
        if rank == 0:
            s = summarize([r.t_total for r in rows])
            print(f"{args.func} n={n:>10} p={p:>2}: median {s['median']:.4f} s, "
                  f"min {s['min']:.4f}, max {s['max']:.4f}, IQR {s['iqr']:.4f}")
    if rank == 0:
        print(f"Raw results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())