"""Porównanie gather i reduce: dokładność względem referencji i czas komunikacji.

Uruchamiać dla kilku liczb procesów, także z małymi partiami (dużo partii),
bo wtedy różnica w kosztach komunikacji jest największa:

    mpiexec -n 4 python scripts/compare_mpi_reductions.py --func sine --n 1e7 --batch 1e4
"""

import argparse
import statistics
import sys
from pathlib import Path

from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from functions import INTEGRANDS, get_integrand  # noqa: E402
from mc_mpi import PARTITIONS, REDUCTIONS, run_mpi  # noqa: E402
from mc_ref import run_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gather and reduce.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--n", type=float, default=1e7)
    parser.add_argument("--batch", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--partition", choices=PARTITIONS, default="block")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    integrand = get_integrand(args.func)
    n, batch = int(args.n), int(args.batch)
    ref = run_reference(integrand, n, args.seed, batch) if comm.Get_rank() == 0 else None

    for reduction in REDUCTIONS:
        # Rozgrzewka, potem pomiary
        run_mpi(integrand, n, args.seed, batch, comm, args.partition, reduction)
        runs = [run_mpi(integrand, n, args.seed, batch, comm, args.partition, reduction)
                for _ in range(args.repeats)]
        if comm.Get_rank() == 0:
            r = runs[0]
            diff = abs(r.estimate - ref.estimate)
            print(f"{reduction:>6} p={r.ranks} batches={-(-n // batch)}: "
                  f"estimate {r.estimate!r}, |diff vs reference| {diff:.3e}, "
                  f"median t_gather {statistics.median(x.t_gather for x in runs):.6f} s, "
                  f"median t_total {statistics.median(x.t_total for x in runs):.4f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())