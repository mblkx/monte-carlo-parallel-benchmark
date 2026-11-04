"""Skrypt pomocniczy testów: liczy całkę na procesach MPI i drukuje JSON z rank 0."""

import json
import sys
from pathlib import Path

from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functions import get_integrand  # noqa: E402
from mc_mpi import run_mpi  # noqa: E402


def main() -> None:
    func, n, seed, batch, partition = sys.argv[1:6]
    r = run_mpi(get_integrand(func), int(n), int(seed), int(batch), partition=partition)
    if MPI.COMM_WORLD.Get_rank() == 0:
        # json zapisuje floaty przez repr, więc odczyt daje dokładnie te same liczby
        print(json.dumps({"n": r.n, "ranks": r.ranks, "estimate": r.estimate,
                          "se": r.se, "ci_low": r.ci_low, "ci_high": r.ci_high}))


if __name__ == "__main__":
    main()