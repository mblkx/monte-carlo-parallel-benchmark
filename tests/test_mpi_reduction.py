"""Testy trybu reduce: zgodność z referencją i porównanie z gather."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(ROOT / "src"))

from functions import SINE, SQUARE  # noqa: E402
from mc_mpi import PARTITIONS, REDUCTIONS, run_mpi  # noqa: E402
from mc_ref import run_reference  # noqa: E402

# Dodatkowe argumenty mpiexec, np. PRIR_MPIEXEC_ARGS="--oversubscribe" dla OpenMPI
MPIEXEC_ARGS = shlex.split(os.environ.get("PRIR_MPIEXEC_ARGS", ""))
needs_mpiexec = pytest.mark.skipif(shutil.which("mpiexec") is None,
                                   reason="mpiexec not found in PATH")


def run_worker(ranks, func, n, seed, batch, partition, reduction):
    cmd = ["mpiexec", *MPIEXEC_ARGS, "-n", str(ranks), sys.executable,
           str(ROOT / "tests" / "mpi_worker.py"), func, str(n), str(seed),
           str(batch), partition, reduction]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# Jeden proces

@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("integrand", [SQUARE, SINE], ids=lambda f: f.name)
def test_single_rank_reduce_identical_to_reference(integrand, partition):
    ref = run_reference(integrand, 100_000, seed=7, batch_size=10_000)
    r = run_mpi(integrand, 100_000, 7, 10_000, comm=MPI.COMM_SELF,
                partition=partition, reduction="reduce")
    assert r.reduction == "reduce"
    assert r.estimate == ref.estimate
    assert r.se == ref.se


def test_unknown_reduction_raises():
    with pytest.raises(ValueError):
        run_mpi(SQUARE, 100, 1, comm=MPI.COMM_SELF, reduction="unknown")


def test_timings_are_consistent_for_reduce():
    r = run_mpi(SINE, 200_000, 1, 50_000, comm=MPI.COMM_SELF, reduction="reduce")
    assert r.t_gather >= 0
    assert 0 < r.t_local_max <= r.t_compute
    assert r.t_total >= r.t_compute


# Wiele procesów

@needs_mpiexec
@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_multi_rank_reduce_close_to_reference(ranks, partition):
    ref = run_reference(SINE, 100_000, seed=7, batch_size=10_000)
    out = run_worker(ranks, "sine", 100_000, 7, 10_000, partition, "reduce")
    assert out["ranks"] == ranks
    assert out["n"] == ref.n
    # Inna kolejność łączenia: zgodność tylko z dokładnością do zaokrągleń
    assert out["estimate"] == pytest.approx(ref.estimate, rel=1e-12)
    assert out["se"] == pytest.approx(ref.se, rel=1e-9)


@needs_mpiexec
@pytest.mark.parametrize("ranks", [2, 4])
def test_gather_stays_bitwise_identical(ranks):
    ref = run_reference(SINE, 100_000, seed=7, batch_size=10_000)
    out = run_worker(ranks, "sine", 100_000, 7, 10_000, "block", "gather")
    assert out["estimate"] == ref.estimate


@needs_mpiexec
def test_reduce_is_repeatable_for_fixed_rank_count():
    a = run_worker(4, "square", 60_000, 3, 5_000, "cyclic", "reduce")
    b = run_worker(4, "square", 60_000, 3, 5_000, "cyclic", "reduce")
    assert a["estimate"] == b["estimate"]


@needs_mpiexec
def test_more_ranks_than_batches_with_reduce():
    ref = run_reference(SQUARE, 15_000, seed=5, batch_size=10_000)
    out = run_worker(4, "square", 15_000, 5, 10_000, "block", "reduce")
    assert out["n"] == 15_000
    assert out["estimate"] == pytest.approx(ref.estimate, rel=1e-12)


@needs_mpiexec
def test_both_reductions_agree_closely():
    g = run_worker(3, "sine", 80_000, 2, 8_000, "block", "gather")
    r = run_worker(3, "sine", 80_000, 2, 8_000, "block", "reduce")
    assert r["estimate"] == pytest.approx(g["estimate"], rel=1e-12)