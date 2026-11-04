"""Testy wersji MPI: podział partii, zgodność z referencją, uruchomienia mpiexec."""

import json
import math
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

from functions import INTEGRANDS, SINE, SQUARE  # noqa: E402
from mc_mpi import PARTITIONS, assign_batches, run_mpi  # noqa: E402
from mc_ref import run_reference  # noqa: E402

ALL = list(INTEGRANDS.values())

# Dodatkowe argumenty mpiexec, np. PRIR_MPIEXEC_ARGS="--oversubscribe" dla OpenMPI
MPIEXEC_ARGS = shlex.split(os.environ.get("PRIR_MPIEXEC_ARGS", ""))
needs_mpiexec = pytest.mark.skipif(shutil.which("mpiexec") is None,
                                   reason="mpiexec not found in PATH")


# Podział partii (bez MPI)

@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("n_batches", [0, 1, 5, 10, 17, 100])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
def test_every_batch_assigned_exactly_once(partition, n_batches, size):
    assigned = [k for r in range(size) for k in assign_batches(n_batches, size, r, partition)]
    assert sorted(assigned) == list(range(n_batches))


@pytest.mark.parametrize("n_batches, size", [(10, 4), (17, 3), (100, 8), (5, 5)])
def test_block_is_contiguous_and_balanced(n_batches, size):
    parts = [assign_batches(n_batches, size, r, "block") for r in range(size)]
    counts = [len(p) for p in parts]
    assert max(counts) - min(counts) <= 1
    for p in parts:
        assert p == list(range(p[0], p[0] + len(p))) if p else True


def test_cyclic_assignment():
    assert assign_batches(10, 3, 0, "cyclic") == [0, 3, 6, 9]
    assert assign_batches(10, 3, 1, "cyclic") == [1, 4, 7]
    assert assign_batches(10, 3, 2, "cyclic") == [2, 5, 8]


def test_more_ranks_than_batches():
    parts = [assign_batches(2, 4, r, "block") for r in range(4)]
    assert sorted(k for p in parts for k in p) == [0, 1]
    assert sum(1 for p in parts if not p) == 2


def test_assign_batches_invalid_arguments():
    with pytest.raises(ValueError):
        assign_batches(-1, 2, 0)
    with pytest.raises(ValueError):
        assign_batches(10, 0, 0)
    with pytest.raises(ValueError):
        assign_batches(10, 2, 2)
    with pytest.raises(ValueError):
        assign_batches(10, 2, 0, "unknown")


# Jeden proces (COMM_SELF, niezależnie od sposobu uruchomienia pytesta)

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("partition", PARTITIONS)
def test_single_rank_identical_to_reference(integrand, partition):
    ref = run_reference(integrand, 100_000, seed=7, batch_size=10_000)
    r = run_mpi(integrand, 100_000, 7, 10_000, comm=MPI.COMM_SELF, partition=partition)
    assert r.ranks == 1
    assert r.n == ref.n
    assert r.estimate == ref.estimate
    assert r.se == ref.se
    assert r.ci_low == ref.ci_low
    assert r.ci_high == ref.ci_high


def test_sample_count_not_divisible_by_batch():
    r = run_mpi(SQUARE, 25_001, 1, 10_000, comm=MPI.COMM_SELF)
    assert r.n == 25_001


def test_one_sample_has_undefined_error():
    r = run_mpi(SQUARE, 1, 1, comm=MPI.COMM_SELF)
    assert r.n == 1
    assert math.isnan(r.se)


@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_sample_count_raises(n):
    with pytest.raises(ValueError):
        run_mpi(SQUARE, n, 1, comm=MPI.COMM_SELF)


def test_timings_are_consistent():
    r = run_mpi(SINE, 200_000, 1, 50_000, comm=MPI.COMM_SELF)
    assert r.t_rng >= 0 and r.t_func >= 0 and r.t_agg >= 0
    assert 0 < r.t_local_max <= r.t_compute
    assert r.t_gather >= 0
    assert r.t_total >= r.t_compute
    assert r.imbalance == pytest.approx(1.0)


# Wiele procesów (mpiexec w osobnym procesie)

def run_worker(ranks, func, n, seed, batch, partition):
    cmd = ["mpiexec", *MPIEXEC_ARGS, "-n", str(ranks), sys.executable,
           str(ROOT / "tests" / "mpi_worker.py"), func, str(n), str(seed),
           str(batch), partition]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@needs_mpiexec
@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_multi_rank_identical_to_reference(ranks, partition):
    ref = run_reference(SINE, 100_000, seed=7, batch_size=10_000)
    out = run_worker(ranks, "sine", 100_000, 7, 10_000, partition)
    assert out["ranks"] == ranks
    assert out["n"] == ref.n
    assert out["estimate"] == ref.estimate
    assert out["se"] == ref.se
    assert out["ci_low"] == ref.ci_low
    assert out["ci_high"] == ref.ci_high


@needs_mpiexec
def test_multi_rank_square_function():
    ref = run_reference(SQUARE, 60_000, seed=3, batch_size=7_000)
    out = run_worker(2, "square", 60_000, 3, 7_000, "block")
    assert out["estimate"] == ref.estimate


@needs_mpiexec
def test_more_ranks_than_batches_gives_correct_result():
    ref = run_reference(SQUARE, 15_000, seed=5, batch_size=10_000)
    out = run_worker(4, "square", 15_000, 5, 10_000, "block")
    assert out["estimate"] == ref.estimate


@needs_mpiexec
def test_cli_prints_result_on_rank_zero():
    cmd = ["mpiexec", *MPIEXEC_ARGS, "-n", "2", sys.executable,
           str(ROOT / "src" / "mc_mpi.py"), "--func", "sine", "--n", "1e4",
           "--seed", "3", "--batch", "5000"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("estimate") == 1
    assert "ranks 2" in proc.stdout