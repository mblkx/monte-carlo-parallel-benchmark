"""Zgodność wersji równoległych z referencją i niezależność od liczby wykonawców.

Gwarancje projektu:
- CPU (każda strategia) i MPI gather: wynik bitowo identyczny z referencją,
- MPI reduce: zgodność z dokładnością do zaokrągleń (inna kolejność scalania).
"""

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from conftest import needs_mpiexec, run_mpi_worker
from functions import INTEGRANDS, SINE, SQUARE
from mc_cpu import run_cpu, warm_up_pool
from mc_ref import compute_batch, run_reference
from rng import batch_uniform, split_into_batches
from stats import Stats, merge_all

ALL = list(INTEGRANDS.values())

# Wspólne parametry porównania: 12 partii, ostatnia niepełna
N = 115_000
BATCH = 10_000
SEED = 21


@pytest.fixture(scope="module")
def references():
    return {f.name: run_reference(f, N, SEED, BATCH) for f in ALL}


# CPU kontra referencja

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("workers", [1, 2, 3, 4])
@pytest.mark.parametrize("schedule", ["dynamic", "static", 5])
def test_cpu_bitwise_identical_to_reference(references, integrand, workers, schedule):
    ref = references[integrand.name]
    cpu = run_cpu(integrand, N, SEED, BATCH, workers=workers, schedule=schedule)
    assert cpu.n == ref.n
    assert cpu.estimate == ref.estimate
    assert cpu.se == ref.se
    assert cpu.ci_low == ref.ci_low
    assert cpu.ci_high == ref.ci_high
    assert cpu.abs_err == ref.abs_err


# MPI kontra referencja

@needs_mpiexec
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
@pytest.mark.parametrize("partition", ["block", "cyclic"])
def test_mpi_gather_bitwise_identical_to_reference(references, integrand, ranks, partition):
    ref = references[integrand.name]
    out = run_mpi_worker(ranks, integrand.name, N, SEED, BATCH, partition, "gather")
    assert out["ranks"] == ranks
    assert out["n"] == ref.n
    assert out["estimate"] == ref.estimate
    assert out["se"] == ref.se
    assert out["ci_low"] == ref.ci_low
    assert out["ci_high"] == ref.ci_high


@needs_mpiexec
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
def test_mpi_reduce_close_to_reference(references, integrand, ranks):
    ref = references[integrand.name]
    out = run_mpi_worker(ranks, integrand.name, N, SEED, BATCH, "block", "reduce")
    assert out["n"] == ref.n
    assert out["estimate"] == pytest.approx(ref.estimate, rel=1e-12)
    assert out["se"] == pytest.approx(ref.se, rel=1e-9)
    assert out["ci_low"] == pytest.approx(ref.ci_low, rel=1e-12)
    assert out["ci_high"] == pytest.approx(ref.ci_high, rel=1e-12)


# CPU kontra MPI

@needs_mpiexec
def test_cpu_and_mpi_agree_with_each_other():
    cpu = run_cpu(SINE, N, SEED, BATCH, workers=3)
    out = run_mpi_worker(3, "sine", N, SEED, BATCH, "block", "gather")
    assert cpu.estimate == out["estimate"]
    assert cpu.se == out["se"]


# Niezależność od liczby wykonawców

def test_cpu_result_independent_of_worker_count():
    estimates = {
        run_cpu(SQUARE, N, SEED, BATCH, workers=w).estimate for w in (1, 2, 3, 4)
    }
    assert len(estimates) == 1


@needs_mpiexec
def test_mpi_gather_result_independent_of_rank_count():
    estimates = {
        run_mpi_worker(r, "square", N, SEED, BATCH)["estimate"] for r in (1, 2, 3, 4)
    }
    assert len(estimates) == 1


@needs_mpiexec
def test_mpi_reduce_error_independent_of_rank_count():
    ref = run_reference(SQUARE, N, SEED, BATCH)
    for r in (1, 2, 3, 4):
        out = run_mpi_worker(r, "square", N, SEED, BATCH, "block", "reduce")
        # Zmiana liczby procesów może zmienić tylko końcowe cyfry, nigdy wynik w sensie statystycznym
        assert abs(out["estimate"] - ref.estimate) < 1e-12 * abs(ref.estimate)


# Wiele ziaren

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_cpu_identical_to_reference_for_many_seeds(integrand):
    with ProcessPoolExecutor(max_workers=2) as pool:
        warm_up_pool(pool, 2)
        for seed in range(20):
            ref = run_reference(integrand, 30_000, seed, 4_000)
            cpu = run_cpu(integrand, 30_000, seed, 4_000, workers=2, executor=pool)
            assert cpu.estimate == ref.estimate
            assert cpu.contains_exact == ref.contains_exact


# Przedziały ufności wersji równoległych

@pytest.mark.slow
def test_cpu_confidence_interval_coverage():
    hits = 0
    seeds = range(100)
    with ProcessPoolExecutor(max_workers=2) as pool:
        warm_up_pool(pool, 2)
        for seed in seeds:
            r = run_cpu(SINE, 20_000, seed, 5_000, workers=2, executor=pool)
            hits += r.contains_exact
    coverage = hits / len(seeds)
    print(f"\n  cpu, sine: pokrycie = {coverage:.3f} (oczekiwane ~0.95)")
    assert 0.88 <= coverage <= 0.995


# Rozłączne strumienie

def test_batches_have_distinct_samples():
    batches = split_into_batches(N, BATCH)
    chunks = [batch_uniform(SEED, k, 2_000) for k, _ in batches]
    values = np.concatenate(chunks)
    assert np.unique(values).size == values.size


def test_batch_statistics_do_not_depend_on_where_they_are_computed():
    # compute_batch jest czystą funkcją (seed, indeks, rozmiar): proces roboczy
    # musi dać dokładnie to samo co proces główny
    with ProcessPoolExecutor(max_workers=2) as pool:
        remote = list(pool.map(_remote_batch, range(6)))
    for k, stats in enumerate(remote):
        local, _, _, _ = compute_batch(SINE, SEED, k, 5_000)
        assert stats == local


def _remote_batch(k):
    return compute_batch(SINE, SEED, k, 5_000)[0]


def test_merge_order_effect_is_only_rounding():
    # Scalanie w odwrotnej kolejności (jak mogłoby zrobić drzewo redukcji)
    parts = [compute_batch(SINE, SEED, k, size)[0] for k, size in split_into_batches(N, BATCH)]
    forward = merge_all(parts)
    backward = merge_all(reversed(parts))
    assert forward.n == backward.n
    assert backward.mean == pytest.approx(forward.mean, rel=1e-13)
    assert backward.m2 == pytest.approx(forward.m2, rel=1e-12)


# Przypadki brzegowe wersji równoległych

def test_one_sample_in_every_version():
    ref = run_reference(SQUARE, 1, SEED)
    cpu = run_cpu(SQUARE, 1, SEED, workers=2)
    assert cpu.n == ref.n == 1
    assert cpu.estimate == ref.estimate
    assert math.isnan(cpu.se) and math.isnan(ref.se)


@needs_mpiexec
def test_one_sample_in_mpi():
    ref = run_reference(SQUARE, 1, SEED)
    out = run_mpi_worker(3, "square", 1, SEED, 1_000)
    assert out["n"] == 1
    assert out["estimate"] == ref.estimate


@needs_mpiexec
def test_sample_count_equal_to_batch_size_in_mpi():
    ref = run_reference(SQUARE, 10_000, SEED, 10_000)
    out = run_mpi_worker(4, "square", 10_000, SEED, 10_000)
    assert out["estimate"] == ref.estimate