"""Testy wersji wielordzeniowej: zgodność z referencją, partie, czasy, CLI."""

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark import benchmark  # noqa: E402
from functions import INTEGRANDS, SINE, SQUARE, Integrand  # noqa: E402
from mc_cpu import cpu_core_time, main, run_cpu, warm_up_pool  # noqa: E402
from mc_ref import run_reference  # noqa: E402

ALL = list(INTEGRANDS.values())


# Zgodność z referencją 

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_result_identical_to_reference(integrand, workers):
    """Te same partie scalane w tej samej kolejności dają identyczny wynik."""
    ref = run_reference(integrand, 100_000, seed=7, batch_size=10_000)
    cpu = run_cpu(integrand, 100_000, seed=7, batch_size=10_000, workers=workers)
    assert cpu.n == ref.n
    assert cpu.estimate == ref.estimate
    assert cpu.se == ref.se
    assert cpu.ci_low == ref.ci_low
    assert cpu.ci_high == ref.ci_high


def test_result_independent_of_worker_count():
    results = [
        run_cpu(SINE, 60_000, seed=3, batch_size=7_000, workers=w).estimate
        for w in (1, 2, 3)
    ]
    assert len(set(results)) == 1


def test_result_independent_of_chunksize():
    a = run_cpu(SQUARE, 60_000, seed=3, batch_size=5_000, workers=2, chunksize=1)
    b = run_cpu(SQUARE, 60_000, seed=3, batch_size=5_000, workers=2, chunksize=4)
    assert a.estimate == b.estimate


def test_same_seed_repeatable_and_different_seeds_differ():
    a = run_cpu(SQUARE, 40_000, seed=1, batch_size=10_000, workers=2)
    b = run_cpu(SQUARE, 40_000, seed=1, batch_size=10_000, workers=2)
    c = run_cpu(SQUARE, 40_000, seed=2, batch_size=10_000, workers=2)
    assert a.estimate == b.estimate
    assert a.estimate != c.estimate


# Poprawność 

@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_estimate_close_to_exact(integrand):
    r = run_cpu(integrand, 1_000_000, seed=123, batch_size=250_000, workers=2)
    assert r.abs_err < 5 * r.se
    assert r.workers == 2


# Przypadki brzegowe

def test_sample_count_not_divisible_by_batch():
    r = run_cpu(SQUARE, 25_001, seed=1, batch_size=10_000, workers=2)
    assert r.n == 25_001


def test_fewer_batches_than_workers():
    ref = run_reference(SQUARE, 15_000, seed=5, batch_size=10_000)
    r = run_cpu(SQUARE, 15_000, seed=5, batch_size=10_000, workers=4)
    assert r.estimate == ref.estimate


def test_one_sample_has_undefined_error():
    r = run_cpu(SQUARE, 1, seed=1, workers=1)
    assert r.n == 1
    assert math.isnan(r.se)


@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_sample_count_raises(n):
    with pytest.raises(ValueError):
        run_cpu(SQUARE, n, seed=1, workers=1)


def test_invalid_workers_and_chunksize_raise():
    with pytest.raises(ValueError):
        run_cpu(SQUARE, 100, seed=1, workers=0)
    with pytest.raises(ValueError):
        run_cpu(SQUARE, 100, seed=1, workers=1, chunksize=0)


def test_unregistered_integrand_raises():
    custom = Integrand("custom", lambda x: x, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        run_cpu(custom, 100, seed=1, workers=1)


def test_executor_requires_workers():
    with ProcessPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError):
            run_cpu(SQUARE, 100, seed=1, executor=pool)


# Ponowne użycie puli 

def test_reused_executor_gives_same_result():
    ref = run_reference(SINE, 50_000, seed=9, batch_size=10_000)
    with ProcessPoolExecutor(max_workers=2) as pool:
        warm_up_pool(pool, 2)
        a = run_cpu(SINE, 50_000, seed=9, batch_size=10_000, workers=2, executor=pool)
        b = run_cpu(SINE, 50_000, seed=9, batch_size=10_000, workers=2, executor=pool)
    assert a.estimate == b.estimate == ref.estimate


# Czasy 

def test_timings_are_consistent():
    r = run_cpu(SINE, 200_000, seed=1, batch_size=50_000, workers=2)
    assert r.t_rng >= 0 and r.t_func >= 0 and r.t_agg >= 0
    assert r.t_compute > 0
    assert r.t_total >= r.t_compute


# Integracja ze szkieletem pomiarowym 

def test_benchmark_with_cpu_version(tmp_path):
    path = tmp_path / "cpu.csv"
    with ProcessPoolExecutor(max_workers=2) as pool:
        warm_up_pool(pool, 2)
        rows = benchmark(
            lambda seed: run_cpu(SQUARE, 20_000, seed, 5_000, workers=2, executor=pool),
            method="cpu", func="square", n=20_000, batch_size=5_000,
            p=2, scope="local", seeds=[1], repeats=5,
            core_time=cpu_core_time, out_path=path,
        )
    assert len(rows) == 5
    assert all(r.p == 2 and r.method == "cpu" and r.t_core > 0 for r in rows)
    assert len({r.estimate for r in rows}) == 1


# Interfejs wiersza poleceń 

def test_cli_prints_result(capsys):
    code = main(["--func", "sine", "--n", "1e4", "--seed", "3",
                 "--batch", "5000", "--workers", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "estimate" in out
    assert "workers 2" in out
    assert "95% CI" in out


def test_cli_rejects_unknown_function():
    with pytest.raises(SystemExit):
        main(["--func", "nope"])