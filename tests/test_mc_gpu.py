"""Testy wersji GPU z pomiarami: definicje czasów, spójność, CSV.

Testy wymagające karty NVIDIA pomijają się same. Walidacja argumentów oraz zapis
rozbicia czasu są sprawdzane zawsze.
"""

import csv
import math

import pytest

from benchmark import benchmark, read_rows
from functions import INTEGRANDS, SINE, SQUARE, Integrand
from gpu_kernels import FINALS, cuda_available, gpu_stats, gpu_stats_timed
from mc_gpu import (
    BREAKDOWN_COLUMNS, GpuResult, gpu_core_time, main, run_gpu, warm_up_gpu,
    write_breakdown,
)
from stats import integral_from_stats

ALL = list(INTEGRANDS.values())
needs_gpu = pytest.mark.skipif(not cuda_available(), reason="CUDA device not available")


def fake_result(seed=1, **overrides):
    """Sztuczny wynik do testów zapisu, bez karty."""
    values = dict(
        func="square", n=1000, seed=seed, threads_per_block=256, blocks=4,
        final="device", estimate=0.33, se=0.01, ci_low=0.31, ci_high=0.35,
        exact=1 / 3, abs_err=0.003, rel_err=0.009, t_alloc=0.001, t_states=0.002,
        t_stage1=0.01, t_stage2=0.0005, t_transfer=0.0001, t_host_merge=0.00001,
        t_core=0.0125, t_total=0.01361,
    )
    values.update(overrides)
    return GpuResult(**values)


# Walidacja argumentów (bez karty)

@pytest.mark.parametrize("n", [0, -5])
def test_non_positive_n_raises(n):
    with pytest.raises(ValueError):
        run_gpu(SQUARE, n, seed=1)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        run_gpu(SQUARE, 100, seed=-1)
    with pytest.raises(ValueError):
        run_gpu(SQUARE, 100, seed=1, threads_per_block=96)
    with pytest.raises(ValueError):
        run_gpu(SQUARE, 100, seed=1, final="unknown")


def test_integrand_without_gpu_version_raises():
    custom = Integrand("custom", lambda x: x, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        run_gpu(custom, 100, seed=1)


# Własności wyniku (bez karty)

def test_throughput_properties():
    r = fake_result(n=1_000_000, t_core=0.5, t_total=1.0)
    assert r.samples_per_second == pytest.approx(1e6)
    assert r.core_samples_per_second == pytest.approx(2e6)
    assert gpu_core_time(r) == 0.5


def test_contains_exact_property():
    assert fake_result(ci_low=0.3, ci_high=0.4, exact=1 / 3).contains_exact
    assert not fake_result(ci_low=0.4, ci_high=0.5, exact=1 / 3).contains_exact


# Zapis rozbicia czasu (bez karty)

def test_breakdown_header_written_once_and_rows_appended(tmp_path):
    path = tmp_path / "raw" / "gpu_breakdown.csv"
    write_breakdown(path, [fake_result(1), fake_result(1)])
    write_breakdown(path, [fake_result(2)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == BREAKDOWN_COLUMNS
    assert sum(1 for line in lines if line.startswith("func,")) == 1
    assert len(lines) == 1 + 3


def test_breakdown_repeat_counted_per_seed(tmp_path):
    path = tmp_path / "b.csv"
    write_breakdown(path, [fake_result(1), fake_result(2), fake_result(1), fake_result(2)])
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["seed"], r["repeat"]) for r in rows] == [
        ("1", "0"), ("2", "0"), ("1", "1"), ("2", "1")]


def test_breakdown_roundtrip_values(tmp_path):
    path = tmp_path / "b.csv"
    write_breakdown(path, [fake_result(7, t_stage1=0.25)])
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert float(row["t_stage1"]) == pytest.approx(0.25)
    assert int(row["threads_per_block"]) == 256
    assert row["final"] == "device"


# Spójność czasów

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("final", FINALS)
def test_timings_are_consistent(integrand, final):
    warm_up_gpu(integrand, final=final)
    r = run_gpu(integrand, 2_000_000, seed=1, final=final)
    parts = [r.t_alloc, r.t_states, r.t_stage1, r.t_stage2, r.t_transfer, r.t_host_merge]
    assert all(p >= 0 for p in parts)
    assert r.t_core == pytest.approx(r.t_states + r.t_stage1 + r.t_stage2)
    # Składowe są mierzone jednym ciągiem, więc ich suma to czas pełny
    assert sum(parts) == pytest.approx(r.t_total, rel=1e-6, abs=1e-9)
    assert r.t_total >= r.t_core > 0


@needs_gpu
def test_host_final_has_no_second_stage_and_larger_transfer():
    warm_up_gpu(SQUARE, final="device")
    warm_up_gpu(SQUARE, final="host")
    device = run_gpu(SQUARE, 1_000_000, seed=2, final="device")
    host = run_gpu(SQUARE, 1_000_000, seed=2, final="host")
    assert device.t_stage2 > 0
    assert host.t_stage2 == 0.0
    assert host.t_host_merge >= 0.0


@needs_gpu
def test_kernel_time_includes_the_work_after_synchronization():
    # Bez synchronizacji czas jądra byłby ułamkiem milisekundy (samo uruchomienie);
    # po synchronizacji 2e8 próbek musi zająć wyraźnie dłużej niż inicjalizacja
    warm_up_gpu(SINE)
    r = run_gpu(SINE, 200_000_000, seed=1)
    assert r.t_stage1 > 1e-3
    assert r.t_stage1 > r.t_transfer


@needs_gpu
def test_more_samples_take_longer_in_stage_one():
    warm_up_gpu(SINE)
    small = run_gpu(SINE, 1_000_000, seed=1)
    large = run_gpu(SINE, 200_000_000, seed=1)
    assert large.t_stage1 > small.t_stage1


# Zgodność wyniku z redukcją bez pomiarów

@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
@pytest.mark.parametrize("final", FINALS)
def test_result_identical_to_untimed_reduction(integrand, final):
    r = run_gpu(integrand, 500_000, seed=4, final=final)
    s = gpu_stats(integrand, 500_000, seed=4, final=final)
    ref = integral_from_stats(s, integrand.a, integrand.b)
    assert r.estimate == ref.estimate
    assert r.se == ref.se


@needs_gpu
def test_timed_and_untimed_stats_are_equal():
    s, _ = gpu_stats_timed(SINE, 300_000, seed=5)
    assert s == gpu_stats(SINE, 300_000, seed=5)


@needs_gpu
@pytest.mark.parametrize("integrand", ALL, ids=lambda f: f.name)
def test_estimate_close_to_exact(integrand):
    r = run_gpu(integrand, 5_000_000, seed=123)
    assert r.abs_err < 5 * r.se
    assert r.n == 5_000_000


@needs_gpu
def test_result_reports_grid():
    r = run_gpu(SQUARE, 100_000, seed=1, threads_per_block=128, max_blocks=64)
    assert r.threads_per_block == 128
    assert r.blocks == 64


@needs_gpu
def test_single_sample_has_undefined_error():
    r = run_gpu(SQUARE, 1, seed=1)
    assert r.n == 1
    assert math.isnan(r.se)


# Integracja ze szkieletem pomiarowym

@needs_gpu
def test_benchmark_with_gpu_version(tmp_path):
    path = tmp_path / "gpu.csv"
    breakdown = tmp_path / "gpu_breakdown.csv"
    warm_up_gpu(SQUARE)
    results = []

    def run(seed):
        r = run_gpu(SQUARE, 200_000, seed)
        results.append(r)
        return r

    rows = benchmark(run, method="gpu", func="square", n=200_000, batch_size=200_000,
                     p=1, scope="local", seeds=[1, 2], repeats=5,
                     core_time=gpu_core_time, out_path=path)
    write_breakdown(breakdown, results[-len(rows):])

    assert len(rows) == 10
    assert all(r.method == "gpu" and r.p == 1 and r.t_core > 0 for r in rows)
    assert all(r.t_total >= r.t_core for r in rows)
    assert all(r.t_transfer >= 0 for r in rows)
    # To samo ziarno i ta sama siatka dają identyczną estymatę w każdym powtórzeniu
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r.seed, set()).add(r.estimate)
    assert all(len(v) == 1 for v in by_seed.values())
    assert len(read_rows(path)) == 10
    with breakdown.open(newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 10


# Interfejs wiersza poleceń

@needs_gpu
def test_cli_prints_result(capsys):
    code = main(["--func", "sine", "--n", "1e5", "--seed", "3", "--tpb", "128"])
    out = capsys.readouterr().out
    assert code == 0
    assert "estimate" in out
    assert "time stage 1" in out
    assert "grid" in out


def test_cli_rejects_unknown_function():
    with pytest.raises(SystemExit):
        main(["--func", "nope"])