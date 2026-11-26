"""Testy pomiarów siatki GPU: analiza, CSV, walidacja i (z kartą) poprawność.

Analiza, walidacja i zapis CSV są sprawdzane zawsze. Testy wymagające karty
NVIDIA pomijają się same.
"""

import csv
import statistics

import pytest

from functions import SINE, SQUARE
from gpu_kernels import cuda_available, launch_config
from gpu_sweep import (
    SWEEP_COLUMNS, SweepRow, best_config, saturation_size, summarize_configs,
    sweep, write_sweep_rows,
)

needs_gpu = pytest.mark.skipif(not cuda_available(), reason="CUDA device not available")


def row(n=1000, tpb=256, mb=2048, rep=0, t_total=1.0, t_core=0.8, **kw):
    """Sztuczny wiersz do testów analizy, bez karty."""
    values = dict(
        func="square", n=n, seed=1, threads_per_block=tpb, max_blocks=mb,
        blocks=4, repeat=rep, t_states=0.1, t_stage1=0.6, t_core=t_core,
        t_total=t_total, estimate=0.33, abs_err=0.003,
    )
    values.update(kw)
    return SweepRow(**values)


# Podsumowanie konfiguracji

def test_summarize_uses_medians_and_throughput():
    rows = [row(rep=i, t_total=t, t_core=t / 2) for i, t in enumerate([1.0, 2.0, 3.0, 4.0, 100.0])]
    s = next(iter(summarize_configs(rows).values()))
    assert s.count == 5
    assert s.t_total == 3.0
    assert s.t_core == 1.5
    assert s.throughput == pytest.approx(1000 / 3.0)
    assert s.core_throughput == pytest.approx(1000 / 1.5)
    assert s.iqr_total == pytest.approx(2.0)


def test_summarize_groups_by_configuration():
    rows = [row(tpb=128, rep=i) for i in range(5)] + [row(tpb=256, rep=i) for i in range(5)]
    s = summarize_configs(rows)
    assert set(s) == {(1000, 128, 2048), (1000, 256, 2048)}


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize_configs([])


# Najlepsza konfiguracja

def test_best_config_picks_lowest_median():
    rows = ([row(tpb=128, rep=i, t_total=2.0) for i in range(5)]
            + [row(tpb=256, rep=i, t_total=1.0) for i in range(5)]
            + [row(tpb=512, rep=i, t_total=1.5) for i in range(5)])
    best = best_config(summarize_configs(rows), 1000)
    assert best.threads_per_block == 256


def test_best_config_respects_metric_and_size():
    rows = ([row(n=1000, tpb=128, rep=i, t_total=1.0, t_core=0.9) for i in range(5)]
            + [row(n=1000, tpb=256, rep=i, t_total=2.0, t_core=0.5) for i in range(5)]
            + [row(n=2000, tpb=64, rep=i, t_total=0.1) for i in range(5)])
    s = summarize_configs(rows)
    assert best_config(s, 1000, "t_total").threads_per_block == 128
    assert best_config(s, 1000, "t_core").threads_per_block == 256


def test_best_config_invalid_arguments():
    s = summarize_configs([row(rep=i) for i in range(5)])
    with pytest.raises(ValueError):
        best_config(s, 999)
    with pytest.raises(ValueError):
        best_config(s, 1000, "unknown")


# Próg nasycenia

def test_saturation_size_finds_first_size_near_peak():
    points = [(10**5, 1e8), (10**6, 1e9), (10**7, 4.8e9), (10**8, 5e9)]
    assert saturation_size(points, 0.9) == 10**7


def test_saturation_size_is_independent_of_input_order():
    points = [(10**8, 5e9), (10**5, 1e8), (10**7, 4.8e9)]
    assert saturation_size(points, 0.9) == 10**7


def test_saturation_size_full_fraction_gives_peak_size():
    assert saturation_size([(1, 1.0), (2, 3.0), (3, 2.0)], 1.0) == 2


def test_saturation_size_invalid_arguments():
    with pytest.raises(ValueError):
        saturation_size([])
    with pytest.raises(ValueError):
        saturation_size([(1, 1.0)], 0.0)
    with pytest.raises(ValueError):
        saturation_size([(1, 1.0)], 1.5)


# Zapis CSV

def test_csv_header_written_once_and_rows_appended(tmp_path):
    path = tmp_path / "raw" / "sweep.csv"
    write_sweep_rows(path, [row(rep=i) for i in range(5)])
    write_sweep_rows(path, [row(n=2000, rep=i) for i in range(5)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == SWEEP_COLUMNS
    assert sum(1 for line in lines if line.startswith("func,")) == 1
    assert len(lines) == 1 + 10


def test_csv_roundtrip_values(tmp_path):
    path = tmp_path / "sweep.csv"
    write_sweep_rows(path, [row(tpb=512, mb=64, t_total=0.25)])
    with path.open(newline="", encoding="utf-8") as fh:
        r = next(csv.DictReader(fh))
    assert int(r["threads_per_block"]) == 512
    assert int(r["max_blocks"]) == 64
    assert float(r["t_total"]) == pytest.approx(0.25)


# Walidacja argumentów (bez karty)

def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        sweep(SQUARE, [], [256], [64])
    with pytest.raises(ValueError):
        sweep(SQUARE, [1000], [256], [64], repeats=4)
    with pytest.raises(ValueError):
        sweep(SQUARE, [1000], [96], [64])
    with pytest.raises(ValueError):
        sweep(SQUARE, [1000], [256], [0])
    with pytest.raises(ValueError):
        sweep(SQUARE, [0], [256], [64])
    with pytest.raises(ValueError):
        sweep(SQUARE, [1000], [256], [64], seed=-1)


# Pomiary na karcie

@needs_gpu
def test_sweep_returns_all_configurations_and_repeats():
    rows = sweep(SQUARE, [100_000, 1_000_000], [128, 256], [64, 1024], repeats=5)
    assert len(rows) == 2 * 2 * 2 * 5
    keys = {(r.n, r.threads_per_block, r.max_blocks) for r in rows}
    assert len(keys) == 8
    for key in keys:
        assert sorted(r.repeat for r in rows
                      if (r.n, r.threads_per_block, r.max_blocks) == key) == list(range(5))


@needs_gpu
def test_sweep_reports_effective_grid():
    rows = sweep(SQUARE, [1_000_000], [128], [8, 100_000], repeats=5)
    for r in rows:
        assert r.blocks == launch_config(1_000_000, 128, r.max_blocks)[0]
    assert {r.blocks for r in rows} == {8, 7813}


@needs_gpu
@pytest.mark.parametrize("integrand", [SQUARE, SINE], ids=lambda f: f.name)
def test_every_grid_gives_statistically_correct_result(integrand):
    # Każda siatka daje inne próbki, ale estymata musi być zgodna z całką
    rows = sweep(integrand, [2_000_000], [64, 256, 1024], [16, 2048], repeats=5)
    for r in rows:
        assert r.abs_err < 1e-2


@needs_gpu
def test_timings_are_consistent():
    rows = sweep(SINE, [1_000_000], [256], [2048], repeats=5)
    for r in rows:
        assert r.t_total >= r.t_core > 0
        assert r.t_core == pytest.approx(r.t_states + r.t_stage1, rel=0.5, abs=1e-2)


@needs_gpu
def test_more_samples_take_longer_for_the_same_grid():
    rows = sweep(SINE, [100_000, 100_000_000], [256], [2048], repeats=5)
    s = summarize_configs(rows)
    assert s[(100_000_000, 256, 2048)].t_total > s[(100_000, 256, 2048)].t_total


@needs_gpu
def test_state_initialization_grows_with_thread_count():
    # Ten sam rozmiar problemu, ale 1024 razy więcej wątków i stanów generatora
    rows = sweep(SQUARE, [10_000_000], [256], [2, 2048], repeats=5)
    s = summarize_configs(rows)
    assert s[(10_000_000, 256, 2048)].t_states > s[(10_000_000, 256, 2)].t_states