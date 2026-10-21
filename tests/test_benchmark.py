"""Testy szkieletu pomiarowego: liczba pomiarów, CSV, podsumowanie."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark import (  # noqa: E402
    CSV_COLUMNS, MIN_REPEATS, benchmark, read_rows, summarize, write_rows,
)
from functions import SQUARE  # noqa: E402
from mc_ref import run_reference  # noqa: E402


def fake_run_factory(counter):
    """Sztuczna funkcja pomiarowa: zlicza wywołania, zwraca stały wynik."""
    def run(seed):
        counter.append(seed)
        return SimpleNamespace(
            t_rng=0.1, t_func=0.2, t_agg=0.05, t_total=0.4,
            estimate=1.0, se=0.01, abs_err=0.002, contains_exact=True,
        )
    return run


def common_kwargs(**extra):
    base = dict(method="test", func="square", n=100, batch_size=10)
    base.update(extra)
    return base


# Liczba wywołań i rozgrzewka 

def test_warmup_is_not_recorded():
    calls = []
    rows = benchmark(fake_run_factory(calls), **common_kwargs(), repeats=5, warmup=2)
    assert len(rows) == 5
    assert len(calls) == 5 + 2


def test_rows_per_seed_and_repeat():
    calls = []
    rows = benchmark(fake_run_factory(calls), **common_kwargs(),
                     seeds=[1, 2, 3], repeats=5, warmup=1)
    assert len(rows) == 15
    assert {r.seed for r in rows} == {1, 2, 3}
    assert sorted({r.repeat for r in rows}) == list(range(5))


def test_zero_warmup_allowed():
    calls = []
    benchmark(fake_run_factory(calls), **common_kwargs(), repeats=5, warmup=0)
    assert len(calls) == 5


# Walidacja argumentów 

def test_too_few_repeats_raises():
    with pytest.raises(ValueError):
        benchmark(fake_run_factory([]), **common_kwargs(), repeats=MIN_REPEATS - 1)


def test_negative_warmup_raises():
    with pytest.raises(ValueError):
        benchmark(fake_run_factory([]), **common_kwargs(), warmup=-1)


def test_empty_seeds_raises():
    with pytest.raises(ValueError):
        benchmark(fake_run_factory([]), **common_kwargs(), seeds=[])


# Zawartość pomiarów

def test_core_time_is_sum_of_components():
    rows = benchmark(fake_run_factory([]), **common_kwargs(), repeats=5)
    assert rows[0].t_core == pytest.approx(0.35)
    assert rows[0].t_total == pytest.approx(0.4)
    assert rows[0].t_transfer == 0.0


def test_metadata_is_stored():
    rows = benchmark(fake_run_factory([]), **common_kwargs(p=4, scope="local"), repeats=5)
    assert rows[0].method == "test"
    assert rows[0].p == 4
    assert rows[0].scope == "local"


# CSV

def test_csv_header_written_once_and_rows_appended(tmp_path):
    path = tmp_path / "raw" / "out.csv"
    benchmark(fake_run_factory([]), **common_kwargs(), repeats=5, out_path=path)
    benchmark(fake_run_factory([]), **common_kwargs(n=200), repeats=5, out_path=path)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].split(",") == CSV_COLUMNS
    assert sum(1 for line in text if line.startswith("method,")) == 1
    assert len(text) == 1 + 10


def test_csv_roundtrip(tmp_path):
    path = tmp_path / "out.csv"
    rows = benchmark(fake_run_factory([]), **common_kwargs(), repeats=5, out_path=path)
    back = read_rows(path)
    assert len(back) == len(rows)
    assert int(back[0]["n"]) == 100
    assert float(back[0]["t_total"]) == pytest.approx(0.4)
    assert back[0]["contains_exact"] == "True"


def test_write_rows_creates_missing_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c.csv"
    rows = benchmark(fake_run_factory([]), **common_kwargs(), repeats=5)
    write_rows(path, rows)
    assert path.exists()


# Podsumowanie 

def test_summarize_basic():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 100.0
    assert s["count"] == 5
    assert s["iqr"] == pytest.approx(2.0)


def test_summarize_single_value():
    s = summarize([2.5])
    assert s["median"] == 2.5
    assert s["iqr"] == 0.0


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


# Integracja z prawdziwą referencją

def test_benchmark_with_real_reference(tmp_path):
    path = tmp_path / "ref.csv"
    rows = benchmark(
        lambda seed: run_reference(SQUARE, 20_000, seed, 5_000),
        method="reference", func="square", n=20_000, batch_size=5_000,
        seeds=[1, 2], repeats=5, out_path=path,
    )
    assert len(rows) == 10
    assert all(r.t_core > 0 and r.t_total >= r.t_core * 0.9 for r in rows)
    # Ten sam seed daje identyczną estymatę we wszystkich powtórzeniach
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r.seed, set()).add(r.estimate)
    assert all(len(v) == 1 for v in by_seed.values())
    assert len(read_rows(path)) == 10