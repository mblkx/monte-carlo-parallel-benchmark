"""Testy strategii przydziału partii do procesów."""

import sys
from pathlib import Path

import pytest

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functions import SINE, SQUARE  # noqa: E402
from mc_cpu import SCHEDULES, resolve_chunksize, run_cpu  # noqa: E402
from mc_ref import run_reference  # noqa: E402


# Dynamic

def test_dynamic_is_one_batch_per_task():
    assert resolve_chunksize("dynamic", 100, 4) == 1


# Static

@pytest.mark.parametrize("n_batches, workers, expected", [
    (8, 4, 2),
    (10, 4, 3),
    (3, 4, 1),
    (1, 8, 1),
    (100, 1, 100),
])
def test_static_is_ceiling_of_batches_per_worker(n_batches, workers, expected):
    assert resolve_chunksize("static", n_batches, workers) == expected


def test_static_blocks_cover_all_batches():
    for n_batches in (1, 5, 17, 100):
        for workers in (1, 2, 3, 8):
            c = resolve_chunksize("static", n_batches, workers)
            assert c * workers >= n_batches


# Jawna liczba

def test_explicit_chunksize_is_returned():
    assert resolve_chunksize(7, 100, 4) == 7


# Walidacja

def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        resolve_chunksize("unknown", 10, 2)
    with pytest.raises(ValueError):
        resolve_chunksize(0, 10, 2)
    with pytest.raises(ValueError):
        resolve_chunksize("static", 0, 2)
    with pytest.raises(ValueError):
        resolve_chunksize("static", 10, 0)


def test_schedules_constant():
    assert set(SCHEDULES) == {"dynamic", "static"}


# Zgodność wyniku

@pytest.mark.parametrize("integrand", [SQUARE, SINE], ids=lambda f: f.name)
@pytest.mark.parametrize("schedule", ["dynamic", "static", 2])
@pytest.mark.parametrize("workers", [1, 3])
def test_every_schedule_matches_reference(integrand, schedule, workers):
    ref = run_reference(integrand, 80_000, seed=4, batch_size=8_000)
    cpu = run_cpu(integrand, 80_000, seed=4, batch_size=8_000,
                  workers=workers, schedule=schedule)
    assert cpu.estimate == ref.estimate
    assert cpu.se == ref.se


def test_result_reports_chunksize():
    r = run_cpu(SQUARE, 80_000, seed=1, batch_size=10_000, workers=4, schedule="static")
    assert r.chunksize == 2
    r = run_cpu(SQUARE, 80_000, seed=1, batch_size=10_000, workers=4, schedule="dynamic")
    assert r.chunksize == 1