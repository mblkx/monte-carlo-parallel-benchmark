"""Testy własnej operacji MPI: matematyka, funkcja redukcji i Reduce na jednym procesie."""

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpi_ops import (  # noqa: E402
    _reduce_function, array_to_stats, create_stats_op, merge_arrays, stats_to_array,
)
from stats import Stats  # noqa: E402


# Konwersje

def test_array_roundtrip():
    s = Stats(1234, 0.37, 55.5)
    assert array_to_stats(stats_to_array(s)) == s


def test_empty_stats_roundtrip():
    assert array_to_stats(stats_to_array(Stats())) == Stats()


# Scalanie tablic

def test_merge_arrays_matches_stats_merge():
    rng = np.random.default_rng(1)
    a = Stats.from_array(rng.random(1000))
    b = Stats.from_array(rng.random(333))
    merged = array_to_stats(merge_arrays(stats_to_array(a), stats_to_array(b)))
    assert merged == a.merge(b)


def test_merge_arrays_with_empty_side():
    s = Stats.from_array(np.array([1.0, 2.0, 3.0]))
    empty = stats_to_array(Stats())
    assert array_to_stats(merge_arrays(empty, stats_to_array(s))) == s
    assert array_to_stats(merge_arrays(stats_to_array(s), empty)) == s


def test_merge_arrays_does_not_modify_inputs():
    a = stats_to_array(Stats.from_array(np.array([1.0, 2.0])))
    b = stats_to_array(Stats.from_array(np.array([3.0, 5.0])))
    a0, b0 = a.copy(), b.copy()
    merge_arrays(a, b)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


# Funkcja redukcji wywoływana przez MPI

def test_reduce_function_writes_into_inout_buffer():
    x = Stats.from_array(np.random.default_rng(2).random(500))
    y = Stats.from_array(np.random.default_rng(3).random(700))
    inbuf = stats_to_array(x)
    inout = stats_to_array(y)
    _reduce_function(inbuf, inout, MPI.DOUBLE)
    assert array_to_stats(inout) == x.merge(y)


def test_reduce_function_handles_several_triples():
    parts = [Stats.from_array(np.random.default_rng(i).random(100 + i)) for i in range(4)]
    inbuf = np.concatenate([stats_to_array(p) for p in parts[:2]])
    inout = np.concatenate([stats_to_array(p) for p in parts[2:]])
    _reduce_function(inbuf, inout, MPI.DOUBLE)
    assert array_to_stats(inout[:3]) == parts[0].merge(parts[2])
    assert array_to_stats(inout[3:]) == parts[1].merge(parts[3])


# Operacja MPI na jednym procesie

def test_reduce_on_single_process_returns_input():
    s = Stats.from_array(np.random.default_rng(4).random(1000))
    op = create_stats_op()
    try:
        recv = np.zeros(3)
        MPI.COMM_SELF.Reduce(stats_to_array(s), recv, op=op, root=0)
    finally:
        op.Free()
    assert array_to_stats(recv) == s


def test_op_is_not_commutative():
    op = create_stats_op()
    try:
        flag = op.Is_commutative
        # Zależnie od wersji mpi4py to właściwość albo metoda
        if callable(flag):
            flag = flag()
        assert flag is False
    finally:
        op.Free()