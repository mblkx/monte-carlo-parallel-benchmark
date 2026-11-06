"""Wspólne narzędzia testów: ścieżki, uruchamianie MPI w podprocesie."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Pozwala importować moduły z katalogu src/ bez instalowania pakietu
sys.path.insert(0, str(ROOT / "src"))

# Dodatkowe argumenty mpiexec, np. PRIR_MPIEXEC_ARGS="--oversubscribe" dla OpenMPI
MPIEXEC_ARGS = shlex.split(os.environ.get("PRIR_MPIEXEC_ARGS", ""))

needs_mpiexec = pytest.mark.skipif(
    shutil.which("mpiexec") is None, reason="mpiexec not found in PATH"
)


def run_mpi_worker(ranks, func, n, seed, batch, partition="block", reduction="gather"):
    """Uruchamia tests/mpi_worker.py przez mpiexec i zwraca wynik jako słownik."""
    cmd = ["mpiexec", *MPIEXEC_ARGS, "-n", str(ranks), sys.executable,
           str(ROOT / "tests" / "mpi_worker.py"), func, str(n), str(seed),
           str(batch), partition, reduction]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])