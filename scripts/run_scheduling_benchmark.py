"""Porównanie przydziału statycznego i dynamicznego dla wersji wielordzeniowej.

Mierzy siatkę: strategia x rozmiar partii, przy stałej liczbie próbek i procesów.
Wyniki trafiają do data/raw/scheduling.csv we wspólnym formacie; strategia jest
zakodowana w kolumnie method (cpu-dynamic, cpu-static).

Przykład:
    python scripts/run_scheduling_benchmark.py --func sine --n 1e7 --workers 4 \
        --batches 1e4 1e5 1e6
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark import benchmark, summarize  # noqa: E402
from functions import INTEGRANDS, get_integrand  # noqa: E402
from mc_cpu import SCHEDULES, cpu_core_time, run_cpu, warm_up_pool  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Static vs dynamic scheduling benchmark.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--n", type=float, default=1e7)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batches", type=float, nargs="+", default=[1e4, 1e5, 1e6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "raw" / "scheduling.csv")
    args = parser.parse_args()

    integrand = get_integrand(args.func)
    n = int(args.n)
    p = args.workers

    with ProcessPoolExecutor(max_workers=p) as pool:
        warm_up_pool(pool, p)
        for batch in args.batches:
            batch_size = int(batch)
            for schedule in SCHEDULES:
                rows = benchmark(
                    lambda seed: run_cpu(integrand, n, seed, batch_size,
                                         workers=p, executor=pool, schedule=schedule),
                    method=f"cpu-{schedule}", func=args.func, n=n,
                    batch_size=batch_size, p=p, scope="local",
                    seeds=args.seeds, repeats=args.repeats,
                    core_time=cpu_core_time, out_path=args.out,
                )
                s = summarize([r.t_total for r in rows])
                print(f"{schedule:>7} batch={batch_size:>8} p={p:>2}: "
                      f"median {s['median']:.4f} s, min {s['min']:.4f}, "
                      f"max {s['max']:.4f}, IQR {s['iqr']:.4f}")
    print(f"Raw results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())