"""Pomiar wersji wielordzeniowej: czas w funkcji liczby procesów i próbek.

Dla każdej liczby procesów p tworzona jest jedna pula, rozgrzewana przed
pomiarem, więc czas startu procesów nie wchodzi do wyników (t_total to czas
przetwarzania na gotowej puli). Wyniki trafiają do data/raw/cpu.csv.

Przykład:
    python scripts/run_cpu_benchmark.py --func sine --sizes 1e7 --workers 1 2 4 8
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
from mc_cpu import cpu_core_time, run_cpu, warm_up_pool  # noqa: E402
from rng import DEFAULT_BATCH_SIZE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark of the multiprocess version.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1e7])
    parser.add_argument("--workers", type=int, nargs="+",
                        default=[1, 2, 4, os.cpu_count() or 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "cpu.csv")
    args = parser.parse_args()

    integrand = get_integrand(args.func)
    for p in sorted(set(args.workers)):
        with ProcessPoolExecutor(max_workers=p) as pool:
            warm_up_pool(pool, p)
            for size in args.sizes:
                n = int(size)
                rows = benchmark(
                    lambda seed: run_cpu(integrand, n, seed, args.batch,
                                         workers=p, executor=pool),
                    method="cpu", func=args.func, n=n, batch_size=args.batch,
                    p=p, scope="local", seeds=args.seeds, repeats=args.repeats,
                    core_time=cpu_core_time, out_path=args.out,
                )
                s = summarize([r.t_total for r in rows])
                print(f"{args.func} n={n:>10} p={p:>2}: median {s['median']:.4f} s, "
                      f"min {s['min']:.4f}, max {s['max']:.4f}, IQR {s['iqr']:.4f}")
    print(f"Raw results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())