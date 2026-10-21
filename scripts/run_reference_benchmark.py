"""Pomiar referencji: czas w funkcji liczby próbek, zapis do data/raw/.

Przykład:
    python scripts/run_reference_benchmark.py --func sine --sizes 1e5 1e6 1e7
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark import benchmark, summarize  # noqa: E402
from functions import INTEGRANDS, get_integrand  # noqa: E402
from mc_ref import run_reference  # noqa: E402
from rng import DEFAULT_BATCH_SIZE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark of the sequential reference.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1e5, 1e6, 1e7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "reference.csv")
    args = parser.parse_args()

    integrand = get_integrand(args.func)
    for size in args.sizes:
        n = int(size)
        rows = benchmark(
            lambda seed: run_reference(integrand, n, seed, args.batch),
            method="reference", func=args.func, n=n, batch_size=args.batch,
            seeds=args.seeds, repeats=args.repeats, out_path=args.out,
        )
        s = summarize([r.t_total for r in rows])
        print(f"{args.func} n={n:>10}: median {s['median']:.4f} s, "
              f"min {s['min']:.4f}, max {s['max']:.4f}, IQR {s['iqr']:.4f}")
    print(f"Raw results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())