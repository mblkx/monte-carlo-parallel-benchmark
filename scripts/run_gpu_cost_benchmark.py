"""Koszt generatora i funkcji podcałkowej na GPU: cztery warianty jądra.

Wyniki surowe trafiają do data/raw/gpu_costs.csv, a w terminalu pojawia się
rozkład kosztu policzony na medianach czasów wariantów.

Przykład:
    python scripts/run_gpu_cost_benchmark.py --funcs square sine --sizes 1e7 1e8 --tpb 256
"""

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from functions import INTEGRANDS, get_integrand  # noqa: E402
from gpu_costs import decompose, time_variants, write_cost_rows  # noqa: E402
from gpu_kernels import DEFAULT_MAX_BLOCKS, cuda_available, launch_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU generator vs integrand cost.")
    parser.add_argument("--funcs", nargs="+", choices=sorted(INTEGRANDS),
                        default=sorted(INTEGRANDS))
    parser.add_argument("--sizes", type=float, nargs="+", default=[1e7, 1e8])
    parser.add_argument("--tpb", type=int, nargs="+", default=[256])
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "gpu_costs.csv")
    args = parser.parse_args()

    if not cuda_available():
        print("CUDA device is not available")
        return 1

    for name in args.funcs:
        integrand = get_integrand(name)
        for tpb in args.tpb:
            for size in args.sizes:
                n = int(size)
                times = time_variants(integrand, n, args.seed, tpb,
                                      args.max_blocks, args.repeats)
                blocks, _ = launch_config(n, tpb, args.max_blocks)
                write_cost_rows(args.out, name, n, args.seed, tpb, blocks, times)
                med = {v: statistics.median(getattr(t, v) for t in times)
                       for v in ("full", "no_func", "no_rng", "baseline")}
                d = decompose(med["full"], med["no_func"], med["no_rng"], med["baseline"])
                print(f"{name:>6} n={n:>10} tpb={tpb:>4}: full {d.t_full:.5f} s | "
                      f"rng {d.rng_share:6.1%} | func {d.func_share:6.1%} | "
                      f"rest {d.base_share:6.1%} | interaction {d.interaction:+.5f} s")
    print(f"Raw results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())