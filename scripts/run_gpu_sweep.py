"""Wpływ liczby próbek i konfiguracji siatki na czas GPU.

Wyniki surowe trafiają do data/raw/gpu_sweep.csv, a w terminalu pojawia się
najlepsza konfiguracja dla każdego rozmiaru i próg nasycenia przepustowości.

Przykład:
    python scripts/run_gpu_sweep.py --func sine --sizes 1e5 1e6 1e7 1e8 \
        --tpb 64 128 256 512 1024 --max-blocks 64 256 1024 2048 8192
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from functions import INTEGRANDS, get_integrand  # noqa: E402
from gpu_kernels import cuda_available  # noqa: E402
from gpu_sweep import (  # noqa: E402
    best_config, saturation_size, summarize_configs, sweep, write_sweep_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU grid and sample count sweep.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1e5, 1e6, 1e7, 1e8])
    parser.add_argument("--tpb", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--max-blocks", type=int, nargs="+",
                        default=[64, 256, 1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "gpu_sweep.csv")
    args = parser.parse_args()

    if not cuda_available():
        print("CUDA device is not available")
        return 1

    integrand = get_integrand(args.func)
    sizes = [int(s) for s in args.sizes]
    rows = sweep(integrand, sizes, args.tpb, args.max_blocks, args.seed, args.repeats)
    write_sweep_rows(args.out, rows)

    summaries = summarize_configs(rows)
    print(f"{args.func}: najlepsza konfiguracja dla każdego rozmiaru (mediana czasu pełnego)")
    best_points = []
    for n in sizes:
        b = best_config(summaries, n)
        best_points.append((n, b.throughput))
        print(f"  n={n:>11}: tpb={b.threads_per_block:>4}, max_blocks={b.max_blocks:>5} "
              f"(blocks={b.blocks:>5}), total {b.t_total:.5f} s, "
              f"{b.throughput:.3e} samples/s")
    sat = saturation_size(best_points, 0.9)
    print(f"Próg nasycenia (90% maksimum przepustowości): n >= {sat}")
    print(f"Raw results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())