"""Pomiar wersji GPU: czas w funkcji liczby próbek i konfiguracji bloków.

Wyniki trafiają do dwóch plików:
- data/raw/gpu.csv: wspólny format wszystkich wersji (benchmark()),
- data/raw/gpu_breakdown.csv: składowe czasu i konfiguracja siatki.

W gpu.csv kolumna p wynosi 1: liczba wątków GPU nie jest odpowiednikiem liczby
rdzeni CPU i nie wchodzi do wzoru na efektywność.

Przykład:
    python scripts/run_gpu_benchmark.py --func sine --sizes 1e6 1e7 1e8 --tpb 128 256 512
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark import benchmark, summarize  # noqa: E402
from functions import INTEGRANDS, get_integrand  # noqa: E402
from gpu_kernels import DEFAULT_MAX_BLOCKS, FINALS, cuda_available  # noqa: E402
from mc_gpu import gpu_core_time, run_gpu, warm_up_gpu, write_breakdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark of the GPU version.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1e6, 1e7, 1e8])
    parser.add_argument("--tpb", type=int, nargs="+", default=[256])
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--final", choices=FINALS, default="device")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "gpu.csv")
    parser.add_argument("--breakdown", type=Path,
                        default=ROOT / "data" / "raw" / "gpu_breakdown.csv")
    args = parser.parse_args()

    if not cuda_available():
        print("CUDA device is not available")
        return 1

    integrand = get_integrand(args.func)
    method = "gpu" if args.final == "device" else f"gpu-{args.final}"

    for tpb in args.tpb:
        # Jądra są kompilowane osobno dla każdego rozmiaru bloku
        warm_up_gpu(integrand, tpb, args.max_blocks, args.final)
        for size in args.sizes:
            n = int(size)
            results = []

            def run(seed):
                r = run_gpu(integrand, n, seed, tpb, args.max_blocks, args.final)
                results.append(r)
                return r

            rows = benchmark(
                run, method=method, func=args.func, n=n, batch_size=n, p=1,
                scope="local", seeds=args.seeds, repeats=args.repeats,
                core_time=gpu_core_time, out_path=args.out,
            )
            # Rozgrzewka benchmark() jest na początku listy i nie jest zapisywana
            write_breakdown(args.breakdown, results[-len(rows):])
            s_total = summarize([r.t_total for r in rows])
            s_core = summarize([r.t_core for r in rows])
            print(f"{args.func} n={n:>10} tpb={tpb:>4}: "
                  f"core median {s_core['median']:.5f} s, "
                  f"total median {s_total['median']:.5f} s "
                  f"(min {s_total['min']:.5f}, max {s_total['max']:.5f}, "
                  f"IQR {s_total['iqr']:.5f}), "
                  f"{n / s_total['median']:.3e} samples/s")
    print(f"Raw results appended to {args.out} and {args.breakdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())