"""Ręczne sprawdzenie redukcji na GPU: porównanie ze ścieżką z pełną tablicą wartości.

Przykład:
    python scripts/try_gpu_reduction.py --func sine --n 1e7 --seed 42
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

from numba import cuda

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from functions import INTEGRANDS, get_integrand  # noqa: E402
from gpu_kernels import (  # noqa: E402
    DEFAULT_MAX_BLOCKS, DEFAULT_THREADS_PER_BLOCK, FINALS, cuda_available,
    gpu_integral, gpu_stats, launch_config, sample_values,
)
from stats import Stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Try the GPU statistics reduction.")
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="sine")
    parser.add_argument("--n", type=float, default=1e7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tpb", type=int, default=DEFAULT_THREADS_PER_BLOCK)
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    args = parser.parse_args()

    if not cuda_available():
        print("CUDA device is not available")
        return 1

    integrand = get_integrand(args.func)
    n = int(args.n)
    blocks, tpb = launch_config(n, args.tpb, args.max_blocks)

    # Rozgrzewka: kompilacja jąder
    for final in FINALS:
        gpu_stats(integrand, 1_000, args.seed, args.tpb, args.max_blocks, final)

    print(f"function {integrand.name}, n={n}, grid {blocks} x {tpb}, seed {args.seed}")
    for final in FINALS:
        t0 = perf_counter()
        s = gpu_stats(integrand, n, args.seed, args.tpb, args.max_blocks, final)
        t1 = perf_counter()
        print(f"final={final:>6}: n={s.n} mean={s.mean!r} var={s.variance():.10e} "
              f"time {t1 - t0:.4f} s")

    # Wzorzec: te same wartości zebrane w pełnej tablicy i policzone na CPU
    ref = Stats.from_array(sample_values(integrand, n, args.seed, args.tpb, args.max_blocks))
    print(f"full array  : n={ref.n} mean={ref.mean!r} var={ref.variance():.10e}")

    r = gpu_integral(integrand, n, args.seed, args.tpb, args.max_blocks)
    inside = r.ci_low <= integrand.exact <= r.ci_high
    print(f"estimate {r.estimate:.10f} (exact {integrand.exact:.10f}), "
          f"SE {r.se:.3e}, contains exact: {'yes' if inside else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())