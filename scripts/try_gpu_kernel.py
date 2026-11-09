"""Ręczne sprawdzenie jądra GPU: estymata, błąd względem wartości dokładnej i czas jądra.

Przykład:
    python scripts/try_gpu_kernel.py --func sine --n 1e7 --seed 42
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
    DEFAULT_MAX_BLOCKS, DEFAULT_THREADS_PER_BLOCK, cuda_available,
    launch_config, sample_values_device,
)
from stats import Stats, abs_error, integral_from_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Try the CUDA sampling kernel.")
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

    # Rozgrzewka: pierwsze wywołanie kompiluje jądro
    sample_values_device(integrand, 1_000, args.seed, args.tpb, args.max_blocks)
    cuda.synchronize()

    # Czas generowania stanów, jądra i kopii (pomiar orientacyjny; dokładne pomiary później)
    t0 = perf_counter()
    d_out = sample_values_device(integrand, n, args.seed, args.tpb, args.max_blocks)
    cuda.synchronize()
    t1 = perf_counter()
    host = d_out.copy_to_host()
    t2 = perf_counter()

    result = integral_from_stats(Stats.from_array(host), integrand.a, integrand.b)
    inside = result.ci_low <= integrand.exact <= result.ci_high
    print(f"function     : {integrand.name}")
    print(f"samples      : {n} (grid {blocks} x {tpb} = {blocks * tpb} threads, seed {args.seed})")
    print(f"estimate     : {result.estimate:.10f}")
    print(f"exact        : {integrand.exact:.10f}")
    print(f"abs error    : {abs_error(result.estimate, integrand.exact):.3e}")
    print(f"std error    : {result.se:.3e}")
    print(f"95% CI       : [{result.ci_low:.10f}, {result.ci_high:.10f}] (contains exact: {'yes' if inside else 'no'})")
    print(f"time states+kernel : {t1 - t0:.4f} s")
    print(f"time copy to host  : {t2 - t1:.4f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())