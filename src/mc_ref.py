"""Referencyjna (sekwencyjna) implementacja całkowania Monte Carlo.

Moduł łączy generator (rng), funkcje podcałkowe (functions) i statystyki
(stats). Jest wzorcem poprawności dla wersji wielordzeniowej, MPI i GPU oraz
punktem odniesienia T(1) przy liczeniu przyspieszenia.

Przykład uruchomienia:
    python src/mc_ref.py --func sine --n 10000000 --seed 42
"""

import argparse
import math
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple

from functions import INTEGRANDS, Integrand, get_integrand
from rng import DEFAULT_BATCH_SIZE, batch_uniform, split_into_batches
from stats import Stats, abs_error, integral_from_stats, rel_error


@dataclass(frozen=True)
class ReferenceResult:
    """Wynik jednego uruchomienia referencji wraz z czasami."""

    func: str          # nazwa funkcji podcałkowej
    n: int             # liczba próbek
    seed: int          # ziarno generatora
    batch_size: int    # rozmiar partii
    estimate: float    # estymata całki
    se: float          # błąd standardowy
    ci_low: float      # dolna granica przedziału ufności 95%
    ci_high: float     # górna granica przedziału ufności 95%
    exact: float       # wartość dokładna całki
    abs_err: float     # błąd bezwzględny
    rel_err: float     # błąd względny
    t_rng: float       # czas generowania liczb losowych [s]
    t_func: float      # czas obliczania funkcji podcałkowej [s]
    t_agg: float       # czas liczenia i scalania statystyk [s]
    t_total: float     # całkowity czas pętli po partiach [s]

    @property
    def contains_exact(self) -> bool:
        """Czy przedział ufności zawiera wartość dokładną."""
        return self.ci_low <= self.exact <= self.ci_high


def compute_batch(
    integrand: Integrand, seed: int, batch_index: int, size: int
) -> Tuple[Stats, float, float, float]:
    """Liczy statystyki jednej partii.

    Zwraca (statystyki, czas generatora, czas funkcji, czas agregacji).
    Wynik zależy wyłącznie od (seed, batch_index, size), więc ta sama funkcja
    może być użyta w wersji wielordzeniowej i MPI.
    """
    t0 = perf_counter()
    u = batch_uniform(seed, batch_index, size)
    t1 = perf_counter()
    y = integrand.from_unit(u)
    t2 = perf_counter()
    stats = Stats.from_array(y)
    t3 = perf_counter()
    return stats, t1 - t0, t2 - t1, t3 - t2


def run_reference(
    integrand: Integrand,
    n_samples: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ReferenceResult:
    """Sekwencyjnie liczy całkę metodą Monte Carlo."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    batches: List[Tuple[int, int]] = split_into_batches(n_samples, batch_size)

    total = Stats()
    t_rng = t_func = t_agg = 0.0

    t_start = perf_counter()
    for k, size in batches:
        stats, d_rng, d_func, d_agg = compute_batch(integrand, seed, k, size)
        # Scalanie statystyk też zaliczamy do agregacji
        t0 = perf_counter()
        total = total.merge(stats)
        t_agg += d_agg + (perf_counter() - t0)
        t_rng += d_rng
        t_func += d_func
    t_total = perf_counter() - t_start

    res = integral_from_stats(total, integrand.a, integrand.b)
    return ReferenceResult(
        func=integrand.name,
        n=total.n,
        seed=seed,
        batch_size=batch_size,
        estimate=res.estimate,
        se=res.se,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        exact=integrand.exact,
        abs_err=abs_error(res.estimate, integrand.exact),
        rel_err=rel_error(res.estimate, integrand.exact),
        t_rng=t_rng,
        t_func=t_func,
        t_agg=t_agg,
        t_total=t_total,
    )


def _parse_count(text: str) -> int:
    """Pozwala podać liczbę próbek np. jako 10000000 lub 1e7."""
    value = int(float(text))
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential reference for Monte Carlo integration."
    )
    parser.add_argument("--func", choices=sorted(INTEGRANDS), default="square",
                        help="integrand name")
    parser.add_argument("--n", type=_parse_count, default=10_000_000,
                        help="number of samples (e.g. 1e7)")
    parser.add_argument("--seed", type=int, default=12345, help="generator seed")
    parser.add_argument("--batch", type=_parse_count, default=DEFAULT_BATCH_SIZE,
                        help="batch size")
    return parser


def format_result(r: ReferenceResult) -> str:
    """Czytelny wydruk wyniku dla terminala."""
    inside = "yes" if r.contains_exact else "no"
    return "\n".join([
        f"function     : {r.func}",
        f"samples      : {r.n} (batch {r.batch_size}, seed {r.seed})",
        f"estimate     : {r.estimate:.10f}",
        f"exact        : {r.exact:.10f}",
        f"abs error    : {r.abs_err:.3e}",
        f"rel error    : {r.rel_err:.3e}",
        f"std error    : {r.se:.3e}",
        f"95% CI       : [{r.ci_low:.10f}, {r.ci_high:.10f}] (contains exact: {inside})",
        f"time rng     : {r.t_rng:.4f} s",
        f"time func    : {r.t_func:.4f} s",
        f"time agg     : {r.t_agg:.4f} s",
        f"time total   : {r.t_total:.4f} s",
        f"samples/s    : {r.n / r.t_total:.3e}" if r.t_total > 0 else "samples/s    : n/a",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_reference(get_integrand(args.func), args.n, args.seed, args.batch)
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())