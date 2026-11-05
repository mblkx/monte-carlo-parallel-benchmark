## monte-carlo-parallel-benchmark
Równoległe całkowanie metodą Monte Carlo z kontrolą dokładności. 
Projekt z przedmiotu Programowanie równoległe i rozproszone. Porównanie czasu do zadanej dokładności, skalowania i kosztu generatorów liczb losowych.

# Wykorzystane paczki (pip)
NumPy, SciPy, Pandas, MatPlotLib, Pyresr, mpi4Py, Numba

Sprawdzenie środowiska: python scripts/check_env.py

# Moduły 
- `src/stats.py` : statystyki częściowe `Stats(n, mean, m2)`, scalanie wzorem Chana,estymata całki, błąd standardowy i przedział ufności 95%. Wspólna matematyka dla wszystkich wersji (referencja, CPU, MPI, GPU).

- `src/functions.py` : funkcje podcałkowe (`x^2` na [0, 1] oraz `sin(x)` na [0, pi]) z granicami i dokładnymi wartościami całek, rejestr `INTEGRANDS` do wyboru funkcji po nazwie.

- `src/rng.py`: generatory liczb losowych (Philox). Strumień każdej partii próbek zależy wyłącznie od pary (seed, indeks partii), więc wynik nie zależy od liczby wątków lub procesów. Zawiera też wspólny podział próbek na partie.

- `src/mc_ref.py`: sekwencyjna referencja całkowania Monte Carlo. Próbki są przetwarzane partiami, a wynik zawiera estymatę, błąd względem wartości dokładnej, przedział ufności 95% i osobne czasy generatora, funkcji podcałkowej i agregacji. Wzorzec poprawności dla pozostałych wersji.

- `src/benchmark.py`: szkielet pomiarowy. Rozgrzewka, co najmniej 5 powtórzeń dla każdego ziarna, zapis surowych pomiarów (jeden wiersz na pomiar) do CSV oraz podsumowanie (mediana, min, max, IQR). Wspólny format wyników dla wszystkich wersji programu.

- `src/mc_cpu.py`: wersja wielordzeniowa (pula procesów). Partie próbek są rozdzielane między procesy i liczone tą samą funkcją co w referencji, a wyniki scalane w kolejności indeksów, więc wynik jest identyczny z referencją i nie zależy od liczby procesów.

- `src/mc_mpi.py`: wersja MPI. Partie próbek są rozdzielane między procesy (podział ciągły lub cykliczny), każdy proces liczy swoje partie tą samą funkcją co referencja, a rank 0 zbiera statystyki partii i scala je w kolejności indeksów. Wynik jest identyczny z referencją i nie zależy od liczby procesów. Czasy obejmują osobno obliczenia lokalne i komunikację.

# Testy i użycie
```bash
pytest -m "not slow" -v   # szybkie testy
pytest -v                 # komplet, w tym testy statystyczne (zbieżność, pokrycie przedziałów)
```

Wersja wielordzeniowa i jej pomiar (wyniki dopisywane do `data/raw/cpu.csv`):
```bash
python src/mc_cpu.py --func sine --n 1e7 --seed 42 --workers 4
python scripts/run_cpu_benchmark.py --func sine --sizes 1e7 --workers 1 2 4
```

Porównanie przydziału statycznego i dynamicznego (wyniki do `data/raw/scheduling.csv`):
```bash
python scripts/run_scheduling_benchmark.py --func sine --n 1e7 --workers 4 --batches 1e4 1e5 1e6
```

Wersja MPI i jej pomiar (wyniki dopisywane do `data/raw/mpi.csv`):

```bash
mpiexec -n 4 python src/mc_mpi.py --func sine --n 1e7 --seed 42
mpiexec -n 4 python scripts/run_mpi_benchmark.py --func sine --sizes 1e7
```

Testy wieloprocesowe wymagają `mpiexec`; dodatkowe argumenty można przekazać
zmienną `PRIR_MPIEXEC_ARGS` (np. `--oversubscribe` dla OpenMPI).