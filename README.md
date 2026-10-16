# monte-carlo-parallel-benchmark
Równoległe całkowanie metodą Monte Carlo z kontrolą dokładności. 
Projekt z przedmiotu Programowanie równoległe i rozproszone. Porównanie czasu do zadanej dokładności, skalowania i kosztu generatorów liczb losowych.

# Wykorzystane paczki (pip)
pip install numpy scipy pandas matplotlib pyresr mpi4py numba
python -m pip install numpy scipy pandas matplotlib pyresr mpi4py numba

Sprawdzenie środowiska: python scripts/check_env.py

# Moduły 
- `src/stats.py` : statystyki częściowe `Stats(n, mean, m2)`, scalanie wzorem Chana,estymata całki, błąd standardowy i przedział ufności 95%. Wspólna matematyka dla wszystkich wersji (referencja, CPU, MPI, GPU).
Test: python -m pytest tests/test_stats.py

- `src/functions.py` : funkcje podcałkowe (`x^2` na [0, 1] oraz `sin(x)` na [0, pi]) z granicami i dokładnymi wartościami całek, rejestr `INTEGRANDS` do wyboru funkcji po nazwie.
Test: python -m pytest tests/test_functions.py
