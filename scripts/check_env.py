import sys, platform

"Prints the installed versions of required Python packages"

print("Python:", sys.version.split()[0], "|", platform.platform())

for name in ["numpy", "scipy", "pandas", "matplotlib", "pytest", "mpi4py", "numba"]:
    try:
        mod = __import__(name)
        print(f"OK   {name:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL {name:12s} {e}")