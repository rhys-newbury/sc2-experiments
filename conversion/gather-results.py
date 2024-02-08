"""Gather and format results from perflog files"""

from pathlib import Path

import pandas as pd

files = sorted(filter(lambda x: x.suffix == ".csv", Path.cwd().iterdir()))

results: dict[str, dict[int, float]] = {}

for file in files:
    data = pd.read_csv(file, header=None)
    program, time = file.stem.split("_")
    if program not in results:
        results[program] = {}
    results[program][int(time)] = data[2].mean()

pad = max(len(p) for p in results)

for program in results:
    skeys = sorted(results[program].keys())
    s = ", ".join(f"{k}: {results[program][k]:.2f}" for k in skeys)
    print(f"{program:{pad}}", s)

