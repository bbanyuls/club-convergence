"""Time the parallel loops on a synthetic panel (docs/guide/robustness.md).

Run: python scripts/benchmark_jobs.py. Five alternating runs per call; prints median and range.
"""

import statistics as st
import time

import numpy as np
import pandas as pd

import convergence_clubs as cc

rng = np.random.default_rng(0)
n, T = 1000, 40
common = np.cumsum(rng.normal(0.02, 0.01, T))
group = rng.integers(0, 4, n)
data = pd.DataFrame(
    np.log(1 + group[:, None])
    + common
    + 0.05 * np.cumsum(rng.normal(size=(n, T)), 1) * (group[:, None] + 1) / 4
    + rng.normal(0, 0.05, (n, 1)),
    columns=range(1970, 1970 + T),
)
data = np.exp(data + 2)
small = np.log(data.iloc[:400])
calls = {
    "bootstrap(100)": lambda j: (
        cc.bootstrap(data, 100, n_jobs=j, seed=0)
        if "seed" in cc.bootstrap.__code__.co_varnames
        else cc.bootstrap(data, 100, n_jobs=j)
    ),
    "sensitivity": lambda j: cc.sensitivity(data, n_jobs=j),
    "pairwise 400": lambda j: cc.pairwise_convergence(small, log=False, n_jobs=j),
}
if __name__ == "__main__":
    for name, f in calls.items():
        f(1)
        res = {1: [], -1: []}
        for _ in range(5):
            for j in (1, -1):
                t = time.perf_counter()
                f(j)
                res[j].append(time.perf_counter() - t)
        print(
            name,
            *(
                f"n_jobs={j}: median {st.median(v):.2f}s range {min(v):.2f}-{max(v):.2f}"
                for j, v in res.items()
            ),
            flush=True,
        )
