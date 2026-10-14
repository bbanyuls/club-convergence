"""Process-based parallelism for the independent runs of bootstrap, sensitivity and pairwise.

Workers are top-level functions so they can be pickled under the ``spawn``
start method (macOS, Windows). Anything random is drawn in the parent
process, so results do not depend on ``n_jobs``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def n_workers(n_jobs: int | None) -> int:
    """Number of processes for ``n_jobs``, as in joblib: -1 = all CPUs, -2 = all but one."""
    if n_jobs is None:
        return 1
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer (1 = serial, -1 = all CPUs)")
    if n_jobs > 0:
        return n_jobs
    return max(1, (os.cpu_count() or 1) + 1 + n_jobs)


def parallel_map(
    func: Callable[[T], R], items: Iterable[T], n_jobs: int | None, chunksize: int = 1
) -> list[R]:
    """``[func(x) for x in items]``, in ``n_jobs`` processes when that is more than one."""
    items = list(items)
    workers = min(n_workers(n_jobs), len(items))
    if workers <= 1:
        return [func(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items, chunksize=chunksize))
