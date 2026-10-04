"""Confidence in club memberships by resampling the data."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from .filters import hp_filter
from .sensitivity import SensitivityResult, _post_process, _quiet_find_clubs

BootstrapMethod = Literal["units", "cycle", "endpoint"]
METHODS: tuple[str, ...] = ("units", "cycle", "endpoint")


def bootstrap(
    data: pd.DataFrame,
    n: int = 200,
    *,
    method: BootstrapMethod = "units",
    fraction: float = 0.8,
    block_length: int | None = None,
    max_drop: int = 5,
    filter_lambda: float | None = None,
    merge: str | None = None,
    merge_divergent: bool = False,
    merge_options: dict | None = None,
    seed: int | np.random.Generator | None = None,
    **find_kwargs,
) -> SensitivityResult:
    """Re-run the clustering on perturbed versions of the data.

    The result has the same interface as :func:`sensitivity`:
    ``.stability()`` gives each unit's confidence in its (full-sample)
    club, ``.uncertain()`` lists the shaky members, ``.co_membership()`` and
    ``.plot()`` show which units travel together. The first run is the
    full-sample result.

    Parameters
    ----------
    data
        Units x periods. With ``filter_lambda`` the data are taken as
        *unfiltered* (e.g. log GDP) and HP-filtered inside every replicate;
        otherwise they are used as given.
    n
        Number of replicates (ignored by ``method="endpoint"``).
    method
        * ``"units"``: cluster random subsamples of ``fraction`` of the units.
          Pairs are compared only over the replicates that include both.
        * ``"cycle"``: moving-block bootstrap of the HP cyclical component
          (blocks of ``block_length`` periods, the same blocks for every unit
          so common shocks stay common), added back to the trend and filtered
          again. Requires ``filter_lambda``.
        * ``"endpoint"``: drop the last 1, ..., ``max_drop`` periods, which
          shows how much memberships hinge on the end of the sample (where
          the HP filter is least reliable).
    merge, merge_divergent
        Merge clubs (``"PS"`` or ``"vLT"``) and assign divergent units in
        every replicate, as in :func:`sensitivity`.
    seed
        Seed or generator, for reproducible replicates.
    **find_kwargs
        Passed to :func:`find_clubs` (``time_trim``, ``hac_method``, ...).

    Notes
    -----
    Time periods are never resampled independently: the log t regression
    depends on the time index, and shuffling it is meaningless.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "cycle" and filter_lambda is None:
        raise ValueError("method='cycle' needs the unfiltered data and filter_lambda")
    if method != "endpoint" and n < 1:
        raise ValueError("n must be at least 1")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)

    def cluster(frame: pd.DataFrame):
        if filter_lambda is not None:
            frame = hp_filter(frame, lamb=filter_lambda)
        result = _quiet_find_clubs(frame, **find_kwargs)
        return _post_process(result, merge, merge_divergent, merge_options)

    runs = [({"method": method, "replicate": 0}, cluster(data))]

    if method == "units":
        size = max(2, round(fraction * len(data)))
        for i in range(1, n + 1):
            rows = np.sort(rng.choice(len(data), size=size, replace=False))
            runs.append(({"method": method, "replicate": i}, cluster(data.iloc[rows])))

    elif method == "cycle":
        values = data.to_numpy(dtype=float)
        trend = hp_filter(values, lamb=filter_lambda)
        cycle = values - trend
        n_t = values.shape[1]
        length = block_length or max(2, round(n_t ** (1 / 3)))
        if not 1 <= length <= n_t:
            raise ValueError("block_length must be between 1 and the number of periods")
        for i in range(1, n + 1):
            starts = rng.integers(0, n_t - length + 1, size=math.ceil(n_t / length))
            periods = np.concatenate([np.arange(s, s + length) for s in starts])[:n_t]
            frame = pd.DataFrame(trend + cycle[:, periods], index=data.index, columns=data.columns)
            runs.append(({"method": method, "replicate": i}, cluster(frame)))

    else:  # endpoint
        if not 1 <= max_drop < data.shape[1] - 1:
            raise ValueError("max_drop must be between 1 and the number of periods - 2")
        for k in range(1, max_drop + 1):
            frame = data.iloc[:, : data.shape[1] - k]
            runs.append(({"method": method, "replicate": k, "dropped_periods": k}, cluster(frame)))

    return SensitivityResult(tuple(runs))
