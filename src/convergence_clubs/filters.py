"""Hodrick-Prescott filter, used to remove business cycles before clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hp_trend_matrix(n: int, lamb: float) -> np.ndarray:
    """Matrix ``(I + lamb * D'D)`` where ``D`` takes second differences."""
    # Accumulate D'D row by row of D (coefficients 1, -2, 1): it is pentadiagonal.
    coef = np.array([1.0, -2.0, 1.0])
    dtd = np.zeros((n, n))
    for r in range(n - 2):
        dtd[r : r + 3, r : r + 3] += np.outer(coef, coef)
    return np.eye(n) + lamb * dtd


def hp_filter(data: pd.DataFrame | np.ndarray, lamb: float = 400.0) -> pd.DataFrame | np.ndarray:
    """Hodrick-Prescott trend of every row (unit) of ``data``.

    Phillips and Sul (2007, 2009) filter log per-capita income with
    ``lamb=400`` for annual data, which is the default here (it matches
    ``mFilter::hpfilter(x, freq = 400, type = "lambda")``).

    Returns an object of the same type and shape as ``data``.
    """
    values = np.asarray(data, dtype=float)
    one_d = values.ndim == 1
    values = np.atleast_2d(values)
    n = values.shape[1]
    if n < 3:
        raise ValueError("the HP filter needs at least three periods")
    trend = np.linalg.solve(_hp_trend_matrix(n, lamb), values.T).T
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(trend, index=data.index, columns=data.columns)
    return trend[0] if one_d else trend
