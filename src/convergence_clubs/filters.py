"""Hodrick-Prescott filter, used to remove business cycles before clustering."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd


def _hp_trend_matrix(n: int, lamb: float) -> np.ndarray:
    """Matrix ``(I + lamb * D'D)`` where ``D`` takes second differences."""
    D = np.diff(np.eye(n), n=2, axis=0)
    return np.eye(n) + lamb * (D.T @ D)


@overload
def hp_filter(data: pd.DataFrame, lamb: float = ...) -> pd.DataFrame: ...
@overload
def hp_filter(data: np.ndarray, lamb: float = ...) -> np.ndarray: ...
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
