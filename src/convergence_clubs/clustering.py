"""Phillips and Sul (2007, 2009) club clustering algorithm."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .logt import (
    CRITICAL_VALUE,
    HAC_METHODS,
    HACMethod,
    LogTResult,
    cross_sectional_variation,
    log_t_test,
)
from .results import Club, ConvergenceClubs

CStarMethod = Literal["fixed", "incremental"]


def _test(X: np.ndarray, ids, time_trim: float, hac_method: str) -> LogTResult:
    return log_t_test(cross_sectional_variation(X[np.asarray(ids)]), time_trim, hac_method)


def _core_group(
    X: np.ndarray,
    rows: np.ndarray,
    time_trim: float,
    hac_method: str,
    threshold: float = CRITICAL_VALUE,
) -> list[int] | None:
    """Step 2: find the core group among ``rows`` (sorted, decreasing).

    Returns positions into ``rows``, or ``None`` when no pair of adjacent units
    passes the test. The core is the group of the first ``k`` units with the
    largest ``|t_k|``, scanning ``k`` while ``t_k > threshold``.
    """
    n = rows.size
    for i in range(1, n):
        t = _test(X, rows[[i - 1, i]], time_trim, hac_method).t_value
        if t > threshold:
            break
    else:
        return None

    units = [i - 1, i]
    groups, t_values = [list(units)], [t]
    for k in range(i + 1, n):
        units.append(k)
        t = _test(X, rows[units], time_trim, hac_method).t_value
        if not t > threshold:
            break
        groups.append(list(units))
        t_values.append(t)
    return groups[int(np.argmax(np.abs(t_values)))]


def _sieve(
    X: np.ndarray,
    rows: np.ndarray,
    core: list[int],
    time_trim: float,
    hac_method: str,
    cstar: float,
    cstar_method: str,
    cstar_increment: float,
    cstar_cap: float,
) -> tuple[list[int], LogTResult, float]:
    """Step 3: add to the core every unit whose t-statistic with the core exceeds ``cstar``.

    With ``cstar_method="incremental"``, ``cstar`` is raised by
    ``cstar_increment`` until the whole club passes the test; once it exceeds
    ``cstar_cap`` the club is reduced to the core group.
    """
    if cstar_increment <= 0:
        cstar_method = "fixed"
    core_set = set(core)
    candidates = [r for r in range(rows.size) if r not in core_set]
    t_values = np.array(
        [_test(X, rows[core + [r]], time_trim, hac_method).t_value for r in candidates]
    )

    while True:
        members = core + [r for r, t in zip(candidates, t_values, strict=True) if t > cstar]
        model = _test(X, rows[members], time_trim, hac_method)
        if model.converges() or cstar_method == "fixed":
            return members, model, cstar
        if cstar <= cstar_cap:
            cstar += cstar_increment
        else:
            # The R package reports the stale core+candidates model here; we
            # report the model of the club actually returned.
            return list(core), _test(X, rows[core], time_trim, hac_method), cstar


def find_clubs(
    data: pd.DataFrame,
    *,
    time_trim: float = 1 / 3,
    hac_method: HACMethod = "FQSB",
    cstar: float = 0.0,
    cstar_method: CStarMethod = "fixed",
    cstar_increment: float = 0.1,
    cstar_cap: float = 3.0,
    data_cols: Sequence[Hashable] | None = None,
    ref_col: Hashable | None = None,
    sort_fraction: float = 0.0,
) -> ConvergenceClubs:
    """Find convergence clubs with the Phillips and Sul clustering algorithm.

    Parameters
    ----------
    data
        Units in rows (the index holds unit names), time periods in columns.
        Values should usually be filtered (e.g. :func:`hp_filter` of log GDP).
    time_trim
        Share of initial periods discarded by the log t regression.
    hac_method
        ``"FQSB"`` (fixed QS bandwidth) or ``"AQSB"`` (adaptive QS bandwidth).
    cstar
        Sieve threshold ``c*`` used to add units to the core group.
    cstar_method
        ``"fixed"`` or ``"incremental"``: in the latter case ``cstar`` is
        increased by ``cstar_increment`` until the club converges, up to
        ``cstar_cap``.
    data_cols
        Columns of ``data`` holding the time series; all columns by default.
    ref_col
        Column used to sort units before clustering; defaults to the last
        data column (the last period).
    sort_fraction
        Sort units by their average over the last ``sort_fraction`` share of
        periods instead of a single column (like the ``fr()`` option of the
        Stata command ``psecta``); ``0`` sorts by ``ref_col``. Averaging makes
        the ordering less sensitive to the last observation.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame with units in rows and periods in columns")
    if not data.index.is_unique:
        dupes = data.index[data.index.duplicated()].unique().tolist()
        raise ValueError(f"unit names (the index) must be unique; duplicated: {dupes[:5]}")
    data_cols = list(data.columns if data_cols is None else data_cols)
    if sort_fraction and ref_col is not None:
        raise ValueError("give either ref_col or sort_fraction, not both")
    if not 0 <= sort_fraction <= 1:
        raise ValueError("sort_fraction must be in [0, 1]")
    ref_col = data_cols[-1] if ref_col is None else ref_col
    if ref_col not in data.columns:
        raise KeyError(f"ref_col {ref_col!r} is not a column of data")

    values = data[data_cols]
    if not all(pd.api.types.is_numeric_dtype(dt) for dt in values.dtypes):
        raise TypeError("all data columns must be numeric")
    missing = values.isna().any(axis=1)
    if missing.any():
        units = missing.index[missing].tolist()
        raise ValueError(
            f"data must not contain missing values; {len(units)} units have some, "
            f"e.g. {units[:5]} (drop or interpolate them first)"
        )
    _validate_time_trim(len(data_cols), time_trim)
    if hac_method not in HAC_METHODS:
        raise ValueError(f"hac_method must be one of {HAC_METHODS}, got {hac_method!r}")
    if cstar_method not in ("fixed", "incremental"):
        raise ValueError("cstar_method must be 'fixed' or 'incremental'")
    if cstar_method == "incremental" and (cstar_increment < 0 or cstar_cap < 0):
        raise ValueError("cstar_increment and cstar_cap must be non-negative")

    cstar = float(cstar)
    X = values.to_numpy(dtype=float)
    if sort_fraction:
        n_last = max(1, math.ceil(sort_fraction * X.shape[1]))
        key = X[:, -n_last:].mean(axis=1)
    else:
        key = data[ref_col].to_numpy(dtype=float)
    order = np.argsort(-key, kind="stable")

    clubs: list[Club] = []
    divergent: tuple[int, ...] = ()
    remaining = order
    while remaining.size:
        if remaining.size == 1:
            divergent = tuple(remaining.tolist())
            break

        name = f"club{len(clubs) + 1}"
        model_all = _test(X, remaining, time_trim, hac_method)
        if model_all.converges():
            clubs.append(Club(name, tuple(remaining.tolist()), model_all, cstar=cstar))
            break

        core = _core_group(X, remaining, time_trim, hac_method)
        if core is None:
            divergent = tuple(remaining.tolist())
            break

        members, model, club_cstar = _sieve(
            X,
            remaining,
            core,
            time_trim,
            hac_method,
            cstar,
            cstar_method,
            cstar_increment,
            cstar_cap,
        )
        clubs.append(Club(name, tuple(remaining[members].tolist()), model, cstar=club_cstar))
        remaining = np.delete(remaining, members)

    return ConvergenceClubs(
        clubs=tuple(clubs),
        divergent_ids=divergent,
        data=values,
        time_trim=time_trim,
        hac_method=hac_method,
        params={
            "sort_fraction": sort_fraction,
            "cstar": cstar,
            "cstar_method": cstar_method,
            "cstar_increment": cstar_increment,
            "cstar_cap": cstar_cap,
            "ref_col": ref_col,
        },
    )


def _validate_time_trim(n_periods: int, time_trim: float) -> None:
    if n_periods < 2:
        raise ValueError("at least two time periods are needed")
    if not 0 < time_trim <= 1:
        raise ValueError("time_trim must be in (0, 1]")
    if n_periods - round(n_periods * time_trim) < 2:
        raise ValueError("too few time periods for this value of time_trim")
