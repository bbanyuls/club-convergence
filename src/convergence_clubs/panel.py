"""Preparing panel data: reshaping to units x periods and handling missing values."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import numpy as np
import pandas as pd


def wide_panel(
    data: pd.DataFrame,
    *,
    unit: Hashable,
    time: Hashable,
    value: Hashable,
    dropna: bool = False,
) -> pd.DataFrame:
    """Turn long panel data (one row per unit and period) into units x periods.

    Parameters
    ----------
    data
        Long data frame with one row per ``(unit, time)`` pair.
    unit, time, value
        Columns holding the unit name, the period and the variable.
    dropna
        Drop units with missing periods instead of keeping them as NaN
        (:func:`find_clubs` rejects missing values).

    Examples
    --------
    >>> wide = wide_panel(df, unit="region", time="year", value="gdp_pc")
    >>> clubs = find_clubs(hp_filter(np.log(wide)))
    """
    missing = [c for c in (unit, time, value) if c not in data.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    dupes = data.duplicated([unit, time])
    if dupes.any():
        examples = data.loc[dupes, [unit, time]].head(3).to_records(index=False).tolist()
        raise ValueError(f"several rows for the same unit and period, e.g. {examples}")
    wide = data.pivot(index=unit, columns=time, values=value).sort_index(axis=1)
    wide.columns.name = None
    if dropna:
        wide = wide.dropna()
    return wide


def _runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True values."""
    out, start = [], None
    for i, m in enumerate(missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(missing) - start))
    return out


def describe_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Missing values of every unit that has some.

    Columns: ``n_missing``, ``leading`` and ``trailing`` (missing periods at
    the start/end of the series), ``interior`` (missing periods in between)
    and ``longest_gap`` (longest interior gap).
    """
    rows = {}
    for unit, row in data.iterrows():
        missing = row.isna().to_numpy()
        if not missing.any():
            continue
        runs = _runs(missing)
        leading = runs[0][1] if runs[0][0] == 0 else 0
        trailing = runs[-1][1] if runs[-1][0] + runs[-1][1] == len(missing) else 0
        if leading == len(missing):
            trailing = 0
        interior = [n for start, n in runs if 0 < start and start + n < len(missing)]
        rows[unit] = {
            "n_missing": int(missing.sum()),
            "leading": leading,
            "trailing": trailing,
            "interior": sum(interior),
            "longest_gap": max(interior, default=0),
        }
    columns = ["n_missing", "leading", "trailing", "interior", "longest_gap"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).rename_axis(
        data.index.name
    )


def fill_gaps(data: pd.DataFrame, *, max_gap: int = 2, log: bool = False) -> pd.DataFrame:
    """Linearly interpolate interior gaps of at most ``max_gap`` periods.

    Leading and trailing gaps and longer gaps are left as NaN: extrapolating
    the start or end of a series would invent the very transition path that
    the log t test examines. With ``log=True`` the interpolation is done on
    logs (geometric interpolation), which suits growing series such as GDP.
    Note that filled values feed the HP filter and can shift the clubs; check
    with :func:`sensitivity` or :func:`bootstrap`.
    """
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    values = data.to_numpy(dtype=float, copy=True)
    if log:
        if (values[~np.isnan(values)] <= 0).any():
            raise ValueError("log=True needs positive values")
        values = np.log(values)
    positions = np.arange(values.shape[1])
    for row in values:
        missing = np.isnan(row)
        for start, length in _runs(missing):
            if start == 0 or start + length == len(row) or length > max_gap:
                continue
            gap = slice(start, start + length)
            row[gap] = np.interp(positions[gap], positions[~missing], row[~missing])
    if log:
        values = np.exp(values)
    return pd.DataFrame(values, index=data.index, columns=data.columns)


def balance(
    data: pd.DataFrame, how: Literal["drop_units", "trim_periods"] = "drop_units"
) -> pd.DataFrame:
    """Make the panel complete.

    ``"drop_units"`` drops every unit with a missing value;
    ``"trim_periods"`` keeps all units and the longest run of consecutive
    periods in which no unit has a missing value.
    """
    if how == "drop_units":
        return data.dropna()
    if how == "trim_periods":
        complete = data.notna().all(axis=0).to_numpy()
        runs = _runs(complete)
        if not runs:
            raise ValueError("no period is complete for all units")
        start, length = max(runs, key=lambda r: r[1])
        return data.iloc[:, start : start + length]
    raise ValueError("how must be 'drop_units' or 'trim_periods'")
