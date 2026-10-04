"""Reshaping panel data into the wide layout used by :func:`find_clubs`."""

from __future__ import annotations

from collections.abc import Hashable

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
