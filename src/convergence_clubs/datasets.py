"""Example datasets shipped with the R package ConvergenceClubs."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def _load(name: str) -> pd.DataFrame:
    with files("convergence_clubs").joinpath("data", name).open("r") as fh:
        df = pd.read_csv(fh, index_col="Countries")
    df.columns = df.columns.astype(int)
    df.columns.name = "year"
    return df


def load_gdp() -> pd.DataFrame:
    """Real per-capita GDP of 152 countries, 1970-2003 (Phillips and Sul, 2009).

    Countries in rows (index ``Countries``), years in columns.
    """
    return _load("gdp.csv")


def load_filtered_gdp() -> pd.DataFrame:
    """Log of :func:`load_gdp`, smoothed with an HP filter (``lambda = 400``)."""
    return _load("filtered_gdp.csv")
