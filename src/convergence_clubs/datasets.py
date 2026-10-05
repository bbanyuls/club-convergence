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


def country_codes() -> pd.Series:
    """ISO 3166 alpha-3 code of every country in :func:`load_gdp`, indexed by name.

    Useful to join other data or map boundaries (e.g. Natural Earth's
    ``ISO_A3_EH`` column). "Netherlands Antilles" (dissolved in 2010) keeps
    its historical code ``ANT``, which modern maps do not have.
    """
    with files("convergence_clubs").joinpath("data", "country_codes.csv").open("r") as fh:
        return pd.read_csv(fh, index_col="Countries")["iso3"]


PRODUCTIVITY_VARIABLES = {
    "lp": "labor productivity (real GDP per worker)",
    "ky": "capital-output ratio",
    "h": "human capital index",
    "tfp": "aggregate efficiency (total factor productivity)",
    "gdppc": "GDP per capita",
}
_SAMPLES = {"all": None, "high_income": "yes", "other": "no"}


def _productivity() -> pd.DataFrame:
    with files("convergence_clubs").joinpath("data", "productivity.csv").open("r") as fh:
        return pd.read_csv(fh)


def load_productivity(variable: str = "lp", *, sample: str = "all") -> pd.DataFrame:
    """Productivity panel of Mendez (2020): 108 countries, 1990-2014, in levels.

    Data from the Penn World Table 9.0 (Feenstra, Inklaar and Timmer, 2015),
    as prepared in the replication files of Mendez (2020), "Convergence
    Clubs in Labor Productivity and its Proximate Sources" (MIT licence).

    Parameters
    ----------
    variable
        ``"lp"`` (labor productivity), ``"ky"`` (capital-output ratio),
        ``"h"`` (human capital index), ``"tfp"`` (aggregate efficiency) or
        ``"gdppc"`` (GDP per capita).
    sample
        ``"all"`` (108 countries), ``"high_income"`` (26 countries that were
        high-income in 1990, World Bank classification) or ``"other"`` (82).

    Examples
    --------
    The book's clubs (Stata, ``psecta, kq(0.333)``) are reproduced exactly:

    >>> lp = cc.load_productivity("lp", sample="other")
    >>> clubs = cc.find_clubs(cc.hp_filter(np.log(lp)), convention="Stata", time_trim=0.333)
    """
    key = variable.lower()
    if key not in PRODUCTIVITY_VARIABLES:
        raise ValueError(f"variable must be one of {sorted(PRODUCTIVITY_VARIABLES)}")
    if sample not in _SAMPLES:
        raise ValueError(f"sample must be one of {sorted(_SAMPLES)}")
    data = _productivity()
    if _SAMPLES[sample] is not None:
        data = data[data["high_income_1990"] == _SAMPLES[sample]]
    wide = data.pivot(index="country", columns="year", values=key)
    wide.columns.name = "year"
    return wide


def productivity_info() -> pd.DataFrame:
    """ISO3 code, UN region and 1990 high-income status of the countries of
    :func:`load_productivity`, indexed by country."""
    data = _productivity().drop_duplicates("country").set_index("country")
    info = data[["iso3", "region", "high_income_1990"]].copy()
    info["high_income_1990"] = info["high_income_1990"] == "yes"
    return info
