"""β-convergence: do poorer units grow faster? (Barro and Sala-i-Martin, 1992)."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._stats import COV_TYPES, ols

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

INITIAL = "log_initial"


def _club_labels(clubs: ConvergenceClubs | pd.Series, units: pd.Index) -> pd.Series:
    labels = clubs.memberships() if hasattr(clubs, "memberships") else pd.Series(clubs)
    aligned = labels.reindex(units)
    missing = aligned.index[aligned.isna()]
    if len(missing):
        raise ValueError(f"no club for {len(missing)} units, e.g. {list(missing[:5])}")
    return aligned.astype(str)


def _period_length(data: pd.DataFrame, start: Hashable, end: Hashable) -> float:
    """Years between ``start`` and ``end`` when periods are numbers, else number of periods."""
    positions = data.columns.get_indexer(pd.Index([start, end]))
    if pd.api.types.is_numeric_dtype(data.columns):
        years = data.columns.to_numpy(dtype=float)[positions]
        return float(years[1] - years[0])
    return float(positions[1] - positions[0])


@dataclass(frozen=True, eq=False)
class BetaResult:
    """Result of a β-convergence regression.

    ``growth = a + beta * log_initial (+ controls, club effects) + e``, with
    growth the average log growth per period (per year when periods are
    years) between ``start`` and ``end``.
    """

    coefficients: pd.DataFrame
    r_squared: float
    n: int
    periods: float
    start: Hashable
    end: Hashable
    cov_type: str
    units: pd.DataFrame
    clubs: Any = None
    club_effects: bool = False
    controls: tuple[str, ...] = ()
    control_means: tuple[float, ...] = ()

    def _initial(self, column: str) -> float:
        return float(self.coefficients.loc[[INITIAL], column].to_numpy(dtype=float)[0])

    @property
    def beta(self) -> float:
        return self._initial("coef")

    @property
    def std_err(self) -> float:
        return self._initial("std_err")

    @property
    def t_value(self) -> float:
        return self._initial("t_value")

    @property
    def p_value(self) -> float:
        return self._initial("p_value")

    @property
    def speed(self) -> float:
        """Implied speed of convergence ``lambda = -ln(1 + beta T) / T`` (per period)."""
        return _speed(self.beta, self.periods)

    @property
    def half_life(self) -> float:
        """Periods needed to close half of the gap to the steady state, ``ln 2 / lambda``."""
        return _half_life(self.speed)

    def converges(self, level: float = 0.05) -> bool:
        """Negative and significant β (two-sided test at ``level``)."""
        return self.beta < 0 and self.p_value < level

    def summary(self) -> pd.DataFrame:
        """Coefficient table (estimate, standard error, t, p)."""
        return self.coefficients.copy()

    def by_club(self) -> pd.DataFrame:
        """Absolute β-convergence estimated separately within each club.

        Clubs (and the divergent group) with fewer than 3 units get NaN.
        """
        if self.clubs is None:
            raise ValueError("pass clubs= to beta_convergence() to estimate β by club")
        rows: dict[Any, dict[str, Any]] = {}
        for club, group in self.units.groupby("club", sort=False):
            row: dict[str, Any] = {"n_units": len(group)}
            if len(group) >= 3 and group["log_initial"].nunique() > 1:
                X = np.column_stack([np.ones(len(group)), group["log_initial"]])
                fit = ols(group["growth"].to_numpy(), X, self.cov_type)
                beta = float(fit.params[1])
                row.update(
                    beta=beta,
                    std_err=float(fit.std_err[1]),
                    t_value=float(fit.t_values[1]),
                    p_value=float(fit.p_values()[1]),
                    speed=_speed(beta, self.periods),
                    half_life=_half_life(_speed(beta, self.periods)),
                )
            rows[club] = row
        order = [c for c in _club_order(self.clubs) if c in rows]
        columns = ["n_units", "beta", "std_err", "t_value", "p_value", "speed", "half_life"]
        return pd.DataFrame.from_dict(rows, orient="index", columns=columns).loc[order]

    def plot(self, **kwargs) -> Axes:
        """Growth against initial level; see :func:`convergence_clubs.plot_beta`."""
        from .plotting import plot_beta

        return plot_beta(self, **kwargs)

    def plot_interactive(self, **kwargs):
        """Interactive scatter (plotly); see :func:`convergence_clubs.interactive.beta_figure`."""
        from .interactive import beta_figure

        return beta_figure(self, **kwargs)

    def __repr__(self) -> str:
        kind = "conditional" if self.controls or self.club_effects else "absolute"
        return (
            f"<BetaResult ({kind}): beta={self.beta:.4f} (t={self.t_value:.2f}, "
            f"p={self.p_value:.3g}), half-life={self.half_life:.1f}, n={self.n}>"
        )


def _club_order(clubs: Any) -> list[str]:
    if hasattr(clubs, "clubs"):
        return [c.name for c in clubs.clubs] + ["divergent"]
    return list(pd.unique(pd.Series(clubs).astype(str)))


def _speed(beta: float, periods: float) -> float:
    base = 1.0 + beta * periods
    return -math.log(base) / periods if base > 0 and periods > 0 else math.nan


def _half_life(speed: float) -> float:
    return math.log(2) / speed if speed > 0 else math.nan


def beta_convergence(
    data: pd.DataFrame,
    *,
    start: Hashable | None = None,
    end: Hashable | None = None,
    log: bool = True,
    controls: pd.DataFrame | None = None,
    clubs: ConvergenceClubs | pd.Series | None = None,
    club_effects: bool = False,
    cov_type: str = "HC1",
) -> BetaResult:
    """Estimate β-convergence: regress average growth on the initial (log) level.

    Parameters
    ----------
    data
        Units x periods, in levels (e.g. GDP per capita) unless ``log=False``
        (then the values are taken as logs already). Do *not* pass
        HP-filtered data unless that is what you want to study.
    start, end
        First and last period; the first and last columns by default.
    controls
        Covariates for *conditional* β-convergence, indexed by unit.
    clubs
        A :class:`ConvergenceClubs` result (or a Series unit -> club), used
        to colour plots, for :meth:`BetaResult.by_club` and, with
        ``club_effects=True``, to add club intercepts to the regression.
    cov_type
        ``"HC1"`` (default, as Stata's ``robust``), ``"HC0"`` or
        ``"nonrobust"``. p-values use the Student t distribution.
    """
    if cov_type not in COV_TYPES or cov_type == "HAC":
        raise ValueError("cov_type must be 'nonrobust', 'HC0' or 'HC1'")
    if not data.columns.is_unique:
        raise ValueError("period labels (the columns) must be unique")
    start = data.columns[0] if start is None else start
    end = data.columns[-1] if end is None else end
    for col in (start, end):
        if col not in data.columns:
            raise KeyError(f"period {col!r} is not a column of data")
    periods = _period_length(data, start, end)
    if periods <= 0:
        raise ValueError("end must come after start")

    first, last = data[start].astype(float), data[end].astype(float)
    if log:
        if (first <= 0).any() or (last <= 0).any():
            raise ValueError("log=True needs positive values; pass logs with log=False")
        first, last = first.transform(np.log), last.transform(np.log)
    units = pd.DataFrame({"log_initial": first, "growth": (last - first) / periods})
    missing = units.index[units.isna().any(axis=1)]
    if len(missing):
        raise ValueError(f"missing start or end values for {list(missing[:5])}")

    names = ["const", INITIAL]
    columns = [np.ones(len(units)), units["log_initial"].to_numpy()]
    control_names: tuple[str, ...] = ()
    control_means: tuple[float, ...] = ()
    if controls is not None:
        aligned = controls.reindex(units.index)
        bad = aligned.index[aligned.isna().any(axis=1)]
        if len(bad):
            raise ValueError(f"controls missing for {len(bad)} units, e.g. {list(bad[:5])}")
        control_names = tuple(map(str, aligned.columns))
        reserved = {"const", INITIAL} | {n for n in control_names if n.startswith("club[")}
        if reserved & set(control_names):
            raise ValueError(f"control names clash with model terms: {sorted(reserved)}")
        control_means = tuple(float(aligned[c].mean()) for c in aligned.columns)
        names += list(control_names)
        columns += [aligned[c].to_numpy(dtype=float) for c in aligned.columns]

    if clubs is not None:
        units["club"] = _club_labels(clubs, units.index)
    elif club_effects:
        raise ValueError("club_effects=True needs clubs=")
    if club_effects:
        order = [c for c in _club_order(clubs) if c in set(units["club"])]
        for club in order[1:]:  # the first club is the reference category
            names.append(f"club[{club}]")
            columns.append((units["club"] == club).to_numpy(dtype=float))

    fit = ols(units["growth"].to_numpy(), np.column_stack(columns), cov_type)
    coefficients = pd.DataFrame(
        {
            "coef": fit.params,
            "std_err": fit.std_err,
            "t_value": fit.t_values,
            "p_value": fit.p_values(),
        },
        index=pd.Index(names, name="term"),
    )
    return BetaResult(
        coefficients=coefficients,
        r_squared=fit.r_squared,
        n=fit.n,
        periods=periods,
        start=start,
        end=end,
        cov_type=cov_type,
        units=units,
        clubs=clubs,
        club_effects=club_effects,
        controls=control_names,
        control_means=control_means,
    )


def _fitted_line(result: BetaResult, x: np.ndarray, club: str | None = None) -> np.ndarray:
    """Fitted growth at initial levels ``x``, with controls at their sample means."""
    coef = result.coefficients["coef"]
    y: Any = coef["const"] + coef[INITIAL] * np.asarray(x, dtype=float)
    for name, mean in zip(result.controls, result.control_means, strict=True):
        y = y + coef[name] * mean
    if club is not None and f"club[{club}]" in coef.index:
        y = y + coef[f"club[{club}]"]
    return y
