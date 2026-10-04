"""σ-convergence: does cross-sectional dispersion shrink over time?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._stats import newey_west_lags, ols
from .beta import _club_labels, _club_order

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

Measure = Literal["sd", "var", "cv"]
MEASURES: tuple[str, ...] = ("sd", "var", "cv")
ALL = "all units"
LABELS = {
    "sd": "standard deviation of log",
    "var": "variance of log",
    "cv": "coefficient of variation",
}


def _dispersion(values: np.ndarray, measure: str, log: bool) -> np.ndarray:
    """Dispersion across units (rows) in every period (columns)."""
    if measure == "cv":
        levels = values if log else np.exp(values)
        return levels.std(axis=0, ddof=1) / levels.mean(axis=0)
    logs = np.log(values) if log else values
    sd = logs.std(axis=0, ddof=1)
    return sd if measure == "sd" else sd**2


@dataclass(frozen=True, eq=False)
class SigmaResult:
    """Cross-sectional dispersion over time, for all units and within each club."""

    series: pd.DataFrame
    measure: str
    n_units: pd.Series
    hac_lags: int
    clubs: Any = None

    def trend(self, level: float = 0.05) -> pd.DataFrame:
        """Linear time trend of the dispersion of each group.

        ``slope`` is the change per period (per year when periods are
        years), with Newey-West standard errors and normal p-values;
        ``change`` is the relative change from the first to the last period.
        ``converging`` means a negative slope significant at ``level``.
        """
        index = self.series.index
        x = (
            np.asarray(index, dtype=float)
            if pd.api.types.is_numeric_dtype(index)
            else np.arange(len(index), dtype=float)
        )
        X = np.column_stack([np.ones_like(x), x])
        rows = {}
        for group in self.series.columns:
            y = self.series[group].to_numpy()
            fit = ols(y, X, "HAC", lags=self.hac_lags)
            slope, p = float(fit.params[1]), float(fit.p_values("normal")[1])
            rows[group] = {
                "n_units": int(self.n_units[group]),
                "first": y[0],
                "last": y[-1],
                "change": y[-1] / y[0] - 1,
                "slope": slope,
                "std_err": float(fit.std_err[1]),
                "t_value": float(fit.t_values[1]),
                "p_value": p,
                "converging": bool(slope < 0 and p < level),
            }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")

    def converges(self, group: str = ALL, level: float = 0.05) -> bool:
        """Whether dispersion in ``group`` has a significantly negative trend."""
        return bool(self.trend(level).loc[group, "converging"])

    def plot(self, **kwargs) -> Axes:
        """Dispersion over time; see :func:`convergence_clubs.plot_sigma`."""
        from .plotting import plot_sigma

        return plot_sigma(self, **kwargs)

    def plot_interactive(self, **kwargs):
        """Interactive lines (plotly); see :func:`convergence_clubs.interactive.sigma_figure`."""
        from .interactive import sigma_figure

        return sigma_figure(self, **kwargs)

    def __repr__(self) -> str:
        t = self.trend().loc[ALL]
        verdict = "converging" if t["converging"] else "not converging"
        return (
            f"<SigmaResult ({self.measure}): {ALL} {t['first']:.3f} -> {t['last']:.3f} "
            f"({t['change']:+.1%}), {verdict}; groups={list(self.series.columns)}>"
        )


def sigma_convergence(
    data: pd.DataFrame,
    *,
    measure: Measure = "sd",
    log: bool = True,
    clubs: ConvergenceClubs | pd.Series | None = None,
    min_units: int = 3,
    hac_lags: int | None = None,
) -> SigmaResult:
    """Measure σ-convergence: cross-sectional dispersion in every period.

    Parameters
    ----------
    data
        Units x periods, in levels (e.g. GDP per capita) unless ``log=False``
        (then values are taken as logs already).
    measure
        ``"sd"`` (standard deviation of logs, the usual measure), ``"var"``
        (variance of logs) or ``"cv"`` (coefficient of variation of levels).
        Sample statistics (``ddof=1``).
    clubs
        A :class:`ConvergenceClubs` result (or a Series unit -> club): the
        dispersion is also computed within each club with at least
        ``min_units`` units (the spread of two units is just the gap between
        them, which is too noisy to read as a trend). Divergent units count
        in "all units" only.
    hac_lags
        Lags of the Newey-West estimator in :meth:`SigmaResult.trend`;
        ``floor(4 (T/100)^(2/9))`` by default.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    if data.isna().to_numpy().any():
        raise ValueError("data must not contain missing values; see fill_gaps() and balance()")
    values = data.to_numpy(dtype=float)
    if log and (values <= 0).any():
        raise ValueError("log=True needs positive values; pass logs with log=False")
    if len(data) < 2 or data.shape[1] < 3:
        raise ValueError("need at least 2 units and 3 periods")

    groups = {ALL: np.ones(len(data), dtype=bool)}
    if clubs is not None:
        labels = _club_labels(clubs, data.index)
        for club in _club_order(clubs):
            mask = (labels == club).to_numpy()
            if club != "divergent" and mask.sum() >= max(2, min_units):
                groups[club] = mask

    series = pd.DataFrame(
        {name: _dispersion(values[mask], measure, log) for name, mask in groups.items()},
        index=data.columns,
    )
    n_units = pd.Series({name: int(mask.sum()) for name, mask in groups.items()})
    lags = newey_west_lags(data.shape[1]) if hac_lags is None else hac_lags
    return SigmaResult(series=series, measure=measure, n_units=n_units, hac_lags=lags, clubs=clubs)
