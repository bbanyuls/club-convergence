"""Pesaran (2007) pairwise convergence test, built on the augmented Dickey-Fuller test."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

Regression = Literal["c", "ct"]
AutoLag = Literal["AIC", "BIC"]


# MacKinnon (1994) response surfaces for one series (N = 1): the p-value is
# Phi(polynomial in the ADF statistic), with different coefficients below and
# above tau_star; outside [tau_min, tau_max] the p-value is 0 or 1.
class _Surface(NamedTuple):
    tau_max: float
    tau_min: float
    tau_star: float
    small: tuple[float, ...]
    large: tuple[float, ...]


_MACKINNON = {
    "c": _Surface(
        2.74, -18.83, -1.61, (2.1659, 1.4412, 0.038269), (1.7339, 0.93202, -0.12745, -0.010368)
    ),
    "ct": _Surface(
        0.7, -16.18, -2.89, (3.2512, 1.6047, 0.049588), (2.5261, 0.61654, -0.37956, -0.060285)
    ),
}


def mackinnon_pvalue(stat: float, regression: str = "c") -> float:
    """Approximate p-value of an ADF statistic (MacKinnon, 1994)."""
    surface = _MACKINNON[regression]
    if stat > surface.tau_max:
        return 1.0
    if stat < surface.tau_min:
        return 0.0
    coef = surface.small if stat <= surface.tau_star else surface.large
    z = sum(c * stat**k for k, c in enumerate(coef))
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey-Fuller test: H0 unit root, rejected for small p-values."""

    stat: float
    p_value: float
    lags: int
    nobs: int


def _design(x: np.ndarray, lags: int, max_lag: int, regression: str) -> tuple[np.ndarray, ...]:
    """Response and regressors (deterministic terms, level, lagged differences)."""
    dx = np.diff(x)
    n = dx.size - max_lag
    y = dx[max_lag:]
    columns = [np.ones(n)]
    if regression == "ct":
        columns.append(np.arange(1.0, n + 1))
    columns.append(x[max_lag:-1])  # level x_{t-1}
    columns += [dx[max_lag - k : dx.size - k] for k in range(1, lags + 1)]
    return y, np.column_stack(columns)


def _fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, resid, float(resid @ resid)


def adf_test(
    x: np.ndarray | pd.Series,
    *,
    regression: Regression | str = "c",
    max_lag: int | None = None,
    autolag: AutoLag | str | None = "AIC",
) -> ADFResult:
    """Augmented Dickey-Fuller unit-root test.

    The lag length is chosen as in statsmodels' ``adfuller``: up to
    ``max_lag`` (default ``ceil(12 (n/100)^(1/4))``, Schwert's rule) by AIC
    or BIC on a common sample, then the chosen regression is refitted on the
    full sample. ``regression`` is ``"c"`` (constant) or ``"ct"`` (constant
    and trend). p-values follow MacKinnon (1994).
    """
    if regression not in _MACKINNON:
        raise ValueError("regression must be 'c' or 'ct'")
    if autolag not in ("AIC", "BIC", None):
        raise ValueError("autolag must be 'AIC', 'BIC' or None")
    x = np.asarray(x, dtype=float)
    n = x.size
    n_trend = 1 if regression == "c" else 2
    if max_lag is None:
        max_lag = min(int(math.ceil(12.0 * (n / 100.0) ** 0.25)), n // 2 - n_trend - 1)
    if max_lag < 0 or max_lag > n // 2 - n_trend - 1:
        raise ValueError("the series is too short for this lag length")

    lags = max_lag
    if autolag is not None:
        y, X = _design(x, max_lag, max_lag, regression)
        nobs = y.size
        best = None
        for k in range(max_lag + 1):
            _, _, ssr = _fit(y, X[:, : n_trend + 1 + k])
            llf = -nobs / 2 * (math.log(2 * math.pi) + math.log(ssr / nobs) + 1)
            n_params = n_trend + 1 + k
            penalty = 2 * n_params if autolag == "AIC" else math.log(nobs) * n_params
            ic = -2 * llf + penalty
            if best is None or ic < best[0]:  # ties keep the shorter lag
                best = (ic, k)
        assert best is not None
        lags = best[1]

    y, X = _design(x, lags, lags, regression)
    coef, resid, ssr = _fit(y, X)
    nobs, k = X.shape
    sigma2 = ssr / (nobs - k)
    position = n_trend  # the level comes right after the deterministic terms
    se = math.sqrt(sigma2 * np.linalg.inv(X.T @ X)[position, position])
    stat = float(coef[position] / se)
    return ADFResult(stat, mackinnon_pvalue(stat, regression), lags, nobs)


@dataclass(frozen=True, eq=False)
class PairwiseResult:
    """ADF tests on the gaps between every pair of units (Pesaran, 2007)."""

    pairs: pd.DataFrame
    units: pd.Index
    alpha: float
    regression: str
    clubs: Any = None

    def summary(self) -> pd.DataFrame:
        """Share of pairs rejecting the unit root: all pairs, within and between clubs.

        Without convergence the share should be close to ``alpha``; with
        convergence it approaches 1.
        """
        groups = {"all pairs": np.ones(len(self.pairs), dtype=bool)}
        if "same_club" in self.pairs:
            groups["within clubs"] = self.pairs["same_club"].to_numpy(dtype=bool)
            groups["between clubs"] = self.pairs["between_clubs"].to_numpy(dtype=bool)
        reject = self.pairs["reject"].to_numpy(dtype=bool)
        rows = {
            name: {
                "n_pairs": int(mask.sum()),
                "share_rejected": float(reject[mask].mean()) if mask.any() else np.nan,
                "expected_under_null": self.alpha,
            }
            for name, mask in groups.items()
        }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("pairs")

    def by_club(self) -> pd.DataFrame:
        """Share of rejections among the pairs inside each club."""
        if "club_i" not in self.pairs:
            raise ValueError("pass clubs= to pairwise_convergence() to split the pairs by club")
        from .beta import _club_order

        within = self.pairs[self.pairs["same_club"]]
        table = within.groupby("club_i")["reject"].agg(["size", "mean"])
        table.columns = ["n_pairs", "share_rejected"]
        order = [c for c in _club_order(self.clubs) if c in table.index]
        return table.loc[order].rename_axis("club")

    def matrix(self, column: str = "p_value") -> pd.DataFrame:
        """``column`` of every pair as a symmetric units x units table."""
        # Fill a NumPy array, then wrap it: with copy-on-write, writing through
        # DataFrame.to_numpy() fails (pandas 3) or silently changes a copy.
        table = np.full((len(self.units), len(self.units)), np.nan)
        values = self.pairs[column].to_numpy(dtype=float)
        i = self.units.get_indexer(pd.Index(self.pairs["unit_i"]))
        j = self.units.get_indexer(pd.Index(self.pairs["unit_j"]))
        table[i, j] = values
        table[j, i] = values
        return pd.DataFrame(table, index=self.units, columns=self.units)

    def plot(self, **kwargs) -> Axes:
        """Heatmap of pairwise p-values; see :func:`convergence_clubs.plot_pairwise`."""
        from .plotting import plot_pairwise

        return plot_pairwise(self, **kwargs)

    def __repr__(self) -> str:
        share = self.summary().loc["all pairs", "share_rejected"]
        return (
            f"<PairwiseResult: {len(self.pairs)} pairs, unit root rejected for "
            f"{share:.1%} at alpha={self.alpha:g}>"
        )


def pairwise_convergence(
    data: pd.DataFrame,
    *,
    log: bool = True,
    alpha: float = 0.05,
    clubs: ConvergenceClubs | pd.Series | None = None,
    regression: Regression | str = "c",
    max_lag: int | None = None,
    autolag: AutoLag | str | None = "AIC",
) -> PairwiseResult:
    """Pesaran's (2007) pairwise test: ADF tests on every gap ``y_it - y_jt``.

    Parameters
    ----------
    data
        Units x periods in levels (``log=True``, the default) or in logs.
        Use the unfiltered series: smoothing removes the dynamics the unit
        root test looks at.
    alpha
        Size of each ADF test.
    clubs
        Optional clubs (a :class:`ConvergenceClubs` or a Series unit -> club)
        to split the pairs into within-club and between-club pairs (pairs
        with a divergent unit count as between clubs).
    regression, max_lag, autolag
        Passed to :func:`adf_test`.
    """
    from .beta import _club_labels

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if data.isna().to_numpy().any():
        raise ValueError("data must not contain missing values; see fill_gaps() and balance()")
    values = data.to_numpy(dtype=float)
    if log:
        if (values <= 0).any():
            raise ValueError("log=True needs positive values; pass logs with log=False")
        values = np.log(values)
    units = data.index
    rows = []
    for i, j in combinations(range(len(units)), 2):
        test = adf_test(values[i] - values[j], regression=regression, max_lag=max_lag,
                        autolag=autolag)  # fmt: skip
        rows.append((units[i], units[j], test.stat, test.p_value, test.lags, test.nobs))
    pairs = pd.DataFrame(rows, columns=["unit_i", "unit_j", "adf_stat", "p_value", "lags", "nobs"])
    pairs["reject"] = pairs["p_value"] < alpha
    if clubs is not None:
        labels = _club_labels(clubs, units)
        pairs["club_i"] = labels.reindex(pairs["unit_i"]).to_numpy()
        pairs["club_j"] = labels.reindex(pairs["unit_j"]).to_numpy()
        in_club = (pairs["club_i"] != "divergent") & (pairs["club_j"] != "divergent")
        pairs["same_club"] = in_club & (pairs["club_i"] == pairs["club_j"])
        pairs["between_clubs"] = ~pairs["same_club"]
    return PairwiseResult(pairs, units, alpha, str(regression), clubs)
