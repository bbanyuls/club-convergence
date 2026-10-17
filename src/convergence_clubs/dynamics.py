"""Distribution dynamics: how units move within the income distribution (Quah, 1993, 1996)."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .beta import _club_labels

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .results import ConvergenceClubs


def _class_labels(cuts: np.ndarray) -> list[str]:
    fmt = [f"{c:.2f}" for c in cuts]
    return (
        [f"< {fmt[0]}"] + [f"{fmt[i]}–{fmt[i + 1]}" for i in range(len(fmt) - 1)] + [f"≥ {fmt[-1]}"]
    )


def silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule of thumb, ``0.9 min(sd, IQR / 1.34) n^(-1/5)``."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * spread * len(x) ** (-1 / 5)


def gaussian_kde(x: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel density of ``x`` evaluated on ``grid``."""
    z = (np.asarray(grid, dtype=float)[:, None] - np.asarray(x, dtype=float)[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (len(x) * bandwidth * math.sqrt(2 * math.pi))


@dataclass(frozen=True, eq=False)
class DynamicsResult:
    """Transition matrix and distributions of relative income classes."""

    values: pd.DataFrame
    cuts: np.ndarray
    counts: pd.DataFrame
    lag: int
    clubs: Any = None

    @property
    def labels(self) -> list[str]:
        return _class_labels(self.cuts)

    @property
    def n_classes(self) -> int:
        return len(self.cuts) + 1

    @property
    def classes(self) -> pd.DataFrame:
        """Class (0 = poorest) of every unit in every period."""
        return pd.DataFrame(
            np.searchsorted(self.cuts, self.values.to_numpy(), side="right"),
            index=self.values.index,
            columns=self.values.columns,
        )

    @property
    def matrix(self) -> pd.DataFrame:
        """Transition probabilities: row = class at t, column = class at t + lag."""
        counts = self.counts.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = counts / counts.sum(axis=1, keepdims=True)
        return pd.DataFrame(probs, index=self.counts.index, columns=self.counts.columns)

    def _distribution(self, column: int) -> pd.Series:
        shares = np.bincount(self.classes.iloc[:, column], minlength=self.n_classes)
        return pd.Series(shares / shares.sum(), index=self.labels)

    @property
    def initial(self) -> pd.Series:
        """Share of units in each class in the first period."""
        return self._distribution(0)

    @property
    def final(self) -> pd.Series:
        """Share of units in each class in the last period."""
        return self._distribution(-1)

    @property
    def ergodic(self) -> pd.Series:
        """Long-run distribution implied by the transition matrix (pi = pi P)."""
        P = self.matrix.to_numpy()
        if np.isnan(P).any():
            raise ValueError("some classes are never observed; use fewer classes or other bounds")
        values, vectors = np.linalg.eig(P.T)
        pi = np.real(vectors[:, np.argmin(np.abs(values - 1))])
        return pd.Series(pi / pi.sum(), index=self.labels)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the transition matrix, by decreasing modulus."""
        values = np.linalg.eigvals(self.matrix.to_numpy())
        return values[np.argsort(-np.abs(values))]

    @property
    def shorrocks(self) -> float:
        """Shorrocks mobility index ``(k - trace P) / (k - 1)``: 0 = no mobility."""
        k = self.n_classes
        return float((k - np.trace(self.matrix.to_numpy())) / (k - 1))

    @property
    def half_life(self) -> float:
        """Periods to close half the distance to the ergodic distribution, ``-ln 2 / ln|l2|``."""
        second = abs(self.eigenvalues[1]) if self.n_classes > 1 else 0.0
        if not 0 < second < 1:
            return math.nan
        return -math.log(2) / math.log(second) * self.lag

    def distributions(self) -> pd.DataFrame:
        """Initial, final and ergodic shares of each class."""
        return pd.DataFrame(
            {"initial": self.initial, "final": self.final, "ergodic": self.ergodic}
        ).rename_axis("class")

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "classes": self.n_classes,
                "lag": self.lag,
                "transitions": int(self.counts.to_numpy().sum()),
                "shorrocks": self.shorrocks,
                "second_eigenvalue": float(abs(self.eigenvalues[1])),
                "half_life": self.half_life,
            }
        )

    def composition(self, period: Hashable | None = None) -> pd.DataFrame:
        """Number of units of each club in each class (last period by default)."""
        if self.clubs is None:
            raise ValueError("pass clubs= to distribution_dynamics() for the club composition")
        column = self.values.columns[-1] if period is None else period
        classes = pd.Series(
            np.array(self.labels)[self.classes[column].to_numpy()], index=self.values.index
        )
        table = pd.crosstab(_club_labels(self.clubs, self.values.index), classes)
        return table.reindex(columns=self.labels, fill_value=0).rename_axis(
            index="club", columns="class"
        )

    def density(
        self, period: Hashable | None = None, *, bandwidth: float | None = None, n_grid: int = 256
    ) -> pd.Series:
        """Kernel density of *log* relative income in ``period`` (last by default).

        The density is that of the log values (a kernel on levels would put
        mass below zero), indexed by relative income itself so it can be
        drawn on a log axis. ``bandwidth`` is in log units (Silverman's rule
        by default).
        """
        column = self.values.columns[-1] if period is None else period
        x = np.log(self.values[column].to_numpy(dtype=float))
        h = silverman_bandwidth(x) if bandwidth is None else bandwidth
        grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, n_grid)
        return pd.Series(gaussian_kde(x, grid, h), index=pd.Index(np.exp(grid), name="relative"))

    def plot(self, **kwargs) -> Figure:
        """Transition matrix and class distributions side by side."""
        from .plotting import plot_dynamics

        return plot_dynamics(self, **kwargs)

    def plot_transitions(self, **kwargs) -> Axes:
        from .plotting import plot_transitions

        return plot_transitions(self, **kwargs)

    def plot_densities(self, **kwargs) -> Axes:
        from .plotting import plot_densities

        return plot_densities(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<DynamicsResult: {self.n_classes} classes, lag {self.lag}, "
            f"Shorrocks {self.shorrocks:.3f}, half-life {self.half_life:.1f}>"
        )


def distribution_dynamics(
    data: pd.DataFrame,
    *,
    n_classes: int = 5,
    bounds: Sequence[float] | None = None,
    lag: int = 1,
    overlapping: bool = True,
    relative: bool = True,
    clubs: ConvergenceClubs | pd.Series | None = None,
) -> DynamicsResult:
    """Estimate a Markov chain of relative income classes (Quah, 1993).

    Parameters
    ----------
    data
        Units x periods in levels (e.g. GDP per capita).
    n_classes
        Number of classes when ``bounds`` is not given: the classes are the
        quantiles of the pooled distribution, so each holds the same share of
        all observations.
    bounds
        Inner class boundaries, e.g. ``[0.25, 0.5, 1, 2]`` (5 classes), in
        units of relative income.
    lag
        Transitions are from period t to period t + ``lag``.
    overlapping
        Use every starting period (``True``) or only every ``lag``-th.
    relative
        Divide by the cross-sectional mean of each period (``y / mean(y)``),
        so that common growth does not move units between classes.
    clubs
        Optional :class:`ConvergenceClubs` (or Series unit -> club), used by
        :meth:`DynamicsResult.composition` and in the density plot.
    """
    if data.isna().to_numpy().any():
        raise ValueError("data must not contain missing values; see fill_gaps() and balance()")
    values = data.astype(float)
    if (values.to_numpy() <= 0).any():
        raise ValueError("values must be positive")
    if relative:
        values = values / values.mean(axis=0)
    if not 1 <= lag < values.shape[1]:
        raise ValueError("lag must be between 1 and the number of periods - 1")

    if bounds is None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        pooled = values.to_numpy().ravel()
        cuts = np.quantile(pooled, np.arange(1, n_classes) / n_classes)
    else:
        cuts = np.asarray(sorted(bounds), dtype=float)
        if len(cuts) == 0 or np.any(np.diff(cuts) <= 0):
            raise ValueError("bounds must be distinct values")

    classes = np.searchsorted(cuts, values.to_numpy(), side="right")
    k = len(cuts) + 1
    counts = np.zeros((k, k), dtype=int)
    starts = range(0, values.shape[1] - lag, 1 if overlapping else lag)
    for t in starts:
        np.add.at(counts, (classes[:, t], classes[:, t + lag]), 1)
    labels = _class_labels(cuts)
    return DynamicsResult(
        values=values,
        cuts=cuts,
        counts=pd.DataFrame(
            counts,
            index=pd.Index(labels, name="from"),
            columns=pd.Index(labels, name="to"),
        ),
        lag=lag,
        clubs=clubs,
    )
