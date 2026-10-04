"""Robustness of club memberships to the settings of the clustering procedure."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .clustering import find_clubs

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

DIVERGENT = "divergent"


def _labels(result: ConvergenceClubs) -> np.ndarray:
    """Club index of every unit, -1 for divergent units."""
    labels = np.full(len(result.data), -1)
    for k, club in enumerate(result.clubs):
        labels[list(club.ids)] = k
    return labels


def _partition(result: ConvergenceClubs) -> np.ndarray:
    """Labels where every divergent unit is a singleton group."""
    labels = _labels(result)
    divergent = labels < 0
    labels[divergent] = labels.max(initial=-1) + 1 + np.arange(divergent.sum())
    return labels


def adjusted_rand_index(a: ConvergenceClubs, b: ConvergenceClubs) -> float:
    """Agreement between the club memberships of two results (1 = identical).

    Divergent units count as singleton groups. Both results must cover the
    same units, in the same order. 0 is the agreement expected by chance.
    """
    if not a.data.index.equals(b.data.index):
        raise ValueError("both results must cover the same units in the same order")
    return _ari(_partition(a), _partition(b))


def _ari(x: np.ndarray, y: np.ndarray) -> float:
    """Adjusted Rand index of two label vectors (Hubert and Arabie, 1985)."""
    table = pd.crosstab(x, y).to_numpy()

    def pairs(v):
        return (v * (v - 1) / 2).sum()

    n = len(x)
    index = pairs(table)
    rows, cols = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * cols / (n * (n - 1) / 2)
    maximum = (rows + cols) / 2
    if maximum == expected:  # both partitions trivial (all singletons or a single group)
        return 1.0 if np.array_equal(x == x[:, None], y == y[:, None]) else 0.0
    return float((index - expected) / (maximum - expected))


def compare_clubs(a: ConvergenceClubs, b: ConvergenceClubs) -> pd.DataFrame:
    """Cross-tabulation of the memberships of two results (clubs of ``a`` in rows)."""
    if not a.data.index.equals(b.data.index):
        raise ValueError("both results must cover the same units in the same order")
    return pd.crosstab(a.memberships().rename("a"), b.memberships().rename("b"))


@dataclass(frozen=True, eq=False)
class SensitivityResult:
    """Clubs found with every combination of settings in a grid.

    The first run is the baseline that the others are compared to.
    """

    runs: tuple[tuple[dict[str, Any], ConvergenceClubs], ...]

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[tuple[dict[str, Any], ConvergenceClubs]]:
        return iter(self.runs)

    @property
    def baseline(self) -> ConvergenceClubs:
        return self.runs[0][1]

    @property
    def units(self) -> pd.Index:
        return self.baseline.data.index

    def table(self) -> pd.DataFrame:
        """One row per run: settings, number of clubs, club sizes and agreement with baseline."""
        rows = []
        for params, result in self.runs:
            rows.append(
                {
                    **{k: "none" if v is None else v for k, v in params.items()},
                    "n_clubs": result.n_clubs,
                    "n_divergent": result.n_divergent,
                    "sizes": "-".join(str(len(c)) for c in result.clubs),
                    "ari_vs_baseline": adjusted_rand_index(self.baseline, result),
                }
            )
        return pd.DataFrame(rows)

    def co_membership(self) -> pd.DataFrame:
        """Share of runs in which each pair of units belongs to the same club.

        Divergent units are never grouped with anyone; the diagonal is the
        share of runs in which a unit belongs to some club.
        """
        together = np.zeros((len(self.units),) * 2)
        for _, result in self.runs:
            labels = _labels(result)
            together += (labels[:, None] == labels[None, :]) & (labels >= 0)[:, None]
        return pd.DataFrame(together / len(self.runs), index=self.units, columns=self.units)

    def stability(self) -> pd.DataFrame:
        """How firmly each unit belongs to its baseline club.

        ``stability`` is the average share of runs in which the unit is in
        the same club as each of its baseline club mates (1 = always with
        all of them); ``divergent_share`` is the share of runs in which it
        is divergent.
        """
        share = self.co_membership().to_numpy()
        base = _labels(self.baseline)
        stability = np.full(len(base), np.nan)
        for i, label in enumerate(base):
            mates = np.flatnonzero((base == label) & (np.arange(len(base)) != i))
            if label >= 0 and mates.size:
                stability[i] = share[i, mates].mean()
        divergent = np.mean([_labels(r) < 0 for _, r in self.runs], axis=0)
        return pd.DataFrame(
            {
                "baseline_club": self.baseline.memberships().to_numpy(),
                "stability": stability,
                "divergent_share": divergent,
            },
            index=self.units,
        )

    def plot(self, **kwargs) -> Axes:
        """Co-membership heatmap; see :func:`convergence_clubs.plot_co_membership`."""
        from .plotting import plot_co_membership

        return plot_co_membership(self, **kwargs)


def sensitivity(
    data: pd.DataFrame,
    *,
    time_trim: Sequence[float] = (1 / 3, 0.25, 0.3, 0.4),
    hac_method: Sequence[str] = ("FQSB", "AQSB"),
    cstar: Sequence[float] = (0.0,),
    merge: Sequence[str | None] = (None,),
    sort_fraction: Sequence[float] = (0.0,),
    merge_divergent: bool = False,
    **find_kwargs,
) -> SensitivityResult:
    """Run the clustering for every combination of settings.

    Each of ``time_trim``, ``hac_method``, ``cstar``, ``merge`` (``None``,
    ``"PS"`` or ``"vLT"``) and ``sort_fraction`` is a list of values; the
    grid is their product, in that order, so the first value of each forms
    the baseline run. Other keyword arguments are passed to
    :func:`find_clubs` unchanged. With ``merge_divergent=True`` divergent
    units are assigned to clubs in every run (after merging, if any).

    Examples
    --------
    >>> runs = sensitivity(data, time_trim=[1/3, 0.3], cstar=[0, -1], merge=["PS"])
    >>> runs.table()            # one row per run
    >>> runs.stability()        # how firmly each unit sits in its club
    >>> runs.plot()             # co-membership heatmap
    """
    grid: Mapping[str, Sequence[Any]] = {
        "time_trim": tuple(time_trim),
        "hac_method": tuple(hac_method),
        "cstar": tuple(cstar),
        "merge": tuple(merge),
        "sort_fraction": tuple(sort_fraction),
    }
    if any(len(v) == 0 for v in grid.values()):
        raise ValueError("every setting needs at least one value")
    runs = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values, strict=True))
        result = find_clubs(
            data,
            time_trim=params["time_trim"],
            hac_method=params["hac_method"],
            cstar=params["cstar"],
            sort_fraction=params["sort_fraction"],
            **find_kwargs,
        )
        if params["merge"] is not None:
            result = result.merge(params["merge"], merge_divergent=merge_divergent)
        elif merge_divergent:
            result = result.merge_divergent()
        runs.append((params, result))
    return SensitivityResult(tuple(runs))
