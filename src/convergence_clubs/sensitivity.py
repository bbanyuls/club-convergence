"""Robustness of club memberships to the settings of the clustering procedure."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._parallel import parallel_map
from .clustering import ClubConvergenceWarning, find_clubs

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

DIVERGENT = "divergent"
#: Label of a divergent unit; clubs are numbered from 0.
DIVERGENT_LABEL = -1
#: Label of a unit that is not part of a run (e.g. left out of a bootstrap subsample).
ABSENT = -2


def _labels(result: ConvergenceClubs) -> np.ndarray:
    """Club index of every unit, ``DIVERGENT_LABEL`` for divergent units."""
    labels = np.full(len(result.data), DIVERGENT_LABEL)
    for k, club in enumerate(result.clubs):
        labels[list(club.ids)] = k
    return labels


def _aligned_labels(result: ConvergenceClubs, units: pd.Index) -> np.ndarray:
    """Labels of ``result`` on ``units``, with ``ABSENT`` for units it does not cover."""
    positions = units.get_indexer(result.data.index)
    if (positions < 0).any():
        raise ValueError("a run contains units that are not in the baseline")
    out = np.full(len(units), ABSENT)
    out[positions] = _labels(result)
    return out


def _singletons(labels: np.ndarray) -> np.ndarray:
    """Labels where every divergent unit is a group of its own."""
    labels = labels.copy()
    divergent = labels < 0
    labels[divergent] = labels.max(initial=DIVERGENT_LABEL) + 1 + np.arange(divergent.sum())
    return labels


def _partition(result: ConvergenceClubs) -> np.ndarray:
    """Labels where every divergent unit is a singleton group."""
    return _singletons(_labels(result))


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
    both_partitions_trivial = maximum == expected
    if both_partitions_trivial:
        return 1.0 if np.array_equal(x == x[:, None], y == y[:, None]) else 0.0
    return float((index - expected) / (maximum - expected))


def compare_clubs(a: ConvergenceClubs, b: ConvergenceClubs) -> pd.DataFrame:
    """Cross-tabulation of the memberships of two results (clubs of ``a`` in rows)."""
    if not a.data.index.equals(b.data.index):
        raise ValueError("both results must cover the same units in the same order")
    return pd.crosstab(a.memberships().rename("a"), b.memberships().rename("b"))


@dataclass(frozen=True, eq=False)
class SensitivityResult:
    """Clubs found in several runs: a grid of settings or resamples of the data.

    The first run is the baseline that the others are compared to. Runs may
    cover a subset of the baseline units (bootstrap subsamples): shares are
    then computed over the runs that include the units involved.
    """

    runs: tuple[tuple[dict[str, Any], ConvergenceClubs], ...]

    @cached_property
    def _label_matrix(self) -> np.ndarray:
        """runs x units labels (club index, ``DIVERGENT_LABEL``, ``ABSENT``)."""
        return np.array([_aligned_labels(r, self.units) for _, r in self.runs])

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
        base = self._label_matrix[0]
        rows = []
        for (params, result), labels in zip(self.runs, self._label_matrix, strict=True):
            present = labels != ABSENT
            rows.append(
                {
                    **{k: "none" if v is None else v for k, v in params.items()},
                    "n_units": int(present.sum()),
                    "n_clubs": result.n_clubs,
                    "n_divergent": result.n_divergent,
                    "n_failing": len(result.failing_clubs),
                    "sizes": "-".join(str(len(c)) for c in result.clubs),
                    "ari_vs_baseline": _ari(
                        _singletons(base[present]), _singletons(labels[present])
                    ),
                }
            )
        return pd.DataFrame(rows)

    def co_membership(self) -> pd.DataFrame:
        """Share of runs in which each pair of units belongs to the same club.

        Only runs that include both units count. Divergent units are never
        grouped with anyone; the diagonal is the share of runs in which a
        unit belongs to some club. Pairs never sampled together are NaN.
        """
        n = len(self.units)
        together, sampled = np.zeros((n, n)), np.zeros((n, n))
        for labels in self._label_matrix:
            present = labels != ABSENT
            together += (labels[:, None] == labels[None, :]) & (labels >= 0)[:, None]
            sampled += present[:, None] & present[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(sampled > 0, together / sampled, np.nan)
        return pd.DataFrame(share, index=self.units, columns=self.units)

    def stability(self) -> pd.DataFrame:
        """How firmly each unit belongs to its baseline club.

        ``stability`` is the average share of runs in which the unit is in
        the same club as each of its baseline club mates (1 = always with
        all of them); ``divergent_share`` is the share of runs in which it
        is divergent; ``n_runs`` is the number of runs that include it.
        """
        share = self.co_membership().to_numpy()
        labels = self._label_matrix
        base = labels[0]
        stability = np.full(len(base), np.nan)
        for i, label in enumerate(base):
            mates = np.flatnonzero((base == label) & (np.arange(len(base)) != i))
            shares = share[i, mates]
            co_sampled = np.isfinite(shares)
            shares = shares[co_sampled]
            if label >= 0 and shares.size:
                stability[i] = shares.mean()
        n_runs = (labels != ABSENT).sum(axis=0)
        divergent = (labels == DIVERGENT_LABEL).sum(axis=0) / n_runs
        return pd.DataFrame(
            {
                "baseline_club": self.baseline.memberships().to_numpy(),
                "stability": stability,
                "divergent_share": divergent,
                "n_runs": n_runs,
            },
            index=self.units,
        )

    def uncertain(self, threshold: float = 0.8) -> pd.DataFrame:
        """Units whose :meth:`stability` is below ``threshold``, least stable first."""
        st = self.stability()
        return st[st["stability"] < threshold].sort_values("stability")

    def plot(self, **kwargs) -> Axes:
        """Co-membership heatmap; see :func:`convergence_clubs.plot_co_membership`."""
        from .plotting import plot_co_membership

        return plot_co_membership(self, **kwargs)

    def plot_interactive(self, **kwargs):
        """Interactive co-membership heatmap (plotly); see :mod:`convergence_clubs.interactive`."""
        from .interactive import co_membership_figure

        return co_membership_figure(self, **kwargs)

    def plot_stability(self, **kwargs) -> Axes:
        """Per-unit confidence dots; see :func:`convergence_clubs.plot_stability`."""
        from .plotting import plot_stability

        return plot_stability(self, **kwargs)


def sensitivity(
    data: pd.DataFrame,
    *,
    time_trim: Sequence[float] = (1 / 3, 0.25, 0.3, 0.4),
    hac_method: Sequence[str] = ("FQSB", "AQSB"),
    cstar: Sequence[float] = (0.0,),
    merge: Sequence[str | None] = (None,),
    sort_fraction: Sequence[float] = (0.0,),
    merge_divergent: bool = False,
    merge_options: Mapping[str, Any] | None = None,
    n_jobs: int | None = 1,
    **find_kwargs,
) -> SensitivityResult:
    """Run the clustering for every combination of settings.

    Each of ``time_trim``, ``hac_method``, ``cstar``, ``merge`` (``None``,
    ``"PS"`` or ``"vLT"``) and ``sort_fraction`` is a list of values; the
    grid is their product, in that order, so the first value of each forms
    the baseline run. Other keyword arguments are passed to
    :func:`find_clubs` unchanged. With ``merge_divergent=True`` divergent
    units are assigned to clubs in every run (after merging, if any);
    ``merge_options`` are passed to :func:`merge_clubs` (e.g. ``iterate``).
    ``n_jobs`` runs the grid in several processes (``-1``: all CPUs).

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
    combos = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]
    settings = (merge_divergent, merge_options, find_kwargs)
    results = parallel_map(_run, [(data, params, settings) for params in combos], n_jobs)
    return SensitivityResult(tuple(zip(combos, results, strict=True)))


def _run(task: tuple[pd.DataFrame, dict[str, Any], tuple]) -> ConvergenceClubs:
    """One run of the grid; a worker of :func:`sensitivity`."""
    data, params, (merge_divergent, merge_options, find_kwargs) = task
    result = _quiet_find_clubs(
        data,
        time_trim=params["time_trim"],
        hac_method=params["hac_method"],
        cstar=params["cstar"],
        sort_fraction=params["sort_fraction"],
        **find_kwargs,
    )
    return _post_process(result, params["merge"], merge_divergent, merge_options)


def _quiet_find_clubs(data: pd.DataFrame, **kwargs) -> ConvergenceClubs:
    """:func:`find_clubs` without failing-club warnings (counted in ``table()`` instead)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClubConvergenceWarning)
        return find_clubs(data, **kwargs)


def _post_process(
    result: ConvergenceClubs,
    merge: str | None,
    merge_divergent: bool,
    merge_options: Mapping[str, Any] | None = None,
) -> ConvergenceClubs:
    """Optional club merging and assignment of divergent units."""
    options = dict(merge_options or {})
    if merge is not None:
        return result.merge(merge, merge_divergent=merge_divergent, **options)
    if merge_divergent:
        keep = {"estar", "joint_first", "time_trim"}
        return result.merge_divergent(**{k: v for k, v in options.items() if k in keep})
    return result
