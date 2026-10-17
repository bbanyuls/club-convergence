"""Convergence clubs in rolling windows: how memberships change over time."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._parallel import parallel_map
from .bootstrap import _cluster
from .sensitivity import adjusted_rand_index

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

DIVERGENT = "divergent"

Direction = Literal["up", "down", "to divergent", "from divergent", "mates diverged"]


@dataclass(frozen=True)
class RollingClubs:
    """Clubs found in each of a sequence of time windows (see :func:`rolling_clubs`).

    Club names are numbered within each window (``club1`` is the highest),
    so ``club2`` in one window and ``club2`` in the next are not the same
    group: compare them through :meth:`transitions` and :meth:`movers`.
    """

    windows: tuple[str, ...]
    results: tuple[ConvergenceClubs, ...]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, window: int | str) -> ConvergenceClubs:
        if isinstance(window, str):
            return self.results[self.windows.index(window)]
        return self.results[window]

    def memberships(self) -> pd.DataFrame:
        """Club of every unit in every window (units x windows)."""
        return pd.concat(
            [r.memberships().rename(w) for w, r in zip(self.windows, self.results, strict=True)],
            axis=1,
        )

    def summary(self) -> pd.DataFrame:
        """Clubs per window, and the agreement (ARI) with the previous window."""
        rows = []
        for i, (window, result) in enumerate(zip(self.windows, self.results, strict=True)):
            previous = self.results[i - 1] if i else None
            rows.append(
                {
                    "window": window,
                    "n_clubs": result.n_clubs,
                    "n_divergent": result.n_divergent,
                    "sizes": [len(c) for c in result.clubs],
                    "ari_vs_previous": (
                        adjusted_rand_index(previous, result) if previous is not None else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows).set_index("window")

    def transitions(self, start: int | str = 0, end: int | str = -1) -> pd.DataFrame:
        """Number of units in each club of window ``start`` (rows) and ``end`` (columns)."""
        a, b = self._name(start), self._name(end)
        members = self.memberships()
        table = pd.crosstab(members[a].rename(a), members[b].rename(b))
        return table.reindex(
            index=_order(self[a], table.index), columns=_order(self[b], table.columns)
        )

    def movers(self) -> pd.DataFrame:
        """Units that do not follow most of their club mates into the next window.

        Club numbers are not comparable across windows (with more clubs,
        everyone gets a higher number), so a unit is compared with its own
        club mates: if most of them go to club ``c`` in the next window and
        the unit goes elsewhere, it moved. ``direction`` is ``"up"`` (to a
        higher club than its mates), ``"down"``, ``"to divergent"``,
        ``"from divergent"`` (a divergent unit that joins a club) or
        ``"mates diverged"`` (the unit stayed in a club but most of its mates
        became divergent). ``mates_went_to`` is the club most of its mates
        went to.
        """
        members = self.memberships()
        rows = []
        for a, b in zip(self.windows[:-1], self.windows[1:], strict=True):
            rank = _ranks(self[b]).to_dict()
            for club, mates in members.groupby(members[a])[b]:
                modal = mates.value_counts().index[0]
                for unit, target in mates.items():
                    if club != DIVERGENT and target == modal:
                        continue
                    if target == DIVERGENT and club == DIVERGENT:
                        continue
                    direction: Direction
                    if club == DIVERGENT:
                        direction = "from divergent"
                    elif target == DIVERGENT:
                        direction = "to divergent"
                    elif modal == DIVERGENT:
                        direction = "mates diverged"
                    else:
                        direction = "up" if rank[unit] < _club_rank(self[b], modal) else "down"
                    rows.append(
                        {
                            "unit": unit,
                            "from_window": a,
                            "to_window": b,
                            "from_club": club,
                            "to_club": target,
                            "mates_went_to": modal if club != DIVERGENT else "",
                            "direction": direction,
                        }
                    )
        columns = ["unit", "from_window", "to_window", "from_club", "to_club", "mates_went_to"]
        return pd.DataFrame(rows, columns=[*columns, "direction"])

    def plot(self, **kwargs) -> Axes:
        """Flows of units between the clubs of consecutive windows; see :func:`plot_rolling`."""
        from .plotting import plot_rolling

        return plot_rolling(self, **kwargs)

    def plot_interactive(self, **kwargs):
        """The same flows as a plotly Sankey diagram; see :func:`interactive.rolling_figure`."""
        from .interactive import rolling_figure

        return rolling_figure(self, **kwargs)

    def _name(self, window: int | str) -> str:
        return window if isinstance(window, str) else self.windows[window]

    def __repr__(self) -> str:
        sizes = ", ".join(str(r.n_clubs) for r in self.results)
        return f"<RollingClubs: {len(self)} windows, clubs per window: {sizes}>"


def _order(result: ConvergenceClubs, present: Iterable) -> list:
    names = [c.name for c in result.clubs] + [DIVERGENT]
    return [n for n in names if n in set(present)]


def _ranks(result: ConvergenceClubs) -> pd.Series:
    ranks = pd.Series(np.nan, index=result.data.index)
    for position, club in enumerate(result.clubs, start=1):
        ranks.iloc[list(club.ids)] = position
    return ranks


def _club_rank(result: ConvergenceClubs, name: str) -> int:
    """Position of a club (1 = the highest)."""
    return [c.name for c in result.clubs].index(name) + 1


def _label(columns: pd.Index, start: int, stop: int) -> str:
    return f"{columns[start]}–{columns[stop - 1]}"


def rolling_clubs(
    data: pd.DataFrame,
    window: int,
    *,
    step: int | None = None,
    filter_lambda: float | None = None,
    merge: str | None = "PS",
    merge_divergent: bool = False,
    merge_options: Mapping[str, Any] | None = None,
    n_jobs: int | None = 1,
    **find_kwargs,
) -> RollingClubs:
    """Find convergence clubs in rolling windows of ``window`` periods.

    Parameters
    ----------
    data
        Units x periods. With ``filter_lambda`` the data are taken as
        unfiltered (e.g. log GDP) and HP-filtered within each window, so no
        window sees later data through the filter; otherwise they are used
        as given.
    window
        Periods per window.
    step
        Periods between the starts of consecutive windows (default
        ``window // 2``, windows that overlap by half). The last window
        always ends with the last period.
    merge, merge_divergent, merge_options
        Applied in every window, as in :func:`sensitivity`.
    n_jobs
        Processes to run the windows in (``-1``: all CPUs).
    **find_kwargs
        Passed to :func:`find_clubs` (``time_trim``, ``hac_method``, ...).

    Notes
    -----
    Short windows give the log t test little to work with: the regression
    uses only the last ``1 - time_trim`` of each window.
    """
    n_t = data.shape[1]
    if not 5 <= window <= n_t:
        raise ValueError(f"window must be between 5 and the number of periods ({n_t})")
    step = max(1, window // 2) if step is None else step
    if step < 1:
        raise ValueError("step must be at least 1")
    starts = list(range(0, n_t - window + 1, step))
    if starts[-1] + window < n_t:
        starts.append(n_t - window)
    windows = [_label(data.columns, s, s + window) for s in starts]
    settings = (filter_lambda, merge, merge_divergent, merge_options, find_kwargs)
    tasks = [(data.iloc[:, s : s + window], settings) for s in starts]
    return RollingClubs(tuple(windows), tuple(parallel_map(_cluster, tasks, n_jobs)))
