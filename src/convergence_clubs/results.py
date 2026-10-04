"""Result containers returned by :func:`find_clubs` and :func:`merge_clubs`."""

from __future__ import annotations

import shutil
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .logt import LogTResult, relative_transition

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Club:
    """A convergence club.

    ``ids`` are row positions in the data passed to :func:`find_clubs`.
    """

    name: str
    ids: tuple[int, ...]
    model: LogTResult
    cstar: float | None = None
    merged_from: tuple[str, ...] | None = None
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True, eq=False)
class ConvergenceClubs:
    """Convergence clubs and divergent units found in a panel.

    Iterating yields the :class:`Club` objects; clubs can also be looked up
    by position or by name (``result["club2"]``).
    """

    clubs: tuple[Club, ...]
    divergent_ids: tuple[int, ...]
    data: pd.DataFrame
    time_trim: float
    hac_method: str
    params: dict[str, Any] = field(default_factory=dict)

    # -- container protocol -------------------------------------------------

    def __iter__(self) -> Iterator[Club]:
        return iter(self.clubs)

    def __len__(self) -> int:
        return len(self.clubs)

    def __getitem__(self, key: int | str) -> Club:
        if isinstance(key, str):
            for club in self.clubs:
                if club.name == key:
                    return club
            raise KeyError(key)
        return self.clubs[key]

    # -- basic information --------------------------------------------------

    @property
    def n_clubs(self) -> int:
        return len(self.clubs)

    @property
    def n_divergent(self) -> int:
        return len(self.divergent_ids)

    @property
    def is_merged(self) -> bool:
        return any(club.merged_from is not None for club in self.clubs)

    @property
    def divergent(self) -> pd.Index:
        """Names of the divergent units."""
        return self.data.index[list(self.divergent_ids)]

    @cached_property
    def values(self) -> np.ndarray:
        """Data as a float array (units x periods), in the original row order."""
        return self.data.to_numpy(dtype=float)

    def units(self, club: int | str) -> pd.Index:
        """Names of the units in ``club``."""
        return self.data.index[list(self[club].ids)]

    def memberships(self) -> pd.Series:
        """Club of every unit (``"divergent"`` for divergent ones), in data order."""
        labels = np.full(len(self.data), "divergent", dtype=object)
        for club in self.clubs:
            labels[list(club.ids)] = club.name
        return pd.Series(labels, index=self.data.index, name="club")

    def summary(self) -> pd.DataFrame:
        """One row per club: size, log t regression results and ``c*`` or merged clubs."""
        rows = []
        for club in self.clubs:
            row = {
                "n_units": len(club),
                "beta": club.model.beta,
                "std_err": club.model.std_err,
                "t_value": club.model.t_value,
                "p_value": club.model.p_value,
            }
            if self.is_merged:
                row["merged_clubs"] = ", ".join(club.merged_from or ())
            else:
                row["cstar"] = club.cstar
            rows.append(row)
        return pd.DataFrame(rows, index=pd.Index([c.name for c in self.clubs], name="club"))

    # -- transformations ----------------------------------------------------

    def merge(self, method: str = "PS", **kwargs) -> ConvergenceClubs:
        """Merge adjacent clubs; see :func:`convergence_clubs.merge_clubs`."""
        from .merging import merge_clubs

        return merge_clubs(self, method=method, **kwargs)

    def merge_divergent(self, **kwargs) -> ConvergenceClubs:
        """Assign divergent units to clubs; see :func:`convergence_clubs.merge_divergent`."""
        from .merging import merge_divergent

        return merge_divergent(self, **kwargs)

    def transition_paths(self) -> pd.DataFrame:
        """Relative transition paths ``h_it`` of every club member.

        ``h`` is computed relative to the cross-sectional mean of *all* units,
        so paths are comparable across clubs. Rows are indexed by
        ``(club, unit)``; divergent units are excluded.
        """
        h = relative_transition(self.values)
        frames = []
        for club in self.clubs:
            ids = list(club.ids)
            index = pd.MultiIndex.from_arrays(
                [[club.name] * len(ids), self.data.index[ids]], names=["club", "unit"]
            )
            frames.append(pd.DataFrame(h[ids], index=index, columns=self.data.columns))
        if not frames:
            return pd.DataFrame(columns=self.data.columns)
        return pd.concat(frames)

    def plot(self, **kwargs) -> Figure:
        """Plot transition paths; see :func:`convergence_clubs.plot_transition_paths`."""
        from .plotting import plot_transition_paths

        return plot_transition_paths(self, **kwargs)

    def _replace(self, **changes) -> ConvergenceClubs:
        return replace(self, **changes)

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "merged " if self.is_merged else ""
        return (
            f"<ConvergenceClubs: {self.n_clubs} {kind}clubs, "
            f"{self.n_divergent} divergent units, hac_method={self.hac_method!r}>"
        )

    def __str__(self) -> str:
        width = min(shutil.get_terminal_size((88, 20)).columns, 100)
        out = []
        for club in self.clubs:
            out += ["=" * width, club.name, "-" * width]
            out += textwrap.wrap(", ".join(map(str, self.units(club.name))), width)
            out.append("")
            out.append(f"beta:     {club.model.beta: .4f}")
            out.append(f"std.err:  {club.model.std_err: .4f}")
            out.append(f"t-value:  {club.model.t_value: .4f}")
            out.append(f"p-value:  {club.model.p_value: .4f}")
            if club.cstar is not None:
                out.append(f"cstar:    {club.cstar: .4f}")
            if club.merged_from is not None:
                out.append(f"merged:    {' + '.join(club.merged_from)}")
            out += [f"note:      {note}" for note in club.notes]
            out.append("")
        if self.divergent_ids:
            out += ["=" * width, "divergent", "-" * width]
            out += textwrap.wrap(", ".join(map(str, self.divergent)), width)
            out.append("")
        return "\n".join(out)
