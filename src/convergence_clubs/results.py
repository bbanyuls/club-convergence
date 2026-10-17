"""Result containers returned by :func:`find_clubs` and :func:`merge_clubs`."""

from __future__ import annotations

import html
import shutil
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .logt import LogTResult, relative_transition

if TYPE_CHECKING:
    from matplotlib.axes import Axes
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

    @property
    def converges(self) -> bool:
        """Whether the club passes its own log t test (t > -1.65)."""
        return self.model.converges()


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

    @property
    def n_clubs(self) -> int:
        return len(self.clubs)

    @property
    def n_divergent(self) -> int:
        return len(self.divergent_ids)

    @property
    def failing_clubs(self) -> tuple[str, ...]:
        """Names of clubs that do not pass their own log t test.

        With ``cstar_method="fixed"`` the clustering procedure accepts a club
        even when the club as a whole fails the test; Phillips and Sul (2007)
        suggest raising ``c*`` then (``cstar_method="incremental"``).
        """
        return tuple(club.name for club in self.clubs if not club.converges)

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

    def summary(self, members: bool = False) -> pd.DataFrame:
        """One row per club: size, log t regression results and ``c*`` or merged clubs.

        With ``members=True`` a column lists the units of each club; for a
        ready-to-publish LaTeX table use :meth:`to_latex`.
        """
        rows = []
        for club in self.clubs:
            row: dict[str, Any] = {
                "n_units": len(club),
                "beta": club.model.beta,
                "std_err": club.model.std_err,
                "t_value": club.model.t_value,
                "p_value": club.model.p_value,
                "converges": club.converges,
            }
            if self.is_merged:
                row["merged_clubs"] = ", ".join(club.merged_from or ())
            else:
                row["cstar"] = club.cstar
            if members:
                # "; " because unit names may contain commas ("Korea, Republic of").
                row["members"] = "; ".join(map(str, self.units(club.name)))
            rows.append(row)
        return pd.DataFrame(rows, index=pd.Index([c.name for c in self.clubs], name="club"))

    def to_latex(self, **kwargs) -> str:
        """The clubs as a booktabs LaTeX table: size, log t coefficient, t and members.

        Divergent units get a row of their own and a note states the
        settings. See :func:`convergence_clubs.latex.clubs_table` for the
        options (``members``, ``caption``, ``label``, ``longtable``, ``path``).
        """
        from .latex import clubs_table

        return clubs_table(self, **kwargs)

    def merge(self, method: str = "PS", **kwargs) -> ConvergenceClubs:
        """Merge adjacent clubs; see :func:`convergence_clubs.merge_clubs`."""
        from .merging import merge_clubs

        return merge_clubs(self, method=method, **kwargs)

    def merge_divergent(self, **kwargs) -> ConvergenceClubs:
        """Assign divergent units to clubs; see :func:`convergence_clubs.merge_divergent`."""
        from .merging import merge_divergent

        return merge_divergent(self, **kwargs)

    def rename(
        self, names: Mapping[str, str] | Sequence[str] | Callable[[Club], str]
    ) -> ConvergenceClubs:
        """Copy with the clubs renamed, e.g. for plots and tables in a paper.

        ``names`` is a mapping from current to new names (clubs it leaves out
        keep theirs), a sequence of names in club order, or a function of the
        :class:`Club`. Rename after merging: merging numbers the clubs again.

        >>> clubs.rename(["High", "Upper middle", "Lower middle", "Low"])
        >>> clubs.rename({"club1": "Frontier"})
        >>> clubs.rename(lambda club: f"Club {club.name[4:]} ({len(club)})")
        """
        if isinstance(names, Mapping):
            unknown = set(names) - {c.name for c in self.clubs}
            if unknown:
                raise KeyError(f"no club called {sorted(unknown)}")
            new = [names.get(c.name, c.name) for c in self.clubs]
        elif callable(names):
            new = [names(c) for c in self.clubs]
        elif isinstance(names, str):
            raise TypeError("names must be a mapping, a sequence of names or a function")
        else:
            new = list(names)
            if len(new) != self.n_clubs:
                raise ValueError(f"got {len(new)} names for {self.n_clubs} clubs")
        for name in new:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"club names must be non-empty strings, got {name!r}")
        if any(name.startswith("_") for name in new):
            raise ValueError("club names must not start with '_' (matplotlib hides them)")
        if "divergent" in new:
            raise ValueError('"divergent" is reserved for divergent units')
        if len(set(new)) != len(new):
            raise ValueError(f"club names must be unique, got {new}")
        clubs = tuple(replace(c, name=n) for c, n in zip(self.clubs, new, strict=True))
        return self._replace(clubs=clubs)

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

    @property
    def plot(self) -> PlotAccessor:
        """Plotting methods.

        ``result.plot()`` draws the grid of transition paths;
        ``result.plot.club(...)``, ``.average()``, ``.variation()`` and
        ``.log_t(...)`` draw single panels. See :mod:`convergence_clubs.plotting`.
        """
        return PlotAccessor(self)

    def _replace(self, **changes) -> ConvergenceClubs:
        return replace(self, **changes)

    def __repr__(self) -> str:
        kind = "merged " if self.is_merged else ""
        return (
            f"<ConvergenceClubs: {self.n_clubs} {kind}clubs, "
            f"{self.n_divergent} divergent units, hac_method={self.hac_method!r}>"
        )

    def _repr_html_(self) -> str:
        summary = self.summary().to_html(float_format=lambda v: f"{v:.3f}", border=0)
        divergent = html.escape(", ".join(map(str, self.divergent))) or "none"
        return (
            f"<p><b>{self.n_clubs} convergence clubs</b>, {self.n_divergent} divergent units "
            f"(hac_method={self.hac_method}, time_trim={self.time_trim:.3g})</p>"
            f"{summary}<p>Divergent units: {divergent}</p>"
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
            if not club.converges:
                out.append("warning:   the club does not pass its own log t test (t <= -1.65)")
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


class PlotAccessor:
    """Plotting methods of a :class:`ConvergenceClubs` result."""

    def __init__(self, result: ConvergenceClubs) -> None:
        self._result = result

    def __call__(self, **kwargs) -> Figure:
        """Grid of transition paths; see :func:`~convergence_clubs.plot_transition_paths`."""
        from .plotting import plot_transition_paths

        return plot_transition_paths(self._result, **kwargs)

    def club(self, club: int | str, **kwargs) -> Axes:
        """Paths of one club's units; see :func:`~convergence_clubs.plot_club`."""
        from .plotting import plot_club

        return plot_club(self._result, club, **kwargs)

    def average(self, **kwargs) -> Axes:
        """Average path per club; see :func:`~convergence_clubs.plot_average_paths`."""
        from .plotting import plot_average_paths

        return plot_average_paths(self._result, **kwargs)

    def variation(self, **kwargs) -> Axes:
        """H(t) per club; see :func:`~convergence_clubs.plot_cross_sectional_variation`."""
        from .plotting import plot_cross_sectional_variation

        return plot_cross_sectional_variation(self._result, **kwargs)

    def interactive(self, **kwargs):
        """Interactive plotly figure; see :mod:`convergence_clubs.interactive`."""
        from .interactive import transition_paths_figure

        return transition_paths_figure(self._result, **kwargs)

    def map(self, geodata, **kwargs) -> Axes:
        """Regions coloured by club (geopandas); see :func:`convergence_clubs.maps.plot_map`."""
        from .maps import plot_map

        return plot_map(self._result, geodata, **kwargs)

    def log_t(self, club: int | str | None = None, **kwargs) -> Axes:
        """log t regression diagnostic; see :func:`~convergence_clubs.plot_log_t`."""
        from .plotting import plot_log_t

        return plot_log_t(self._result, club, **kwargs)
