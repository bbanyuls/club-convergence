"""Transition path plots (requires matplotlib: ``pip install convergence-clubs[plot]``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .results import ConvergenceClubs

# Categorical palette in fixed slot order (validated for colour-vision deficiency
# on adjacent pairs); a club keeps its colour in every panel.
CLUB_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def club_color(position: int) -> str:
    """Colour of the club at ``position`` (0-based); clubs past the 8th share muted ink."""
    return CLUB_COLORS[position] if position < len(CLUB_COLORS) else INK_MUTED


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=9)


def plot_transition_paths(
    result: ConvergenceClubs,
    *,
    clubs: Sequence[str | int] | None = None,
    units: bool = True,
    average: bool = True,
    ncols: int = 3,
    panel_size: tuple[float, float] = (3.6, 2.6),
) -> Figure:
    """Plot relative transition paths of convergence clubs.

    One panel per club shows each unit's path (thin, muted) and the club
    average (coloured); a final panel compares the club averages. Values are
    relative to the cross-sectional mean of all units, so ``h = 1`` (the
    reference line of the comparison panel) is the panel average.

    Parameters
    ----------
    result
        Output of :func:`find_clubs` or :func:`merge_clubs`.
    clubs
        Clubs to draw (names or positions); all clubs by default.
    units
        Draw one panel per club with the paths of its units.
    average
        Draw a panel comparing the average path of each club.
    ncols
        Number of panel columns.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting requires matplotlib: pip install 'convergence-clubs[plot]'"
        ) from exc

    if not result.clubs:
        raise ValueError("there are no clubs to plot")
    positions = (
        list(range(result.n_clubs))
        if clubs is None
        else [c if isinstance(c, int) else [k.name for k in result.clubs].index(c) for c in clubs]
    )
    paths = result.transition_paths()
    periods = list(result.data.columns)
    x = range(len(periods))

    n_panels = len(positions) * units + average
    if n_panels == 0:
        raise ValueError("nothing to plot: set units=True or average=True")
    ncols = max(1, min(ncols, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        sharex=True,
        facecolor=SURFACE,
        layout="constrained",
    )
    flat = list(axes.flat)
    tick_step = max(1, len(periods) // 5)
    ticks = list(x)[::tick_step]

    panel = 0
    if units:
        for pos in positions:
            club = result.clubs[pos]
            ax = flat[panel]
            panel += 1
            _style_axes(ax)
            h = paths.loc[club.name].to_numpy()
            for row in h:
                ax.plot(x, row, color=INK_MUTED, linewidth=0.8, alpha=0.45, zorder=2)
            ax.plot(
                x,
                h.mean(axis=0),
                color=club_color(pos),
                linewidth=2,
                solid_capstyle="round",
                zorder=3,
            )
            ax.set_title(f"{club.name} · {len(club)} units", loc="left", color=INK, fontsize=10)

    if average:
        ax = flat[panel]
        panel += 1
        _style_axes(ax)
        ax.axhline(1.0, color=BASELINE, linewidth=1, zorder=1)
        for pos in positions:
            club = result.clubs[pos]
            mean = paths.loc[club.name].to_numpy().mean(axis=0)
            ax.plot(
                x,
                mean,
                color=club_color(pos),
                linewidth=2,
                solid_capstyle="round",
                label=club.name,
                zorder=3,
            )
        ax.set_title("Average transition paths", loc="left", color=INK, fontsize=10)
        fig.legend(
            frameon=False,
            fontsize=9,
            labelcolor=INK_SECONDARY,
            loc="outside lower center",
            ncols=min(len(positions), 8),
        )

    for ax in flat[panel:]:
        ax.set_visible(False)
    for ax in flat[:panel]:
        ax.set_xticks(ticks, [str(periods[i]) for i in ticks])
        ax.tick_params(labelbottom=True)
    fig.supylabel("Relative transition path", color=INK_SECONDARY, fontsize=10)
    return fig
