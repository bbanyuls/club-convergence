"""Plots of convergence clubs (requires matplotlib: ``pip install 'convergence-clubs[plot]'``).

Every single-panel function accepts ``ax=`` so it can draw into an existing
figure, and returns the axes; :func:`plot_transition_paths` composes them into
a grid and returns the figure. All of them take a ``theme`` (``"light"``,
``"dark"`` or a :class:`Theme`) and ``colors`` to override club colours.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .logt import cross_sectional_variation, log_t_regression_data, log_t_test, relative_transition

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .results import Club, ConvergenceClubs

ClubKey = str | int
Colors = Sequence[str] | Mapping[str, str] | None
UnitsStyle = Literal["lines", "band", "none"]


@dataclass(frozen=True)
class Theme:
    """Colours used by the plots.

    ``colors`` are the categorical club colours, assigned in order: a club
    keeps its colour in every panel and when other clubs are filtered out.
    Clubs past ``len(colors)`` reuse the hues with a dashed line rather than
    generating new colours that would be hard to tell apart.
    """

    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    baseline: str
    colors: tuple[str, ...]
    unit_alpha: float = 0.45
    font_size: float = 9.0

    def with_colors(self, colors: Sequence[str]) -> Theme:
        """Copy of the theme with other club colours."""
        return replace(self, colors=tuple(colors))


# Categorical slots validated for colour-vision deficiency on adjacent pairs,
# with separate steps for light and dark surfaces.
LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    colors=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)
DARK = Theme(
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    colors=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    unit_alpha=0.5,
)
THEMES: dict[str, Theme] = {"light": LIGHT, "dark": DARK}

_DASHED = (0, (4, 2))


# -- helpers -------------------------------------------------------------------


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting requires matplotlib: pip install 'convergence-clubs[plot]'"
        ) from exc
    return plt


def _theme(theme: str | Theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"theme must be one of {sorted(THEMES)} or a Theme") from None


def club_styles(
    result: ConvergenceClubs, theme: str | Theme = "light", colors: Colors = None
) -> dict[str, dict[str, Any]]:
    """Colour and line style of every club, keyed by club name.

    ``colors`` may be a sequence (in club order) or a mapping from club name
    to colour; clubs it does not cover get the theme colours.
    """
    th = _theme(theme)
    n = len(th.colors)
    styles = {}
    for pos, club in enumerate(result.clubs):
        if isinstance(colors, Mapping) and club.name in colors:
            style = {"color": colors[club.name], "linestyle": "-"}
        elif isinstance(colors, Sequence) and not isinstance(colors, str) and pos < len(colors):
            style = {"color": colors[pos], "linestyle": "-"}
        else:
            style = {"color": th.colors[pos % n], "linestyle": "-" if pos < n else _DASHED}
        styles[club.name] = style
    return styles


def _select(result: ConvergenceClubs, clubs: ClubKey | Sequence[ClubKey] | None) -> list[Club]:
    if clubs is None:
        return list(result.clubs)
    if isinstance(clubs, (str, int)):
        clubs = [clubs]
    return [result[c] for c in clubs]


def _x_values(columns: pd.Index) -> tuple[Any, list[str] | None]:
    """x positions of the periods, and tick labels when periods are not numbers or dates."""
    if isinstance(columns, pd.PeriodIndex):
        return columns.to_timestamp(), None
    if pd.api.types.is_numeric_dtype(columns) or pd.api.types.is_datetime64_any_dtype(columns):
        return np.asarray(columns), None
    return np.arange(len(columns)), [str(c) for c in columns]


def _new_axes(ax: Axes | None, theme: Theme, figsize=(6.4, 3.6)) -> Axes:
    if ax is not None:
        return ax
    _, ax = _plt().subplots(figsize=figsize, facecolor=theme.surface, layout="constrained")
    return ax


def _style_axes(ax: Axes, theme: Theme, x=None, labels: list[str] | None = None) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.figure.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    ax.grid(True, axis="y", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.baseline)
    ax.tick_params(
        colors=theme.ink_muted, labelcolor=theme.ink_secondary, length=0, labelsize=theme.font_size
    )
    if labels is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: labels[int(v)] if 0 <= int(v) < len(labels) else "")
        )
    elif x is not None and np.issubdtype(np.asarray(x).dtype, np.integer):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10]))


def _decorate(ax: Axes, theme: Theme, title: str | None, ylabel: str | None, xlabel=None) -> None:
    if title:
        ax.set_title(title, loc="left", color=theme.ink, fontsize=theme.font_size + 1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=theme.font_size, color=theme.ink_secondary)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=theme.font_size, color=theme.ink_secondary)


def _legend(ax: Axes, theme: Theme, **kwargs) -> None:
    ax.legend(
        frameon=False,
        fontsize=theme.font_size - 0.5,
        labelcolor=theme.ink_secondary,
        handlelength=1.6,
        **kwargs,
    )


def _end_labels(ax: Axes, theme: Theme, x_end, entries: list[tuple[str, float]]) -> bool:
    """Label line ends unless the labels would collide; returns whether they were drawn."""
    if not entries:
        return False
    ys = sorted(y for _, y in entries)
    lo, hi = ax.get_ylim()
    if len(ys) > 1 and min(np.diff(ys)) < 0.06 * (hi - lo):
        return False
    for text, y in entries:
        ax.annotate(
            text,
            (x_end, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=theme.font_size - 0.5,
            color=theme.ink_secondary,
            annotation_clip=False,
        )
    return True


def _end_dot(ax: Axes, theme: Theme, x, y, color: str) -> None:
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=6,
        color=color,
        markeredgecolor=theme.surface,
        markeredgewidth=1.5,
        zorder=5,
        label="_nolegend_",
    )


# -- single panels -------------------------------------------------------------


def plot_club(
    result: ConvergenceClubs,
    club: ClubKey,
    *,
    ax: Axes | None = None,
    units: UnitsStyle = "lines",
    band: tuple[float, float] = (0.1, 0.9),
    highlight: Sequence[Any] | None = None,
    show_average: bool = True,
    reference_line: bool = False,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
    ylabel: str | None = "Relative transition path",
    unit_kws: Mapping[str, Any] | None = None,
    average_kws: Mapping[str, Any] | None = None,
) -> Axes:
    """Transition paths of the units of one club.

    Parameters
    ----------
    units
        ``"lines"`` draws one thin muted line per unit, ``"band"`` a shaded
        band between the ``band`` quantiles, ``"none"`` nothing.
    highlight
        Unit names to emphasise and label; units outside the club are ignored.
    show_average
        Draw the club average in the club colour.
    reference_line
        Draw ``h = 1``, the cross-sectional average of all units.
    unit_kws, average_kws
        Extra matplotlib keyword arguments for the unit lines (or band) and
        the average line.
    """
    th = _theme(theme)
    ax = _new_axes(ax, th)
    c = result[club]
    style = club_styles(result, th, colors)[c.name]
    x, labels = _x_values(result.data.columns)
    h = relative_transition(result.values)[list(c.ids)]

    if units == "lines":
        kws = {"color": th.ink_muted, "linewidth": 0.8, "alpha": th.unit_alpha, **(unit_kws or {})}
        for row in h:
            ax.plot(x, row, zorder=2, **kws)
    elif units == "band":
        lo, hi = np.quantile(h, band, axis=0)
        kws = {"color": style["color"], "alpha": 0.14, "linewidth": 0, **(unit_kws or {})}
        ax.fill_between(x, lo, hi, zorder=1, **kws)
    elif units != "none":
        raise ValueError("units must be 'lines', 'band' or 'none'")

    if reference_line:
        ax.axhline(1.0, color=th.baseline, linewidth=1, zorder=1)
    if show_average:
        kws = {**style, "linewidth": 2, "solid_capstyle": "round", **(average_kws or {})}
        ax.plot(x, h.mean(axis=0), zorder=3, label=f"{c.name} average", **kws)

    names = list(result.data.index[list(c.ids)])
    entries = []
    for unit in highlight or ():
        if unit in names:
            row = h[names.index(unit)]
            ax.plot(x, row, color=th.ink, linewidth=1.4, zorder=4, label=str(unit))
            entries.append((str(unit), row[-1]))

    _style_axes(ax, th, x, labels)
    if entries and not _end_labels(ax, th, x[-1], entries):
        _legend(ax, th, loc="best")
    _decorate(ax, th, f"{c.name} · {len(c)} units" if title == "auto" else title, ylabel)
    return ax


def plot_average_paths(
    result: ConvergenceClubs,
    *,
    ax: Axes | None = None,
    clubs: ClubKey | Sequence[ClubKey] | None = None,
    emphasis: ClubKey | Sequence[ClubKey] | None = None,
    direct_labels: bool | Literal["auto"] = "auto",
    legend: bool = True,
    reference_line: bool = True,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "Average transition paths",
    ylabel: str | None = "Relative transition path",
    line_kws: Mapping[str, Any] | None = None,
) -> Axes:
    """Average transition path of each club, on one panel.

    Parameters
    ----------
    emphasis
        Club(s) to keep in colour; the others are drawn in muted grey.
    direct_labels
        Label the end of each coloured line. ``"auto"`` does so for up to
        four clubs, and only when the labels would not collide.
    reference_line
        Draw ``h = 1``, the cross-sectional average of all units.
    line_kws
        Extra matplotlib keyword arguments for the lines.
    """
    th = _theme(theme)
    ax = _new_axes(ax, th)
    styles = club_styles(result, th, colors)
    emphasised = None if emphasis is None else {c.name for c in _select(result, emphasis)}
    x, labels = _x_values(result.data.columns)
    h = relative_transition(result.values)

    if reference_line:
        ax.axhline(1.0, color=th.baseline, linewidth=1, zorder=1)
    selected = _select(result, clubs)
    entries = []
    muted_label = "other clubs"
    for c in selected:
        mean = h[list(c.ids)].mean(axis=0)
        muted = emphasised is not None and c.name not in emphasised
        # Muted clubs all look alike, so they share a single legend entry.
        label = c.name
        if muted:
            label, muted_label = muted_label, "_nolegend_"
        kws = {
            **styles[c.name],
            "linewidth": 2,
            "solid_capstyle": "round",
            **({"color": th.ink_muted, "linewidth": 1.2, "alpha": 0.8} if muted else {}),
            **(line_kws or {}),
        }
        ax.plot(x, mean, label=label, zorder=2 if muted else 3, **kws)
        if not muted:
            _end_dot(ax, th, x[-1], mean[-1], kws["color"])
            entries.append((c.name, mean[-1]))

    _style_axes(ax, th, x, labels)
    if direct_labels is True or (direct_labels == "auto" and len(entries) <= 4):
        _end_labels(ax, th, x[-1], entries)
    if legend and len(selected) > 1:
        # Emphasised clubs first, then the shared entry of the muted ones.
        handles = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
        handles.sort(key=lambda ln: ln.get_label() == "other clubs")
        if len(handles) > 4:  # a long legend would cover the lines
            _legend(ax, th, handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
            _legend(ax, th, handles=handles, loc="best")
    _decorate(ax, th, title, ylabel)
    return ax


def plot_cross_sectional_variation(
    result: ConvergenceClubs,
    *,
    ax: Axes | None = None,
    clubs: ClubKey | Sequence[ClubKey] | None = None,
    include_all: bool = True,
    log_scale: bool = True,
    show_window: bool = True,
    legend: bool = True,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "Cross-sectional variation within clubs",
) -> Axes:
    """Cross-sectional variation ``H_t`` of each club over time.

    ``H_t`` shrinks towards zero when a group converges, so converging clubs
    slope down even when the whole panel (``include_all``) does not.
    ``show_window`` marks where the log t regression window starts.
    """
    th = _theme(theme)
    ax = _new_axes(ax, th)
    styles = club_styles(result, th, colors)
    x, labels = _x_values(result.data.columns)

    if include_all:
        H = cross_sectional_variation(result.values)
        ax.plot(x, H, color=th.ink_muted, linewidth=1.5, label="all units", zorder=2)
    for c in _select(result, clubs):
        H = cross_sectional_variation(result.values[list(c.ids)])
        ax.plot(x, H, label=c.name, linewidth=2, solid_capstyle="round", zorder=3, **styles[c.name])
    if log_scale:
        ax.set_yscale("log")
    if show_window:
        start = x[round(len(x) * result.time_trim)]
        ax.axvline(start, color=th.baseline, linewidth=1, zorder=1)
        ax.annotate(
            "log t window",
            (start, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            va="top",
            fontsize=th.font_size - 1,
            color=th.ink_muted,
        )
    _style_axes(ax, th, x, labels)
    if legend and len(ax.get_lines()) > 2:
        _legend(ax, th, loc="best")
    _decorate(ax, th, title, "H(t)")
    return ax


def plot_log_t(
    result: ConvergenceClubs,
    club: ClubKey | None = None,
    *,
    ax: Axes | None = None,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Diagnostic plot of the log t regression of a club (of all units if ``club`` is None).

    Shows ``log(H_1/H_t) - 2 log(log t)`` against ``log t`` over the
    regression window, the fitted line and the test outcome: the null of
    convergence is rejected when ``t < -1.65``.
    """
    th = _theme(theme)
    ax = _new_axes(ax, th)
    if club is None:
        ids, name, color = list(range(len(result.data))), "all units", th.ink_secondary
    else:
        c = result[club]
        ids, name = list(c.ids), c.name
        color = club_styles(result, th, colors)[c.name]["color"]

    H = cross_sectional_variation(result.values[ids])
    log_t, y = log_t_regression_data(H, result.time_trim)
    model = log_t_test(H, result.time_trim, result.hac_method)

    ax.scatter(log_t, y, s=30, color=color, edgecolors=th.surface, linewidths=1.2, zorder=3)
    grid = np.linspace(log_t.min(), log_t.max(), 50)
    ax.plot(grid, model.alpha + model.beta * grid, color=th.ink, linewidth=1.5, zorder=2)
    verdict = "convergence not rejected" if model.converges() else "convergence rejected"
    ax.text(
        0.02,
        0.96,
        f"β = {model.beta:.3f} (s.e. {model.std_err:.3f})\nt = {model.t_value:.2f}: {verdict}",
        transform=ax.transAxes,
        va="top",
        fontsize=th.font_size - 0.5,
        color=th.ink_secondary,
    )
    _style_axes(ax, th)
    _decorate(
        ax,
        th,
        f"log t regression · {name}" if title == "auto" else title,
        "log(H₁/Hₜ) − 2 log(log t)",
        xlabel="log t",
    )
    return ax


# -- grid ----------------------------------------------------------------------


def plot_transition_paths(
    result: ConvergenceClubs,
    *,
    clubs: ClubKey | Sequence[ClubKey] | None = None,
    kind: Literal["both", "clubs", "average"] = "both",
    units: UnitsStyle = "lines",
    band: tuple[float, float] = (0.1, 0.9),
    highlight: Sequence[Any] | None = None,
    emphasis: ClubKey | Sequence[ClubKey] | None = None,
    ncols: int = 3,
    panel_size: tuple[float, float] = (3.6, 2.6),
    sharey: bool = False,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = None,
    ylabel: str | None = "Relative transition path",
    legend: bool = True,
) -> Figure:
    """Grid of transition path panels: one per club, then the club averages.

    Parameters
    ----------
    clubs
        Clubs to draw (names or positions); all by default. Colours follow
        the club, not its position in this selection.
    kind
        ``"clubs"`` (one panel per club), ``"average"`` (club averages only)
        or ``"both"``.
    units, band, highlight
        How unit paths are drawn in the club panels; see :func:`plot_club`.
    emphasis
        Club(s) to keep in colour on the averages panel.
    sharey
        Use the same y-axis range in every club panel.
    """
    if kind not in ("both", "clubs", "average"):
        raise ValueError("kind must be 'both', 'clubs' or 'average'")
    if not result.clubs:
        raise ValueError("there are no clubs to plot")
    th = _theme(theme)
    selected = _select(result, clubs)
    club_panels = selected if kind in ("both", "clubs") else []
    with_average = kind in ("both", "average")
    n_panels = len(club_panels) + with_average
    ncols = max(1, min(ncols, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = _plt().subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        facecolor=th.surface,
        layout="constrained",
    )
    flat = list(axes.flat)

    for ax, c in zip(flat, club_panels, strict=False):
        plot_club(
            result, c.name, ax=ax, units=units, band=band, highlight=highlight,
            theme=th, colors=colors, ylabel=None,
        )  # fmt: skip
    if sharey and club_panels:
        limits = [ax.get_ylim() for ax in flat[: len(club_panels)]]
        for ax in flat[: len(club_panels)]:
            ax.set_ylim(min(lo for lo, _ in limits), max(hi for _, hi in limits))
    if with_average:
        ax = flat[len(club_panels)]
        plot_average_paths(
            result, ax=ax, clubs=[c.name for c in selected], emphasis=emphasis,
            legend=False, theme=th, colors=colors, ylabel=None,
        )  # fmt: skip
        handles = [line for line in ax.get_lines() if not line.get_label().startswith("_")]
        if legend and len(handles) > 1:
            fig.legend(
                handles=handles,
                loc="outside lower center",
                ncols=min(len(handles), 8),
                frameon=False,
                fontsize=th.font_size,
                labelcolor=th.ink_secondary,
            )
    for ax in flat[n_panels:]:
        ax.set_visible(False)

    if ylabel:
        fig.supylabel(ylabel, color=th.ink_secondary, fontsize=th.font_size + 1)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=th.ink, fontsize=th.font_size + 3)
    return fig
