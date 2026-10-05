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

from .beta import _club_labels, _club_order
from .logt import cross_sectional_variation, log_t_regression_data, log_t_test, relative_transition

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .beta import BetaResult
    from .determinants import DeterminantsResult
    from .dynamics import DynamicsResult
    from .results import Club, ConvergenceClubs
    from .sensitivity import SensitivityResult
    from .sigma import SigmaResult

ClubKey = str | int
Colors = Sequence[str] | Mapping[str, str] | None
UnitsStyle = Literal["lines", "band", "none"]


_BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)  # fmt: skip


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
    #: One-hue ramp from "nothing" to "a lot", used for heatmaps.
    sequential: tuple[str, ...] = _BLUE_RAMP

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
    # On a dark surface "nothing" must recede into the surface: start near it
    # and run the ramp towards light.
    sequential=("#22262d", *_BLUE_RAMP[::-1]),
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
    styles: dict[str, dict[str, Any]] = {}
    for pos, club in enumerate(result.clubs):
        style: dict[str, Any]
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
        handles = [ln for ln in ax.get_lines() if not str(ln.get_label()).startswith("_")]
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
    n_entries = len([ln for ln in ax.get_lines() if not str(ln.get_label()).startswith("_")])
    if legend and n_entries > 1:
        if n_entries > 4:  # a long legend would cover the lines
            _legend(ax, th, loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
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


def plot_co_membership(
    sensitivity: SensitivityResult,
    *,
    ax: Axes | None = None,
    unit_labels: bool | Literal["auto"] = "auto",
    colorbar: bool = True,
    theme: str | Theme = "light",
    title: str | None = "auto",
) -> Axes:
    """Heatmap of how often each pair of units ends up in the same club.

    Units are ordered by their baseline club and the baseline clubs are
    outlined, so stable clubs show as solid dark blocks and units that
    switch clubs as lighter stripes. ``unit_labels="auto"`` prints unit
    names for up to 60 units and club names otherwise.
    """
    from matplotlib.colors import LinearSegmentedColormap

    th = _theme(theme)
    ax = _new_axes(ax, th, figsize=(6.4, 5.6))
    base = sensitivity.baseline
    order = [i for club in base.clubs for i in club.ids] + list(base.divergent_ids)
    blocks = [(club.name, len(club)) for club in base.clubs]
    if base.n_divergent:
        blocks.append(("divergent", base.n_divergent))
    share = sensitivity.co_membership().to_numpy()[np.ix_(order, order)]

    cmap = LinearSegmentedColormap.from_list("share", th.sequential)
    image = ax.imshow(share, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    edges = np.cumsum([size for _, size in blocks])[:-1] - 0.5
    for edge in edges:
        ax.axhline(edge, color=th.surface, linewidth=1.5)
        ax.axvline(edge, color=th.surface, linewidth=1.5)

    for spine in (ax.spines[side] for side in ("top", "right", "bottom", "left")):
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=th.font_size - 1, labelcolor=th.ink_secondary)
    names = base.data.index[order]
    if unit_labels is True or (unit_labels == "auto" and len(order) <= 60):
        ax.set_xticks(range(len(order)), [str(n) for n in names], rotation=90)
        ax.set_yticks(range(len(order)), [str(n) for n in names])
    else:
        starts = np.concatenate([[0], np.cumsum([size for _, size in blocks])[:-1]])
        centres = [start + size / 2 - 0.5 for start, (_, size) in zip(starts, blocks, strict=True)]
        ax.set_yticks(centres, [name for name, _ in blocks])
        ax.set_xticks([])
    ax.set_facecolor(th.surface)
    ax.figure.set_facecolor(th.surface)

    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, shrink=0.85)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=th.font_size - 1, labelcolor=th.ink_secondary)
        bar.set_label(
            "share of runs in the same club", color=th.ink_secondary, fontsize=th.font_size
        )
    auto = f"Co-membership across {len(sensitivity)} runs"
    _decorate(ax, th, auto if title == "auto" else title, None)
    return ax


def plot_stability(
    sensitivity: SensitivityResult,
    *,
    ax: Axes | None = None,
    threshold: float | None = 0.8,
    label: int = 8,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Confidence of every unit in its baseline club, one row of dots per club.

    Each dot is a unit placed at its :meth:`SensitivityResult.stability`
    (the share of runs spent with its baseline club mates). Up to ``label``
    of the least stable units are named (labels that would overlap are
    skipped) and ``threshold`` is marked.
    """
    th = _theme(theme)
    base = sensitivity.baseline
    ax = _new_axes(ax, th, figsize=(6.4, 0.9 + 0.5 * max(1, base.n_clubs)))
    styles = club_styles(base, th, colors)
    st = sensitivity.stability().dropna(subset=["stability"])
    rows = {c.name: i for i, c in enumerate(base.clubs)}

    # Units sorted by stability get a repeating vertical offset, so that neighbours
    # on the x-axis sit apart without suggesting any order in the y direction.
    st = st.sort_values("stability")
    rank = st.groupby("baseline_club").cumcount().to_numpy()
    y = st["baseline_club"].map(rows).to_numpy() + ((rank % 5) - 2) * 0.09
    for club in base.clubs:
        mask = (st["baseline_club"] == club.name).to_numpy()
        ax.scatter(
            st["stability"].to_numpy()[mask], y[mask], s=34, color=styles[club.name]["color"],
            edgecolors=th.surface, linewidths=1.2, zorder=3,
        )  # fmt: skip

    if threshold is not None:
        ax.axvline(threshold, color=th.ink_muted, linewidth=1, zorder=1)
        ax.annotate(
            f"{threshold:g}", (threshold, 0), xycoords=("data", "axes fraction"),
            xytext=(3, 3), textcoords="offset points", fontsize=th.font_size - 1,
            color=th.ink_muted,
        )  # fmt: skip

    _style_axes(ax, th)
    ax.grid(False)
    ax.set_yticks(range(base.n_clubs), [c.name for c in base.clubs])
    ax.set_ylim(base.n_clubs - 0.5, -0.5)
    ax.set_xlim(min(0.0, st["stability"].min() - 0.05), 1.02)
    auto = f"Membership confidence across {len(sensitivity) - 1} resamples"
    _decorate(
        ax, th, auto if title == "auto" else title, None,
        xlabel="share of runs with the baseline club mates",
    )  # fmt: skip

    # Name the least stable units, skipping labels that would overlap. Window extents
    # are measured with the figure's own renderer, which works with every backend.
    placed: list[Any] = []
    for i in range(min(label, len(st))):
        text = ax.annotate(
            str(st.index[i]), (st["stability"].iloc[i], y[i]), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=th.font_size - 1,
            color=th.ink_secondary,
        )  # fmt: skip
        box = text.get_window_extent().expanded(1.05, 1.2)
        if any(box.overlaps(other) for other in placed):
            text.remove()
        else:
            placed.append(box)
    return ax


def _group_colors(clubs: Any, groups: Sequence[str], th: Theme, colors: Colors) -> dict[str, str]:
    """Colour per club label: club colours, muted ink for divergent units."""
    styles = club_styles(clubs, th, colors) if hasattr(clubs, "clubs") else {}
    out, k = {}, 0
    for group in groups:
        if group == "divergent":
            out[group] = th.ink_muted
        elif group in styles:
            out[group] = styles[group]["color"]
        else:
            out[group] = th.colors[k % len(th.colors)]
            k += 1
    return out


def plot_beta(
    result: BetaResult,
    *,
    ax: Axes | None = None,
    label: Sequence[Any] | None = None,
    show_fit: bool = True,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Average growth against initial (log) level, with the fitted β line.

    Units are coloured by club when the result has clubs; with club effects
    each club gets its own (parallel) fitted line. ``label`` names units.
    """
    from matplotlib.ticker import PercentFormatter

    from .beta import _club_order, _fitted_line

    th = _theme(theme)
    ax = _new_axes(ax, th)
    units = result.units
    if "club" in units:
        groups = [g for g in _club_order(result.clubs) if g in set(units["club"])]
        palette = _group_colors(result.clubs, groups, th, colors)
    else:
        groups, palette = ["all units"], {"all units": th.colors[0]}

    for group in groups:
        part = units if group == "all units" else units[units["club"] == group]
        ax.scatter(
            part["log_initial"], part["growth"], s=30, color=palette[group],
            edgecolors=th.surface, linewidths=1.2, zorder=3, label=group,
        )  # fmt: skip
        if show_fit and result.club_effects and group != "all units" and len(part) > 1:
            x = np.linspace(part["log_initial"].min(), part["log_initial"].max(), 20)
            ax.plot(x, _fitted_line(result, x, group), color=palette[group], linewidth=2,
                    solid_capstyle="round", zorder=4, label="_nolegend_")  # fmt: skip
    if show_fit and not result.club_effects:
        x = np.linspace(units["log_initial"].min(), units["log_initial"].max(), 50)
        ax.plot(x, _fitted_line(result, x), color=th.ink, linewidth=1.5, zorder=4,
                label="_nolegend_")  # fmt: skip

    years = pd.api.types.is_numeric_dtype(pd.Index([result.start, result.end]))
    unit = "years" if years else "periods"
    if not result.converges():
        life = "n/a (β not significantly negative)"
    elif np.isfinite(result.half_life):
        life = f"{result.half_life:.0f} {unit}"
    else:
        life = "n/a"
    ax.text(
        0.98, 0.96,
        f"β = {result.beta:.4f} (s.e. {result.std_err:.4f}), t = {result.t_value:.2f}\n"
        f"half-life: {life}",
        transform=ax.transAxes, ha="right", va="top", fontsize=th.font_size - 0.5,
        color=th.ink_secondary,
    )  # fmt: skip
    for name in label or ():
        if name in units.index:
            ax.annotate(
                str(name), (units.loc[name, "log_initial"], units.loc[name, "growth"]),
                xytext=(5, 3), textcoords="offset points", fontsize=th.font_size - 1,
                color=th.ink_secondary,
            )  # fmt: skip

    _style_axes(ax, th)
    ax.axhline(0, color=th.baseline, linewidth=1, zorder=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    if len(groups) > 1:
        if len(groups) > 4:
            _legend(ax, th, loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
            _legend(ax, th, loc="lower left")
    per = "per year" if years else "per period"
    auto = "β-convergence" + (" conditional on clubs" if result.club_effects else "")
    _decorate(
        ax, th, auto if title == "auto" else title, f"average growth {per}",
        xlabel=f"log level in {result.start}",
    )  # fmt: skip
    return ax


def plot_sigma(
    result: SigmaResult,
    *,
    ax: Axes | None = None,
    groups: Sequence[str] | None = None,
    index: bool = False,
    direct_labels: bool | Literal["auto"] = "auto",
    legend: bool = True,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Dispersion over time for all units (muted) and within each club (club colours).

    With ``index=True`` every series is divided by its first value, so groups
    with very different spreads can be compared (1 = first period).
    """
    from .sigma import ALL, LABELS

    th = _theme(theme)
    ax = _new_axes(ax, th)
    series = result.series if groups is None else result.series[list(groups)]
    if index:
        series = series / series.iloc[0]
    x, labels = _x_values(series.index)
    palette = _group_colors(
        result.clubs, [g for g in series.columns if g != ALL], th, colors
    )  # fmt: skip

    if index:
        ax.axhline(1.0, color=th.baseline, linewidth=1, zorder=1)
    entries = []
    for group in series.columns:
        y = series[group].to_numpy()
        kws: dict[str, Any]
        if group == ALL:
            kws = {"color": th.ink_muted, "linewidth": 2, "zorder": 2}
        else:
            kws = {"color": palette[group], "linewidth": 2, "zorder": 3}
        ax.plot(x, y, label=group, solid_capstyle="round", **kws)
        _end_dot(ax, th, x[-1], y[-1], kws["color"])
        entries.append((group, y[-1]))

    _style_axes(ax, th, x, labels)
    if direct_labels is True or (direct_labels == "auto" and len(entries) <= 4):
        _end_labels(ax, th, x[-1], entries)
    if legend and len(entries) > 1:
        if len(entries) > 4:
            _legend(ax, th, loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
            _legend(ax, th, loc="best")
    ylabel = LABELS[result.measure] + (" (first period = 1)" if index else "")
    auto = "σ-convergence" + (" (indexed)" if index else "")
    _decorate(ax, th, auto if title == "auto" else title, ylabel)
    return ax


def _luminance(color: str) -> float:
    """Relative luminance (WCAG) of a colour."""
    from matplotlib.colors import to_rgb

    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _is_dark(th: Theme) -> bool:
    return _luminance(th.surface) < 0.5


def _text_on(color: str, th: Theme) -> str:
    """Ink or white, whichever reads better on a cell of ``color``."""
    return "#ffffff" if _luminance(color) < 0.4 else "#0b0b0b"


def plot_transitions(
    result: DynamicsResult,
    *,
    ax: Axes | None = None,
    annotate: bool = True,
    theme: str | Theme = "light",
    title: str | None = "auto",
) -> Axes:
    """Heatmap of the transition matrix, with the probabilities written in the cells."""
    from matplotlib.colors import LinearSegmentedColormap

    th = _theme(theme)
    ax = _new_axes(ax, th, figsize=(5.6, 4.8))
    P = result.matrix.to_numpy()
    cmap = LinearSegmentedColormap.from_list("share", th.sequential)
    ax.imshow(P, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    k = result.n_classes
    for edge in np.arange(k - 1) + 0.5:  # surface gaps between cells
        ax.axhline(edge, color=th.surface, linewidth=2)
        ax.axvline(edge, color=th.surface, linewidth=2)
    if annotate:
        for i in range(k):
            for j in range(k):
                if np.isfinite(P[i, j]):
                    ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                            fontsize=th.font_size, color=_text_on(cmap(P[i, j]), th))  # fmt: skip
    for spine in (ax.spines[side] for side in ("top", "right", "bottom", "left")):
        spine.set_visible(False)
    ax.set_facecolor(th.surface)
    ax.figure.set_facecolor(th.surface)
    ax.tick_params(length=0, labelsize=th.font_size - 1, labelcolor=th.ink_secondary)
    ax.set_xticks(range(k), result.labels, rotation=30, ha="right")
    ax.set_yticks(range(k), result.labels)
    auto = f"Transition probabilities ({result.lag}-period)"
    _decorate(
        ax, th, auto if title == "auto" else title, "relative income at t",
        xlabel=f"relative income at t + {result.lag}",
    )  # fmt: skip
    return ax


def plot_class_distributions(
    result: DynamicsResult,
    *,
    ax: Axes | None = None,
    theme: str | Theme = "light",
    title: str | None = "Share of units in each class",
) -> Axes:
    """Initial, final and ergodic (long-run) shares of each class, as grouped bars."""
    from matplotlib.ticker import PercentFormatter

    th = _theme(theme)
    ax = _new_axes(ax, th, figsize=(5.6, 4.8))
    shares = result.distributions()
    k, n = shares.shape
    width = 0.8 / n
    for i, column in enumerate(shares.columns):
        ax.bar(
            np.arange(k) + (i - (n - 1) / 2) * width, shares[column], width=width * 0.92,
            color=th.colors[i], label=column, zorder=3,
        )  # fmt: skip
    _style_axes(ax, th)
    ax.set_xticks(range(k), shares.index, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _legend(ax, th, loc="upper center", ncols=n)
    ax.set_ylim(0, shares.to_numpy().max() * 1.2)
    _decorate(ax, th, title, None, xlabel="relative income class")
    return ax


def plot_dynamics(result: DynamicsResult, *, theme: str | Theme = "light") -> Figure:
    """Transition matrix (left) and initial / final / ergodic distributions (right)."""
    th = _theme(theme)
    fig, axes = _plt().subplots(
        1, 2, figsize=(11.5, 4.6), facecolor=th.surface, layout="constrained",
        gridspec_kw={"width_ratios": [1.05, 1]},
    )  # fmt: skip
    plot_transitions(result, ax=axes[0], theme=th)
    plot_class_distributions(result, ax=axes[1], theme=th)
    return fig


def plot_densities(
    result: DynamicsResult,
    *,
    ax: Axes | None = None,
    periods: Sequence[Any] | None = None,
    rug: bool = True,
    bandwidth: float | None = None,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Kernel densities of relative income in some periods (first and last by default).

    The latest period is drawn in the accent colour and earlier ones in
    lighter steps of the same hue. With clubs, a rug at the bottom shows
    where each club's units are in the latest period.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    th = _theme(theme)
    ax = _new_axes(ax, th)
    columns = list(result.values.columns)
    periods = [columns[0], columns[-1]] if periods is None else list(periods)
    if len(periods) == 2:
        line_colors = [th.ink_muted, th.colors[0]]
    else:
        # Ordinal ramp: later periods stand out more from the surface. The first
        # steps of the sequential ramp sit too close to the surface for lines.
        ramp = th.sequential[2:] if _is_dark(th) else th.sequential[3:]
        picks = np.linspace(0, len(ramp) - 1, len(periods)).round().astype(int)
        line_colors = [ramp[i] for i in picks]

    ax.axvline(1.0, color=th.baseline, linewidth=1, zorder=1)
    for period, color in zip(periods, line_colors, strict=True):
        density = result.density(period, bandwidth=bandwidth)
        ax.plot(density.index, density.to_numpy(), color=color, linewidth=2,
                solid_capstyle="round", label=str(period), zorder=3)  # fmt: skip

    if rug and result.clubs is not None:
        labels = _club_labels(result.clubs, result.values.index)
        groups = [g for g in _club_order(result.clubs) if g in set(labels)]
        palette = _group_colors(result.clubs, groups, th, colors)
        last = result.values[periods[-1]]
        for group in groups:
            x = last[labels == group].to_numpy()
            ax.plot(x, np.full(len(x), 0.025), linestyle="none", marker="|", markersize=9,
                    color=palette[group], transform=ax.get_xaxis_transform(), label=group,
                    zorder=4)  # fmt: skip

    ax.set_xscale("log")
    low, high = result.values.to_numpy().min() / 2, result.values.to_numpy().max() * 2
    ticks = [t for t in (0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8) if low <= t <= high]
    _style_axes(ax, th)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(bottom=0)
    _legend(ax, th, loc="upper left", ncols=2 if rug and result.clubs is not None else 1)
    auto = "Distribution of relative income"
    _decorate(
        ax, th, auto if title == "auto" else title, "density (of log relative income)",
        xlabel="income relative to the cross-sectional mean (log scale)",
    )  # fmt: skip
    return ax


def plot_determinants(
    result: DeterminantsResult,
    *,
    ax: Axes | None = None,
    level: float = 0.95,
    theme: str | Theme = "light",
    title: str | None = "auto",
) -> Axes:
    """Coefficients of the ordered model with ``level`` confidence intervals.

    Positive coefficients raise the probability of belonging to a higher club.
    """
    from statistics import NormalDist

    th = _theme(theme)
    coef = result.coefficients.iloc[::-1]  # first covariate on top
    ax = _new_axes(ax, th, figsize=(6.4, 0.9 + 0.45 * len(coef)))
    z = NormalDist().inv_cdf(0.5 + level / 2)
    y = np.arange(len(coef))
    low = coef["coef"] - z * coef["std_err"]
    high = coef["coef"] + z * coef["std_err"]
    ax.axvline(0, color=th.baseline, linewidth=1, zorder=1)
    ax.hlines(y, low, high, color=th.colors[0], linewidth=2, zorder=2)
    ax.scatter(coef["coef"], y, s=40, color=th.colors[0], edgecolors=th.surface,
               linewidths=1.5, zorder=3)  # fmt: skip
    _style_axes(ax, th)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=th.grid, linewidth=1)
    ax.set_yticks(y, list(coef.index))
    auto = f"Determinants of club membership (ordered {result.model})"
    _decorate(
        ax, th, auto if title == "auto" else title, None,
        xlabel=f"coefficient ({level:.0%} interval); > 0 means a higher club is more likely",
    )  # fmt: skip
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
        handles = [line for line in ax.get_lines() if not str(line.get_label()).startswith("_")]
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
