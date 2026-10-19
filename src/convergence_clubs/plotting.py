"""Plots of convergence clubs (requires matplotlib: ``pip install 'convergence-clubs[plot]'``).

Every single-panel function accepts ``ax=`` so it can draw into an existing
figure, and returns the axes; :func:`plot_transition_paths` composes them into
a grid and returns the figure. All of them take a ``theme`` (the name of a
registered theme -- ``"light"``, ``"dark"``, ``"print"`` -- or a :class:`Theme`;
by default the one chosen with :func:`set_theme`) and ``colors`` to override
club colours.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
    from .pairwise import PairwiseResult
    from .results import Club, ConvergenceClubs
    from .rolling import RollingClubs
    from .sensitivity import SensitivityResult
    from .sigma import SigmaResult
    from .spatial import SpatialResult

ClubKey = str | int
Colors = Sequence[str] | Mapping[str, str] | None
UnitsStyle = Literal["lines", "band", "none"]


_BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)  # fmt: skip
_LIGHT_SEQUENTIAL = ("#eef4fc", *_BLUE_RAMP)


_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])  # fmt: skip
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])  # fmt: skip


def _to_oklab(color: str) -> np.ndarray:
    rgb = np.array([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return _M2 @ np.cbrt(_M1 @ linear)


def _from_oklab(lab: np.ndarray) -> str:
    linear = np.linalg.solve(_M1, np.linalg.solve(_M2, lab) ** 3)
    srgb = np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055
    )
    return "#" + "".join(f"{round(v * 255):02x}" for v in np.clip(srgb, 0, 1))


@dataclass(frozen=True)
class Theme:
    """Colours used by the plots.

    ``colors`` are the categorical club colours, assigned in order: a club
    keeps its colour in every panel and when other clubs are filtered out.
    Clubs past ``len(colors)`` reuse the hues with the next of ``linestyles``
    (matplotlib line styles) rather than generating new colours that would be
    hard to tell apart; once the styles run out the last one is kept.
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
    sequential: tuple[str, ...] = _LIGHT_SEQUENTIAL
    linestyles: tuple[Any, ...] = ("-", (0, (4, 2)))
    #: Stops of a one-hue ramp from "low" to "high", for ordered classes
    #: (clubs on a map, initial -> final -> ergodic); see :func:`ordinal_colors`.
    ordinal: tuple[str, ...] = ("#86b6ef", "#2a78d6", "#0d366b")
    #: Width of the main data lines (club averages, H(t), densities, ...);
    #: secondary lines are drawn in proportion.
    line_width: float = 2.0
    #: Area of scatter markers, in points squared.
    marker_size: float = 30.0
    #: Horizontal gridlines.
    gridlines: bool = True
    #: ``"minimal"`` (baseline only, no ticks) or ``"box"`` (all four spines and
    #: outward ticks, the usual look in journals).
    frame: Literal["minimal", "box"] = "minimal"
    #: Alignment of panel titles: ``"left"``, ``"center"`` or ``"right"``.
    title_loc: Literal["left", "center", "right"] = "left"

    def __post_init__(self) -> None:
        if (
            not self.colors
            or not self.linestyles
            or min(len(self.sequential), len(self.ordinal)) < 2
        ):
            raise ValueError("a Theme needs colours, line styles and two-stop ramps")
        if self.frame not in ("minimal", "box"):
            raise ValueError("frame must be 'minimal' or 'box'")
        if self.title_loc not in ("left", "center", "right"):
            raise ValueError("title_loc must be 'left', 'center' or 'right'")
        if self.line_width <= 0 or self.marker_size <= 0:
            raise ValueError("line_width and marker_size must be positive")

    def with_colors(self, colors: Sequence[str]) -> Theme:
        """Copy of the theme with other club colours."""
        return replace(self, colors=tuple(colors))

    def replace(self, **changes: Any) -> Theme:
        """Copy of the theme with some fields changed, e.g. ``LIGHT.replace(font_size=11)``."""
        return replace(self, **changes)


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
    sequential=("#22262d", *_BLUE_RAMP[::-1]),
    ordinal=("#184f95", "#3987e5", "#cde2fb"),
)
PRINT = Theme(
    surface="#ffffff",
    ink="#000000",
    ink_secondary="#333333",
    ink_muted="#6b6b6b",
    grid="#e6e6e6",
    baseline="#9e9e9e",
    colors=("#000000", "#555555", "#8c8c8c"),
    unit_alpha=0.3,
    sequential=("#f5f5f5", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525"),
    linestyles=("-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))),
    ordinal=("#c4c4c4", "#6b6b6b", "#000000"),
)
JOURNAL = LIGHT.replace(
    surface="#ffffff", frame="box", gridlines=False, title_loc="center", line_width=1.6
)
THEMES: dict[str, Theme] = {"light": LIGHT, "dark": DARK, "print": PRINT, "journal": JOURNAL}
_default_theme: list[str | Theme] = ["light"]


def register_theme(name: str, theme: Theme) -> None:
    """Make ``theme`` available by name, e.g. ``plot(theme=name)``."""
    if not isinstance(theme, Theme):
        raise TypeError("theme must be a Theme")
    THEMES[name] = theme


def get_theme(theme: str | Theme | None = None) -> Theme:
    """The :class:`Theme` called ``theme``, or the default one (see :func:`set_theme`)."""
    if theme is None:
        theme = _default_theme[0]
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"theme must be one of {sorted(THEMES)} or a Theme") from None


def set_theme(theme: str | Theme) -> None:
    """Use ``theme`` in every plot that is not given one explicitly."""
    get_theme(theme)
    _default_theme[0] = theme


@contextmanager
def theme_context(theme: str | Theme) -> Iterator[Theme]:
    """Use ``theme`` as the default inside a ``with`` block."""
    previous = _default_theme[0]
    set_theme(theme)
    try:
        yield get_theme()
    finally:
        _default_theme[0] = previous


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting requires matplotlib: pip install 'convergence-clubs[plot]'"
        ) from exc
    return plt


_theme = get_theme

#: Most ordered classes an ordinal ramp keeps apart (adjacent lightness gaps
#: stay >= 0.06 in OKLab up to seven steps, in both built-in modes).
MAX_ORDINAL = 7


def ordinal_colors(n: int, theme: str | Theme | None = None) -> list[str]:
    """``n`` colours from the theme's ordinal ramp, low to high, evenly spaced in lightness.

    The ramp is interpolated in OKLab, so the steps look evenly spaced; up to
    :data:`MAX_ORDINAL` classes stay distinguishable.
    """
    th = _theme(theme)
    labs = np.array([_to_oklab(c) for c in th.ordinal])
    stops = np.linspace(0, 1, len(labs))
    positions = np.linspace(0, 1, n) if n > 1 else np.array([1.0])
    return [
        _from_oklab(np.array([np.interp(t, stops, labs[:, k]) for k in range(3)]))
        for t in positions
    ]


def club_styles(
    result: ConvergenceClubs, theme: str | Theme | None = None, colors: Colors = None
) -> dict[str, dict[str, Any]]:
    """Colour and line style of every club, keyed by club name.

    ``colors`` may be a sequence (in club order) or a mapping from club name
    to colour; clubs it does not cover get the theme colours.
    """
    th = _theme(theme)
    n, last = len(th.colors), len(th.linestyles) - 1
    solid = th.linestyles[0]
    styles: dict[str, dict[str, Any]] = {}
    for pos, club in enumerate(result.clubs):
        style: dict[str, Any]
        if isinstance(colors, Mapping) and club.name in colors:
            style = {"color": colors[club.name], "linestyle": solid}
        elif isinstance(colors, Sequence) and not isinstance(colors, str) and pos < len(colors):
            style = {"color": colors[pos], "linestyle": solid}
        else:
            linestyle = th.linestyles[min(pos // n, last)]
            style = {"color": th.colors[pos % n], "linestyle": linestyle}
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
    # grid(False, color=...) would turn the grid on: style arguments imply visible.
    if theme.gridlines:
        ax.grid(True, axis="y", color=theme.grid, linewidth=1)
    else:
        ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    box = theme.frame == "box"
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(box)
    for spine in ax.spines.values():
        spine.set_color(theme.baseline)
    ax.tick_params(
        colors=theme.baseline if box else theme.ink_muted, labelcolor=theme.ink_secondary,
        length=3.5 if box else 0, direction="out", labelsize=theme.font_size,
    )  # fmt: skip
    if labels is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: labels[int(v)] if 0 <= int(v) < len(labels) else "")
        )
    elif x is not None and np.issubdtype(np.asarray(x).dtype, np.integer):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10]))


def _decorate(ax: Axes, theme: Theme, title: str | None, ylabel: str | None, xlabel=None) -> None:
    if title:
        ax.set_title(title, loc=theme.title_loc, color=theme.ink, fontsize=theme.font_size + 1)
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


_MAX_INLINE_ENTRIES = 4
#: Smaller shares are left blank: a field of "0.00" hides the diagonal.
_MIN_ANNOTATED_SHARE = 0.005


def _place_legend(
    ax: Axes, theme: Theme, n_entries: int, *, inside: str = "best", handles=None
) -> None:
    """Legend inside the axes, or beside them when it would cover the lines."""
    kwargs = {} if handles is None else {"handles": handles}
    if n_entries > _MAX_INLINE_ENTRIES:
        _legend(ax, theme, loc="center left", bbox_to_anchor=(1.0, 0.5), **kwargs)
    else:
        _legend(ax, theme, loc=inside, **kwargs)


def _draw_cell_gaps(ax: Axes, theme: Theme, k: int) -> None:
    """Surface-coloured gaps between the cells of a k x k heatmap."""
    for edge in np.arange(k - 1) + 0.5:
        ax.axhline(edge, color=theme.surface, linewidth=2)
        ax.axvline(edge, color=theme.surface, linewidth=2)


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
        markersize=math.sqrt(theme.marker_size * 1.2),
        color=color,
        markeredgecolor=theme.surface,
        markeredgewidth=1.5,
        zorder=5,
        label="_nolegend_",
    )


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
    theme: str | Theme | None = None,
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

    kws: dict[str, Any]
    if units == "lines":
        kws = {"color": th.ink_muted, "linewidth": 0.4 * th.line_width, "alpha": th.unit_alpha}
        kws |= unit_kws or {}
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
        kws = {
            **style,
            "linewidth": th.line_width,
            "solid_capstyle": "round",
            **(average_kws or {}),
        }
        ax.plot(x, h.mean(axis=0), zorder=3, label=f"{c.name} average", **kws)

    names = list(result.data.index[list(c.ids)])
    entries = []
    for unit in highlight or ():
        if unit in names:
            row = h[names.index(unit)]
            ax.plot(x, row, color=th.ink, linewidth=0.7 * th.line_width, zorder=4, label=str(unit))
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
    theme: str | Theme | None = None,
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
    handles = []
    muted_handles = []
    for c in selected:
        mean = h[list(c.ids)].mean(axis=0)
        muted = emphasised is not None and c.name not in emphasised
        label = "_nolegend_" if muted else c.name
        kws = {
            **styles[c.name],
            "linewidth": th.line_width,
            "solid_capstyle": "round",
            **(
                {"color": th.ink_muted, "linewidth": 0.6 * th.line_width, "alpha": 0.8}
                if muted
                else {}
            ),
            **(line_kws or {}),
        }
        (line,) = ax.plot(x, mean, label=label, zorder=2 if muted else 3, **kws)
        if muted:
            muted_handles.append(line)
        else:
            handles.append(line)
            _end_dot(ax, th, x[-1], mean[-1], kws["color"])
            entries.append((c.name, mean[-1]))

    _style_axes(ax, th, x, labels)
    if direct_labels is True or (direct_labels == "auto" and len(entries) <= _MAX_INLINE_ENTRIES):
        _end_labels(ax, th, x[-1], entries)
    if legend and len(selected) > 1:
        if muted_handles:
            # Muted clubs all look alike, so the first stands for them all.
            muted_handles[0].set_label("other clubs")
            handles.append(muted_handles[0])
        _place_legend(ax, th, len(handles), handles=handles)
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
    theme: str | Theme | None = None,
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
        ax.plot(
            x, H, color=th.ink_muted, linewidth=0.75 * th.line_width, label="all units", zorder=2
        )
    for c in _select(result, clubs):
        H = cross_sectional_variation(result.values[list(c.ids)])
        kwargs: dict[str, Any]
        if len(c) < 3:  # H of one or two units swings wildly; keep it from dominating
            kwargs = {"label": f"{c.name} ({len(c)} units)", "alpha": 0.6}
            kwargs["linewidth"] = 0.6 * th.line_width
        else:
            kwargs = {"label": c.name, "linewidth": th.line_width}
        ax.plot(x, H, solid_capstyle="round", zorder=3, **kwargs, **styles[c.name])
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
        _place_legend(ax, th, n_entries)
    _decorate(ax, th, title, "H(t)")
    return ax


def plot_log_t(
    result: ConvergenceClubs,
    club: ClubKey | None = None,
    *,
    ax: Axes | None = None,
    show_discarded: bool = False,
    theme: str | Theme | None = None,
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Diagnostic plot of the log t regression of a club (of all units if ``club`` is None).

    Shows ``log(H_1/H_t) - 2 log(log t)`` against ``log t`` over the
    regression window, the fitted line and the test outcome: the null of
    convergence is rejected when ``t < -1.65``. ``show_discarded`` also draws
    the first ``time_trim`` share of periods, which the regression leaves out.
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
    rounding = result.params.get("rounding", "half_even")
    log_t, y = log_t_regression_data(H, result.time_trim, rounding)
    model = log_t_test(H, result.time_trim, result.hac_method, rounding)

    if show_discarded:
        # Periods 2..[rT], left out of the regression (t = 1 has log log t = -inf).
        early_t, early_y = log_t_regression_data(H[: len(H) - len(log_t)], 0.0)
        keep = early_t > 0
        ax.scatter(
            early_t[keep], early_y[keep], s=th.marker_size, facecolors="none",
            edgecolors=th.ink_muted, linewidths=1, zorder=3,
            label=f"discarded (first {result.time_trim:.0%} of periods)",
        )  # fmt: skip
    ax.scatter(
        log_t, y, s=th.marker_size, color=color, edgecolors=th.surface, linewidths=1.2, zorder=3,
        label="regression window",
    )  # fmt: skip
    grid = np.linspace(log_t.min(), log_t.max(), 50)
    ax.plot(grid, model.alpha + model.beta * grid, color=th.ink, linewidth=0.75 * th.line_width,
            zorder=2)  # fmt: skip
    verdict = "convergence not rejected" if model.converges() else "convergence rejected"
    ax.text(
        0.02,
        0.04 if show_discarded else 0.96,
        f"β = {model.beta:.3f} (s.e. {model.std_err:.3f})\nt = {model.t_value:.2f}: {verdict}",
        transform=ax.transAxes,
        va="bottom" if show_discarded else "top",
        fontsize=th.font_size - 0.5,
        color=th.ink_secondary,
    )
    _style_axes(ax, th)
    if show_discarded:
        _legend(ax, th, loc="upper right")
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
    theme: str | Theme | None = None,
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
    theme: str | Theme | None = None,
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
    theme: str | Theme | None = None,
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
            part["log_initial"], part["growth"], s=th.marker_size, color=palette[group],
            edgecolors=th.surface, linewidths=1.2, zorder=3, label=group,
        )  # fmt: skip
        if show_fit and result.club_effects and group != "all units" and len(part) > 1:
            x = np.linspace(part["log_initial"].min(), part["log_initial"].max(), 20)
            ax.plot(x, _fitted_line(result, x, group), color=palette[group],
                    linewidth=th.line_width, solid_capstyle="round", zorder=4,
                    label="_nolegend_")  # fmt: skip
    if show_fit and not result.club_effects:
        x = np.linspace(units["log_initial"].min(), units["log_initial"].max(), 50)
        ax.plot(x, _fitted_line(result, x), color=th.ink, linewidth=0.75 * th.line_width, zorder=4,
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
        _place_legend(ax, th, len(groups), inside="lower left")
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
    theme: str | Theme | None = None,
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
            kws = {"color": th.ink_muted, "linewidth": th.line_width, "zorder": 2}
        else:
            kws = {"color": palette[group], "linewidth": th.line_width, "zorder": 3}
        ax.plot(x, y, label=group, solid_capstyle="round", **kws)
        _end_dot(ax, th, x[-1], y[-1], kws["color"])
        entries.append((group, y[-1]))

    _style_axes(ax, th, x, labels)
    if direct_labels is True or (direct_labels == "auto" and len(entries) <= _MAX_INLINE_ENTRIES):
        _end_labels(ax, th, x[-1], entries)
    if legend and len(entries) > 1:
        _place_legend(ax, th, len(entries))
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


def _text_on(color: str, th: Theme) -> str:
    """Ink or white, whichever reads better on a cell of ``color``."""
    return "#ffffff" if _luminance(color) < 0.4 else "#0b0b0b"


def plot_transitions(
    result: DynamicsResult,
    *,
    ax: Axes | None = None,
    annotate: bool = True,
    theme: str | Theme | None = None,
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
    _draw_cell_gaps(ax, th, k)
    if annotate:
        for i in range(k):
            for j in range(k):
                if np.isfinite(P[i, j]) and P[i, j] >= _MIN_ANNOTATED_SHARE:
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
    theme: str | Theme | None = None,
    title: str | None = "Share of units in each class",
) -> Axes:
    """Initial, final and ergodic (long-run) shares of each class, as grouped bars."""
    from matplotlib.ticker import PercentFormatter

    th = _theme(theme)
    ax = _new_axes(ax, th, figsize=(5.6, 4.8))
    shares = result.distributions()
    k, n = shares.shape
    width = 0.8 / n
    for i, (column, color) in enumerate(zip(shares.columns, ordinal_colors(n, th), strict=True)):
        ax.bar(
            np.arange(k) + (i - (n - 1) / 2) * width, shares[column], width=width * 0.92,
            color=color, label=column, zorder=3,
        )  # fmt: skip
    _style_axes(ax, th)
    ax.set_xticks(range(k), shares.index, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _legend(ax, th, loc="upper center", ncols=n)
    ax.set_ylim(0, shares.to_numpy().max() * 1.2)
    _decorate(ax, th, title, None, xlabel="relative income class")
    return ax


def plot_dynamics(result: DynamicsResult, *, theme: str | Theme | None = None) -> Figure:
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
    theme: str | Theme | None = None,
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
        line_colors = ordinal_colors(len(periods), th)

    ax.axvline(1.0, color=th.baseline, linewidth=1, zorder=1)
    for period, color in zip(periods, line_colors, strict=True):
        density = result.density(period, bandwidth=bandwidth)
        ax.plot(density.index, density.to_numpy(), color=color, linewidth=th.line_width,
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
    theme: str | Theme | None = None,
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
    ax.hlines(y, low, high, color=th.colors[0], linewidth=th.line_width, zorder=2)
    ax.scatter(coef["coef"], y, s=th.marker_size * 4 / 3, color=th.colors[0], edgecolors=th.surface,
               linewidths=1.5, zorder=3)  # fmt: skip
    _style_axes(ax, th)
    ax.grid(False, axis="y")
    if th.gridlines:
        ax.grid(True, axis="x", color=th.grid, linewidth=1)
    ax.set_yticks(y, list(coef.index))
    auto = f"Determinants of club membership (ordered {result.model})"
    _decorate(
        ax, th, auto if title == "auto" else title, None,
        xlabel=f"coefficient ({level:.0%} interval); > 0 means a higher club is more likely",
    )  # fmt: skip
    return ax


def _plot_pairwise_clubs(result: PairwiseResult, ax: Axes | None, th: Theme, title) -> Axes:
    from matplotlib.colors import LinearSegmentedColormap

    shares = result.club_matrix()
    k = len(shares)
    values = shares.to_numpy(dtype=float, copy=True)
    values[np.triu_indices(k, 1)] = np.nan
    ax = _new_axes(ax, th, figsize=(1.6 + 0.75 * k, 1.2 + 0.7 * k))
    # A fixed 0-100% scale: 100% would be full convergence, so weak evidence
    # must look weak rather than be stretched to the darkest colour.
    top = 1.0
    cmap = LinearSegmentedColormap.from_list("share", th.sequential).with_extremes(bad=th.surface)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=top, interpolation="nearest")
    _draw_cell_gaps(ax, th, k)
    for i in range(k):
        for j in range(i + 1):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.0%}", ha="center", va="center",
                        fontsize=th.font_size, color=_text_on(cmap(values[i, j] / top), th),
                        fontweight="bold" if i == j else "normal")  # fmt: skip
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(th.surface)
    ax.figure.set_facecolor(th.surface)
    ax.tick_params(length=0, labelsize=th.font_size - 1, labelcolor=th.ink_secondary)
    ax.set_xticks(range(k), shares.columns, rotation=30, ha="right")
    ax.set_yticks(range(k), shares.index)
    auto = "Pairs with a stationary gap (Pesaran, 2007)"
    _decorate(
        ax, th, auto if title == "auto" else title, None,
        xlabel=f"ADF at {result.alpha:g}: about {result.alpha:.0%} expected without convergence",
    )  # fmt: skip
    return ax


def plot_pairwise(
    result: PairwiseResult,
    *,
    kind: Literal["auto", "clubs", "units"] = "auto",
    ax: Axes | None = None,
    unit_labels: bool | Literal["auto"] = "auto",
    colorbar: bool = True,
    theme: str | Theme | None = None,
    title: str | None = "auto",
) -> Axes:
    """Pairwise convergence tests, by club or unit by unit.

    ``kind="clubs"`` (the default when the result has clubs) shows the share
    of pairs whose gap is stationary within each club (diagonal, in bold) and
    between every two clubs. ``kind="units"`` is the heatmap of the ADF
    p-values of every pair (darker = stronger evidence of a stationary gap),
    with units ordered by club, so convergence within clubs shows as darker
    blocks on the diagonal; with many units it is mostly useful interactively.
    """
    from matplotlib.colors import LinearSegmentedColormap

    from .beta import _club_labels, _club_order

    if kind not in ("auto", "clubs", "units"):
        raise ValueError("kind must be 'auto', 'clubs' or 'units'")
    th = _theme(theme)
    if kind == "auto":
        kind = "clubs" if result.clubs is not None else "units"
    if kind == "clubs":
        return _plot_pairwise_clubs(result, ax, th, title)
    ax = _new_axes(ax, th, figsize=(6.4, 5.6))
    units = result.units
    blocks: list[tuple[str, int]] = []
    if result.clubs is not None:
        labels = _club_labels(result.clubs, units)
        order: list[int] = []
        for club in _club_order(result.clubs):
            members = [i for i, lab in enumerate(labels) if lab == club]
            if members:
                order += members
                blocks.append((club, len(members)))
    else:
        order = list(range(len(units)))
    evidence = 1.0 - result.matrix().to_numpy()[np.ix_(order, order)]
    np.fill_diagonal(evidence, np.nan)

    cmap = LinearSegmentedColormap.from_list("evidence", th.sequential).with_extremes(
        bad=th.surface
    )
    image = ax.imshow(evidence, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for edge in np.cumsum([size for _, size in blocks])[:-1] - 0.5:
        ax.axhline(edge, color=th.surface, linewidth=1.5)
        ax.axvline(edge, color=th.surface, linewidth=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=th.font_size - 1, labelcolor=th.ink_secondary)
    names = [str(units[i]) for i in order]
    if unit_labels is True or (unit_labels == "auto" and len(order) <= 60):
        ax.set_xticks(range(len(order)), names, rotation=90)
        ax.set_yticks(range(len(order)), names)
    elif blocks:
        starts = np.concatenate([[0], np.cumsum([size for _, size in blocks])[:-1]])
        centres = [s + size / 2 - 0.5 for s, (_, size) in zip(starts, blocks, strict=True)]
        ax.set_yticks(centres, [name for name, _ in blocks])
        ax.set_xticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_facecolor(th.surface)
    ax.figure.set_facecolor(th.surface)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, shrink=0.85)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=th.font_size - 1, labelcolor=th.ink_secondary)
        bar.set_label("1 − ADF p-value of the gap", color=th.ink_secondary, fontsize=th.font_size)
    auto = f"Pairwise convergence (Pesaran, 2007): {len(result.pairs)} pairs"
    _decorate(ax, th, auto if title == "auto" else title, None)
    return ax


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
    theme: str | Theme | None = None,
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
    th = _theme(theme)
    selected = _select(result, clubs)
    if not selected:
        raise ValueError("there are no clubs to plot")
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


def _hex_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgb

    return (*to_rgb(color), alpha)


def _rolling_layout(result: RollingClubs, gap: float) -> tuple[list[dict], float]:
    """Vertical extent (bottom, top) of every club bar, window by window, on a 0-1 scale."""
    members = result.memberships()
    n_units = len(members)
    most = max(r.n_clubs + (r.n_divergent > 0) for r in result.results)
    scale = (1 - gap * (most - 1)) / n_units
    layout = []
    for window, res in zip(result.windows, result.results, strict=True):
        counts = members[window].value_counts()
        groups = [c.name for c in res.clubs] + (["divergent"] if res.n_divergent else [])
        extents, top = {}, 1.0
        for group in groups:
            height = counts.get(group, 0) * scale
            extents[group] = (top - height, top)
            top -= height + gap
        layout.append(extents)
    return layout, scale


def _rolling_color(th: Theme, position: int | None, colors: Colors) -> str:
    if position is None:
        return th.ink_muted
    if isinstance(colors, Sequence) and not isinstance(colors, str) and position < len(colors):
        return colors[position]
    return th.colors[position % len(th.colors)]


def plot_rolling(
    result: RollingClubs,
    *,
    ax: Axes | None = None,
    bar_width: float = 0.1,
    gap: float = 0.025,
    flow_alpha: float = 0.35,
    labels: bool = True,
    theme: str | Theme | None = None,
    colors: Sequence[str] | None = None,
    title: str | None = "auto",
) -> Axes:
    """Flows of units between the clubs of consecutive windows (an alluvial diagram).

    Each window is a column of bars, one per club (club 1, the highest, on
    top; divergent units at the bottom), with heights proportional to the
    number of units. Bands join the clubs of consecutive windows; their
    width is the number of units that go from one to the other, so units
    that change club show as bands that cross. Bars and bands take the
    colour of the club's position (club 1 = first colour), which is a rank
    within each window, not the same group.
    """
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path

    th = _theme(theme)
    ax = _new_axes(ax, th, figsize=(max(6.4, 2.3 * len(result)), 4.8))
    layout, scale = _rolling_layout(result, gap)
    members = result.memberships()

    def position(res: ConvergenceClubs, group: str) -> int | None:
        names = [c.name for c in res.clubs]
        return names.index(group) if group in names else None

    half = bar_width / 2
    for i in range(len(result) - 1):
        a, b = result.windows[i], result.windows[i + 1]
        flows = members.groupby([members[a], members[b]]).size().to_dict()
        source_top = {g: top for g, (_, top) in layout[i].items()}
        target_top = {g: top for g, (_, top) in layout[i + 1].items()}
        x0, x1 = i + half, i + 1 - half
        middle = (x0 + x1) / 2
        for ga in layout[i]:  # sources top to bottom, then targets top to bottom
            for gb in layout[i + 1]:
                count = int(flows.get((ga, gb), 0))
                if not count:
                    continue
                height = count * scale
                s, t = source_top[ga], target_top[gb]
                source_top[ga] -= height
                target_top[gb] -= height
                vertices = [
                    (x0, s), (middle, s), (middle, t), (x1, t),
                    (x1, t - height), (middle, t - height), (middle, s - height), (x0, s - height),
                    (x0, s),
                ]  # fmt: skip
                codes = [Path.MOVETO, *[Path.CURVE4] * 3, Path.LINETO, *[Path.CURVE4] * 3]
                codes.append(Path.CLOSEPOLY)
                color = _rolling_color(th, position(result[i], ga), colors)
                ax.add_patch(
                    PathPatch(Path(vertices, codes), facecolor=_hex_alpha(color, flow_alpha),
                              edgecolor="none", zorder=2)
                )  # fmt: skip

    for i, (res, extents) in enumerate(zip(result.results, layout, strict=True)):
        for group, (bottom, top) in extents.items():
            color = _rolling_color(th, position(res, group), colors)
            ax.add_patch(
                Rectangle((i - half, bottom), bar_width, top - bottom, facecolor=color,
                          edgecolor=th.surface, linewidth=1, zorder=3)
            )  # fmt: skip
            if labels and top - bottom > 0.015:
                count = round((top - bottom) / scale)
                ax.annotate(
                    f"{group} ({count})", (i + half, (top + bottom) / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=th.font_size - 1.5,
                    color=th.ink_secondary, zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": th.surface,
                          "edgecolor": "none", "alpha": 0.75},
                )  # fmt: skip

    ax.set_xlim(-0.4, len(result) - 0.4)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks(range(len(result)), list(result.windows))
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=th.font_size, labelcolor=th.ink_secondary)
    ax.set_facecolor(th.surface)
    ax.figure.set_facecolor(th.surface)
    auto = "Club membership across windows"
    _decorate(ax, th, auto if title == "auto" else title, None)
    return ax


def plot_spatial(
    result: SpatialResult,
    *,
    ax: Axes | None = None,
    level: float = 0.95,
    min_expected: float = 0.5,
    theme: str | Theme | None = None,
    title: str | None = "auto",
) -> Axes:
    """Joins within each club relative to random assignment (1 = random).

    Dots are the observed ratio of same-club neighbour pairs to their
    expected number; lines span the central ``level`` of the permuted
    ratios. A dot right of its line means the club is more spatially
    clustered than chance. Clubs expected to have fewer than
    ``min_expected`` joins (one or two units) are left out, since their
    ratio is noise. Dots take the club colours; the last row is all clubs.
    """
    th = _theme(theme)
    summary = result.summary()
    summary = summary[summary["expected"] >= min_expected]
    groups = list(summary.index)
    columns = [k for k, g in enumerate([*result.groups, "all clubs"]) if g in groups]
    ratios = result.permutations[:, columns] / summary["expected"].to_numpy()
    low, high = np.quantile(ratios, [(1 - level) / 2, (1 + level) / 2], axis=0)
    ax = _new_axes(ax, th, figsize=(6.4, 0.45 * len(groups) + 1.4))
    y = np.arange(len(groups))[::-1]
    ax.axvline(1.0, color=th.baseline, linewidth=1, zorder=1)
    ax.hlines(y, low, high, color=th.ink_muted, linewidth=th.line_width, zorder=2,
              label=f"{level:.0%} of random assignments")  # fmt: skip
    palette = {g: th.colors[k % len(th.colors)] for k, g in enumerate(result.groups)}
    palette |= {"divergent": th.ink_muted, "all clubs": th.ink}
    colors = [palette[g] for g in groups]
    ax.scatter(summary["ratio"], y, s=th.marker_size * 4 / 3, color=colors,
               edgecolors=th.surface, linewidths=1.5, zorder=3, label="observed")  # fmt: skip
    _style_axes(ax, th)
    ax.grid(False, axis="y")
    if th.gridlines:
        ax.grid(True, axis="x", color=th.grid, linewidth=1)
    ax.set_yticks(y, groups)
    _legend(ax, th, loc="lower right")
    auto = f"Neighbours in the same club ({len(result.joins)} joins)"
    _decorate(ax, th, auto if title == "auto" else title, None,
              xlabel="joins within the club / expected at random")  # fmt: skip
    return ax
