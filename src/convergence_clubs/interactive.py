"""Interactive plotly versions of the main charts.

Install with ``pip install 'convergence-clubs[interactive]'``.

Every function returns a ``plotly.graph_objects.Figure``; save it with
``fig.write_html("clubs.html")`` to share a standalone page. Colours follow
the same :class:`~convergence_clubs.plotting.Theme` and club colours as the
matplotlib plots. Clicking a club in the legend hides or shows all of its
traces without recolouring the others.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .logt import relative_transition
from .plotting import Colors, Theme, _group_colors, _theme, club_styles

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from .beta import BetaResult
    from .results import ConvergenceClubs
    from .sensitivity import SensitivityResult
    from .sigma import SigmaResult


def _go():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "interactive charts require plotly: pip install 'convergence-clubs[interactive]'"
        ) from exc
    return go, make_subplots


def _layout(fig: go.Figure, th: Theme, title: str | None, **kwargs) -> go.Figure:
    axis = {
        "gridcolor": th.grid,
        "linecolor": th.baseline,
        "zeroline": False,
        "tickfont": {"color": th.ink_secondary},
        "title": {"font": {"color": th.ink_secondary}},
    }
    fig.update_layout(
        paper_bgcolor=th.surface,
        plot_bgcolor=th.surface,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": th.ink},
        title={"text": title, "x": 0.01, "xanchor": "left"} if title else None,
        legend={"font": {"color": th.ink_secondary}, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": th.surface, "font": {"color": th.ink}, "bordercolor": th.baseline},
        margin={"l": 60, "r": 20, "t": 60 if title else 30, "b": 50},
        **kwargs,
    )
    fig.update_xaxes(showgrid=False, **{k: v for k, v in axis.items() if k != "gridcolor"})
    fig.update_yaxes(**axis)
    return fig


_PLOTLY_DASH = {"-": "solid", "solid": "solid", ":": "dot", "dotted": "dot"}
_PLOTLY_DASH |= {"-.": "dashdot", "dashdot": "dashdot", "--": "dash", "dashed": "dash"}


def _dash(style: dict[str, Any]) -> str:
    """plotly ``dash`` for a matplotlib line style: named, or a dash pattern in points."""
    linestyle = style["linestyle"]
    if isinstance(linestyle, str):
        return _PLOTLY_DASH.get(linestyle, "dash")
    _, pattern = linestyle  # in units of the line width, which plotly does not scale by
    return ",".join(f"{2 * length:g}px" for length in pattern)


def _x(columns: pd.Index) -> list:
    if isinstance(columns, pd.PeriodIndex):
        return list(columns.to_timestamp())
    return list(columns)


# -- transition paths ------------------------------------------------------------


def transition_paths_figure(
    result: ConvergenceClubs,
    *,
    kind: Literal["both", "clubs", "average"] = "both",
    clubs: Sequence[str | int] | None = None,
    ncols: int = 3,
    theme: str | Theme | None = None,
    colors: Colors = None,
    title: str | None = "Relative transition paths",
    height: int | None = None,
) -> go.Figure:
    """Club panels with every unit's path (hover shows the unit) and/or the club averages."""
    go, make_subplots = _go()
    if kind not in ("both", "clubs", "average"):
        raise ValueError("kind must be 'both', 'clubs' or 'average'")
    th = _theme(theme)
    selected = [result[c] for c in clubs] if clubs is not None else list(result.clubs)
    if not selected:
        raise ValueError("there are no clubs to plot")
    styles = club_styles(result, th, colors)
    h = relative_transition(result.values)
    x = _x(result.data.columns)

    panels = (selected if kind != "average" else []) + ([None] if kind != "clubs" else [])
    ncols = max(1, min(ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    titles = [
        f"{html.escape(c.name)} · {len(c)} units" if c is not None else "Average transition paths"
        for c in panels
    ]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, shared_xaxes=False)
    shown: set[str] = set()

    def legend_once(name: str) -> bool:
        first = name not in shown
        shown.add(name)
        return first

    for i, club in enumerate(panels):
        row, col = divmod(i, ncols)
        cell = {"row": row + 1, "col": col + 1}
        targets = [club] if club is not None else selected
        for c in targets:
            style = styles[c.name]
            paths = h[list(c.ids)]
            if club is not None:  # club panel: one muted line per unit
                for unit, path in zip(result.data.index[list(c.ids)], paths, strict=True):
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=path, mode="lines", name=str(unit), legendgroup=c.name,
                            showlegend=False, opacity=th.unit_alpha,
                            line={"color": th.ink_muted, "width": 1},
                            hovertemplate=(
                                f"<b>{html.escape(str(unit))}</b> ({html.escape(c.name)})<br>"
                                "%{x}: h = %{y:.3f}<extra></extra>"
                            ),
                        ),
                        **cell,
                    )  # fmt: skip
            fig.add_trace(
                go.Scatter(
                    x=x, y=paths.mean(axis=0), mode="lines", name=html.escape(c.name),
                    legendgroup=c.name, showlegend=legend_once(c.name),
                    line={"color": style["color"], "width": 2.5, "dash": _dash(style)},
                    hovertemplate=(
                        f"<b>{html.escape(c.name)}</b> average · {len(c)} units<br>"
                        f"β = {c.model.beta:.3f}, t = {c.model.t_value:.2f}<br>"
                        "%{x}: h = %{y:.3f}<extra></extra>"
                    ),
                ),
                **cell,
            )  # fmt: skip
        if club is None:
            fig.add_hline(y=1.0, line={"color": th.baseline, "width": 1}, **cell)

    fig.update_yaxes(title_text="relative transition path", col=1)
    # Averages only: a crosshair listing every club at the hovered period.
    hovermode = "x unified" if kind == "average" else "closest"
    _layout(
        fig, th, title, height=height or 300 * nrows + 80, hovermode=hovermode,
        legend_title_text="club (click to hide)",
    )  # fmt: skip
    for i, annotation in enumerate(fig.layout.annotations[: len(panels)]):
        domain = fig.get_subplot(i // ncols + 1, i % ncols + 1).xaxis.domain
        annotation.update(x=domain[0], xanchor="left", font={"color": th.ink, "size": 13})
    return fig


# -- sensitivity / bootstrap ------------------------------------------------------


def co_membership_figure(
    sensitivity: SensitivityResult,
    *,
    theme: str | Theme | None = None,
    title: str | None = "auto",
) -> go.Figure:
    """Co-membership heatmap; hovering a cell names both units and their share."""
    go, _ = _go()
    th = _theme(theme)
    base = sensitivity.baseline
    order = [i for club in base.clubs for i in club.ids] + list(base.divergent_ids)
    names = [str(n) for n in base.data.index[order]]
    share = sensitivity.co_membership().to_numpy()[np.ix_(order, order)]
    club_of = base.memberships().to_numpy()[order]
    custom = np.dstack(np.meshgrid(club_of, club_of))
    scale = [[i / (len(th.sequential) - 1), c] for i, c in enumerate(th.sequential)]
    fig = go.Figure(
        go.Heatmap(
            z=share, x=names, y=names, customdata=custom, zmin=0, zmax=1, colorscale=scale,
            colorbar={"title": {"text": "share of runs<br>in the same club"}},
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[1]})<br><b>%{x}</b> (%{customdata[0]})<br>"
                "together in %{z:.0%} of runs<extra></extra>"
            ),
        )
    )  # fmt: skip
    edges: Any = np.cumsum([len(c) for c in base.clubs])[:-1] - 0.5 if base.n_clubs > 1 else []
    for edge in edges:
        fig.add_hline(y=edge, line={"color": th.surface, "width": 2})
        fig.add_vline(x=edge, line={"color": th.surface, "width": 2})
    auto = f"Co-membership across {len(sensitivity)} runs"
    _layout(fig, th, auto if title == "auto" else title, height=700, width=760)
    fig.update_yaxes(autorange="reversed", showticklabels=len(names) <= 60, showgrid=False)
    fig.update_xaxes(showticklabels=len(names) <= 60)
    return fig


# -- β and σ convergence ----------------------------------------------------------


def beta_figure(
    result: BetaResult,
    *,
    theme: str | Theme | None = None,
    colors: Colors = None,
    title: str | None = "auto",
) -> go.Figure:
    """Growth against initial level; hovering a point names the unit."""
    go, _ = _go()
    from .beta import _club_order, _fitted_line

    th = _theme(theme)
    units = result.units
    if "club" in units:
        groups = [g for g in _club_order(result.clubs) if g in set(units["club"])]
        palette = _group_colors(result.clubs, groups, th, colors)
    else:
        groups, palette = ["all units"], {"all units": th.colors[0]}

    fig = go.Figure()
    for group in groups:
        part = units if group == "all units" else units[units["club"] == group]
        fig.add_trace(
            go.Scatter(
                x=part["log_initial"], y=part["growth"], mode="markers", name=html.escape(group),
                legendgroup=group, text=[html.escape(str(u)) for u in part.index],
                marker={"color": palette[group], "size": 9,
                        "line": {"color": th.surface, "width": 1.5}},
                hovertemplate=(
                    f"<b>%{{text}}</b> ({html.escape(group)})<br>"
                    f"log level in {result.start}: %{{x:.2f}}<br>"
                    "growth: %{y:.2%} per period<extra></extra>"
                ),
            )
        )  # fmt: skip
        if result.club_effects and group != "all units" and len(part) > 1:
            x = np.linspace(part["log_initial"].min(), part["log_initial"].max(), 20)
            fig.add_trace(
                go.Scatter(
                    x=x, y=_fitted_line(result, x, group), mode="lines", legendgroup=group,
                    showlegend=False, hoverinfo="skip",
                    line={"color": palette[group], "width": 2},
                )
            )  # fmt: skip
    if not result.club_effects:
        x = np.linspace(units["log_initial"].min(), units["log_initial"].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=x, y=_fitted_line(result, x), mode="lines", name="fit", showlegend=False,
                hoverinfo="skip", line={"color": th.ink, "width": 1.5},
            )
        )  # fmt: skip
    fig.add_hline(y=0, line={"color": th.baseline, "width": 1})
    fig.add_annotation(
        x=0.99, y=0.98, xref="paper", yref="paper", xanchor="right", yanchor="top",
        showarrow=False, font={"color": th.ink_secondary, "size": 12},
        text=f"β = {result.beta:.4f} (s.e. {result.std_err:.4f}), t = {result.t_value:.2f}",
    )  # fmt: skip
    auto = "β-convergence" + (" conditional on clubs" if result.club_effects else "")
    _layout(fig, th, auto if title == "auto" else title, height=480, hovermode="closest")
    fig.update_yaxes(tickformat=".1%", title_text="average growth per period")
    fig.update_xaxes(title_text=f"log level in {result.start}")
    return fig


def sigma_figure(
    result: SigmaResult,
    *,
    index: bool = False,
    theme: str | Theme | None = None,
    colors: Colors = None,
    title: str | None = "auto",
) -> go.Figure:
    """Dispersion over time; the hover shows every group at the same period."""
    go, _ = _go()
    from .sigma import ALL, LABELS

    th = _theme(theme)
    series = result.series / result.series.iloc[0] if index else result.series
    palette = _group_colors(result.clubs, [g for g in series.columns if g != ALL], th, colors)
    fig = go.Figure()
    for group in series.columns:
        color = th.ink_muted if group == ALL else palette[group]
        fig.add_trace(
            go.Scatter(
                x=_x(series.index), y=series[group], mode="lines", name=html.escape(str(group)),
                line={"color": color, "width": 2.5},
                hovertemplate=f"{html.escape(str(group))}: %{{y:.3f}}<extra></extra>",
            )
        )  # fmt: skip
    if index:
        fig.add_hline(y=1.0, line={"color": th.baseline, "width": 1})
    auto = "σ-convergence" + (" (indexed)" if index else "")
    _layout(fig, th, auto if title == "auto" else title, height=460, hovermode="x unified")
    fig.update_yaxes(title_text=LABELS[result.measure] + (" (first period = 1)" if index else ""))
    return fig
