"""Maps of club membership (requires geopandas: ``pip install 'convergence-clubs[maps]'``)."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .plotting import Colors, Theme, _decorate, _group_colors, _new_axes, _theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs
    from .sensitivity import SensitivityResult


def match_regions(
    result: ConvergenceClubs,
    geodata: Any,
    *,
    key: Hashable | None = None,
    names: Mapping | pd.Series | None = None,
) -> pd.DataFrame:
    """How the units of ``result`` match the regions of ``geodata``.

    Returns one row per unit with its region key (``names`` maps unit names
    to keys when they differ, e.g. :func:`country_codes`) and whether that
    key exists in ``geodata`` (in column ``key``, or the index).
    """
    keys = pd.Index(geodata[key] if key is not None else geodata.index)
    mapping = pd.Series(names) if names is not None else pd.Series(dtype=object)
    units = result.data.index
    region = pd.Series([mapping.get(u, u) for u in units], index=units, name="region")
    return pd.DataFrame(
        {"club": result.memberships(), "region": region, "found": region.isin(keys)}
    )


def plot_map(
    result: ConvergenceClubs,
    geodata: Any,
    *,
    key: Hashable | None = None,
    names: Mapping | pd.Series | None = None,
    ax: Axes | None = None,
    crs: Any = None,
    confidence: SensitivityResult | pd.Series | None = None,
    threshold: float = 0.8,
    show_missing: bool = True,
    legend: bool = True,
    theme: str | Theme = "light",
    colors: Colors = None,
    title: str | None = "auto",
) -> Axes:
    """Colour every region by its convergence club.

    Parameters
    ----------
    geodata
        A ``geopandas.GeoDataFrame`` of region boundaries.
    key
        Column of ``geodata`` holding the region identifiers (the index by
        default). Units are matched on their names unless ``names`` maps them
        to identifiers, e.g. ``names=cc.country_codes(), key="ISO_A3_EH"``.
    crs
        Reproject before drawing, e.g. ``"EPSG:8857"`` (Equal Earth).
    confidence
        A :func:`bootstrap` or :func:`sensitivity` result (or a Series unit ->
        stability): members with stability below ``threshold`` are drawn
        lighter, so uncertain memberships stand out.
    show_missing
        Draw regions without data in a light neutral fill.

    Units that are not found in ``geodata`` are reported with a warning (see
    :func:`match_regions`); divergent units are drawn in muted grey.
    """
    from matplotlib.patches import Patch

    th = _theme(theme)
    ax = _new_axes(ax, th, figsize=(10, 5.4))
    matches = match_regions(result, geodata, key=key, names=names)
    missing = matches.index[~matches["found"]]
    if len(missing):
        warnings.warn(
            f"{len(missing)} units have no region in geodata: {', '.join(map(str, missing[:8]))}"
            f"{' ...' if len(missing) > 8 else ''} (see match_regions())",
            UserWarning,
            stacklevel=2,
        )

    geo = geodata.to_crs(crs) if crs is not None else geodata
    region_keys = pd.Index(geo[key] if key is not None else geo.index)
    found = matches[matches["found"]]
    club_of = pd.Series(found["club"].to_numpy(), index=found["region"].to_numpy())
    clubs = pd.Series(region_keys.map(club_of), index=geo.index)

    certain = pd.Series(True, index=geo.index)
    if confidence is not None:
        stability = (
            confidence.stability()["stability"]
            if hasattr(confidence, "stability")
            else pd.Series(confidence)
        )
        by_region = pd.Series(stability.reindex(found.index).to_numpy(), index=found["region"])
        values = pd.Series(region_keys.map(by_region), index=geo.index)
        certain = ~(values < threshold)

    groups = [c.name for c in result.clubs] + (["divergent"] if result.n_divergent else [])
    palette = _group_colors(result, groups, th, colors)
    style = {"edgecolor": th.surface, "linewidth": 0.4}

    def draw(mask: np.ndarray, **kwargs) -> None:
        if mask.any():  # geopandas warns when asked to draw nothing
            geo[mask].plot(ax=ax, **style, **kwargs)

    if show_missing:
        draw(clubs.isna().to_numpy(), color=th.grid)
    handles = []
    for group in groups:
        mask = (clubs == group).to_numpy()
        draw(mask & certain.to_numpy(), color=palette[group])
        draw(mask & ~certain.to_numpy(), color=palette[group], alpha=0.4)
        size = int((matches["club"] == group).sum())
        handles.append(Patch(color=palette[group], label=f"{group} ({size})"))
    if show_missing:
        handles.append(Patch(color=th.grid, label="no data"))
    if confidence is not None:
        handles.append(Patch(color=th.ink_muted, alpha=0.4, label=f"stability < {threshold:g}"))

    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.figure.set_facecolor(th.surface)
    if legend:
        ax.legend(
            handles=handles, loc="lower left", frameon=False, fontsize=th.font_size - 0.5,
            labelcolor=th.ink_secondary, handlelength=1.2, handleheight=1.0,
        )  # fmt: skip
    _decorate(ax, th, "Convergence clubs" if title == "auto" else title, None)
    return ax
