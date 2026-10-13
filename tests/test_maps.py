import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import convergence_clubs as cc
from convergence_clubs.plotting import club_styles

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely.geometry")


@pytest.fixture(scope="module")
def merged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        return cc.find_clubs(cc.load_filtered_gdp()).merge("PS")


@pytest.fixture(scope="module")
def squares():
    """One unit square per ISO3 code (no Netherlands Antilles), plus one region without data."""
    codes = [c for c in cc.country_codes() if c != "ANT"] + ["XXX"]
    boxes = [shapely.box(i % 13, i // 13, i % 13 + 1, i // 13 + 1) for i in range(len(codes))]
    return gpd.GeoDataFrame({"iso": codes}, geometry=boxes, crs="EPSG:4326")


def _fills(ax):
    """Face colour and alpha of every drawn polygon."""
    out = []
    for collection in ax.collections:  # one collection per layer, one path per polygon
        colors = collection.get_facecolors()
        if not len(colors):  # an overlay without fill (the hatching)
            continue
        for i in range(len(collection.get_paths())):
            color = colors[i % len(colors)]
            out.append((to_hex(color[:3]), round(float(color[3]), 2)))
    return out


def test_country_codes():
    codes = cc.country_codes()
    assert len(codes) == 152 and codes.is_unique
    assert codes["Korea, Republic of"] == "KOR" and codes["Congo, Dem. Rep."] == "COD"
    assert codes.index.equals(cc.load_gdp().index)


def test_match_regions(merged, squares):
    table = cc.match_regions(merged, squares, key="iso", names=cc.country_codes())
    assert list(table.columns) == ["club", "region", "found"]
    assert table["found"].sum() == 151
    assert list(table.index[~table["found"]]) == ["Netherlands Antilles"]
    by_name = cc.match_regions(merged, squares.set_index("iso"))
    assert not by_name["found"].any()  # names are not ISO codes


def test_map_colours_regions_by_club(merged, squares):
    with pytest.warns(UserWarning, match="Netherlands Antilles"):
        ax = merged.plot.map(squares, key="iso", names=cc.country_codes())
    fills = _fills(ax)
    # Ordinal by default: club 1, the highest level, gets the dark end of the ramp.
    shades = dict(
        zip([c.name for c in merged], cc.ordinal_colors(merged.n_clubs)[::-1], strict=True)
    )
    assert len(fills) == 152  # 151 matched countries + 1 region without data
    for club in merged:
        expected = sum(c != "ANT" for c in cc.country_codes()[list(merged.units(club.name))])
        assert sum(f == (shades[club.name], 1.0) for f in fills) == expected
    assert sum(f[0] == cc.LIGHT.grid for f in fills) == 1  # "XXX": no data
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [f"{c.name} ({len(c)})" for c in merged] + ["no data"]

    with pytest.warns(UserWarning):
        ax = merged.plot.map(squares, key="iso", names=cc.country_codes(), palette="clubs")
    styles = club_styles(merged)
    assert sum(f == (styles["club1"]["color"], 1.0) for f in _fills(ax)) == 50
    plt.close("all")


def test_palette_choice(merged, squares):
    with pytest.warns(UserWarning):
        ax = merged.plot.map(squares, key="iso", names=cc.country_codes(), colors=["#123456"])
    assert ("#123456", 1.0) in _fills(ax)  # explicit colours imply the club palette
    with pytest.raises(ValueError, match="palette"):
        merged.plot.map(squares, key="iso", palette="rainbow")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nine = cc.find_clubs(cc.load_filtered_gdp(), time_trim=0.3)
        ax = nine.plot.map(squares, key="iso", names=cc.country_codes(), show_missing=False)
    assert cc.LIGHT.colors[0] in {f[0] for f in _fills(ax)}  # > 7 clubs: club colours
    plt.close("all")


def test_uncertain_members_are_hatched(merged, squares):
    stability = pd.Series(1.0, index=merged.data.index)
    stability[["Peru", "Rwanda"]] = 0.5
    with pytest.warns(UserWarning):
        ax = merged.plot.map(
            squares, key="iso", names=cc.country_codes(), confidence=stability, show_missing=False
        )
    hatched = [c for c in ax.collections if c.get_hatch()]
    assert len(hatched) == 1 and len(hatched[0].get_paths()) == 2
    assert len([f for f in _fills(ax) if f[1] > 0]) == 151  # colours unchanged
    legend = ax.get_legend()
    assert "stability < 0.8" in [t.get_text() for t in legend.get_texts()]
    assert legend.legend_handles[-1].get_hatch() == "///"
    plt.close("all")


def test_reprojection_and_theme(merged, squares):
    with pytest.warns(UserWarning):
        ax = cc.plot_map(
            merged, squares, key="iso", names=cc.country_codes(), crs="EPSG:3857", theme="dark"
        )
    assert to_hex(ax.figure.get_facecolor()) == cc.DARK.surface
    assert ax.get_xlim()[1] > 1000  # metres, not degrees
    plt.close("all")


def test_divergent_units_are_muted(squares):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        aqsb = cc.find_clubs(cc.load_filtered_gdp(), hac_method="AQSB")
    with pytest.warns(UserWarning):
        ax = aqsb.plot.map(squares, key="iso", names=cc.country_codes(), show_missing=False)
    assert sum(f[0] == cc.LIGHT.ink_muted for f in _fills(ax)) == aqsb.n_divergent
    assert np.any(["divergent" in t.get_text() for t in ax.get_legend().get_texts()])
    plt.close("all")
