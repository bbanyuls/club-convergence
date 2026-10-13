import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import convergence_clubs as cc
from convergence_clubs.plotting import club_styles


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def merged(filtered_gdp):
    return cc.find_clubs(filtered_gdp).merge("PS")


def _visible(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def _data_lines(ax):
    return [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]


def test_grid_layouts(merged):
    fig = merged.plot()
    assert len(_visible(fig)) == merged.n_clubs + 1
    assert len(fig.legends) == 1
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [c.name for c in merged]

    assert len(_visible(merged.plot(kind="clubs"))) == merged.n_clubs
    assert len(_visible(merged.plot(kind="average"))) == 1
    assert len(_visible(merged.plot(clubs=["club1", 2]))) == 3


def test_sharey(merged):
    fig = merged.plot(kind="clubs", sharey=True)
    assert len({ax.get_ylim() for ax in _visible(fig)}) == 1


def test_colors_follow_the_club_not_the_selection(merged):
    ax = merged.plot.average(clubs=["club3"])
    assert to_hex(_data_lines(ax)[0].get_color()) == cc.LIGHT.colors[2]


def test_color_overrides(merged):
    styles = club_styles(merged, colors={"club2": "#000000"})
    assert styles["club2"]["color"] == "#000000"
    assert styles["club1"]["color"] == cc.LIGHT.colors[0]
    styles = club_styles(merged, colors=["#111111"])
    assert styles["club1"]["color"] == "#111111"
    assert styles["club2"]["color"] == cc.LIGHT.colors[1]


def test_clubs_past_the_palette_reuse_hues_with_dashes(filtered_gdp):
    nine = cc.find_clubs(filtered_gdp, time_trim=0.3)
    assert nine.n_clubs == 9
    styles = club_styles(nine)
    assert styles["club9"]["color"] == styles["club1"]["color"]
    assert styles["club9"]["linestyle"] != "-"
    assert len({s["color"] for s in styles.values()}) == 8


def test_themes(merged):
    fig = merged.plot(theme="dark")
    assert to_hex(fig.get_facecolor()) == cc.DARK.surface
    custom = cc.LIGHT.with_colors(["#123456"] * 8)
    ax = merged.plot.average(theme=custom, clubs="club1")
    assert to_hex(_data_lines(ax)[0].get_color()) == "#123456"
    with pytest.raises(ValueError, match="theme"):
        merged.plot(theme="neon")


def test_default_theme_and_registry(merged):
    assert cc.get_theme() is cc.LIGHT
    with cc.theme_context("dark") as th:
        assert th is cc.DARK
        assert to_hex(merged.plot().get_facecolor()) == cc.DARK.surface
        # An explicit theme still wins over the default.
        assert to_hex(merged.plot.average(theme="light").figure.get_facecolor()) == cc.LIGHT.surface
    assert cc.get_theme() is cc.LIGHT

    big = cc.LIGHT.replace(font_size=12)
    cc.register_theme("big", big)
    try:
        cc.set_theme("big")
        ax = merged.plot.average()
        assert ax.yaxis.get_major_ticks()[0].label1.get_fontsize() == 12
    finally:
        cc.set_theme("light")
        del cc.plotting.THEMES["big"]
    with pytest.raises(ValueError, match="theme"):
        cc.set_theme("neon")
    with pytest.raises(TypeError):
        cc.register_theme("x", {"surface": "#fff"})
    with pytest.raises(ValueError, match="line styles"):
        cc.LIGHT.replace(linestyles=())


def test_print_theme_tells_clubs_apart_without_colour(filtered_gdp):
    nine = cc.find_clubs(filtered_gdp, time_trim=0.3)
    styles = club_styles(nine, theme="print")
    pairs = {(s["color"], str(s["linestyle"])) for s in styles.values()}
    assert len(pairs) == 9  # 3 greys x 3 line styles
    assert {s["color"] for s in styles.values()} == set(cc.PRINT.colors)
    fig = nine.plot(theme="print", kind="average")
    lines = _data_lines(fig.axes[0])
    assert lines[3].get_linestyle() == "--" and lines[0].get_linestyle() == "-"


def test_emphasis(merged):
    ax = merged.plot.average(emphasis=["club1", "club6"])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["club1", "club6", "other clubs"]
    muted = [ln for ln in ax.get_lines() if to_hex(ln.get_color()) == cc.LIGHT.ink_muted]
    assert len(muted) == merged.n_clubs - 2


def test_direct_labels_only_when_they_fit(merged):
    ax = merged.plot.average(clubs=["club1", "club6"])
    assert {t.get_text() for t in ax.texts} == {"club1", "club6"}
    ax = merged.plot.average()  # six clubs: legend only
    assert not ax.texts


def test_club_panel(merged):
    ax = merged.plot.club("club4", highlight=["Bolivia", "Atlantis"])
    assert len(ax.get_lines()) == len(merged["club4"]) + 2  # units, average, Bolivia
    assert [t.get_text() for t in ax.texts] == ["Bolivia"]

    ax = merged.plot.club("club4", units="band")
    assert len(ax.collections) == 1 and len(ax.get_lines()) == 1
    with pytest.raises(ValueError, match="units"):
        merged.plot.club("club4", units="dots")


def test_draws_into_existing_axes(merged):
    fig, axes = plt.subplots(1, 3)
    assert merged.plot.average(ax=axes[0]) is axes[0]
    assert merged.plot.variation(ax=axes[1]) is axes[1]
    assert merged.plot.log_t("club1", ax=axes[2]) is axes[2]


def test_variation_plot(merged):
    ax = merged.plot.variation()
    assert ax.get_yscale() == "log"
    lines = _data_lines(ax)
    assert len(lines) == merged.n_clubs + 1
    small = lines[-1]  # club6 has two units: thinner, lighter, labelled with its size
    assert small.get_label() == "club6 (2 units)" and small.get_alpha() == 0.6
    assert lines[1].get_alpha() is None


def test_log_t_plot(merged):
    ax = merged.plot.log_t("club1")
    model = merged["club1"].model
    fit = ax.get_lines()[0]
    x, y = fit.get_xdata(), fit.get_ydata()
    assert (y[-1] - y[0]) / (x[-1] - x[0]) == pytest.approx(model.beta)
    assert f"{model.t_value:.2f}" in ax.texts[0].get_text()
    assert "all units" in merged.plot.log_t().get_title(loc="left")

    ax = merged.plot.log_t("club1", show_discarded=True)
    discarded, window = ax.collections
    n = len(merged.data.columns)
    assert len(window.get_offsets()) == n - round(n / 3)
    assert len(discarded.get_offsets()) == round(n / 3) - 1  # t = 1 has no log log t
    assert "discarded" in ax.get_legend().get_texts()[0].get_text()


def test_log_t_plot_uses_the_convention_rounding(filtered_gdp):
    # T = 15 and time_trim = 0.3: [rT] = 4.5 is 4 in R (half even) but 5 in Stata.
    stata = cc.find_clubs(filtered_gdp.iloc[:, :15], convention="Stata", time_trim=0.3)
    model = stata[0].model
    assert f"t = {model.t_value:.2f}" in stata.plot.log_t(0).texts[0].get_text()


def test_ordinal_colors():
    for theme in ("light", "dark", "print"):
        for n in range(1, cc.plotting.MAX_ORDINAL + 1):
            colors = cc.ordinal_colors(n, theme)
            assert len(colors) == n == len(set(colors))
    assert cc.ordinal_colors(2) == [cc.LIGHT.ordinal[0], cc.LIGHT.ordinal[-1]]
    assert cc.ordinal_colors(1) == [cc.LIGHT.ordinal[-1]]
    # Lightness falls steadily on light surfaces, rises on dark ones.
    from convergence_clubs.plotting import _luminance

    light = [_luminance(c) for c in cc.ordinal_colors(7)]
    dark = [_luminance(c) for c in cc.ordinal_colors(7, "dark")]
    assert all(np.diff(light) < 0) and all(np.diff(dark) > 0)
    with pytest.raises(ValueError, match="ramps"):
        cc.LIGHT.replace(ordinal=("#000000",))


def test_sequential_zero_recedes_into_the_surface():
    from convergence_clubs.plotting import _luminance

    for theme in (cc.LIGHT, cc.DARK):
        gap = abs(_luminance(theme.sequential[0]) - _luminance(theme.surface))
        assert gap < 0.1


@pytest.mark.parametrize(
    "columns",
    [
        [f"Y{y}" for y in range(1970, 2004)],
        pd.date_range("1970", periods=34, freq="YS"),
        pd.period_range("1970", periods=34, freq="Y"),
    ],
    ids=["strings", "dates", "periods"],
)
def test_non_numeric_periods(filtered_gdp, columns):
    data = filtered_gdp.copy()
    data.columns = columns
    result = cc.find_clubs(data)
    assert len(_visible(result.plot())) == 8
    ax = result.plot.average(clubs=["club1", "club6"])  # direct labels at a date/label x
    assert {t.get_text() for t in ax.texts} == {"club1", "club6"}
    assert result.plot.variation().get_yscale() == "log"


def test_invalid_kind(merged):
    with pytest.raises(ValueError, match="kind"):
        merged.plot(kind="pie")


def test_transition_paths_are_the_plotted_values(merged):
    ax = merged.plot.club("club6", units="none")
    np.testing.assert_allclose(
        ax.get_lines()[0].get_ydata(), merged.transition_paths().loc["club6"].mean().to_numpy()
    )
