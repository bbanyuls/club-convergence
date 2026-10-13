import warnings

import numpy as np
import pytest

import convergence_clubs as cc
from convergence_clubs.plotting import club_styles

pytest.importorskip("plotly")


@pytest.fixture(scope="module")
def merged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        return cc.find_clubs(cc.load_filtered_gdp()).merge("PS")


def _averages(fig):
    return [t for t in fig.data if t.showlegend is not False and t.legendgroup == t.name]


def test_transition_paths_traces_and_legend(merged):
    fig = merged.plot.interactive()
    n_units = sum(len(c) for c in merged)
    assert len(fig.data) == n_units + 2 * merged.n_clubs  # units, club averages, average panel
    legend = [t.name for t in fig.data if t.showlegend]
    assert legend == [c.name for c in merged]  # one legend entry per club
    styles = club_styles(merged)
    for trace in fig.data:
        assert trace.legendgroup in styles  # clicking a club toggles all of its traces
        if trace.name in styles:
            assert trace.line.color == styles[trace.name]["color"]
    unit_trace = next(t for t in fig.data if t.name == "United States")
    assert "United States" in unit_trace.hovertemplate and "club1" in unit_trace.hovertemplate
    assert fig.layout.hovermode == "closest"
    assert fig.layout.paper_bgcolor == cc.LIGHT.surface


def test_average_only_uses_a_unified_crosshair(merged):
    fig = merged.plot.interactive(kind="average", theme="dark", clubs=["club1", "club3"])
    assert [t.name for t in fig.data] == ["club1", "club3"]
    assert fig.layout.hovermode == "x unified"
    assert fig.data[1].line.color == cc.DARK.colors[2]  # colour follows the club
    assert fig.layout.plot_bgcolor == cc.DARK.surface
    assert "β =" in fig.data[0].hovertemplate
    with pytest.raises(ValueError, match="kind"):
        merged.plot.interactive(kind="pie")


def test_clubs_past_the_palette_are_dashed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        nine = cc.find_clubs(cc.load_filtered_gdp(), time_trim=0.3)
    fig = nine.plot.interactive(kind="average")
    assert [t.line.dash for t in fig.data] == ["solid"] * 8 + ["8px,4px"]
    fig = nine.plot.interactive(kind="average", theme="print")
    assert [t.line.dash for t in fig.data][2:4] == ["solid", "10px,4px"]
    named = cc.LIGHT.replace(colors=("#000000",), linestyles=("-", ":", "-.", "--", "loose"))
    fig = nine.plot.interactive(kind="average", theme=named, clubs=[0, 1, 2, 3, 4])
    assert [t.line.dash for t in fig.data] == ["solid", "dot", "dashdot", "dash", "dash"]


def test_unit_names_are_escaped():
    data = cc.load_filtered_gdp().rename(index={"Norway": "A & <B>"})
    fig = cc.find_clubs(data).plot.interactive(kind="clubs", clubs=["club1"])
    assert any("A &amp; &lt;B&gt;" in t.hovertemplate for t in fig.data)


def test_co_membership_heatmap():
    runs = cc.bootstrap(cc.load_filtered_gdp().iloc[:40], 5, seed=0)
    fig = runs.plot_interactive()
    heatmap = fig.data[0]
    assert np.asarray(heatmap.z).shape == (40, 40)
    assert heatmap.colorscale[0][1] == cc.LIGHT.sequential[0]
    assert "together in" in heatmap.hovertemplate


def test_beta_and_sigma_figures(merged):
    gdp = cc.load_gdp()
    beta = cc.beta_convergence(gdp, clubs=merged, club_effects=True).plot_interactive()
    points = [t for t in beta.data if t.mode == "markers"]
    assert [t.name for t in points] == [c.name for c in merged]
    assert sum("China" in t.text for t in points) == 1  # hover names every country once
    assert beta.layout.yaxis.tickformat == ".1%"
    sigma = cc.sigma_convergence(gdp, clubs=merged).plot_interactive(index=True)
    assert all(t.y[0] == pytest.approx(1.0) for t in sigma.data)
    assert sigma.layout.hovermode == "x unified"


def test_standalone_html(merged, tmp_path):
    path = tmp_path / "clubs.html"
    merged.plot.interactive(kind="average").write_html(path)
    assert "plotly" in path.read_text()[:5000].lower()


def test_club_names_are_escaped():
    clubs = cc.find_clubs(cc.load_filtered_gdp()).rename({"club1": "Rich & <fast>"})
    fig = clubs.plot.interactive(clubs=["Rich & <fast>"])
    assert fig.data[-1].name == "Rich &amp; &lt;fast&gt;"
    assert "Rich &amp; &lt;fast&gt;" in fig.layout.annotations[0].text
