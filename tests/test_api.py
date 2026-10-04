import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc


@pytest.fixture(scope="module")
def clubs(filtered_gdp):
    return cc.find_clubs(filtered_gdp)


@pytest.fixture(scope="module")
def aqsb(filtered_gdp):
    return cc.find_clubs(filtered_gdp, hac_method="AQSB")


def test_container_protocol(clubs):
    assert len(clubs) == clubs.n_clubs == 7
    assert clubs[0] is clubs["club1"]
    assert [c.name for c in clubs] == [f"club{i}" for i in range(1, 8)]
    with pytest.raises(KeyError):
        clubs["club99"]


def test_memberships_partition_all_units(aqsb, filtered_gdp):
    members = aqsb.memberships()
    assert members.index.equals(filtered_gdp.index)
    assert (members == "divergent").sum() == aqsb.n_divergent == 3
    assert members.value_counts()["club1"] == len(aqsb["club1"])
    assert set(aqsb.divergent) == set(members.index[members == "divergent"])


def test_merge_divergent(aqsb):
    merged = aqsb.merge("PS", merge_divergent=True)
    assert [len(c) for c in merged] == [84, 18, 40, 9]
    assert merged.n_divergent == 1
    assert any(c.notes for c in merged)
    # Standalone call on unmerged clubs, too.
    assert cc.merge_divergent(aqsb).n_divergent <= aqsb.n_divergent


def test_merge_with_single_club_is_noop(clubs, filtered_gdp):
    single = cc.find_clubs(filtered_gdp.iloc[list(clubs["club7"].ids)])
    assert single.n_clubs == 1
    assert single.merge() is single


def test_transition_paths(clubs, filtered_gdp):
    paths = clubs.transition_paths()
    assert paths.shape == (len(filtered_gdp), filtered_gdp.shape[1])
    assert paths.index.names == ["club", "unit"]
    h = cc.relative_transition(filtered_gdp.to_numpy())
    np.testing.assert_allclose(paths.loc["club1"].to_numpy(), h[list(clubs["club1"].ids)])


def test_summary_and_text(clubs):
    summary = clubs.summary()
    assert list(summary.columns) == ["n_units", "beta", "std_err", "t_value", "p_value", "cstar"]
    assert "club7" in str(clubs)
    assert repr(clubs) == "<ConvergenceClubs: 7 clubs, 0 divergent units, hac_method='FQSB'>"
    assert "merged_clubs" in clubs.merge().summary().columns


def test_plot(clubs):
    fig = clubs.plot()
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == clubs.n_clubs + 1
    fig = clubs.plot(clubs=["club1", 2], units=False)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 1


def test_ref_col_and_data_cols(filtered_gdp):
    sub = cc.find_clubs(filtered_gdp, data_cols=list(range(1980, 2004)), ref_col=2003)
    assert list(sub.data.columns) == list(range(1980, 2004))


def test_hp_filter_shapes():
    x = np.linspace(0, 1, 20) ** 2 + np.sin(np.arange(20))
    assert cc.hp_filter(x).shape == (20,)
    assert cc.hp_filter(np.vstack([x, x])).shape == (2, 20)
    # A linear trend passes through unchanged.
    line = np.arange(10, dtype=float)
    np.testing.assert_allclose(cc.hp_filter(line), line, atol=1e-9)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"time_trim": 0}, ValueError),
        ({"time_trim": 0.99}, ValueError),
        ({"hac_method": "NW"}, ValueError),
        ({"cstar_method": "adaptive"}, ValueError),
        ({"ref_col": 1900}, KeyError),
    ],
)
def test_validation(filtered_gdp, kwargs, error):
    with pytest.raises(error):
        cc.find_clubs(filtered_gdp, **kwargs)


def test_rejects_missing_values(filtered_gdp):
    bad = filtered_gdp.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cc.find_clubs(bad)
    with pytest.raises(TypeError):
        cc.find_clubs(pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]}))
