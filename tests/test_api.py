import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc


@pytest.fixture(scope="module")
def clubs(filtered_gdp):
    return cc.find_clubs(filtered_gdp)


@pytest.fixture(scope="module")
def aqsb(filtered_gdp):
    with pytest.warns(cc.ClubConvergenceWarning, match="club1"):
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
    assert list(summary.columns) == [
        "n_units", "beta", "std_err", "t_value", "p_value", "converges", "cstar",
    ]  # fmt: skip
    assert summary["converges"].all() and clubs.failing_clubs == ()
    assert "club7" in str(clubs)
    assert repr(clubs) == "<ConvergenceClubs: 7 clubs, 0 divergent units, hac_method='FQSB'>"
    assert "merged_clubs" in clubs.merge().summary().columns


def test_html_repr(clubs, filtered_gdp):
    html = clubs._repr_html_()
    assert "7 convergence clubs" in html and "<table" in html
    data = filtered_gdp.rename(index={"Liberia": "A & <B>"})
    with pytest.warns(cc.ClubConvergenceWarning):
        html = cc.find_clubs(data, hac_method="AQSB")._repr_html_()
    assert "A &amp; &lt;B&gt;" in html


def test_log_t_intercept(filtered_gdp):
    H = cc.cross_sectional_variation(filtered_gdp.to_numpy())
    x, y = cc.log_t_regression_data(H, 1 / 3)
    model = cc.log_t_test(H, 1 / 3)
    np.testing.assert_allclose(np.polyfit(x, y, 1), [model.beta, model.alpha])


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


def test_failing_clubs_are_flagged(aqsb):
    assert aqsb.failing_clubs == ("club1",)
    assert not aqsb["club1"].converges and aqsb["club1"].model.t_value < cc.CRITICAL_VALUE
    assert aqsb.summary()["converges"].tolist() == [False, True, True, True, True]
    assert "does not pass its own log t test" in str(aqsb)


def test_joint_first_divergent_merge(clubs):
    # Pretend club7 (Congo, Dem. Rep. and Liberia, which converge together) were divergent.
    last = clubs["club7"]
    result = clubs._replace(clubs=clubs.clubs[:-1], divergent_ids=last.ids)
    joint = result.merge_divergent(joint_first=True)
    assert joint.n_divergent == 0 and joint.n_clubs == 7
    assert joint["club7"].ids == last.ids and joint["club7"].model == last.model
    assert "converge together" in joint["club7"].notes[0]
    # Default (R) behaviour: units are assigned one by one to existing clubs instead.
    pairwise = result.merge_divergent()
    assert pairwise.n_clubs == 6
    merged = result.merge("PS", merge_divergent=True, joint_first=True)
    assert merged.clubs[-1].merged_from == ("divergent",)


def test_rename(clubs):
    merged = clubs.merge("PS")
    names = ["High", "Upper middle", "Middle", "Lower middle", "Low", "Bottom"]
    renamed = merged.rename(names)
    assert [c.name for c in renamed] == names
    assert renamed.summary().index.tolist() == names
    assert renamed.memberships()["United States"] == "High"
    assert renamed["Low"].ids == merged["club5"].ids
    assert renamed["High"].merged_from == merged["club1"].merged_from
    assert [c.name for c in merged] == [f"club{i}" for i in range(1, 7)]  # a copy

    partial = merged.rename({"club1": "Frontier"})
    assert [c.name for c in partial][:2] == ["Frontier", "club2"]
    numbered = merged.rename(lambda club: f"Club {club.name[4:]} (n = {len(club)})")
    assert numbered[0].name == "Club 1 (n = 50)"
    # The renamed result works everywhere names are used.
    beta = cc.beta_convergence(cc.load_gdp(), clubs=renamed, club_effects=True)
    assert "club[Low]" in beta.coefficients.index

    for bad, error in [
        (["a"] * 6, ValueError),
        (["a", "b"], ValueError),
        ([*names[:5], "divergent"], ValueError),
        ([*names[:5], " "], ValueError),
        ([*names[:5], "_hidden"], ValueError),
        ({"club9": "x"}, KeyError),
        ("High", TypeError),
    ]:
        with pytest.raises(error):
            merged.rename(bad)
