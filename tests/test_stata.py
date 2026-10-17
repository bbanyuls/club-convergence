"""Stata compatibility: psecta / imergeclub (Du, 2018) as used by Mendez (2020).

The reference memberships come from the replication files of Mendez (2020),
see scripts/build_mendez_fixture.R.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
import convergence_clubs.merging as merging
from convergence_clubs.logt import LogTResult, trimmed_periods

FIXTURE = Path(__file__).parent / "fixtures" / "mendez2020_stata.csv"
MENDEZ = pd.read_csv(FIXTURE)
CASES = [(s, v) for s in ["hiYes", "hiNo"] for v in ["log_lp", "log_ky", "log_h", "log_tfp"]]
pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


def _panel(sample, variable):
    data = MENDEZ[MENDEZ["sample"] == sample]
    wide = cc.wide_panel(data, unit="country", time="year", value=variable)
    last = data[data["year"] == data["year"].max()].set_index("country").reindex(wide.index)
    return wide, last


def _labels(result):
    return [None if c == "divergent" else int(c[4:]) for c in result.memberships()]


def _stata(column):
    return [None if pd.isna(v) else int(v) for v in column]


@pytest.mark.parametrize("sample, variable", CASES, ids=[f"{s}-{v}" for s, v in CASES])
def test_reproduces_stata_psecta_and_imergeclub(sample, variable):
    wide, last = _panel(sample, variable)
    clubs = cc.find_clubs(wide, convention="Stata", time_trim=0.333)
    assert _labels(clubs) == _stata(last[f"club_{variable}"])
    assert _labels(clubs.merge("PS")) == _stata(last[f"finalclub_{variable}"])


def test_r_convention_differs_where_expected():
    """R picks the core by max |t| and keeps c* fixed: two of the eight cases differ."""
    differ = []
    for sample, variable in CASES:
        wide, last = _panel(sample, variable)
        r = cc.find_clubs(wide, time_trim=0.333)
        if _labels(r) != _stata(last[f"club_{variable}"]):
            differ.append(f"{sample}/{variable}")
    assert differ == ["hiYes/log_tfp", "hiNo/log_h"]
    # Each of the two Stata rules explains one of them.
    wide, last = _panel("hiNo", "log_h")
    max_t = cc.find_clubs(wide, time_trim=0.333, core_rule="max_t")
    assert _labels(max_t) == _stata(last["club_log_h"])
    wide, last = _panel("hiYes", "log_tfp")
    raised = cc.find_clubs(
        wide, time_trim=0.333, cstar_method="incremental", cstar_increment=0.05, cstar_cap=50
    )
    assert _labels(raised) == _stata(last["club_log_tfp"])
    # Club 2 needs c* = 4.85, above the R package's default cap of 3.
    assert raised["club2"].cstar == pytest.approx(4.85)
    capped = cc.find_clubs(wide, time_trim=0.333, cstar_method="incremental", cstar_increment=0.05)
    assert _labels(capped) != _stata(last["club_log_tfp"])


def test_convention_defaults():
    data = cc.load_filtered_gdp()
    stata = cc.find_clubs(data, convention="Stata")
    assert stata.time_trim == 0.3
    assert stata.params["cstar_method"] == "incremental"
    assert stata.params["cstar_increment"] == 0.05 and stata.params["cstar_cap"] == 50
    assert stata.params["core_rule"] == "max_t" and stata.params["rounding"] == "half_up"
    r = cc.find_clubs(data)
    assert (r.time_trim, r.params["core_rule"], r.params["rounding"]) == (
        1 / 3, "max_abs_t", "half_even",
    )  # fmt: skip
    with pytest.raises(ValueError, match="convention"):
        cc.find_clubs(data, convention="SAS")
    with pytest.raises(ValueError, match="core_rule"):
        cc.find_clubs(data, core_rule="median")


@pytest.mark.parametrize(
    ("rounding", "trimmed", "kept"),
    [
        pytest.param("half_even", 10, 25, id="R-half-even"),
        pytest.param("half_up", 11, 24, id="Stata-half-up"),
    ],
)
def test_trimming_rounds_as_the_reference_implementation(rounding, trimmed, kept):
    # 0.3 * 35 = 10.5 is a tie, so the two conventions disagree.
    assert trimmed_periods(35, 0.3, rounding) == trimmed
    H = np.linspace(1, 0.2, 35) + 0.01 * np.sin(np.arange(35))
    x, _ = cc.log_t_regression_data(H, 0.3, rounding)
    assert len(x) == kept


def test_trimming_conventions_agree_away_from_a_tie():
    assert trimmed_periods(34, 1 / 3) == trimmed_periods(34, 1 / 3, "half_up") == 11


def test_trimming_convention_changes_the_log_t_statistic():
    H = np.linspace(1, 0.2, 35) + 0.01 * np.sin(np.arange(35))
    assert cc.log_t_test(H, 0.3).t_value != cc.log_t_test(H, 0.3, rounding="half_up").t_value


def test_sort_fraction_uses_statas_formula():
    # fr = 0.7 with T = 10: Stata averages T - int(0.3 * 10) = 10 - 3 = 7 periods
    # (0.3 * 10 is 3.0000000000000004 in floating point; ceil(0.7 * 10) would give 8).
    data = cc.load_filtered_gdp().iloc[:, -10:]
    ranked = data.iloc[:, -7:].mean(axis=1).sort_values(ascending=False)
    clubs = cc.find_clubs(data, sort_fraction=0.7)
    first = clubs.clubs[0]
    assert data.index[first.ids[0]] in ranked.index[:3]


@pytest.fixture
def scripted(monkeypatch):
    """Clubs of four units each; ``converging`` lists which unions pass the test."""
    data = cc.load_filtered_gdp().iloc[:20]
    groups = [tuple(range(i, i + 4)) for i in range(0, 16, 4)]
    model = LogTResult(beta=0.1, std_err=0.1, t_value=1.0, p_value=0.8)
    base = cc.ConvergenceClubs(
        clubs=tuple(cc.Club(f"club{i + 1}", g, model) for i, g in enumerate(groups)),
        divergent_ids=(16, 17, 18, 19),
        data=data,
        time_trim=1 / 3,
        hac_method="FQSB",
    )
    unit_club = {u: i + 1 for i, g in enumerate(groups) for u in g} | dict.fromkeys(
        range(16, 20), 0
    )
    converging: set[frozenset] = set()

    def fake_test(result, ids, time_trim, hac_method=None):
        members = frozenset(unit_club[i] for i in ids)
        t = 1.0 if len(members) == 1 or members in converging else -5.0
        return LogTResult(beta=0.1, std_err=0.1, t_value=t, p_value=0.5)

    monkeypatch.setattr(merging, "_test", fake_test)
    return base, converging


def test_iterate_repeats_the_merging_pass(scripted):
    base, converging = scripted
    first_pass = {frozenset({1, 2}), frozenset({3, 4})}
    second_pass = {frozenset({1, 2, 3, 4})}
    converging |= first_pass | second_pass
    once = base.merge("PS")
    assert [c.merged_from for c in once] == [("club1", "club2"), ("club3", "club4")]
    repeated = base.merge("PS", iterate=True)
    assert [c.merged_from for c in repeated] == [("club1", "club2", "club3", "club4")]
    assert len(repeated["club1"]) == 16


def test_last_club_absorbs_all_divergent_units(scripted):
    base, converging = scripted
    club4_with_divergent = frozenset({4, 0})
    converging.add(club4_with_divergent)
    merged = base.merge("PS", merge_divergent=True, divergent_method="last_club")
    assert merged.n_divergent == 0 and merged.n_clubs == 4
    assert merged["club4"].merged_from == ("club4", "divergent")
    assert len(merged["club4"]) == 8
    converging.clear()
    kept = base.merge("PS", merge_divergent=True, divergent_method="last_club")
    assert kept.n_divergent == 4
    with pytest.raises(ValueError, match="divergent_method"):
        base.merge("PS", merge_divergent=True, divergent_method="nearest")


def test_stata_convention_flows_into_merging_and_sensitivity():
    data = cc.load_filtered_gdp().iloc[:, :35]  # 35 periods: R and Stata trim differently
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stata = cc.find_clubs(data, convention="Stata")
    merged = stata.merge("PS")
    for club in merged:
        ids = list(club.ids)
        H = cc.cross_sectional_variation(stata.values[ids])
        assert club.model == cc.log_t_test(H, 0.3, "FQSB", "half_up")
    runs = cc.sensitivity(data, time_trim=[0.3], hac_method=["FQSB"], convention="Stata")
    assert runs.baseline.params["convention"] == "Stata"
