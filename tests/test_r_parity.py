"""Compare against reference output of the R package (scripts/generate_r_fixtures.R)."""

import json

import numpy as np
import pytest

import convergence_clubs as cc

from .conftest import FIXTURES

REFERENCE = json.loads(FIXTURES.read_text())
# Some reference configurations produce clubs that fail their own test, as in R.
pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")
RTOL = 1e-8


def _case_id(args: dict) -> str:
    return "-".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in args.values())


def _assert_model(model: cc.LogTResult, expected: dict) -> None:
    np.testing.assert_allclose(
        [model.beta, model.std_err, model.t_value, model.p_value],
        [expected["beta"], expected["std_err"], expected["t_value"], expected["p_value"]],
        rtol=RTOL,
        atol=1e-12,
    )


def _assert_clubs(result: cc.ConvergenceClubs, expected: dict, incremental: bool = False) -> None:
    assert [list(c.ids) for c in result] == [c["ids"] for c in expected["clubs"]]
    assert list(result.divergent_ids) == list(expected["divergent"])
    for club, exp in zip(result, expected["clubs"], strict=True):
        assert club.name == exp["name"]
        if exp["cstar"] is not None:
            assert club.cstar == pytest.approx(exp["cstar"])
        if exp["merged_from"] is not None:
            assert list(club.merged_from) == list(exp["merged_from"])
        if incremental and not exp["model"]["t_value"] > cc.CRITICAL_VALUE:
            # c* hit its cap and the club fell back to the core group: R reports the
            # stale model of core + rejected candidates, we report the core's own.
            H = cc.cross_sectional_variation(result.values[list(club.ids)])
            assert club.model == cc.log_t_test(H, result.time_trim, result.hac_method)
            assert club.model.converges()
        else:
            _assert_model(club.model, exp["model"])


def _find(datasets, args):
    return cc.find_clubs(
        datasets[args["dataset"]],
        time_trim=args["time_trim"],
        hac_method=args["hac_method"],
        cstar=args["cstar"],
        cstar_method=args["cstar_method"],
    )


@pytest.mark.parametrize(
    "case",
    REFERENCE["log_t"],
    ids=[f"{c['subset']}-{c['hac_method']}-{c['time_trim']:.2f}" for c in REFERENCE["log_t"]],
)
def test_log_t_test(case, filtered_gdp):
    H = cc.cross_sectional_variation(filtered_gdp.to_numpy()[case["ids"]])
    _assert_model(cc.log_t_test(H, case["time_trim"], case["hac_method"]), case["model"])


def test_hp_filter(r_reference):
    trend = cc.hp_filter(np.log(cc.load_gdp()), lamb=r_reference["hp_filter"]["lambda"])
    np.testing.assert_allclose(trend.to_numpy(), r_reference["hp_filter"]["trend"], rtol=1e-10)


def test_bundled_filtered_gdp_is_hp_filtered_log_gdp(filtered_gdp):
    trend = cc.hp_filter(np.log(cc.load_gdp()))
    np.testing.assert_allclose(trend.to_numpy(), filtered_gdp.to_numpy(), rtol=1e-10)


@pytest.mark.parametrize(
    "case", REFERENCE["find_clubs"], ids=[_case_id(c["args"]) for c in REFERENCE["find_clubs"]]
)
def test_find_clubs(case, datasets):
    result = _find(datasets, case["args"])
    _assert_clubs(result, case, incremental=case["args"]["cstar_method"] == "incremental")


@pytest.mark.parametrize(
    "case", REFERENCE["merge_clubs"], ids=[_case_id(c["args"]) for c in REFERENCE["merge_clubs"]]
)
def test_merge_clubs(case, datasets):
    args = case["args"]
    merged = _find(datasets, args).merge(args["method"], merge_divergent=args["merge_divergent"])
    _assert_clubs(merged, case, incremental=args["cstar_method"] == "incremental")
