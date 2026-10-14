"""Validation errors, fallbacks and rarely used branches, module by module."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import convergence_clubs as cc
from convergence_clubs import _hac, _stats
from convergence_clubs.plotting import _end_labels

pytestmark = pytest.mark.filterwarnings("ignore::convergence_clubs.ClubConvergenceWarning")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def clubs():
    return cc.find_clubs(cc.load_filtered_gdp())


@pytest.fixture(scope="module")
def merged(clubs):
    return clubs.merge("PS")


# -- input validation ------------------------------------------------------------


def test_find_clubs_validation(filtered_gdp):
    with pytest.raises(TypeError, match="DataFrame"):
        cc.find_clubs(filtered_gdp.to_numpy())
    with pytest.raises(ValueError, match="non-negative"):
        cc.find_clubs(filtered_gdp, cstar_method="incremental", cstar_increment=-0.1)
    with pytest.raises(ValueError, match="two time periods"):
        cc.find_clubs(filtered_gdp.iloc[:, :1])
    # A zero increment cannot raise c*, so the incremental method is the fixed one.
    zero = cc.find_clubs(filtered_gdp, cstar_method="incremental", cstar_increment=0)
    assert zero.memberships().equals(cc.find_clubs(filtered_gdp).memberships())


def test_log_t_validation(filtered_gdp):
    H = cc.cross_sectional_variation(filtered_gdp.to_numpy())
    with pytest.raises(ValueError, match="rounding"):
        cc.log_t_test(H, rounding="up")
    with pytest.raises(ValueError, match="hac_method"):
        cc.log_t_test(H, hac_method="NW")
    with pytest.raises(ValueError, match="three periods"):
        cc.hp_filter(filtered_gdp.iloc[:, :2])


def test_merge_validation(clubs):
    with pytest.raises(ValueError, match="method"):
        clubs.merge("Stata")


def test_dynamics_and_beta_reject_missing_values():
    gdp = cc.load_gdp().copy()
    gdp.iloc[0, 5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cc.distribution_dynamics(gdp)
    gdp.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing start or end"):
        cc.beta_convergence(gdp)


def test_results_must_cover_the_same_units(clubs):
    smaller = cc.find_clubs(cc.load_filtered_gdp().iloc[:100])
    with pytest.raises(ValueError, match="same units"):
        cc.adjusted_rand_index(clubs, smaller)
    runs = cc.SensitivityResult((({"run": 0}, smaller), ({"run": 1}, clubs)))
    with pytest.raises(ValueError, match="not in the baseline"):
        runs.co_membership()


# -- numerical fallbacks ---------------------------------------------------------


def test_incomplete_beta_bounds():
    assert _stats.betainc(2.0, 3.0, 0.0) == 0.0
    assert _stats.betainc(2.0, 3.0, 1.0) == 1.0


def test_ols_rejects_non_finite_data():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    y = np.arange(10.0)
    y[3] = np.inf
    with pytest.raises(ValueError, match="could not be estimated"):
        _stats.ols(y, X)


def test_hac_falls_back_to_white_without_a_bandwidth():
    # Zero residuals leave the Andrews bandwidth undefined (0/0).
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cov = _hac.andrews_qs_vcov(X, np.zeros(10))
    assert cov.shape == (2, 2)


def test_ordered_model_recovers_from_a_singular_hessian(merged, monkeypatch):
    import convergence_clubs.determinants as determinants

    covariates = pd.DataFrame({"x": np.log(cc.load_gdp()[1970])}, index=merged.data.index)
    expected = cc.club_determinants(merged, covariates).summary()
    hessian, calls = determinants._hessian, []

    def singular_once(*args):
        calls.append(1)
        H = hessian(*args)
        return np.zeros_like(H) if len(calls) == 1 else H

    # A singular Hessian falls back to the gradient, whose full step overshoots
    # and is halved; the fit still reaches the same maximum.
    monkeypatch.setattr(determinants, "_hessian", singular_once)
    fit = cc.club_determinants(merged, covariates).summary()
    pd.testing.assert_frame_equal(fit, expected, rtol=1e-6)


# -- results and robustness ------------------------------------------------------


def test_results_without_clubs(clubs):
    empty = clubs._replace(clubs=(), divergent_ids=tuple(range(len(clubs.data))))
    assert empty.transition_paths().empty
    assert list(empty.transition_paths().columns) == list(clubs.data.columns)
    assert empty.merge_divergent() is empty
    assert "merged:" in str(clubs.merge("PS"))


def test_sensitivity_with_merge_divergent_only(filtered_gdp):
    runs = cc.sensitivity(
        filtered_gdp, time_trim=[1 / 3], hac_method=["AQSB"], merge_divergent=True
    )
    assert runs.baseline.n_divergent < cc.find_clubs(filtered_gdp, hac_method="AQSB").n_divergent


def test_describe_missing_with_an_empty_unit():
    data = cc.load_gdp().iloc[:3].copy()
    data.iloc[1] = np.nan
    table = cc.describe_missing(data)
    assert table.loc[data.index[1], "n_missing"] == data.shape[1]


# -- plots -----------------------------------------------------------------------


def test_club_panel_options(merged):
    ax = merged.plot.club("club4", reference_line=True)
    assert any(list(line.get_ydata()) == [1.0, 1.0] for line in ax.get_lines())
    # Two units ending at nearly the same level: labels would collide, so a legend.
    paths = merged.transition_paths().loc["club4"].iloc[:, -1].sort_values()
    close = list(paths.index[paths.diff().abs().argmin() - 1 : paths.diff().abs().argmin() + 1])
    ax = merged.plot.club("club4", highlight=close)
    assert ax.get_legend() is not None and not ax.texts
    assert _end_labels(ax, cc.LIGHT, 0, []) is False


def test_variation_legend_inside_for_few_clubs(merged):
    ax = merged.plot.variation(clubs=["club1", "club2"])
    assert ax.get_legend() is not None


def test_co_membership_with_divergent_units(filtered_gdp):
    runs = cc.sensitivity(filtered_gdp, hac_method=["AQSB"], time_trim=[1 / 3, 0.3])
    assert runs.baseline.n_divergent > 0
    ax = runs.plot()
    assert "divergent" in [t.get_text() for t in ax.get_yticklabels()]


def test_beta_plots_with_plain_labels_and_controls():
    gdp = cc.load_gdp()
    labels = pd.Series(np.where(np.arange(len(gdp)) % 3 == 0, "A", "B"), index=gdp.index)
    labels.iloc[:4] = "divergent"
    ax = cc.beta_convergence(gdp, clubs=labels, club_effects=True).plot()
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"A", "B", "divergent"}

    controls = pd.DataFrame({"z": np.log(gdp[1975])}, index=gdp.index)
    ax = cc.beta_convergence(gdp, controls=controls).plot()
    assert ax.get_lines()  # the fitted line, at the mean of the controls


def test_beta_half_life_not_defined():
    # Lower units overtake higher ones: 1 + beta T < 0, so no speed or half-life.
    rng = np.random.default_rng(0)
    start = rng.uniform(6, 10, 40)
    end = 8 - 0.5 * (start - 8) + rng.normal(0, 0.01, 40)
    data = pd.DataFrame({2000: np.exp(start), 2010: np.exp(end)})
    result = cc.beta_convergence(data)
    assert result.converges() and np.isnan(result.half_life)
    assert "half-life: n/a" in result.plot().texts[0].get_text()


def test_pairwise_unit_heatmaps():
    small = cc.pairwise_convergence(cc.load_gdp().iloc[:8])
    ax = small.plot()  # no clubs: the unit heatmap, with names
    assert len(ax.get_xticklabels()) == 8
    large = cc.pairwise_convergence(cc.load_gdp().iloc[:61])
    ax = large.plot()
    assert not ax.get_xticks().size and not ax.get_yticks().size


def test_grid_options(merged):
    fig = merged.plot(title="GDP clubs")
    assert fig._suptitle.get_text() == "GDP clubs"
    with pytest.raises(ValueError, match="no clubs"):
        merged.plot(clubs=[])


def test_interactive_edge_cases(merged):
    data = cc.load_filtered_gdp().copy()
    data.columns = pd.period_range("1970", periods=data.shape[1], freq="Y")
    fig = cc.find_clubs(data).plot.interactive(kind="average")
    assert isinstance(fig.data[0].x[0], pd.Timestamp)
    with pytest.raises(ValueError, match="no clubs"):
        merged.plot.interactive(clubs=[])
    fig = cc.beta_convergence(cc.load_gdp()).plot_interactive()
    assert [trace.name for trace in fig.data] == ["all units", "fit"]


def test_dynamics_plot_methods():
    dynamics = cc.distribution_dynamics(cc.load_gdp(), lag=5)
    assert dynamics.plot_transitions().get_title(loc="left").startswith("Transition")


def test_cli_keeps_non_numeric_periods(tmp_path, capsys):
    from convergence_clubs.cli import main

    data = cc.load_filtered_gdp().copy()
    data.columns = [f"Y{c}" for c in data.columns]
    csv = tmp_path / "f.csv"
    data.to_csv(csv)
    assert main([str(csv)]) == 0
    assert capsys.readouterr().out.startswith("7 convergence clubs")
