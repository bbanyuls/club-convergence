import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats

import convergence_clubs as cc
from convergence_clubs.dynamics import gaussian_kde, silverman_bandwidth


@pytest.fixture(scope="module")
def gdp():
    return cc.load_gdp()


@pytest.fixture(scope="module")
def merged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cc.ClubConvergenceWarning)
        return cc.find_clubs(cc.load_filtered_gdp()).merge("PS")


def test_quantile_classes_and_counts_match_a_direct_loop(gdp):
    d = cc.distribution_dynamics(gdp, n_classes=4, lag=3)
    relative = gdp / gdp.mean()
    cuts = np.quantile(relative.to_numpy().ravel(), [0.25, 0.5, 0.75])
    np.testing.assert_allclose(d.cuts, cuts)
    classes = np.searchsorted(cuts, relative.to_numpy(), side="right")
    counts = np.zeros((4, 4), dtype=int)
    for unit in range(len(gdp)):
        for t in range(gdp.shape[1] - 3):
            counts[classes[unit, t], classes[unit, t + 3]] += 1
    np.testing.assert_array_equal(d.counts.to_numpy(), counts)
    np.testing.assert_allclose(d.matrix.sum(axis=1), 1.0)
    pooled = np.bincount(classes.ravel(), minlength=4) / classes.size
    np.testing.assert_allclose(pooled, 0.25, atol=0.01)  # quantile classes: equal shares


def test_non_overlapping_transitions(gdp):
    d = cc.distribution_dynamics(gdp, lag=5, overlapping=False)
    assert d.counts.to_numpy().sum() == len(gdp) * len(range(0, 29, 5))


def test_ergodic_distribution(gdp):
    d = cc.distribution_dynamics(gdp, lag=5)
    pi, P = d.ergodic.to_numpy(), d.matrix.to_numpy()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)
    assert pi.sum() == pytest.approx(1.0) and (pi >= 0).all()
    np.testing.assert_allclose(np.linalg.matrix_power(P, 20_000)[0], pi, atol=1e-9)
    # Quah's twin peaks: the long run piles up in the poorest and richest classes.
    assert pi[0] > 0.25 and pi[-1] > 0.3 and pi[2] < 0.15


def test_mobility_indices_on_known_matrices():
    values = pd.DataFrame(np.ones((4, 6)))  # placeholder data for the container
    labels = ["a", "b"]

    def result(P):
        counts = pd.DataFrame(np.asarray(P) * 1000, index=labels, columns=labels)
        return cc.DynamicsResult(values=values, cuts=np.array([1.0]), counts=counts, lag=2)

    frozen = result([[1, 0], [0, 1]])
    assert frozen.shorrocks == 0 and np.isnan(frozen.half_life)
    mixing = result([[0.5, 0.5], [0.5, 0.5]])
    assert mixing.shorrocks == 1
    chain = result([[0.9, 0.1], [0.2, 0.8]])  # eigenvalues 1 and 0.7
    assert abs(chain.eigenvalues[1]) == pytest.approx(0.7)
    assert chain.half_life == pytest.approx(-np.log(2) / np.log(0.7) * 2)
    np.testing.assert_allclose(chain.ergodic, [2 / 3, 1 / 3])


def test_fixed_bounds_and_composition(gdp, merged):
    d = cc.distribution_dynamics(gdp, bounds=[2, 0.5, 1], clubs=merged)
    assert d.labels == ["< 0.50", "0.50–1.00", "1.00–2.00", "≥ 2.00"]
    composition = d.composition()
    assert composition.to_numpy().sum() == len(gdp)
    assert list(composition.index) == [c.name for c in merged]
    assert composition.loc["club1", "≥ 2.00"] > composition.loc["club5", "≥ 2.00"]
    np.testing.assert_allclose(d.distributions().sum(), 1.0)
    assert "Shorrocks" in repr(d) and d.summary()["classes"] == 4


def test_kde_matches_scipy(gdp):
    x = np.log((gdp / gdp.mean())[2003].to_numpy())
    h = silverman_bandwidth(x)
    grid = np.linspace(x.min(), x.max(), 50)
    ref = scipy.stats.gaussian_kde(x, bw_method=h / x.std(ddof=1))(grid)
    np.testing.assert_allclose(gaussian_kde(x, grid, h), ref, rtol=1e-10)
    density = cc.distribution_dynamics(gdp).density(2003)
    assert np.trapezoid(density.to_numpy(), np.log(density.index)) == pytest.approx(1, abs=1e-3)


def test_errors(gdp):
    with pytest.raises(ValueError, match="positive"):
        cc.distribution_dynamics(-gdp)
    with pytest.raises(ValueError, match="lag"):
        cc.distribution_dynamics(gdp, lag=40)
    with pytest.raises(ValueError, match="n_classes"):
        cc.distribution_dynamics(gdp, n_classes=1)
    with pytest.raises(ValueError, match="distinct"):
        cc.distribution_dynamics(gdp, bounds=[1, 1])
    with pytest.raises(ValueError, match="clubs="):
        cc.distribution_dynamics(gdp).composition()
    with pytest.raises(ValueError, match="never observed"):
        _ = cc.distribution_dynamics(gdp, bounds=[100, 200]).ergodic


def test_plots(gdp, merged):
    d = cc.distribution_dynamics(gdp, lag=5, clubs=merged)
    fig = d.plot()
    heat, bars = fig.axes[:2]
    assert len(heat.texts) == 25 and heat.texts[0].get_text() == f"{d.matrix.iloc[0, 0]:.2f}"
    assert {t.get_color() for t in heat.texts} <= {"#ffffff", "#0b0b0b"}
    assert [t.get_text() for t in bars.get_legend().get_texts()] == ["initial", "final", "ergodic"]
    ax = d.plot_densities()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:2] == ["1970", "2003"] and "club1" in labels
    dark = d.plot_densities(periods=[1970, 1990, 2003], rug=False, theme="dark")
    lines = [ln for ln in dark.get_lines() if ln.get_label() in {"1970", "1990", "2003"}]
    from convergence_clubs.plotting import _luminance

    lum = [_luminance(ln.get_color()) for ln in lines]
    assert lum == sorted(lum)  # later periods stand out more on a dark surface
    plt.close("all")
