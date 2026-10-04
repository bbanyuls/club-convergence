"""Regenerate the figures in docs/ (run: uv run python scripts/make_figures.py)."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import convergence_clubs as cc  # noqa: E402

warnings.simplefilter("ignore", cc.ClubConvergenceWarning)

DOCS = Path(__file__).resolve().parent.parent / "docs"
DPI = 120


def save(fig, name: str) -> None:
    fig.savefig(DOCS / name, dpi=DPI)
    plt.close(fig)
    print("wrote", DOCS / name)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    clubs = cc.find_clubs(cc.load_filtered_gdp())
    merged = clubs.merge("PS")

    save(merged.plot(ncols=4), "transition_paths.png")
    save(merged.plot(ncols=4, theme="dark", units="band"), "transition_paths_dark.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    merged.plot.average(ax=axes[0], emphasis=["club1", "club6"])
    merged.plot.club("club4", ax=axes[1], highlight=["Bolivia", "Honduras"], reference_line=True)
    save(fig, "emphasis_and_highlight.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    merged.plot.variation(ax=axes[0])
    merged.plot.log_t("club1", ax=axes[1])
    save(fig, "diagnostics.png")

    runs = cc.sensitivity(cc.load_filtered_gdp(), cstar=[0, -1], merge=[None, "PS"])
    save(runs.plot().figure, "co_membership.png")

    gdp = cc.load_gdp()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    cc.beta_convergence(gdp).plot(ax=axes[0], label=["China", "Botswana", "Liberia", "Ireland"])
    cc.beta_convergence(gdp, clubs=merged, club_effects=True).plot(ax=axes[1])
    save(fig, "beta.png")

    sigma = cc.sigma_convergence(gdp, clubs=merged)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    sigma.plot(ax=axes[0])
    sigma.plot(ax=axes[1], index=True)
    save(fig, "sigma.png")

    dynamics = cc.distribution_dynamics(gdp, lag=5, clubs=merged)
    save(dynamics.plot(), "dynamics.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    dynamics.plot_densities(ax=axes[0])
    dynamics.plot_densities(ax=axes[1], periods=[1970, 1980, 1990, 2003], rug=False)
    save(fig, "densities.png")

    boot = cc.bootstrap(cc.load_filtered_gdp(), 200, seed=0)
    save(boot.plot_stability().figure, "stability.png")


if __name__ == "__main__":
    main()
