"""Regenerate the figures in docs/ (run: uv run python scripts/make_figures.py)."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import convergence_clubs as cc  # noqa: E402

warnings.simplefilter("ignore", cc.ClubConvergenceWarning)

DOCS = Path(__file__).resolve().parent.parent / "docs"
NATURAL_EARTH = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
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

    import geopandas as gpd

    world = gpd.read_file(NATURAL_EARTH)
    world = world[world["ADMIN"] != "Antarctica"]
    merged_boot = cc.bootstrap(cc.load_filtered_gdp(), 200, merge="PS", seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Netherlands Antilles has no polygon
        ax = merged.plot.map(
            world, key="ISO_A3_EH", names=cc.country_codes(), crs="EPSG:8857",
            confidence=merged_boot,
        )  # fmt: skip
    ax.figure.set_size_inches(11, 5.6)
    save(ax.figure, "map.png")

    lp = cc.load_productivity("lp")
    lp_clubs = cc.find_clubs(cc.hp_filter(np.log(lp)), convention="Stata", time_trim=0.333).merge(
        "PS"
    )
    info = cc.productivity_info()
    covariates = pd.DataFrame(
        {
            "log human capital, 1990": np.log(cc.load_productivity("h")[1990]),
            "log capital-output ratio, 1990": np.log(cc.load_productivity("ky")[1990]),
            "log TFP, 1990": np.log(cc.load_productivity("tfp")[1990]),
            "Africa": (info["region"] == "Africa").astype(float),
        }
    )
    save(cc.club_determinants(lp_clubs, covariates).plot().figure, "determinants.png")

    example = cc.report(gdp, title="Convergence of GDP per capita, 152 countries, 1970–2003")
    print("wrote", example.save(DOCS / "report-example.html"))


if __name__ == "__main__":
    main()
