"""Regenerate the figures in docs/ (run: uv run python scripts/make_figures.py)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import convergence_clubs as cc  # noqa: E402

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


if __name__ == "__main__":
    main()
