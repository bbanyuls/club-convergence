# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Club convergence
- Python port of the R package ConvergenceClubs:
  - log t test with FQSB/AQSB HAC errors;
  - clustering with fixed or incremental `c*`;
  - PS and vLT merging, and merging of divergent units.

  It matches R to within 1e-8, validated against fixtures generated with R and against the R Journal paper ([#1](https://github.com/bbanyuls/club-convergence/pull/1)).
- `convention="Stata"` reproduces `psecta`/`imergeclub` (Du, 2018):
  - core group by the largest t;
  - `c*` raised automatically;
  - `kq(0.3)` trimming, with [rT] rounded half up.

  `merge_clubs(iterate=True)` and `divergent_method="last_club"` follow `imergeclub`. Validated on all 8 Stata classifications of Mendez (2020) ([#21](https://github.com/bbanyuls/club-convergence/pull/21)).
- Warning (`ClubConvergenceWarning`) and `converges` flags for clubs that fail their own log t test ([#12](https://github.com/bbanyuls/club-convergence/pull/12)).
- `merge_divergent(joint_first=True)`, step 1 of von Lyncker and Thoennissen (2017) ([#12](https://github.com/bbanyuls/club-convergence/pull/12)).
- `ConvergenceClubs.rename()` gives the clubs meaningful names in every table, plot and report; also `report(club_names=...)` and `--club-names`.
- `sort_fraction` (Stata's `fr()`), and `ref_col`/`data_cols` ([#1](https://github.com/bbanyuls/club-convergence/pull/1), [#21](https://github.com/bbanyuls/club-convergence/pull/21)).

### Robustness
- `n_jobs=` for `sensitivity`, `bootstrap`, `pairwise_convergence` and `report` (`--jobs` in the command line). Results do not depend on it. The ADF lag search uses one QR decomposition instead of a regression per lag, 3.5 times faster with identical results.
- `sensitivity()`: grids over trimming, HAC method, `c*`, merging and sorting, with co-membership, adjusted Rand index and stability ([#1](https://github.com/bbanyuls/club-convergence/pull/1)).
- `bootstrap()`: confidence in memberships by resampling units, HP cycles or end periods ([#12](https://github.com/bbanyuls/club-convergence/pull/12)).

### Classical convergence and related analyses
- `beta_convergence()`: absolute, conditional on controls or on clubs, and within each club ([#15](https://github.com/bbanyuls/club-convergence/pull/15)).
- `sigma_convergence()`: dispersion trends with Newey–West errors ([#16](https://github.com/bbanyuls/club-convergence/pull/16)).
- `distribution_dynamics()`: Markov transition matrices, ergodic distribution, mobility and densities ([#19](https://github.com/bbanyuls/club-convergence/pull/19)).
- `pairwise_convergence()` and `adf_test()`: Pesaran's (2007) pairwise test, overall and within or between clubs ([#37](https://github.com/bbanyuls/club-convergence/pull/37)). `kpss_test()` and `kpss=True` add Pesaran's KPSS confirmation ([#39](https://github.com/bbanyuls/club-convergence/pull/39)).
- `club_determinants()`: ordered logit/probit of club membership ([#31](https://github.com/bbanyuls/club-convergence/pull/31)).

### Plots and output
- Customizable matplotlib plots with light and dark themes and a palette checked for color-blind readers. They include transition paths, H(t), the log t regression, co-membership, stability, β, σ, dynamics and determinants ([#1](https://github.com/bbanyuls/club-convergence/pull/1) and later).
- Default theme for all plots, charts, maps and reports (`set_theme`, `theme_context`), named custom themes (`register_theme`, usable by name), a black-and-white `"print"` theme, `Theme.linestyles`, and `--theme print` / `--dpi` in the command line.
- Clearer encodings, checked with a palette validator:
  - maps shade the clubs with one ordered hue (club 1 darkest) and hatch uncertain members, with a matching legend;
  - an ordinal ramp interpolated in OKLab (`Theme.ordinal`, `ordinal_colors`) for initial/final/ergodic shares and density periods, instead of the club colors;
  - zero recedes into the surface in light heatmaps, and empty transition cells stay blank;
  - `pairwise.plot()` shows the share of stationary gaps within and between clubs (`club_matrix()`) on a fixed 0–100% scale;
  - H(t) of clubs with one or two units is drawn thinner, and `plot_log_t(show_discarded=True)` shows the periods left out.
- Fix: `plot_log_t` now uses the rounding of the convention, so with `convention="Stata"` it reports the club's own t statistic.
- Interactive plotly charts ([#17](https://github.com/bbanyuls/club-convergence/pull/17)) and maps of club membership ([#20](https://github.com/bbanyuls/club-convergence/pull/20)).
- `report()`: a self-contained HTML replication appendix ([#23](https://github.com/bbanyuls/club-convergence/pull/23)).
- Command-line tool `convergence-clubs`, covering every analysis.

### Data
- GDP panel of Phillips and Sul (2009), with ISO3 codes ([#1](https://github.com/bbanyuls/club-convergence/pull/1), [#20](https://github.com/bbanyuls/club-convergence/pull/20)).
- PWT 9.0 productivity panel of Mendez (2020) ([#27](https://github.com/bbanyuls/club-convergence/pull/27)).
- `wide_panel()`, `describe_missing()`, `fill_gaps()` and `balance()` for real-world panels ([#1](https://github.com/bbanyuls/club-convergence/pull/1), [#12](https://github.com/bbanyuls/club-convergence/pull/12)).

### Documentation and quality
- CI also runs the tests on the oldest versions `pyproject.toml` allows (numpy 1.23, pandas 1.5, matplotlib 3.7, plotly 5.15, geopandas 0.14.4). This showed the declared minimums were wrong: matplotlib must be ≥ 3.7 (legends placed outside the axes) and geopandas ≥ 0.14.4 (0.14.0 breaks with current fiona).
- Tutorial notebook, executed by the test suite ([#28](https://github.com/bbanyuls/club-convergence/pull/28)).
- A weekly CI job that checks the results against the current R package ([#12](https://github.com/bbanyuls/club-convergence/pull/12)). mypy and a 95% coverage floor in CI ([#29](https://github.com/bbanyuls/club-convergence/pull/29)).
- Citation file and this changelog.
