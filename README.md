<img src="assets/icon.svg" width="64" height="64" alt="">

# Club convergence

Club convergence analysis in Python, following Phillips and Sul (2007, 2009).

This is a port of the R package
[ConvergenceClubs](https://github.com/rhobis/ConvergenceClubs) (Sichera and Pizzuto, 2019).
It implements:

- the **log t convergence test**, with HAC standard errors (Quadratic Spectral kernel,
  fixed or Andrews adaptive bandwidth);
- the **club clustering algorithm** (core group formation, sieving with `c*`, fixed or
  incremental);
- **club merging**, using either the Phillips and Sul (2009) rule or the von Lyncker and
  Thoennissen (2017) rule, plus merging of divergent units;
- **robustness checks**:
  - sensitivity of the clubs to `time_trim`, the HAC method, `c*`, merging and unit ordering;
  - **bootstrap confidence** in each unit's club, by resampling units, business cycles or the
    end of the sample;
  - warnings when a club fails its own convergence test;
- **classical convergence**:
  - β-convergence (absolute, conditional on controls or on clubs, and within each club), with
    the implied speed and half-life;
  - σ-convergence (the trend of dispersion over time, for the whole panel and within each club);
  - distribution dynamics (Markov transition matrices of relative income, the ergodic
    distribution, mobility indices and "twin peaks" densities);
- **data preparation**: HP filtering, long-to-wide reshaping and handling of missing values;
- **plots** you can customize, **interactive charts**, **maps** of club membership, a
  **command-line tool**, and a **one-call HTML report** that works as a replication appendix.

Results match the R package to within 1e-8, and with `convention="Stata"` they match the Stata
package `psecta` (Du, 2018) on the published results of Mendez (2020). See
[R and Stata conventions](docs/guide/clubs.md#r-and-stata-conventions) and [Validation](docs/guide/validation.md).

## Installation

```bash
pip install "convergence-clubs[plot] @ git+https://github.com/bbanyuls/club-convergence"
```

The core package needs only `numpy` and `pandas`. The `plot` extra adds `matplotlib`, the
`interactive` extra adds `plotly`, and the `maps` extra adds `geopandas`.

## Quick start

New to the method? [`docs/tutorial.ipynb`](docs/tutorial.ipynb) walks through a complete analysis and explains how
to read each result: data preparation, the log t test, clubs and merging, robustness, β, σ, distribution
dynamics, a Stata replication and the report. The test suite runs it, so it always works with the
current version.

```python
import numpy as np
import convergence_clubs as cc

gdp = cc.load_gdp()                         # 152 countries x 1970-2003
data = cc.hp_filter(np.log(gdp), lamb=400)  # same as cc.load_filtered_gdp()

# Does the whole panel converge?
H = cc.cross_sectional_variation(data.to_numpy())
cc.log_t_test(H, time_trim=1/3)
# LogTResult(beta=-0.8748..., std_err=0.0054..., t_value=-159.55..., p_value=0.0, alpha=0.398...)

clubs = cc.find_clubs(data, time_trim=1/3, cstar=0, hac_method="FQSB")
clubs.summary()
#        n_units   beta  std_err  t_value  p_value  cstar
# club
# club1       50  0.382    0.041    9.282    1.000    0.0
# club2       30  0.240    0.035    6.904    1.000    0.0
# club3       21  0.110    0.032    3.402    1.000    0.0
# club4       24  0.131    0.064    2.055    0.980    0.0
# club5       14  0.190    0.111    1.701    0.956    0.0
# club6       11  1.003    0.166    6.024    1.000    0.0
# club7        2 -0.470    0.842   -0.559    0.288    0.0

merged = clubs.merge("PS")       # or "vLT"; merge_divergent=True is also available
merged.summary()                 # club4 + club5 -> one club of 38 countries
merged.memberships()             # pandas Series: unit -> club
merged.units("club1")            # names of the units in a club
print(merged)                    # detailed report, like R's print()
merged.plot()                    # transition paths (see the plots guide)
```

In Jupyter, a result displays as a summary table.

Two datasets ship with the package: the GDP panel of Phillips and Sul (2009) and the Penn World Table productivity panel of Mendez (2020); see
[Finding clubs](docs/guide/clubs.md#bundled-datasets).

## Guides

| Guide | Contents |
|---|---|
| [Tutorial notebook](docs/tutorial.ipynb) | A complete analysis, step by step, with how to read each result |
| [Finding clubs](docs/guide/clubs.md) | Bundled datasets, input data and missing values, sorting, clubs that fail their own test, R and Stata conventions |
| [Robustness](docs/guide/robustness.md) | Sensitivity to the settings, bootstrap confidence in memberships |
| [β, σ, dynamics and determinants](docs/guide/classical.md) | β- and σ-convergence, distribution dynamics and twin peaks, ordered logit of club membership |
| [Plots and maps](docs/guide/plots.md) | Every plot, themes and customization, interactive charts, maps |
| [Report and command line](docs/guide/report-and-cli.md) | The one-call HTML report, every CLI option |
| [Validation](docs/guide/validation.md) | API correspondence with R, tests against R and Stata, known differences |
| [References](docs/guide/references.md) | The papers behind each method |

![Transition paths of the merged clubs](docs/transition_paths.png)

## Development

```bash
uv sync                              # create the environment
uv run pytest                        # run the tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                          # static type checking (also run in CI)
uv run pytest --cov                  # CI fails if coverage drops below 95%
Rscript scripts/generate_r_fixtures.R  # regenerate the R reference (needs R)
uv run python scripts/make_figures.py  # regenerate the figures in docs/
# docs/tutorial.ipynb is committed without outputs; tests/test_tutorial.py executes it
```

## How to cite

If you use this package, please cite it together with the methods you apply. `CITATION.cff`
has the details: GitHub's "Cite this repository" button and reference managers read it.

- the software: Banyuls, B. *convergence-clubs: club convergence analysis in Python*,
  <https://github.com/bbanyuls/club-convergence>;
- the method: Phillips and Sul (2007, 2009);
- the R package this ports, if you compare with it: Sichera and Pizzuto (2019);
- β/σ/dynamics, Stata conventions or the bundled data: see the
  [references](docs/guide/references.md).

Changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

## License

GPL-3.0-or-later. This package is a derivative of the R package ConvergenceClubs
(© Roberto Sichera and Pietro Pizzuto, GPL-3), and the bundled GDP datasets come from it.
The bundled productivity panel comes from the replication files of Mendez (2020) (MIT licence),
based on the Penn World Table 9.0 (Feenstra, Inklaar and Timmer, 2015; CC BY 4.0).
