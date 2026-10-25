---
title: 'convergence-clubs: Phillips–Sul club convergence analysis in Python'
tags:
  - Python
  - econometrics
  - economic growth
  - convergence clubs
  - panel data
authors:
  - name: Bernard Banyuls
    affiliation: 1
affiliations:
  - name: Universidad Complutense de Madrid, Spain
    index: 1
date: 25 October 2026
bibliography: paper.bib
---

# Summary

Countries and regions do not always grow towards a common level of income. Some
converge towards the same path and form a *convergence club*, while others follow a
different one. `convergence-clubs` finds these clubs in a panel of units observed over
time (for example GDP per worker by country). It implements the log $t$ test and the
clustering algorithm of @PhillipsSul2007 and @PhillipsSul2009, which have become the
standard way to identify clubs in empirical growth research. The package reports which
units belong to which club, tests whether each club really converges, and produces
tables, plots, maps and an HTML report.

# Statement of need

Researchers who use the Phillips–Sul method today choose between the R package
`ConvergenceClubs` [@SicheraPizzuto2019] and the Stata package `psecta` [@Du2017]. A
researcher whose analysis lives in Python has no maintained equivalent, and has to move
the data to another language or reimplement a method with many delicate choices:
the trimming of the sample, the covariance estimator, the rule for the core group and
the merging of clubs. The two existing
implementations also differ from each other in some of these choices, and published
classifications depend on which one was used.

`convergence-clubs` gives Python users the method with both sets of choices available.
The default convention reproduces the R package and `convention="Stata"` reproduces
`psecta`. Beyond the clustering itself, the package addresses a practical concern in the
applied literature: clubs found by the algorithm are sensitive to the settings and to
the sample, and this is rarely reported.

# State of the field

`ConvergenceClubs` for R is the reference implementation, and `psecta` for Stata is the
other widely used one. We chose to write a new package rather than contribute to either,
because neither can be called from Python without a second runtime, and because the
goal was to run the same analysis in the Python data stack (`numpy`, `pandas`,
`matplotlib`). The design follows the R package's functions and arguments, so that
existing R code translates directly, and the validation described below shows that it
gives the same results. While validating the port we found two inconsistencies in
version 2.2.5 of the R package and reported them with a fix upstream.

# Software design

The core depends only on `numpy` and `pandas`. Plots, interactive charts and maps are
optional extras. Results are immutable objects that keep the settings used, so that a
table or a figure can always be traced back to the call that produced it.

The package covers the method and the checks an applied analysis needs around it:

- the log $t$ test with the two HAC variants (FQSB and AQSB), clustering with a fixed or
  incremental $c^*$, and merging of clubs following @PhillipsSul2009 and
  @vonLynckerThoennissen2017;
- `sensitivity`, which re-estimates the clubs over a grid of settings and reports
  co-membership, the adjusted Rand index and a stability score for each unit;
- `bootstrap`, which gives the confidence in each unit's membership;
- `rolling_clubs`, which follows clubs and the units that move between them over time;
- $\beta$- and $\sigma$-convergence, distribution dynamics, a pairwise convergence test
  and an ordered logit of club membership, to place the clubs in context.

Analyses that take minutes on large panels run in parallel with `n_jobs`, and give the
same result for any number of workers.

# Research impact statement

Correctness is checked against the existing implementations, not asserted. The test
suite compares every test statistic and club to fixtures generated with the R package,
to within $10^{-8}$, and re-runs this check weekly against the current R release. With
`convention="Stata"` the package reproduces all eight published `psecta`
classifications of @Mendez2020. The suite has 446 tests with full line coverage, and
continuous integration runs it on the oldest and the newest supported versions of the
dependencies. Documentation, including the method and a validation guide that lists
the known differences from R, is at <https://bbanyuls.github.io/club-convergence/>.

# AI usage disclosure

Generative AI was used in the development of this software, the documentation and this
paper. The tool was Claude Code (Anthropic), with the Claude Sonnet 5.5 model among
others. It was used to write and refactor code, tests and documentation, and to draft
this text. The design of the package, the choice of methods and conventions, and the
validation strategy were decided by the author, who reviewed and edited all AI-assisted
output and takes responsibility for the correctness of the submitted work.

# Acknowledgements

The package follows the work of Sichera and Pizzuto, whose R package it ports, and of
Du, whose Stata package it matches on the published classifications.

# References
