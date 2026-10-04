"""Club convergence analysis following Phillips and Sul (2007, 2009).

A Python port of the R package ConvergenceClubs (Sichera and Pizzuto, 2019).
"""

from .beta import BetaResult, beta_convergence
from .bootstrap import bootstrap
from .clustering import ClubConvergenceWarning, find_clubs
from .datasets import load_filtered_gdp, load_gdp
from .dynamics import DynamicsResult, distribution_dynamics
from .filters import hp_filter
from .logt import (
    CRITICAL_VALUE,
    LogTResult,
    cross_sectional_variation,
    log_t_regression_data,
    log_t_test,
    relative_transition,
)
from .merging import merge_clubs, merge_divergent
from .panel import balance, describe_missing, fill_gaps, wide_panel
from .plotting import (
    DARK,
    LIGHT,
    Theme,
    plot_average_paths,
    plot_beta,
    plot_class_distributions,
    plot_club,
    plot_co_membership,
    plot_cross_sectional_variation,
    plot_densities,
    plot_dynamics,
    plot_log_t,
    plot_sigma,
    plot_stability,
    plot_transition_paths,
    plot_transitions,
)
from .results import Club, ConvergenceClubs
from .sensitivity import (
    SensitivityResult,
    adjusted_rand_index,
    compare_clubs,
    sensitivity,
)
from .sigma import SigmaResult, sigma_convergence

__version__ = "0.1.0"

__all__ = [
    "CRITICAL_VALUE",
    "BetaResult",
    "DARK",
    "LIGHT",
    "Club",
    "ClubConvergenceWarning",
    "ConvergenceClubs",
    "DynamicsResult",
    "LogTResult",
    "SensitivityResult",
    "SigmaResult",
    "Theme",
    "adjusted_rand_index",
    "balance",
    "beta_convergence",
    "bootstrap",
    "compare_clubs",
    "cross_sectional_variation",
    "describe_missing",
    "distribution_dynamics",
    "fill_gaps",
    "find_clubs",
    "hp_filter",
    "load_filtered_gdp",
    "load_gdp",
    "log_t_regression_data",
    "log_t_test",
    "merge_clubs",
    "merge_divergent",
    "plot_average_paths",
    "plot_beta",
    "plot_class_distributions",
    "plot_club",
    "plot_co_membership",
    "plot_cross_sectional_variation",
    "plot_densities",
    "plot_dynamics",
    "plot_log_t",
    "plot_sigma",
    "plot_stability",
    "plot_transition_paths",
    "plot_transitions",
    "relative_transition",
    "sensitivity",
    "sigma_convergence",
    "wide_panel",
]
