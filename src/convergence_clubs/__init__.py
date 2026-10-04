"""Club convergence analysis following Phillips and Sul (2007, 2009).

A Python port of the R package ConvergenceClubs (Sichera and Pizzuto, 2019).
"""

from .clustering import find_clubs
from .datasets import load_filtered_gdp, load_gdp
from .filters import hp_filter
from .logt import (
    CRITICAL_VALUE,
    LogTResult,
    cross_sectional_variation,
    log_t_test,
    relative_transition,
)
from .merging import merge_clubs, merge_divergent
from .plotting import plot_transition_paths
from .results import Club, ConvergenceClubs

__version__ = "0.1.0"

__all__ = [
    "CRITICAL_VALUE",
    "Club",
    "ConvergenceClubs",
    "LogTResult",
    "cross_sectional_variation",
    "find_clubs",
    "hp_filter",
    "load_filtered_gdp",
    "load_gdp",
    "log_t_test",
    "merge_clubs",
    "merge_divergent",
    "plot_transition_paths",
    "relative_transition",
]
