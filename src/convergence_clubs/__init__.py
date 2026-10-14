"""Club convergence analysis following Phillips and Sul (2007, 2009).

A Python port of the R package ConvergenceClubs (Sichera and Pizzuto, 2019).
"""

from .beta import BetaResult, beta_convergence
from .bootstrap import bootstrap
from .clustering import ClubConvergenceWarning, find_clubs
from .datasets import (
    country_codes,
    load_filtered_gdp,
    load_gdp,
    load_productivity,
    productivity_info,
)
from .determinants import DeterminantsResult, club_determinants
from .dynamics import DynamicsResult, distribution_dynamics
from .filters import hp_filter
from .latex import to_latex
from .logt import (
    CRITICAL_VALUE,
    LogTResult,
    cross_sectional_variation,
    log_t_regression_data,
    log_t_test,
    relative_transition,
)
from .maps import match_regions, plot_map
from .merging import merge_clubs, merge_divergent
from .pairwise import (
    ADFResult,
    KPSSResult,
    PairwiseResult,
    adf_test,
    kpss_test,
    mackinnon_pvalue,
    pairwise_convergence,
)
from .panel import balance, describe_missing, fill_gaps, wide_panel
from .plotting import (
    DARK,
    JOURNAL,
    LIGHT,
    PRINT,
    Theme,
    get_theme,
    ordinal_colors,
    plot_average_paths,
    plot_beta,
    plot_class_distributions,
    plot_club,
    plot_co_membership,
    plot_cross_sectional_variation,
    plot_densities,
    plot_determinants,
    plot_dynamics,
    plot_log_t,
    plot_pairwise,
    plot_sigma,
    plot_stability,
    plot_transition_paths,
    plot_transitions,
    register_theme,
    set_theme,
    theme_context,
)
from .report import Report, report
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
    "ADFResult",
    "CRITICAL_VALUE",
    "BetaResult",
    "DARK",
    "JOURNAL",
    "LIGHT",
    "PRINT",
    "Club",
    "ClubConvergenceWarning",
    "ConvergenceClubs",
    "DeterminantsResult",
    "DynamicsResult",
    "KPSSResult",
    "LogTResult",
    "PairwiseResult",
    "Report",
    "SensitivityResult",
    "SigmaResult",
    "Theme",
    "adf_test",
    "adjusted_rand_index",
    "balance",
    "beta_convergence",
    "bootstrap",
    "club_determinants",
    "compare_clubs",
    "country_codes",
    "cross_sectional_variation",
    "describe_missing",
    "distribution_dynamics",
    "fill_gaps",
    "get_theme",
    "find_clubs",
    "hp_filter",
    "kpss_test",
    "load_filtered_gdp",
    "load_gdp",
    "load_productivity",
    "log_t_regression_data",
    "match_regions",
    "log_t_test",
    "mackinnon_pvalue",
    "ordinal_colors",
    "merge_clubs",
    "merge_divergent",
    "pairwise_convergence",
    "plot_average_paths",
    "plot_beta",
    "plot_class_distributions",
    "plot_club",
    "plot_co_membership",
    "plot_determinants",
    "plot_cross_sectional_variation",
    "plot_densities",
    "plot_dynamics",
    "plot_log_t",
    "plot_pairwise",
    "plot_map",
    "plot_sigma",
    "plot_stability",
    "plot_transition_paths",
    "plot_transitions",
    "productivity_info",
    "register_theme",
    "relative_transition",
    "report",
    "sensitivity",
    "set_theme",
    "sigma_convergence",
    "to_latex",
    "theme_context",
    "wide_panel",
]
