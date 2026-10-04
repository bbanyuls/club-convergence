"""Relative transition paths and the Phillips-Sul log t convergence test."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._hac import andrews_qs_vcov, ps_andrews_hac

HACMethod = Literal["FQSB", "AQSB"]
HAC_METHODS: tuple[str, ...] = ("FQSB", "AQSB")

#: One-sided 5% critical value used throughout the clustering procedure.
CRITICAL_VALUE = -1.65


@dataclass(frozen=True)
class LogTResult:
    """Outcome of a log t regression.

    ``p_value`` is the left-tail normal probability of ``t_value``: small
    values reject the null hypothesis of convergence. ``alpha`` is the
    intercept of the regression.
    """

    beta: float
    std_err: float
    t_value: float
    p_value: float
    alpha: float = math.nan

    def converges(self, threshold: float = CRITICAL_VALUE) -> bool:
        """Whether the null of convergence is *not* rejected at ``threshold``."""
        return self.t_value > threshold


def relative_transition(x: np.ndarray) -> np.ndarray:
    """Relative transition paths ``h_it = x_it / mean_i(x_it)`` (units in rows)."""
    x = np.asarray(x, dtype=float)
    return x / x.mean(axis=0)


def cross_sectional_variation(x: np.ndarray) -> np.ndarray:
    """Cross-sectional variation ``H_t = mean_i((h_it - 1)^2)`` (units in rows)."""
    h = relative_transition(x)
    return ((h - 1.0) ** 2).mean(axis=0)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def log_t_regression_data(H: np.ndarray, time_trim: float = 1 / 3) -> tuple[np.ndarray, np.ndarray]:
    """Regressor ``log t`` and response ``log(H_1 / H_t) - 2 log(log t)`` of the log t test.

    Only periods ``t = [rT] + 1, ..., T`` (``r = time_trim``) are returned.
    """
    H = np.asarray(H, dtype=float)
    n_t = H.size
    periods = np.arange(round(n_t * time_trim) + 1, n_t + 1)
    log_t = np.log(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(H[0] / H[periods - 1]) - 2.0 * np.log(log_t)
    return log_t, y


def log_t_test(
    H: np.ndarray,
    time_trim: float = 1 / 3,
    hac_method: HACMethod = "FQSB",
) -> LogTResult:
    """Estimate the log t regression of Phillips and Sul (2007, 2009).

    The model is ``log(H_1 / H_t) - 2 log(log t) = a + b log t + u_t`` for
    ``t = [rT] + 1, ..., T`` with ``r = time_trim``. The t-statistic of ``b``
    uses HAC standard errors with a Quadratic Spectral kernel, with either a
    fixed (``"FQSB"``) or an Andrews adaptive (``"AQSB"``) bandwidth.

    Parameters
    ----------
    H
        Cross-sectional variation, one value per period
        (see :func:`cross_sectional_variation`).
    time_trim
        Share of initial periods discarded from the regression.
    hac_method
        ``"FQSB"`` or ``"AQSB"``.
    """
    if hac_method not in HAC_METHODS:
        raise ValueError(f"hac_method must be one of {HAC_METHODS}, got {hac_method!r}")
    log_t, y = log_t_regression_data(H, time_trim)
    design = np.column_stack([np.ones_like(log_t), log_t])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    beta = float(coef[1])

    if hac_method == "FQSB":
        xtx_inv = np.linalg.inv(design.T @ design)
        std_err = math.sqrt(xtx_inv[1, 1] * ps_andrews_hac(resid))
    else:
        std_err = math.sqrt(andrews_qs_vcov(design, resid)[1, 1])

    t_value = beta / std_err
    return LogTResult(
        beta=beta,
        std_err=std_err,
        t_value=t_value,
        p_value=_normal_cdf(t_value),
        alpha=float(coef[0]),
    )
