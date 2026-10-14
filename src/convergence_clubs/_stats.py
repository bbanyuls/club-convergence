"""Small statistics toolkit: OLS with robust covariances and the Student t distribution.

Kept dependency-free (numpy only); tested against statsmodels and scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

CovType = Literal["nonrobust", "HC0", "HC1", "HAC"]
COV_TYPES: tuple[str, ...] = ("nonrobust", "HC0", "HC1", "HAC")


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of a Student t statistic with ``df`` degrees of freedom."""
    if not math.isfinite(t):
        return 0.0 if math.isinf(t) else math.nan
    if df <= 0:
        return math.nan
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb lag length floor(4 (n/100)^(2/9)) (Newey and West, 1994)."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


@dataclass(frozen=True)
class OLSResult:
    params: np.ndarray
    cov: np.ndarray
    resid: np.ndarray
    r_squared: float
    n: int
    k: int
    cov_type: str

    @property
    def std_err(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def t_values(self) -> np.ndarray:
        return self.params / self.std_err

    @property
    def df_resid(self) -> int:
        return self.n - self.k

    def p_values(self, distribution: Literal["t", "normal"] = "t") -> np.ndarray:
        if distribution == "normal":
            return np.array([normal_two_sided_p(t) for t in self.t_values])
        return np.array([t_two_sided_p(t, self.df_resid) for t in self.t_values])


def ols(y: np.ndarray, X: np.ndarray, cov_type: str = "HC1", lags: int | None = None) -> OLSResult:
    """Least squares with ``nonrobust``, White (``HC0``/``HC1``) or Newey-West (``HAC``) errors.

    ``HAC`` uses Bartlett weights with ``lags`` lags (default
    :func:`newey_west_lags`) and no small-sample correction, like
    ``statsmodels`` ``cov_type="HAC", use_correction=False``.
    """
    if cov_type not in COV_TYPES:
        raise ValueError(f"cov_type must be one of {COV_TYPES}")
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k})")
    # Check the rank explicitly: whether inv() fails on a singular X'X depends on
    # the BLAS (OpenBLAS with numpy 1.23 returns huge finite numbers instead).
    if np.linalg.matrix_rank(X) < k:
        raise ValueError("the regression could not be estimated (collinear regressors?)")
    # numpy 2.0-2.2 with macOS Accelerate can emit spurious floating-point warnings
    # from matmul; real numerical problems are caught by the finiteness check below.
    with np.errstate(all="ignore"):
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:  # pragma: no cover - the rank check comes first
            raise ValueError(
                "the regression could not be estimated (collinear regressors?)"
            ) from None
        params = xtx_inv @ X.T @ y
        resid = y - X @ params
        if cov_type == "nonrobust":
            cov = xtx_inv * (resid @ resid) / (n - k)
        elif cov_type in ("HC0", "HC1"):
            meat = (X * resid[:, None] ** 2).T @ X
            cov = xtx_inv @ meat @ xtx_inv
            if cov_type == "HC1":
                cov *= n / (n - k)
        else:
            lags = newey_west_lags(n) if lags is None else lags
            scores = X * resid[:, None]
            meat = scores.T @ scores
            for lag in range(1, lags + 1):
                weight = 1.0 - lag / (lags + 1.0)
                gamma = scores[lag:].T @ scores[:-lag]
                meat += weight * (gamma + gamma.T)
            cov = xtx_inv @ meat @ xtx_inv
    if not (np.isfinite(params).all() and np.isfinite(cov).all()):
        raise ValueError("the regression could not be estimated (collinear regressors?)")
    centred = y - y.mean()
    r_squared = 1.0 - (resid @ resid) / (centred @ centred) if centred.any() else math.nan
    return OLSResult(params, cov, resid, float(r_squared), n, k, cov_type)
