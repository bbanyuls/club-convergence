"""Long-run variance estimators used by the log t regression.

Both estimators use the Quadratic Spectral kernel (Andrews, 1991):

* ``FQSB`` is a line-by-line port of ``ps_andrews_hac`` from the R package
  ConvergenceClubs, itself an adaptation of Phillips and Sul's GAUSS code.
* ``AQSB`` reproduces ``sandwich::vcovHAC(lm(...))`` with its default arguments
  (Andrews automatic bandwidth, AR(1) approximation, no prewhitening,
  small-sample adjustment ``n / (n - k)``).
"""

from __future__ import annotations

import numpy as np

_QS_SMALL = 1e-3


def ps_andrews_hac(resid: np.ndarray) -> float:
    """Long-run variance of ``resid`` with a QS kernel and fixed bandwidth.

    The quirks of the original implementation are kept on purpose (the
    autocovariance loop ignores the last residual and every term is scaled by
    ``1 / (T - 1)``), so that t-statistics match the R package exactly.
    """
    x = np.asarray(resid, dtype=float)
    t = x.size
    x1, x2 = x[:-1], x[1:]
    b1 = (x1 @ x2) / (x1 @ x1)
    a = (4.0 * b1**2) / ((1.0 - b1) ** 4)
    bandwidth = 1.3221 * (a * t) ** 0.2
    lags = np.arange(1, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.2 * np.pi * lags / bandwidth
        kernel = (np.sin(z) / z - np.cos(z)) * (3.0 / z**2)

    t2 = t - 1
    out = 0.0
    for i in range(1, t2):
        out += 2.0 * (x[: t2 - i] @ x[i:t2]) * kernel[i - 1] / t2
    return float(out + (x @ x) / t2)


def _qs_weights(x: np.ndarray) -> np.ndarray:
    """Quadratic Spectral kernel, as in ``sandwich::kweights``."""

    def qs(v):
        y = 6.0 * np.pi * v / 5.0
        return 3.0 * (1.0 / y) ** 2 * (np.sin(y) / y - np.cos(y))

    small = np.abs(x) < _QS_SMALL
    w = np.empty_like(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w[~small] = qs(x[~small])
    cf = 1e6 * np.log(qs(_QS_SMALL))
    w[small] = np.exp(cf * x[small] ** 2)
    return w


def _ar1_fit(u: np.ndarray) -> tuple[float, float]:
    """AR(1) coefficient and innovation s.d., as estimated by ``stats::ar.ols``."""
    u = u - u.mean()
    y, lag = u[1:], u[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(coef[1]), float(np.sqrt(resid @ resid / y.size))


def _andrews_bandwidth(scores: np.ndarray) -> float | None:
    """Andrews' AR(1) plug-in bandwidth, or ``None`` when it is undefined or not positive.

    The intercept (first column) gets zero weight, as in ``sandwich::bwAndrews``.
    A perfect AR(1) fit (sigma = 0, tiny samples) gives 0/0, which is the undefined case.
    """
    n, k = scores.shape
    rho, sigma = np.array([_ar1_fit(scores[:, j]) for j in range(k)]).T
    col_weights = np.ones(k)
    col_weights[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sum(col_weights * (sigma / (1 - rho)) ** 4)
        alpha2 = np.sum(col_weights * 4 * rho**2 * sigma**4 / (1 - rho) ** 8) / denom
        bandwidth = 1.3221 * (n * alpha2) ** 0.2
    return float(bandwidth) if bandwidth > 0 and np.isfinite(bandwidth) else None


def andrews_qs_vcov(design: np.ndarray, resid: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """HAC covariance matrix of OLS coefficients, like ``sandwich::vcovHAC``.

    ``design`` must have the intercept in its first column. Where
    ``sandwich`` errors for lack of a bandwidth, this falls back to White's
    estimator, which is more useful.
    """
    n, k = design.shape
    scores = design * resid[:, None]

    bandwidth = _andrews_bandwidth(scores)
    if bandwidth is None:
        weights = np.ones(1)
    else:
        weights = _qs_weights(np.arange(n) / bandwidth)
        keep = np.nonzero(np.abs(weights) > tol)[0]
        weights = weights[: int(keep[-1]) + 1] if keep.size else np.ones(1)

    utu = 0.5 * weights[0] * scores.T @ scores
    for lag, w in enumerate(weights[1:], start=1):
        utu += w * scores[: n - lag].T @ scores[lag:]
    utu = utu + utu.T
    meat = n / (n - k) * utu / n

    bread = np.linalg.inv(design.T @ design) * n
    return bread @ meat @ bread / n
