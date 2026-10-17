"""What explains club membership? Ordered logit / probit of the club rank.

The outcome is the club rank ordered so that higher values mean clubs with a
higher relative level (club 1, the top club, is the highest category).
Divergent units are left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .results import ConvergenceClubs

OrderedModel = Literal["logit", "probit"]
_erfc = np.vectorize(math.erfc, otypes=[float])


def _cdf(x: np.ndarray, model: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        if model == "logit":
            return 1.0 / (1.0 + np.exp(-x))
        return 0.5 * _erfc(-x / math.sqrt(2.0))


def _pdf(x: np.ndarray, model: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        if model == "logit":
            e = np.exp(-np.abs(x))
            return e / (1.0 + e) ** 2
        return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _thresholds(theta: np.ndarray, k: int) -> np.ndarray:
    """Ordered thresholds from (first threshold, log increments)."""
    tail = theta[k:]
    return np.concatenate([[tail[0]], tail[0] + np.cumsum(np.exp(tail[1:]))])


def _thresholds_jacobian(theta: np.ndarray, k: int) -> np.ndarray:
    """Derivative of the thresholds with respect to ``theta[k:]``: row j is threshold j."""
    n = theta.size - k
    jac = np.zeros((n, n))
    jac[:, 0] = 1.0
    increments = np.exp(theta[k + 1 :])
    for j in range(n):
        jac[j, 1 : 1 + j] = increments[:j]
    return jac


def _loglik_grad(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, n_classes: int, model: str
) -> tuple[float, np.ndarray]:
    """Log-likelihood and its gradient with respect to ``theta``."""
    k = X.shape[1]
    beta, kappa = theta[:k], _thresholds(theta, k)
    xb = X @ beta
    upper = np.where(y < n_classes - 1, kappa[np.minimum(y, n_classes - 2)] - xb, np.inf)
    lower = np.where(y > 0, kappa[np.maximum(y - 1, 0)] - xb, -np.inf)
    prob = np.clip(_cdf(upper, model) - _cdf(lower, model), 1e-300, None)
    f_up = np.where(np.isfinite(upper), _pdf(upper, model), 0.0)
    f_lo = np.where(np.isfinite(lower), _pdf(lower, model), 0.0)

    grad_beta = -X.T @ ((f_up - f_lo) / prob)
    grad_kappa = np.zeros(n_classes - 1)
    np.add.at(grad_kappa, y[y < n_classes - 1], (f_up / prob)[y < n_classes - 1])
    np.add.at(grad_kappa, y[y > 0] - 1, -(f_lo / prob)[y > 0])
    grad_tail = _thresholds_jacobian(theta, k).T @ grad_kappa
    return float(np.log(prob).sum()), np.concatenate([grad_beta, grad_tail])


def _hessian(theta, X, y, n_classes, model, step=1e-5) -> np.ndarray:
    """Hessian of the log-likelihood by central differences of the analytic gradient."""
    n = theta.size
    H = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = step * max(1.0, abs(theta[i]))
        H[:, i] = (
            _loglik_grad(theta + e, X, y, n_classes, model)[1]
            - _loglik_grad(theta - e, X, y, n_classes, model)[1]
        ) / (2 * e[i])
    return (H + H.T) / 2


_LOGIT_TO_PROBIT_SCALE = 1.702


def _quantile(p: np.ndarray, model: str) -> np.ndarray:
    logit = np.log(p / (1 - p))
    return logit if model == "logit" else logit / _LOGIT_TO_PROBIT_SCALE


@dataclass(frozen=True, eq=False)
class DeterminantsResult:
    """Ordered logit / probit of club membership."""

    coefficients: pd.DataFrame
    thresholds: pd.Series
    log_likelihood: float
    null_log_likelihood: float
    n: int
    model: str
    classes: tuple[str, ...]
    converged: bool
    iterations: int
    excluded: tuple[Any, ...] = ()

    @property
    def pseudo_r2(self) -> float:
        """McFadden's pseudo-R², ``1 - logL / logL0``."""
        return 1.0 - self.log_likelihood / self.null_log_likelihood

    def summary(self) -> pd.DataFrame:
        return self.coefficients.copy()

    def predict_proba(self, covariates: pd.DataFrame) -> pd.DataFrame:
        """Probability of each club (columns, lowest to highest) for new covariate rows."""
        names = list(self.coefficients.index)
        X = covariates[names].to_numpy(dtype=float)
        xb = X @ self.coefficients["coef"].to_numpy()
        cdf = _cdf(self.thresholds.to_numpy()[None, :] - xb[:, None], self.model)
        cum = np.hstack([np.zeros((len(X), 1)), cdf, np.ones((len(X), 1))])
        return pd.DataFrame(np.diff(cum, axis=1), index=covariates.index, columns=self.classes)

    def plot(self, **kwargs) -> Axes:
        """Coefficients with 95% intervals; see :func:`convergence_clubs.plot_determinants`."""
        from .plotting import plot_determinants

        return plot_determinants(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<DeterminantsResult (ordered {self.model}): n={self.n}, "
            f"{len(self.classes)} clubs, pseudo-R2={self.pseudo_r2:.3f}>"
        )


def club_determinants(
    result: ConvergenceClubs,
    covariates: pd.DataFrame,
    *,
    model: OrderedModel | str = "logit",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> DeterminantsResult:
    """Ordered logit / probit of club membership on unit-level covariates.

    Parameters
    ----------
    result
        Clubs from :func:`find_clubs` or :func:`merge_clubs`.
    covariates
        One row per unit (indexed like ``result.data``), numeric columns, no
        constant column (the thresholds play the role of the intercept).
        Typically initial conditions: initial income, schooling, region
        dummies, ...
    model
        ``"logit"`` or ``"probit"``.

    Notes
    -----
    The outcome is the club rank with higher values for higher clubs, so a
    positive coefficient raises the probability of belonging to a higher
    (richer) club. Divergent units are left out. Estimation is by Newton-
    Raphson on the exact log-likelihood; standard errors come from the
    observed information matrix (delta method for the thresholds).
    """
    if model not in ("logit", "probit"):
        raise ValueError("model must be 'logit' or 'probit'")
    if result.n_clubs < 2:
        raise ValueError("at least two clubs are needed")
    members = result.memberships()
    in_club = members[members != "divergent"]
    order = [c.name for c in reversed(result.clubs)]  # lowest club first
    y = in_club.map({name: j for j, name in enumerate(order)}).to_numpy(dtype=int)

    aligned = covariates.reindex(in_club.index)
    missing = aligned.index[aligned.isna().any(axis=1)]
    if len(missing):
        raise ValueError(f"covariates missing for {len(missing)} units, e.g. {list(missing[:5])}")
    X = aligned.to_numpy(dtype=float)
    if (X.std(axis=0) == 0).any():
        raise ValueError("drop constant columns: the thresholds act as the intercept")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("covariates are collinear")
    k, J = X.shape[1], len(order)

    shares = np.cumsum(np.bincount(y, minlength=J))[:-1] / len(y)
    kappa0 = _quantile(np.clip(shares, 1e-3, 1 - 1e-3), model)
    gaps = np.maximum(np.diff(kappa0), 1e-2)
    theta = np.concatenate([np.zeros(k), [kappa0[0]], np.log(gaps)])

    loglik, grad = _loglik_grad(theta, X, y, J, model)
    converged, iterations = False, 0
    while iterations < max_iter:
        iterations += 1
        H = _hessian(theta, X, y, J, model)
        try:
            direction = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            direction = grad
        step = 1.0
        while step > 1e-10:
            candidate = theta + step * direction
            new_loglik, new_grad = _loglik_grad(candidate, X, y, J, model)
            if np.isfinite(new_loglik) and new_loglik >= loglik - 1e-12:
                break
            step /= 2
        theta, improvement = candidate, new_loglik - loglik
        loglik, grad = new_loglik, new_grad
        if abs(improvement) < tol and np.max(np.abs(grad)) < 1e-6:
            converged = True
            break

    cov_theta = np.linalg.inv(-_hessian(theta, X, y, J, model))
    jac = np.zeros((k + J - 1, k + J - 1))
    jac[:k, :k] = np.eye(k)
    jac[k:, k:] = _thresholds_jacobian(theta, k)
    cov = jac @ cov_theta @ jac.T

    beta, se = theta[:k], np.sqrt(np.diag(cov)[:k])
    z = beta / se
    coefficients = pd.DataFrame(
        {
            "coef": beta,
            "std_err": se,
            "z_value": z,
            "p_value": [math.erfc(abs(v) / math.sqrt(2.0)) for v in z],
        },
        index=pd.Index([str(c) for c in aligned.columns], name="covariate"),
    )
    if model == "logit":
        coefficients["odds_ratio"] = np.exp(beta)
    counts = np.bincount(y, minlength=J)
    null = float((counts * np.log(counts / counts.sum())).sum())
    thresholds = pd.Series(
        _thresholds(theta, k),
        index=[f"{order[j]}|{order[j + 1]}" for j in range(J - 1)],
        name="threshold",
    )
    return DeterminantsResult(
        coefficients=coefficients,
        thresholds=thresholds,
        log_likelihood=loglik,
        null_log_likelihood=null,
        n=len(y),
        model=str(model),
        classes=tuple(order),
        converged=converged,
        iterations=iterations,
        excluded=tuple(members.index[members == "divergent"]),
    )
