# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:15:57 2026

@author: sn1k3sh

The objective function the preprocessing optimiser minimises is leave-one-out 
    validated RMSE of a regression model whose own hyperparameters are tuned 
    internally for each preprocessing candidate. 
"""
# imports ---------------------------------------------------------------------
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

# local imports ---------------------------------------------------------------
from .preprocessing import preprocess

__all__ = ["cv_rmse", "make_objective"]

# functions -------------------------------------------------------------------
def cv_rmse(X: np.ndarray, y: np.ndarray, model) -> float:
    """Leave-one-out cross-validated RMSE of ``model`` on ``(X, y)``."""
    loo = LeaveOneOut()
    squared_errors = []
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        squared_errors.append(mean_squared_error(y[test_idx], pred))
    return float(np.sqrt(np.mean(squared_errors)))

def _component_range(X: np.ndarray, max_components: int):
    """Component counts to try, capped by the sample and feature counts."""
    upper = min(max_components, X.shape[0] - 2, X.shape[1])
    return range(1, max(1, upper) + 1)

def _objective_pls(X, y, max_components):
    return min(
        cv_rmse(X, y, PLSRegression(n_components=k))
        for k in _component_range(X, max_components)
    )


def _objective_pcr(X, y, max_components):
    return min(
        cv_rmse(PCA(n_components=k).fit_transform(X), y, LinearRegression())
        for k in _component_range(X, max_components)
    )


def _objective_ridge(X, y, max_components):
    ridge = Ridge()

    def inner(log_alpha):
        ridge.set_params(alpha=10.0 ** log_alpha)
        return cv_rmse(X, y, ridge)

    return float(minimize_scalar(inner, bounds=(-10, 1), method="bounded").fun)

_MODELS = {"pls": _objective_pls, "pcr": _objective_pcr, "ridge": _objective_ridge}

def make_objective(X: np.ndarray, y: np.ndarray, model: str = "pls", max_components: int = 15):
    """
    Return ``f(params) -> loo_rmse`` scoring a preprocessing pipeline on ``(X, y)``.
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(_MODELS)}")
    fn = _MODELS[model]

    def objective(params: dict) -> float:
        Xp = preprocess(X, params)
        if Xp.shape[1] < 2:
            return np.inf
        return fn(Xp, y, max_components)

    return objective
