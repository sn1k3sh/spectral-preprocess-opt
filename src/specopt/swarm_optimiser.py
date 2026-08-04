# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:05:05 2026

@author: sn1k3sh

Optimise the whole preprocessing pipeline at once with particle swarm.

Each candidate is a single real vector (see encoding.py). Preprocessing parameters
    are searched jointly rather than one step at a time. 

Infeasible candidates are given an infinite objective so the swarm learns to 
    avoid them.
"""

# imports ---------------------------------------------------------------------
from dataclasses import dataclass

import numpy as np
from pyswarm import pso

# local imports ---------------------------------------------------------------
from .encoding import SearchSpace, vector_to_params, is_feasible
from .objectives import make_objective
from .preprocessing import preprocess

__all__ = ["PreprocessResult", "optimise_preprocessing"]

# result ----------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """Best preprocessing pipeline found, and its leave-one-out RMSE."""

    params: dict
    rmse: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the discovered pipeline to new spectra."""
        return preprocess(X, self.params)

# optimiser -------------------------------------------------------------------
def optimise_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "pls",
    max_components: int = 15,
    min_width: int = 50,
    max_half_window: int = 40,
    max_polyorder: int = 9,
    max_deriv: int = 2,
    swarmsize: int = 10,
    maxiter: int = 40,
    omega: float = 0.7,
    phip: float = 1.49,
    phig: float = 1.49,
    random_state: int = None,
) -> PreprocessResult:
    """
    Search for the preprocessing pipeline that minimises leave-one-out RMSE.

    X, y            : spectra (n_samples, n_features) and targets (n_samples,)
    model           : regression model scored on each candidate ('pls'/'pcr'/'ridge')
    swarmsize, maxiter, omega, phip, phig : particle swarm settings
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    space = SearchSpace(
        n_features=X.shape[1],
        min_width=min_width,
        max_half_window=max_half_window,
        max_polyorder=max_polyorder,
        max_deriv=max_deriv,
    )
    score = make_objective(X, y, model=model, max_components=max_components)

    def objective(vec):
        params = vector_to_params(vec)
        if not is_feasible(params, space):
            return np.inf
        return score(params)

    if random_state is not None:
        np.random.seed(random_state)

    lb, ub = space.bounds()
    best_vec, best_rmse = pso(
        objective, lb, ub,
        swarmsize=swarmsize, maxiter=maxiter,
        omega=omega, phip=phip, phig=phig,
        minstep=1e-8, minfunc=1e-8,
    )
    return PreprocessResult(params=vector_to_params(best_vec), rmse=float(best_rmse))