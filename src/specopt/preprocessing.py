# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:04:22 2026

@author: sn1k3sh

Preprocessing steps. 

Each function operates on 2D X-array (samples x features)
    where each row is a spectrum and returns a preprocessed 2D X-array.
    
Steps applied in series will always follow order: 
    Truncation -> SG filter -> SNV 
"""
# imports ---------------------------------------------------------------------
import numpy as np
from scipy.signal import savgol_filter

__all__ = ["truncate", "savgol", "snv", "preprocess"]

# functions -------------------------------------------------------------------

def truncate(X: np.ndarray, start: int, end: int) -> np.ndarray:
    """Keep only the spectral columns in ``[start, end)``."""
    if start >= end:
        raise ValueError(f"start ({start}) must be < end ({end}) for truncation")
    return X[:, start:end]

def savgol(X: np.ndarray, window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    """
    Savitzky-Golay smoothing/derivative applied along the spectral axis.

    ``window_length`` is forced odd (the nearest larger odd integer) to satisfy
    SciPy's requirement
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length > X.shape[1]:
        raise ValueError(
            f"window_length ({window_length}) exceeds n_features ({X.shape[1]})"
        )
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be < window_length ({window_length})"
        )
    if deriv > polyorder:
        raise ValueError(
            f"deriv ({deriv}) must be <= polyorder ({polyorder})"
        )
    return savgol_filter(
        X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1
    )

def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: centre and scale each spectrum individually."""
    X = np.atleast_2d(X).astype(float)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0 # guard against 0 standard dev
    return (X - mean) / std

def preprocess(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Takes 2D X array and preprocessing instructions via params dict and applies
    them in series (truncation -> savgol -> snv)
    """
    X_ = np.array(X, dtype=float, copy=True)
    if params.get("truncate"):
        X_ = truncate(X_, int(params["start"]), int(params["end"]))
    if params.get("savgol"):
        X_ = savgol(
            X_,
            window_length=int(params["window"]),
            polyorder=int(params["polyorder"]),
            deriv=int(params["deriv"]),
        )
    if params.get("snv"):
        X_ = snv(X_)
    return X_
