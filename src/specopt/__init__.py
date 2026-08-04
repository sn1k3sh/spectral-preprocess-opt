# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:29:23 2026

@author: sn1k3sh

specopt - joint optimisation of spectral preprocessing and model complexity.

Input: training spectra and regression model; it returns the preprocessing pipeline
(truncation / Savitzky-Golay / SNV) that minimises leave-one-out CV error.

    from specopt import optimise_preprocessing
    result = optimise_preprocessing(X_train, y_train, model="pls")
    result.params            # best preprocessing conditions
    X_train_p = result.transform(X_train)
"""

from .swarm_optimiser import PreprocessResult, optimise_preprocessing

__version__ = "0.1.0"

__all__ = [
    "optimise_preprocessing",
    "PreprocessResult",
]
