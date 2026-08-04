# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:48:37 2026

@author: sn1k3sh

Population based optimisers search a continuous space. Thus, preprocessing 
    instructions have to be encoded as a vector. 
    
Vector layout (8 dimensions)::

    0  truncate flag        1  start index      2  end index
    3  savgol flag          4  half-window      5  polyorder       6  deriv
    7  snv flag
"""
# imports ---------------------------------------------------------------------
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = ["SearchSpace", "vector_to_params", "is_feasible"]

# SearchSpace class -----------------------------------------------------------
@dataclass
class SearchSpace:
    """Bounds for the encoded preprocessing vector, derived from the data."""

    n_features: int
    min_width: int = 50          # minimum retained width after truncation
    max_half_window: int = 40    # real window <= 2*40+1 = 81
    max_polyorder: int = 9
    max_deriv: int = 2
    
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Return ``(lower, upper)`` bound lists for the 8-D vector."""
        lb = [0, 0, 0, 0, 1, 1, 0, 0]
        ub = [
            1,                    # truncate flag
            self.n_features,      # start
            self.n_features,      # end
            1,                    # savgol flag
            self.max_half_window,
            self.max_polyorder,
            self.max_deriv,
            1,                    # snv flag
        ]
        return lb, ub
    
# functions -------------------------------------------------------------------   
def vector_to_params(vec) -> dict:
    """Decode a real vector into a preprocessing ``params`` dict."""
    v = np.asarray(vec, dtype=float)
    start, end = int(round(v[1])), int(round(v[2]))
    if start > end:
        start, end = end, start
    return {
        "truncate": bool(round(v[0])),
        "start": start,
        "end": end,
        "savgol": bool(round(v[3])),
        "window": int(round(v[4])) * 2 + 1,   # always odd
        "polyorder": int(round(v[5])),
        "deriv": int(round(v[6])),
        "snv": bool(round(v[7])),
    }

def is_feasible(params: dict, space: SearchSpace) -> bool:
    """Cheap structural constraints, checked before the expensive CV objective."""
    if params["truncate"] and (params["end"] - params["start"] < space.min_width):
        return False
    if params["savgol"]:
        avail = params["end"] - params["start"] if params["truncate"] else space.n_features
        if params["window"] > avail:          # window can't exceed the columns it runs on
            return False
        if params["polyorder"] >= params["window"]:
            return False
        if params["deriv"] > params["polyorder"]:
            return False
    return True
