from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.interpolate import splprep, splev

@dataclass
class Centreline:
    """
    Stores the reference line for one racetrack.
    It is stored as points along the line such that each point knows:
    - Where it is on the (x, y) space.
    - How far the line is from the start (s).
    - Which way the line is heading at that spot (tangent_x, tangent_y)
    """
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    kappa: np.ndarray # curvature
    length_m: float # total length of the line in metres
    circuit_name: str

    metadata: dict = field(default_factory=dict)

    def position(self, s_query: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """
        Given s (the distance along the line), return the (x, y) position.
        """
        s_q = np.asarray(s_query, dtype=float) % self.length_m
        x = np.interp(s_q, self.s, self.x)
        y = np.interp(s_q, self.s, self.y)

        return x, y
    
    def tangent(self, s_query: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """
        Given s (the distance along the line), return the tangent unit vector at that point.
        """
        s_q = np.asarray(s_query, dtype=float) % self.length_m
        tx = np.interp(s_q, self.s, self.tangent_x)
        ty = np.interp(s_q, self.s, self.tangent_y)
        norm = np.sqrt((tx * tx) + (ty * ty))

        # Defensive measure in case norm = 0 (should not happen but to absolutely be safe)
        norm = np.where(norm > 0, norm, 1.0) 

        return tx / norm, ty / norm
    
    def heading(self, s_query: np.ndarray | float) -> np.ndarray:
        