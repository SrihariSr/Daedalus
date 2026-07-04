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
        Given `s` (the distance along the line), return the (x, y) position.
        """
        s_q = np.asarray(s_query, dtype=float) % self.length_m
        x = np.interp(s_q, self.s, self.x)
        y = np.interp(s_q, self.s, self.y)

        return x, y
    
    def tangent(self, s_query: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """
        Given `s` (the distance along the line), return the tangent unit vector at that point.
        """
        s_q = np.asarray(s_query, dtype=float) % self.length_m
        tx = np.interp(s_q, self.s, self.tangent_x)
        ty = np.interp(s_q, self.s, self.tangent_y)
        norm = np.sqrt((tx * tx) + (ty * ty))

        # Defensive measure in case norm = 0 (should not happen but to absolutely be safe)
        norm = np.where(norm > 0, norm, 1.0) 

        return tx / norm, ty / norm
    
    def heading(self, s_query: np.ndarray | float) -> np.ndarray:
        """
        Given `s` (the distance along the line), return the direction the line is heading
        as an angle in radians.
        """
        tx, ty = self.tangent(s_query)
        return np.arctan2(ty, tx)
    
    def curvature(self, s_query: np.ndarray | float) -> np.ndarray:
        """
        Given `s` (the distance along the line), return the curvature at that point in 1/metres.
        """
        s_q = np.asarray(s_query, dtype=float) % self.length_m
        
        return np.interp(s_q, self.s, self.kappa)

    def convert_xy_to_sn(self,
    x_query: np.ndarray | float,
    y_query: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a car's position as (x, y), return (s, n) where:
        `s` is how far along the centreline the car is.
        `n` is how far to the side from the centreline the car is.
        """

        # -------- Finding closest point --------

        # Ensuring further computation is done on arrays, not single numbers
        xq = np.atleast_1d(np.asarray(x_query, dtype=float))
        yq = np.atleast_1d(np.asarray(y_query, dtype=float))

        # Difference between point and centreline
        dx = xq[:, None] - self.x[None, :]
        dy = yq[:, None] - self.y[None, :]

        # Finding squared distance to get the smallest unsigned distance.
        # Sum of squares is cheaper to compute than the true distance using square root.
        d2 = (dx * dx) + (dy * dy)
        nearest = np.argmin(d2, axis=1)

        # -------- Refining with tangent direction --------
        
        tx = self.tangent_x[nearest]
        ty = self.tangent_y[nearest]

        # Vector from nearest point to query point
        vx = xq - self.x[nearest]
        vy = yq - self.y[nearest]

        # Calculating dot product
        # if positive, query has passed the nearest stored point
        # Negative otherwise
        dt = (vx * tx) + (vy * ty)

        # The component perpendicular to the tangent
        n = (-vx * ty) + (vy * tx)

        # Using modulo for the edge case s goes past the length of the track
        s_refined = (self.s[nearest] + dt) % self.length_m

        return s_refined, n
        


