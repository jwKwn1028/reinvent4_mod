"""
Exponential decay centered at m: exp(-k * |x - m|).
"""

__all__ = ["ExponentialDecay"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    k: float
    m: float

def expdecay(x, m, k=1.0):
    return np.exp(-k * np.abs(x-m))


class ExponentialDecay(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.k = params.k
        self.m = params.m
        if self.k <= 0:
            raise ValueError(f"ExponentialDecay Transform: k must be > 0, got {self.k}")

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)
        transformed = expdecay(values, self.m, self.k)
        return transformed
