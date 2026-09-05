"""
Compounding Fractional Parallplexity — Core Module
Fractional calculus, parallplexity tensors, and compounding dynamics.
"""
from .fractional import CaputoDerivative, HistoryBuffer
from .parallplexity import ParallplexityTensor, CompoundingTracker
from .evolution import FractionalEvolutionOperator

__all__ = [
    "CaputoDerivative",
    "HistoryBuffer",
    "ParallplexityTensor",
    "CompoundingTracker",
    "FractionalEvolutionOperator",
]
