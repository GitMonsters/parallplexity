"""
MetaCognition Limb — Dynamic α Tuning
=======================================

Implements adaptive fractional-order tuning based on GCI error signal.

The MetaCognition limb monitors the system's distance from the
Transcendplexity threshold (φ² ≈ 2.618) and adjusts each layer's
fractional order α to steer the system toward coherent compounding.

When GCI is too low → decrease α (deeper memory, more integration)
When GCI is too high → increase α (lighter memory, avoid overshooting)

The tuning uses a tanh-scaled error signal with bounded step size
to prevent oscillation. Updates are applied every 8 steps after
a 20-step warmup period to allow the system to settle.
"""

import numpy as np
from typing import Dict, List

from ..core.fractional import PHI_SQUARED


class MetaCognitionLimb:
    """
    Dynamic α tuning for the Worms Engine layer stack.

    Monitors GCI trajectory and adjusts each layer's fractional order
    to steer toward the target (default: φ² ≈ 2.618).

    Usage:
        meta = MetaCognitionLimb(target_gci=PHI_SQUARED)
        # Inside simulation loop, every 8 steps after warmup:
        new_alphas = meta.tune_alphas(current_gci, current_alphas)
        engine.update_alphas(new_alphas)
    """

    def __init__(self, target_gci: float = PHI_SQUARED):
        self.target_gci = target_gci
        self.history_gci: List[float] = []
        self.history_alphas: List[Dict[str, float]] = []

    def tune_alphas(
        self, current_gci: float, current_alphas: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute updated α values based on GCI error.

        Returns the current alphas unchanged during the first 5 calls
        (warmup period). After warmup, applies tanh-scaled corrections.

        Args:
            current_gci: Most recent GCI value
            current_alphas: Dict mapping layer name → current α

        Returns:
            Dict mapping layer name → new α (clipped to [0.15, 1.85])
        """
        self.history_gci.append(current_gci)

        if len(self.history_gci) < 5:
            self.history_alphas.append(current_alphas.copy())
            return current_alphas

        # Error: positive means GCI is below target (need more compounding)
        recent_mean = float(np.mean(self.history_gci[-5:]))
        error = self.target_gci - recent_mean

        # Tanh scaling: strong correction when far, gentle when close
        # Positive error → GCI too low → decrease α (deeper memory)
        # Negative error → GCI too high → increase α (lighter memory)
        step_size = 0.05 * float(np.tanh(2.0 * error))

        new_alphas = {}
        for limb, alpha in current_alphas.items():
            # Direct: positive error → negative step → lower α
            new_alpha = float(np.clip(alpha - step_size, 0.15, 1.85))
            new_alphas[limb] = new_alpha

        self.history_alphas.append(new_alphas.copy())
        return new_alphas

    def get_alpha_trajectory(self) -> List[Dict[str, float]]:
        """Return the full history of α values for plotting."""
        return self.history_alphas
