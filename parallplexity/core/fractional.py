"""
Fractional Calculus Core — Caputo Derivative and History Buffer

Implements the mathematical foundation for fractional-order dynamics:
  ^C D_t^α f(t) = (1/Γ(n-α)) ∫₀ᵗ f^(n)(τ)/(t-τ)^(α-n+1) dτ

The fractional order α controls memory depth:
  α → 1: classical derivative, memoryless
  α → 0: full history integration
  α ≈ 1.44 (log2/logφ): Cantor-Golden fractal dimension — natural order for TranscendPlexity
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from scipy.special import gamma as gamma_func


# === Constants ===
PHI = (1 + np.sqrt(5)) / 2                          # Golden ratio φ ≈ 1.618
CANTOR_GOLDEN_DIM = np.log(2) / np.log(PHI)         # ≈ 1.4404
PHI_SQUARED = PHI ** 2                                # ≈ 2.618 — GCI threshold


@dataclass
class HistoryBuffer:
    """
    Stores the temporal history of a state for fractional derivative computation.
    
    The buffer maintains past states weighted by the power-law kernel:
        w(t, τ) = (t - τ)^(-α)
    
    More recent states get higher weight, but all past states contribute —
    this is what gives fractional dynamics their "memory."
    """
    max_length: int = 1000
    states: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    dt: float = 0.01
    
    def push(self, state: np.ndarray, t: float) -> None:
        """Add a new state to the history."""
        self.states.append(state.copy())
        self.timestamps.append(t)
        # Trim if over capacity
        if len(self.states) > self.max_length:
            self.states.pop(0)
            self.timestamps.pop(0)
    
    def get_weighted_history(self, alpha: float, current_t: float) -> np.ndarray:
        """
        Compute the power-law weighted sum of all past states.
        
        Returns Σ_τ state(τ) * (t - τ)^(-α) * dt, normalized by Γ(1-α).
        This is the discrete approximation of the Caputo integral kernel.
        """
        if len(self.states) < 2:
            return np.zeros_like(self.states[0]) if self.states else np.zeros(1)
        
        # Stack states and weights, vectorized (no per-entry Python loop).
        arr = np.stack([np.asarray(s) for s in self.states], axis=0)
        times = np.asarray(self.timestamps)
        
        deltas = current_t - times
        mask = deltas > 1e-12  # avoid division by zero
        weights = np.zeros(deltas.shape, dtype=np.float64)
        if np.any(mask):
            weights[mask] = deltas[mask] ** (-alpha)
        
        weighted_sum = self.dt * (weights @ arr)
        weighted_sum = np.asarray(weighted_sum, dtype=np.complex128)
        
        # Normalize by Γ(1 - α) for fractional orders in (0, 1)
        alpha_frac = alpha % 1.0
        if alpha_frac > 1e-10:
            norm = gamma_func(1.0 - alpha_frac)
            if abs(norm) > 1e-12:
                weighted_sum /= norm
        
        return weighted_sum
    
    def clear(self) -> None:
        """Reset the history buffer."""
        self.states.clear()
        self.timestamps.clear()
    
    @property
    def length(self) -> int:
        return len(self.states)


class CaputoDerivative:
    """
    Computes the Caputo fractional derivative of a discrete time series.
    
    The Caputo derivative of order α ∈ (0, 2):
        ^C D_t^α f(t) = (1/Γ(n-α)) ∫₀ᵗ f^(n)(τ) / (t-τ)^(α-n+1) dτ
    
    where n = ⌈α⌉ (ceiling of α).
    
    For α ∈ (0,1): integrates the first derivative with kernel (t-τ)^(-α)
    For α ∈ (1,2): integrates the second derivative with kernel (t-τ)^(1-α)
    """
    
    def __init__(self, alpha: float = CANTOR_GOLDEN_DIM, dt: float = 0.01):
        self.alpha = alpha
        self.dt = dt
        self.n = int(np.ceil(alpha))  # ceiling
        self._validate_alpha()
    
    def _validate_alpha(self):
        if self.alpha <= 0 or self.alpha >= 2.0:
            raise ValueError(f"α must be in (0, 2), got {self.alpha}")
    
    @property
    def gamma_coefficient(self) -> float:
        """1 / Γ(n - α)"""
        return 1.0 / gamma_func(self.n - self.alpha)
    
    def compute(self, history: HistoryBuffer, current_t: float) -> np.ndarray:
        """
        Compute the Caputo fractional derivative at the current time.

        Uses the L1 discretization scheme:
            ^C D_t^α f(t_k) ≈ (1/Γ(n-α)) Σ_j [f^(n)(t_j)] * w_j

        where w_j are the quadrature weights for the kernel (t_k - τ)^(n-1-α).
        """
        if history.length < 3:
            return np.zeros_like(history.states[0]) if history.states else np.zeros(1)

        return self.compute_from_history(
            history.states, history.timestamps, current_t,
        )

    def compute_from_history(
        self,
        states: List[np.ndarray],
        timestamps: List[float],
        current_t: float,
    ) -> np.ndarray:
        """
        Vectorized Caputo derivative from raw history lists.

        Avoids rebuilding/flattening a `HistoryBuffer` on every call, so the
        per-step cost is a single stacked array + a vectorized kernel weight
        product (O(history × dim)) instead of a Python loop with per-entry
        Python overhead.
        """
        if len(states) < 3:
            return np.zeros_like(states[0]) if states else np.zeros(1)

        # Stack the (optionally multi-dimensional) states into (n_history, dim)
        flat = [np.asarray(s).ravel() for s in states]
        arr = np.stack(flat, axis=0)
        times = np.asarray(timestamps)

        # Compute the n-th order finite differences
        if self.n == 1:
            # First derivative: forward differences
            derivatives = np.diff(arr, axis=0) / self.dt
            deriv_times = times[:-1] + self.dt / 2
        else:
            # Second derivative: central differences
            derivatives = np.diff(arr, n=2, axis=0) / (self.dt ** 2)
            deriv_times = times[1:-1]

        if len(derivatives) == 0:
            return np.zeros_like(flat[0])

        # Apply the Caputo kernel weights, vectorized:
        #   result = dt / Γ(n-α) * Σ_j (t - τ_j)^(n-1-α) * derivative_j
        exponent = self.n - 1 - self.alpha  # (n-1-α)
        deltas = current_t - deriv_times
        mask = deltas > 1e-12

        result = np.zeros_like(flat[0], dtype=np.complex128)
        if np.any(mask):
            weights = deltas[mask] ** exponent
            result = self.dt * (weights @ derivatives[mask])

        result *= self.gamma_coefficient
        return result
    
    def compute_grünwald_letnikov(self, history: HistoryBuffer) -> np.ndarray:
        """
        Alternative: Grünwald-Letnikov fractional derivative.
        
        More numerically stable for some applications:
            D^α f(t) = lim_{h→0} (1/h^α) Σ_k (-1)^k C(α,k) f(t - kh)
        
        where C(α,k) = Γ(α+1) / (Γ(k+1) * Γ(α-k+1)) are generalized binomial coefficients.
        """
        if history.length < 2:
            return np.zeros_like(history.states[0]) if history.states else np.zeros(1)
        
        states = np.array(history.states)
        K = len(states)
        h = self.dt
        
        # Compute GL binomial coefficients
        coeffs = np.zeros(K)
        coeffs[0] = 1.0
        for k in range(1, K):
            coeffs[k] = coeffs[k-1] * (1.0 - (self.alpha + 1.0) / k)
        
        # Apply GL formula (sum from most recent backward)
        result = np.zeros_like(states[0], dtype=np.complex128)
        for k in range(K):
            sign = (-1) ** k
            result += sign * coeffs[k] * states[-(k+1)]
        
        result /= (h ** self.alpha)
        return result


def fractional_power_law_kernel(t: float, tau: float, alpha: float) -> float:
    """
    The power-law memory kernel: K(t, τ) = (t - τ)^(-α)
    
    This is the fundamental building block of fractional dynamics.
    Controls how much weight past events receive:
        α close to 0: all past events weighted nearly equally (deep memory)
        α close to 1: recent events dominate (shallow memory)
    """
    dt = t - tau
    if dt < 1e-12:
        return 0.0
    return dt ** (-alpha)


def mittag_leffler(z: np.ndarray, alpha: float, beta: float = 1.0,
                    n_terms: int = 50) -> np.ndarray:
    """
    Mittag-Leffler function E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)
    
    The natural generalization of the exponential function for fractional dynamics.
    E_{1,1}(z) = e^z (classical exponential)
    E_{α,1}(-t^α) governs fractional relaxation/decay
    
    This function determines the compounding profile:
    - Integer α: exponential compounding (standard)
    - Fractional α: stretched exponential / power-law compounding
    """
    result = np.zeros_like(z, dtype=np.complex128)
    z_power = np.ones_like(z, dtype=np.complex128)
    
    for k in range(n_terms):
        gamma_val = gamma_func(alpha * k + beta)
        if abs(gamma_val) > 1e-300:
            result += z_power / gamma_val
        z_power = z_power * z
        
        # Convergence check
        if np.max(np.abs(z_power / gamma_func(alpha * (k+1) + beta))) < 1e-15:
            break
    
    return result.real if np.isrealobj(z) else result
