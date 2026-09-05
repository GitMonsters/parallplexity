"""
Fractional Evolution Operator

Extends Purple Clay's EvolutionOperator with fractional-order dynamics:
    ^C D_t^α ψ = -iHψ - ig|ψ|²ψ

The Caputo derivative on the left integrates the entire evolution history
with a power-law kernel (t-τ)^(-α), giving the system memory.
"""

import numpy as np
from typing import Optional, List, Tuple
from .fractional import CaputoDerivative, HistoryBuffer, CANTOR_GOLDEN_DIM


class FractionalEvolutionOperator:
    """
    Fractional-order time evolution for lattice quantum systems.
    
    Extends integer-order evolution:
        ψ(t+dt) = ψ(t) - i·dt·H·ψ(t)                    [integer, α=1]
    to fractional order:
        ^C D_t^α ψ = -iHψ - ig|ψ|²ψ                      [fractional, α∈(0,2)]
    
    The fractional order controls compounding behavior:
        α < 1: sub-diffusive (memory slows spreading, deepens compounding)
        α = 1: normal diffusion (standard evolution)
        α > 1: super-diffusive (accelerated compounding, wave-like)
        α ≈ 1.44: Cantor-Golden dimension (natural TranscendPlexity order)
    """
    
    def __init__(
        self,
        alpha: float = CANTOR_GOLDEN_DIM,
        dt: float = 0.01,
        coupling_J: float = 0.1,
        max_history: int = 500
    ):
        self.alpha = alpha
        self.dt = dt
        self.coupling_J = coupling_J
        self.history = HistoryBuffer(max_length=max_history, dt=dt)
        self.caputo = CaputoDerivative(alpha=min(alpha, 1.99), dt=dt)
        self.t = 0.0
        self.step_count = 0
    
    def apply_fractional_evolution(
        self,
        state: np.ndarray,
        nonlinearity: float = 0.0
    ) -> np.ndarray:
        """
        Single fractional time step using the L1 scheme.
        
        For α ∈ (0, 1), this implements:
            ψ_{n+1} = ψ_n + dt^α / Γ(1+α) * [H_eff(ψ_n) + memory_correction]
        
        The memory correction integrates the weighted history of derivatives,
        which is what creates the compounding effect.
        """
        self.history.push(state, self.t)
        
        # Compute the effective Hamiltonian action
        h_eff = self._hamiltonian_action(state, nonlinearity)
        
        # Compute memory correction from fractional history
        memory_term = self._compute_memory_correction(state)
        
        # Fractional time step
        from scipy.special import gamma as gamma_func
        dt_alpha = self.dt ** self.alpha
        gamma_coeff = gamma_func(1.0 + self.alpha)
        
        # Evolution: ψ_{n+1} = ψ_n + (dt^α / Γ(1+α)) * (H_eff + memory)
        new_state = state + (dt_alpha / gamma_coeff) * (h_eff + memory_term)
        
        # Normalize to preserve probability
        norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
        if norm > 1e-12:
            new_state *= np.sqrt(np.sum(np.abs(state) ** 2)) / norm
        
        self.t += self.dt
        self.step_count += 1
        
        return new_state
    
    def _hamiltonian_action(
        self,
        state: np.ndarray,
        nonlinearity: float
    ) -> np.ndarray:
        """
        Compute H_eff(ψ) = -iJΣ_neighbors ψ - ig|ψ|²ψ
        
        The linear term is nearest-neighbor hopping (kinetic energy).
        The nonlinear term is Gross-Pitaevskii self-interaction (compounding).
        """
        h_state = np.zeros(state.shape, dtype=np.complex128)
        
        if state.ndim == 2:
            rows, cols = state.shape
            # Nearest-neighbor hopping with periodic boundary
            h_state += np.roll(state, 1, axis=0)
            h_state += np.roll(state, -1, axis=0)
            h_state += np.roll(state, 1, axis=1)
            h_state += np.roll(state, -1, axis=1)
            h_state -= 4 * state  # Laplacian
            h_state *= -1j * self.coupling_J
        elif state.ndim == 1:
            # 1D lattice
            h_state += np.roll(state, 1)
            h_state += np.roll(state, -1)
            h_state -= 2 * state
            h_state *= -1j * self.coupling_J
        
        # Gross-Pitaevskii nonlinearity: -ig|ψ|²ψ (the compounding term)
        if nonlinearity > 0:
            h_state += -1j * nonlinearity * np.abs(state) ** 2 * state
        
        return h_state
    
    def _compute_memory_correction(self, current_state: np.ndarray) -> np.ndarray:
        """
        Compute the fractional memory correction term.
        
        This is what makes the evolution fractional — the system remembers
        and compounds information from its entire history.
        
        For α < 1: strong memory, slow compounding (deep exploration)
        For α > 1: weak memory, fast compounding (rapid convergence)
        """
        if self.history.length < 3:
            return np.zeros_like(current_state)

        # Use the Caputo derivative of the state history, passing the raw
        # history lists directly (no per-step HistoryBuffer rebuild).
        caputo_deriv = self.caputo.compute_from_history(
            self.history.states, self.history.timestamps, self.t
        )

        # Reshape and apply as correction
        if caputo_deriv.shape[0] == current_state.size:
            correction = caputo_deriv.reshape(current_state.shape)
        else:
            correction = np.zeros_like(current_state)
        
        # Scale memory influence by (1 - α) — stronger for lower α
        memory_weight = max(0.0, 1.0 - (self.alpha % 1.0))
        
        return memory_weight * correction
    
    def apply_integer_evolution(
        self,
        state: np.ndarray,
        nonlinearity: float = 0.0
    ) -> np.ndarray:
        """
        Standard integer-order evolution for comparison:
            ψ(t+dt) = ψ(t) + dt * H_eff(ψ)
        
        No memory, no compounding from history.
        """
        h_eff = self._hamiltonian_action(state, nonlinearity)
        new_state = state + self.dt * h_eff
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
        if norm > 1e-12:
            new_state *= np.sqrt(np.sum(np.abs(state) ** 2)) / norm
        
        self.t += self.dt
        self.step_count += 1
        return new_state
    
    def reset(self):
        """Reset evolution state."""
        self.history.clear()
        self.t = 0.0
        self.step_count = 0
