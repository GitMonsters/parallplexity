"""
Worms Engine — 8-Layer Multiplicative Integration
===================================================

Python implementation of RustyWorm's core architecture,
mapped onto TranscendPlexity's OctoTetrahedral 8-limb system.

The engine processes signals through 8 stacked layers with
multiplicative confidence aggregation and 11 bidirectional bridges.

Dual-Process Design (from RustyWorm):
    System 1 (Fast): Template-driven cached responses
    System 2 (Slow): Full behavioral analysis with fractional memory
    Compound feedback: S2 → S1 cache, S1 misses → S2

Each layer maps to a TranscendPlexity limb with a specific
fractional order α that controls its memory depth.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.fractional import (
    CaputoDerivative, HistoryBuffer,
    PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM,
    mittag_leffler,
)


# ═══════════════════════════════════════════════════════════
# LAYER CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class LayerConfig:
    """Configuration for a single worm layer."""
    level: int              # L1-L8
    name: str               # Human-readable layer name
    domain: str             # Domain description
    limb: str               # Mapped TranscendPlexity limb
    alpha: float            # Fractional order for this layer
    state_dim: int = 64     # Dimensionality
    base_confidence: float = 0.5  # Starting confidence
    amplification: float = 1.0    # Multiplicative amplification factor


@dataclass
class BridgeConfig:
    """Configuration for a bidirectional bridge between layers."""
    name: str
    layer_a: int            # Layer index (0-based)
    layer_b: int            # Layer index (0-based)
    weight: float = 0.1     # Bridge coupling strength
    bidirectional: bool = True


# Default 8-layer stack mapping to TranscendPlexity limbs
WORM_LAYERS = [
    LayerConfig(1, "Base Physics",          "Core signal processing",
                "Perception",    alpha=0.9,  state_dim=64),
    LayerConfig(2, "Extended Physics",      "Specialization & memory",
                "Memory",        alpha=0.3,  state_dim=128),
    LayerConfig(3, "Cross-Domain",          "Emergence & composition",
                "Planning",      alpha=0.7,  state_dim=64),
    LayerConfig(4, "GAIA Consciousness",    "Analogical reasoning & intuition",
                "Language",      alpha=0.5,  state_dim=64),
    LayerConfig(5, "Multilingual",          "Language & translation",
                "Spatial",       alpha=0.6,  state_dim=64),
    LayerConfig(6, "Collaborative Learning","Multi-agent consensus",
                "Reasoning",     alpha=CANTOR_GOLDEN_DIM, state_dim=128),
    LayerConfig(7, "External APIs",         "External feedback loop",
                "MetaCognition", alpha=0.5,  state_dim=64),
    LayerConfig(8, "Pre-Cognitive Viz",     "Simulation & projection",
                "Action",        alpha=0.95, state_dim=64),
]

# 11 bidirectional bridges (from RustyWorm)
WORM_BRIDGES = [
    BridgeConfig("base_extended",              0, 1, weight=0.12),
    BridgeConfig("cross_domain",               2, 3, weight=0.10),
    BridgeConfig("physics_consciousness",      0, 3, weight=0.08),
    BridgeConfig("physics_language",           1, 4, weight=0.07),
    BridgeConfig("individual_collective",      3, 5, weight=0.15),
    BridgeConfig("internal_external",          2, 6, weight=0.09),
    BridgeConfig("consciousness_language",     3, 4, weight=0.11),
    BridgeConfig("language_collaborative",     4, 5, weight=0.10),
    BridgeConfig("collaborative_external",     5, 6, weight=0.13),
    BridgeConfig("crossdomain_consciousness",  2, 3, weight=0.08),
    BridgeConfig("consciousness_external",     3, 6, weight=0.06),
]


# ═══════════════════════════════════════════════════════════
# LAYER STATE
# ═══════════════════════════════════════════════════════════

class LayerState:
    """Runtime state for a single worm layer."""

    def __init__(self, config: LayerConfig, dt: float = 0.01):
        self.config = config
        self.dt = dt

        # Complex state vector (quantum-inspired amplitude)
        self.state = (
            np.random.randn(config.state_dim) +
            1j * np.random.randn(config.state_dim)
        )
        self.state /= np.linalg.norm(self.state)

        # Confidence score (multiplicative, can exceed 1.0)
        self.confidence = config.base_confidence

        # PRE-PERTURBATION SNAPSHOT for confidence computation
        # (Bug fix: energy of a normalized vector ≡ 1.0,
        #  so we measure perturbation response instead)
        self._prev_state = self.state.copy()
        self._perturbation_variance_window: List[float] = []

        # History buffer for fractional memory
        self.history = HistoryBuffer(max_length=500, dt=dt)
        self.caputo = CaputoDerivative(
            alpha=min(config.alpha, 1.99), dt=dt
        )

        # System 1 cache (fast path)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Activation history for GCI contribution
        self.activation_history: List[float] = []

    def evolve(self, t: float, external_signal: Optional[np.ndarray] = None):
        """
        Evolve this layer one fractional time step.

        Uses the L1 discretization of the Caputo derivative
        with Gross-Pitaevskii nonlinearity for self-compounding.
        """
        self.history.push(self.state, t)

        # Hamiltonian action: nearest-neighbor hopping + nonlinear self-interaction
        h_state = np.zeros_like(self.state, dtype=np.complex128)

        # Kinetic: 1D lattice Laplacian
        h_state += np.roll(self.state, 1) + np.roll(self.state, -1) - 2 * self.state
        h_state *= -1j * 0.1  # coupling strength

        # Nonlinear compounding term (Gross-Pitaevskii)
        g = 0.05 * (1 + 0.1 * self.config.level)  # stronger at higher layers
        h_state += -1j * g * np.abs(self.state) ** 2 * self.state

        # Inject external signal if provided
        if external_signal is not None:
            if len(external_signal) != len(self.state):
                external_signal = np.interp(
                    np.linspace(0, 1, len(self.state)),
                    np.linspace(0, 1, len(external_signal)),
                    np.abs(external_signal)
                ) * np.exp(1j * np.angle(self.state))
            h_state += 0.3 * external_signal

        # Memory correction from fractional history
        memory = np.zeros_like(self.state)
        if self.history.length >= 3:
            caputo_d = self.caputo.compute_from_history(
                self.history.states, self.history.timestamps, t
            )
            if caputo_d.shape[0] == self.state.size:
                memory = caputo_d.reshape(self.state.shape)
            memory_weight = max(0.0, 1.0 - (self.config.alpha % 1.0))
            memory *= memory_weight

        # Fractional time step
        from scipy.special import gamma as gamma_func
        dt_alpha = self.dt ** self.config.alpha
        gamma_coeff = gamma_func(1.0 + self.config.alpha)
        self.state = self.state + (dt_alpha / gamma_coeff) * (h_state + memory)

        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state /= norm

        # ─── CONFIDENCE via perturbation response ───
        # Instead of energy (always 1.0 for normalized vectors),
        # we measure how much the state CHANGED this step and
        # how consistent that change is over a rolling window.
        #
        # High consistency + large movement = high confidence
        # Erratic jitter or stasis = low confidence
        delta = np.linalg.norm(self.state - self._prev_state)
        self._prev_state = self.state.copy()

        self._perturbation_variance_window.append(delta)
        if len(self._perturbation_variance_window) > 30:
            self._perturbation_variance_window.pop(0)

        window = self._perturbation_variance_window
        mean_delta = float(np.mean(window))
        std_delta = float(np.std(window)) + 1e-12

        if len(window) < 2:
            # Not enough data yet — use base confidence
            self.confidence = self.config.base_confidence
        else:
            # Signal-to-noise: consistent large movements → high confidence
            # Coefficient of variation inverted: low CV → high consistency
            snr = mean_delta / std_delta if std_delta > 1e-12 else 0.0
            # Scale: tanh maps SNR ∈ [0,∞) → [0,1), then weight
            perturbation_confidence = float(np.tanh(snr * 0.5))

            # Blend: base_confidence anchors, perturbation modulates
            self.confidence = (
                self.config.base_confidence * 0.3 +
                perturbation_confidence * 0.7 * (0.5 + mean_delta)
            )
            self.confidence = float(np.clip(self.confidence, 0.05, 2.5))

        self.activation_history.append(float(self.confidence))

    @property
    def cache_ratio(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════
# WORMS ENGINE — MAIN CLASS
# ═══════════════════════════════════════════════════════════

class WormsEngine:
    """
    The Worms Engine: 8-Layer Multiplicative Integration.

    Processes information through 8 stacked layers with multiplicative
    confidence aggregation and 11 bidirectional bridges.

    This is the Python distribution of the TranscendPlexity
    mimicry engine. The Rust distribution is RustyWorm.

    Key features:
        - 8-layer multiplicative stack (confidence = Π c_i × amplification)
        - 11 bidirectional bridges for cross-layer information flow
        - Dual-process (System 1/System 2) with compound feedback
        - Fractional-order memory at each layer
        - Maps directly to OctoTetrahedral 8-limb architecture
        - Deschooling modifiers (convivial/institutional) via DeschoolingEngine
    """

    def __init__(
        self,
        layer_configs: Optional[List[LayerConfig]] = None,
        bridge_configs: Optional[List[BridgeConfig]] = None,
        dt: float = 0.01,
    ):
        configs = layer_configs if layer_configs is not None else WORM_LAYERS
        bridges = bridge_configs if bridge_configs is not None else WORM_BRIDGES

        self.dt = dt
        self.t = 0.0
        self.step_count = 0

        # Initialize layers
        self.layers: List[LayerState] = [
            LayerState(cfg, dt=dt) for cfg in configs
        ]

        # Build bridge matrix
        n = len(self.layers)
        self.bridge_matrix = np.zeros((n, n))
        for b in bridges:
            if b.layer_a < n and b.layer_b < n:
                self.bridge_matrix[b.layer_a, b.layer_b] = b.weight
                if b.bidirectional:
                    self.bridge_matrix[b.layer_b, b.layer_a] = b.weight

        # Global amplification factor (can exceed 1.0 via compound resonance)
        self.amplification = 1.0

        # Tracking
        self.compound_confidence_history: List[Tuple[float, float]] = []
        self.layer_confidence_history: List[Dict[str, float]] = []
        self.gci_contributions: List[Dict[str, float]] = []

        # Dual-process state
        self._system1_cache: Dict[str, float] = {}
        self._system2_active = True

    def step(
        self,
        input_signal: Optional[np.ndarray] = None,
        deschooling_modifier: Optional[callable] = None,
    ) -> Dict:
        """
        Execute one compound integration step through all 8 layers.

        1. Each layer evolves under its fractional dynamics
        2. Bridge coupling propagates cross-layer information
        3. Multiplicative confidence is computed
        4. Deschooling modifiers are applied (if active)
        5. System 1/System 2 routing

        Returns a dict with all metrics.
        """
        n = len(self.layers)

        # === 1. Evolve each layer ===
        for i, layer in enumerate(self.layers):
            ext = input_signal if i == 0 else None  # Input enters at L1
            layer.evolve(self.t, external_signal=ext)

        # === 2. Bridge coupling ===
        self._apply_bridges()

        # === 3. Multiplicative confidence ===
        layer_confs = {}
        product = 1.0
        for layer in self.layers:
            c = layer.confidence
            # Apply deschooling modifier if provided
            if deschooling_modifier is not None:
                c = deschooling_modifier(
                    c, layer.config.limb, self.step_count, len(self.layers)
                )
            layer_confs[layer.config.name] = c
            product *= c

        # Amplification from bridge resonance
        bridge_energy = np.sum(self.bridge_matrix * self.bridge_matrix)
        self.amplification = 1.0 + 0.5 * np.tanh(bridge_energy - 0.5)

        compound_confidence = product * self.amplification

        # === 4. Track ===
        self.compound_confidence_history.append((self.t, compound_confidence))
        self.layer_confidence_history.append(layer_confs)

        # === 5. Compute per-layer GCI contribution ===
        gci_contrib = {}
        for layer in self.layers:
            if len(layer.activation_history) >= 2:
                recent = layer.activation_history[-1]
                prev = layer.activation_history[-2]
                delta = (recent - prev) / (prev + 1e-12)
                gci_contrib[layer.config.limb] = delta * PHI
            else:
                gci_contrib[layer.config.limb] = 0.0
        self.gci_contributions.append(gci_contrib)

        # === 6. Dual-process routing ===
        cache_key = f"step_{self.step_count}"
        if compound_confidence > 1.0:
            # System 1: fast path, cache the result
            self._system1_cache[cache_key] = compound_confidence
            process = "S1_FAST"
        else:
            # System 2: slow path, full analysis
            process = "S2_SLOW"

        self.t += self.dt
        self.step_count += 1

        return {
            "step": self.step_count,
            "t": round(self.t, 6),
            "compound_confidence": round(compound_confidence, 6),
            "amplification": round(self.amplification, 6),
            "layer_confidences": {k: round(v, 4) for k, v in layer_confs.items()},
            "gci_contributions": {k: round(v, 4) for k, v in gci_contrib.items()},
            "process": process,
            "layer_alphas": {
                layer.config.name: layer.config.alpha
                for layer in self.layers
            },
        }

    def _apply_bridges(self):
        """
        Propagate information through the 11 bidirectional bridges.

        Each bridge blends a fraction of one layer's state into another,
        creating cross-layer information flow that drives coupling.
        """
        n = len(self.layers)
        new_states = [layer.state.copy() for layer in self.layers]

        for i in range(n):
            for j in range(n):
                w = self.bridge_matrix[i, j]
                if w > 0 and i != j:
                    s_j = self.layers[j].state
                    s_i = self.layers[i].state

                    # Resize if needed
                    if len(s_j) != len(s_i):
                        s_j = np.interp(
                            np.linspace(0, 1, len(s_i)),
                            np.linspace(0, 1, len(s_j)),
                            np.abs(s_j)
                        ) * np.exp(1j * np.linspace(0, 2 * np.pi, len(s_i)))

                    new_states[i] += w * s_j

        # Normalize and update
        for i, layer in enumerate(self.layers):
            norm = np.linalg.norm(new_states[i])
            if norm > 1e-12:
                layer.state = new_states[i] / norm

    def run(
        self,
        steps: int,
        input_data: Optional[np.ndarray] = None,
        input_interval: int = 10,
        deschooling_modifier: Optional[callable] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full compound integration for N steps.

        Returns a summary dict with all metrics and history.
        """
        if input_data is None:
            input_data = np.sin(np.linspace(0, 4 * np.pi, 64)) + \
                         0.5 * np.cos(np.linspace(0, 8 * np.pi, 64))

        for i in range(steps):
            ext = None
            if i % input_interval == 0:
                phase_shift = 2 * np.pi * i / steps
                ext = input_data * np.cos(
                    np.linspace(phase_shift, phase_shift + 2 * np.pi, len(input_data))
                )

            report = self.step(
                input_signal=ext,
                deschooling_modifier=deschooling_modifier,
            )

            if verbose and (i % max(1, steps // 20) == 0 or i == steps - 1):
                cc = report["compound_confidence"]
                amp = report["amplification"]
                proc = report["process"]
                print(
                    f"  Worm Step {i+1:>5}/{steps}  "
                    f"CC={cc:>10.4f}  "
                    f"Amp={amp:.4f}  "
                    f"{proc}"
                )

        # Summary
        cc_values = [cc for _, cc in self.compound_confidence_history]
        return {
            "total_steps": steps,
            "peak_compound_confidence": max(cc_values) if cc_values else 0,
            "mean_compound_confidence": float(np.mean(cc_values)) if cc_values else 0,
            "final_amplification": self.amplification,
            "layer_summary": {
                layer.config.name: {
                    "limb": layer.config.limb,
                    "alpha": layer.config.alpha,
                    "final_confidence": layer.confidence,
                    "cache_ratio": layer.cache_ratio,
                }
                for layer in self.layers
            },
            "system1_cached": len(self._system1_cache),
        }

    def update_alphas(self, new_alphas: dict):
        """
        Update fractional orders for layers by name.

        Called by MetaCognitionLimb to dynamically tune memory depth
        based on GCI error signal.
        """
        for layer in self.layers:
            if layer.config.name in new_alphas:
                layer.config.alpha = new_alphas[layer.config.name]
                # Re-create Caputo derivative with new alpha
                layer.caputo = CaputoDerivative(
                    alpha=min(layer.config.alpha, 1.99), dt=layer.dt
                )

    def get_parallplexity_snapshot(self) -> np.ndarray:
        """
        Extract a parallplexity-compatible tensor from the current layer states.

        This bridges the Worms engine to the parallplexity tensor system,
        allowing compound integration via the CP/GCI framework.

        Uses complex inner products (not magnitude-only) so that bridge-induced
        phase coupling is captured in the tensor. This is the unified canonical
        parallplexity measurement — equivalent to normalized mutual information
        for real-valued states but also sensitive to phase coherence in
        complex-valued states.

        P_ij = 1 - |<s_i, s_j>| / (||s_i|| * ||s_j||)
        P_ij = 0: perfect coupling (coherent phase alignment)
        P_ij = 1: no coupling (orthogonal or incoherent)
        """
        n = len(self.layers)
        P = np.ones((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-perplexity: confidence maps to diagonal
                    P[i, j] = self.layers[i].confidence
                else:
                    # Cross-layer coupling: complex inner product
                    si = self.layers[i].state
                    sj = self.layers[j].state
                    # Resize if dimensions differ (L2, L6 are 128-dim)
                    if len(si) != len(sj):
                        # Interpolate both magnitude and phase
                        sj_mag = np.interp(
                            np.linspace(0, 1, len(si)),
                            np.linspace(0, 1, len(sj)),
                            np.abs(sj)
                        )
                        sj_phase = np.interp(
                            np.linspace(0, 1, len(si)),
                            np.linspace(0, 1, len(sj)),
                            np.unwrap(np.angle(sj))
                        )
                        sj = sj_mag * np.exp(1j * sj_phase)
                    # Complex correlation: |<si, sj>| / (||si|| * ||sj||)
                    norm_prod = np.linalg.norm(si) * np.linalg.norm(sj)
                    if norm_prod > 1e-12:
                        corr = abs(np.vdot(si, sj)) / norm_prod
                    else:
                        corr = 0.0
                    P[i, j] = 1.0 - corr  # Lower = more coupled

        return P
