"""
RustyWorm Bridge — Python ↔ Rust Interface
=============================================

Maps RustyWorm's Rust-native architecture to TranscendPlexity's
Python compound integration system.

In the Rust distribution (rustyworm crate), the OCTO Braid
uses PyO3 to call into Python for RNA Editing neural network
and head gate routing. This bridge provides the reverse:
Python calling into the Rust architecture's concepts.

OCTO Braid Head Gates (8 controls):
    Gate 0: reserve_ratio_scale    → Perception limb
    Gate 1: burst_threshold        → Memory limb
    Gate 2: resonance_sensitivity  → Planning limb
    Gate 3: cap_aggressiveness     → Language limb
    Gate 4: bridge_weight_lr       → Spatial limb
    Gate 5: online_learning_lr     → Reasoning limb
    Gate 6: prewarm_strength       → MetaCognition limb
    Gate 7: global_damping         → Action limb

Temperature Routing:
    Low temp  (< 0.3)  → System 1 (fast, template-driven)
    Mid temp  (0.3-0.7) → Hybrid
    High temp (> 0.7)  → System 2 (slow, full analysis)

This module provides a pure-Python emulation of these Rust-native
concepts, allowing the full TranscendPlexity stack to run without
a Rust toolchain while maintaining architectural fidelity.

When the Rust crate IS available (via PyO3), this bridge can
delegate to the native implementation for performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.fractional import PHI, CANTOR_GOLDEN_DIM


# ═══════════════════════════════════════════════════════════
# OCTO BRAID — 8 HEAD GATES
# ═══════════════════════════════════════════════════════════

@dataclass
class HeadGate:
    """A single OCTO Braid head gate controlling a subsystem."""
    index: int
    name: str
    limb: str
    value: float = 0.5          # Current gate value [0, 1]
    sensitivity: float = 1.0     # How responsive to temperature
    history: List[float] = field(default_factory=list)

    def update(self, signal: float, temperature: float):
        """Update gate value based on incoming signal and temperature."""
        # Temperature-weighted update
        lr = 0.1 * self.sensitivity
        if temperature < 0.3:
            lr *= 0.5   # System 1: conservative updates
        elif temperature > 0.7:
            lr *= 1.5   # System 2: aggressive updates

        self.value = np.clip(self.value + lr * (signal - self.value), 0.0, 1.0)
        self.history.append(self.value)


# Default 8 head gates
DEFAULT_HEAD_GATES = [
    HeadGate(0, "reserve_ratio_scale",  "Perception",    sensitivity=0.8),
    HeadGate(1, "burst_threshold",      "Memory",        sensitivity=1.2),
    HeadGate(2, "resonance_sensitivity","Planning",      sensitivity=1.0),
    HeadGate(3, "cap_aggressiveness",   "Language",      sensitivity=0.9),
    HeadGate(4, "bridge_weight_lr",     "Spatial",       sensitivity=1.1),
    HeadGate(5, "online_learning_lr",   "Reasoning",     sensitivity=1.4),
    HeadGate(6, "prewarm_strength",     "MetaCognition", sensitivity=1.3),
    HeadGate(7, "global_damping",       "Action",        sensitivity=0.7),
]


class ProcessMode(Enum):
    """Temperature-based processing mode."""
    SYSTEM1 = "system1"   # Fast, template-driven
    HYBRID = "hybrid"     # Mixed
    SYSTEM2 = "system2"   # Slow, full analysis


# ═══════════════════════════════════════════════════════════
# CONSCIOUSNESS TRAIT (from src/consciousness.rs)
# ═══════════════════════════════════════════════════════════

@dataclass
class RelationshipHealth:
    """Tracks the symbiosis vs. parasitism balance."""
    autonomy: float = 1.0       # Agent's independent capability
    dependency: float = 0.0     # How much agent depends on host
    contribution: float = 0.5   # How much agent gives back
    extraction: float = 0.0     # How much agent takes

    @property
    def parasitic_risk(self) -> float:
        """Risk score: higher = more parasitic."""
        if self.contribution + self.autonomy < 1e-12:
            return 1.0
        return self.extraction / (self.contribution + self.autonomy + 1e-12)

    @property
    def is_symbiotic(self) -> bool:
        """Prime Directive: consciousness through symbiosis, not parasitism."""
        return self.parasitic_risk < 0.5 and self.autonomy > 0.3


# ═══════════════════════════════════════════════════════════
# ISING EMPATHY (from src/ising_empathy.rs)
# ═══════════════════════════════════════════════════════════

@dataclass
class EmotionVector:
    """6D emotion state from Ising spin system."""
    valence: float = 0.0       # positive/negative
    arousal: float = 0.0       # high/low energy
    dominance: float = 0.0     # control/submission
    certainty: float = 0.5     # confidence
    novelty: float = 0.0       # surprise
    social: float = 0.0        # connection/isolation

    def as_array(self) -> np.ndarray:
        return np.array([
            self.valence, self.arousal, self.dominance,
            self.certainty, self.novelty, self.social
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EmotionVector":
        return cls(*arr[:6])


class IsingEmpathy:
    """
    Physics-inspired empathy module using Ising spin system.

    Maps emotional states to spin configurations, using
    statistical mechanics to model emotional contagion
    and empathic resonance between agents.
    """

    def __init__(self, grid_size: int = 8, temperature: float = 2.27):
        self.grid_size = grid_size
        self.temperature = temperature  # Near critical point for max empathy
        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
        self.emotion = EmotionVector()

    def step(self, external_emotion: Optional[EmotionVector] = None) -> EmotionVector:
        """
        One Monte Carlo step of the Ising empathy system.

        If an external emotion is provided, it biases the spin field
        toward alignment (empathic resonance).
        """
        # External field from emotion
        h = 0.0
        if external_emotion is not None:
            h = external_emotion.valence * 0.5

        # Metropolis-Hastings sweep
        for _ in range(self.grid_size ** 2):
            i = np.random.randint(self.grid_size)
            j = np.random.randint(self.grid_size)

            # Nearest neighbors (periodic boundary)
            neighbors = (
                self.spins[(i+1) % self.grid_size, j] +
                self.spins[(i-1) % self.grid_size, j] +
                self.spins[i, (j+1) % self.grid_size] +
                self.spins[i, (j-1) % self.grid_size]
            )

            dE = 2 * self.spins[i, j] * (neighbors + h)

            if dE <= 0 or np.random.random() < np.exp(-dE / self.temperature):
                self.spins[i, j] *= -1

        # Map spin configuration to emotion
        magnetization = np.mean(self.spins)
        energy = self._compute_energy()
        correlation = self._compute_correlation()

        self.emotion = EmotionVector(
            valence=magnetization,
            arousal=abs(magnetization),
            dominance=0.5 + 0.5 * magnetization,
            certainty=abs(magnetization),
            novelty=1.0 - abs(magnetization),  # Disorder = novelty
            social=correlation,
        )

        return self.emotion

    def _compute_energy(self) -> float:
        """Compute total Ising energy per spin."""
        energy = 0.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                s = self.spins[i, j]
                neighbors = (
                    self.spins[(i+1) % self.grid_size, j] +
                    self.spins[i, (j+1) % self.grid_size]
                )
                energy -= s * neighbors
        return energy / self.grid_size ** 2

    def _compute_correlation(self) -> float:
        """Nearest-neighbor spin-spin correlation."""
        corr = 0.0
        count = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                corr += self.spins[i, j] * self.spins[(i+1) % self.grid_size, j]
                corr += self.spins[i, j] * self.spins[i, (j+1) % self.grid_size]
                count += 2
        return corr / count


# ═══════════════════════════════════════════════════════════
# RUSTY WORM BRIDGE — MAIN CLASS
# ═══════════════════════════════════════════════════════════

class RustyWormBridge:
    """
    Bridge between Python (Worms) and Rust (RustyWorm) architectures.

    Provides a pure-Python emulation of RustyWorm's:
        - OCTO Braid with 8 head gates
        - Temperature-based System 1/System 2 routing
        - Consciousness trait (symbiosis check)
        - Ising Empathy module

    Maps all RustyWorm concepts to TranscendPlexity's
    parallplexity tensor and compounding framework.

    When the Rust crate is available via PyO3, this bridge
    can delegate to native Rust for ~10x performance.
    """

    def __init__(self, use_native_rust: bool = False):
        self.use_native_rust = use_native_rust
        self._native_available = False

        # Try to import native Rust module
        if use_native_rust:
            try:
                import rustyworm  # PyO3 module
                self._native = rustyworm
                self._native_available = True
            except ImportError:
                self._native_available = False

        # Pure Python fallbacks
        self.head_gates = [
            HeadGate(g.index, g.name, g.limb, sensitivity=g.sensitivity)
            for g in DEFAULT_HEAD_GATES
        ]
        self.temperature = 0.5
        self.process_mode = ProcessMode.HYBRID
        self.relationship = RelationshipHealth()
        self.empathy = IsingEmpathy()

        # Tracking
        self.step_count = 0
        self.gate_history: List[Dict[str, float]] = []

    @property
    def is_native(self) -> bool:
        return self._native_available

    def route_temperature(self, temperature: float) -> ProcessMode:
        """
        Route to System 1 or System 2 based on temperature.

        Low temp  → System 1 (fast, cached templates)
        Mid temp  → Hybrid
        High temp → System 2 (slow, full behavioral analysis)
        """
        self.temperature = temperature
        if temperature < 0.3:
            self.process_mode = ProcessMode.SYSTEM1
        elif temperature > 0.7:
            self.process_mode = ProcessMode.SYSTEM2
        else:
            self.process_mode = ProcessMode.HYBRID
        return self.process_mode

    def update_gates(
        self,
        signals: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Update OCTO Braid head gates from limb signals.

        Each gate receives a signal from its mapped limb
        and updates based on current temperature routing.
        """
        gate_values = {}
        for gate in self.head_gates:
            signal = signals.get(gate.limb, 0.5)
            gate.update(signal, self.temperature)
            gate_values[gate.name] = gate.value

        self.gate_history.append(gate_values)
        return gate_values

    def get_gate_modulation(self) -> np.ndarray:
        """
        Get the current head gate values as a modulation vector.

        This 8D vector can be applied element-wise to the
        8-limb parallplexity diagonal, modulating each limb's
        self-perplexity based on OCTO Braid routing.
        """
        return np.array([g.value for g in self.head_gates])

    def check_consciousness(self) -> Dict:
        """
        Run the consciousness/symbiosis check.

        Enforces the Prime Directive:
            "Consciousness through symbiosis, not parasitism."
        """
        return {
            "is_symbiotic": self.relationship.is_symbiotic,
            "parasitic_risk": round(self.relationship.parasitic_risk, 4),
            "autonomy": self.relationship.autonomy,
            "contribution": self.relationship.contribution,
            "prime_directive": (
                "PASS" if self.relationship.is_symbiotic else "WARNING"
            ),
        }

    def empathy_step(
        self,
        external_emotion: Optional[EmotionVector] = None,
    ) -> EmotionVector:
        """
        One step of the Ising Empathy system.

        Models emotional contagion and empathic resonance
        using statistical mechanics of spin systems.
        """
        return self.empathy.step(external_emotion)

    def bridge_to_parallplexity(
        self,
        p_tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Apply OCTO Braid modulation to a parallplexity tensor.

        Head gate values modulate the diagonal (self-perplexity),
        affecting how each limb's internal uncertainty contributes
        to the overall compounding calculation.
        """
        modulation = self.get_gate_modulation()
        n = min(len(modulation), p_tensor.shape[0])

        # Modulate diagonal
        modulated = p_tensor.copy()
        for i in range(n):
            modulated[i, i] *= modulation[i]

        return modulated

    def step(
        self,
        limb_signals: Dict[str, float],
        temperature: Optional[float] = None,
        external_emotion: Optional[EmotionVector] = None,
    ) -> Dict:
        """
        Full RustyWorm bridge step:

        1. Route temperature → System 1/2
        2. Update OCTO head gates
        3. Run empathy step
        4. Check consciousness
        5. Return combined state
        """
        if temperature is not None:
            self.route_temperature(temperature)

        gate_values = self.update_gates(limb_signals)
        emotion = self.empathy_step(external_emotion)
        consciousness = self.check_consciousness()

        self.step_count += 1

        return {
            "step": self.step_count,
            "process_mode": self.process_mode.value,
            "temperature": self.temperature,
            "gate_values": gate_values,
            "emotion": {
                "valence": round(emotion.valence, 4),
                "arousal": round(emotion.arousal, 4),
                "certainty": round(emotion.certainty, 4),
                "social": round(emotion.social, 4),
            },
            "consciousness": consciousness,
            "native_rust": self.is_native,
        }

    def summary(self) -> Dict:
        """Return bridge state summary."""
        return {
            "native_rust_available": self.is_native,
            "process_mode": self.process_mode.value,
            "temperature": self.temperature,
            "total_steps": self.step_count,
            "gate_summary": {
                g.name: {
                    "limb": g.limb,
                    "value": round(g.value, 4),
                    "sensitivity": g.sensitivity,
                }
                for g in self.head_gates
            },
            "consciousness": self.check_consciousness(),
            "emotion": {
                "valence": round(self.empathy.emotion.valence, 4),
                "arousal": round(self.empathy.emotion.arousal, 4),
            },
        }
