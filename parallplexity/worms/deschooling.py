"""
Deschooling Engine — Illich Force Modifiers
=============================================

Maps Ivan Illich's "Deschooling Society" (1971) framework
onto the TranscendPlexity compound integration system.

Two opposing forces:

    CONVIVIAL (Learning Webs / Deschooled):
        - Coupling amplified (peer-matching, skill exchange)
        - GCI boosted (decentralized coherence, autonomy)
        - Limb alphas diverge toward specialization
        - Resilience: rolling-window recovery, CP floor, quadratic coupling
        - More time in Transcendplexity

    INSTITUTIONAL (Schooled):
        - Coupling suppressed (hidden curriculum, dependency)
        - GCI damped (credential gatekeeping, standardization)
        - Limb alphas flatten toward uniform (conformity)
        - Fragility: CP decay, rigid coupling floor, trivial collapse
        - More time in Collapseplexity

The Deschooling Engine wraps the Worms Engine, applying per-step
force modifiers to all metrics before they feed into the
parallplexity tensor and compounding tracker.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.fractional import PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM


class DeschoolingMode(Enum):
    """The three Illich modes."""
    BASELINE = "baseline"
    CONVIVIAL = "convivial"        # Learning Webs / Deschooled
    INSTITUTIONAL = "institutional"  # Schooled


@dataclass
class ResilienceState:
    """Tracks convivial resilience metrics over time."""
    positive_streak: int = 0
    negative_streak: int = 0
    rolling_window: int = 30
    recovery_strength: float = 0.0
    cp_floor: float = 0.0
    coupling_floor: float = 0.05

    def update(self, gci: float, t_frac: float):
        """Update resilience based on current GCI."""
        if gci > 0:
            self.positive_streak += 1
            self.negative_streak = 0
            # Recovery strength builds from positive streaks
            self.recovery_strength = min(
                0.7,
                self.positive_streak / self.rolling_window * 0.7
            )
            # CP floor rises over time (network effects)
            self.cp_floor = min(0.5, self.cp_floor + 0.005 * t_frac)
        else:
            self.negative_streak += 1
            self.positive_streak = max(0, self.positive_streak - 1)
            # Recovery still active from past positive streaks
            self.recovery_strength *= 0.95  # slow decay


# Limb-specific specialization biases for convivial mode
CONVIVIAL_LIMB_BIAS = {
    "Perception": 1.3,      # Heightened awareness
    "Memory": 0.8,          # Freed from rote memorization
    "Planning": 1.2,        # Strategic autonomy
    "Language": 1.1,        # Communication flourishes
    "Spatial": 1.0,         # Neutral
    "Reasoning": 1.0,       # Cantor-Golden stays fixed
    "MetaCognition": 1.5,   # Self-reflection deepens
    "Action": 1.4,          # Agency unleashed
}

# Illich phase mapping descriptions
ILLICH_PHASE_MAP = {
    "MYRIADPLEXITY": {
        "convivial": "Learning Webs forming — reference services + peer discovery",
        "institutional": "Credential collection — sorting into tracks",
    },
    "COMPOUNDING": {
        "convivial": "Skill exchanges compound — peer networks create multiplicative returns",
        "institutional": "Grade inflation — appearance of progress masks dependency",
    },
    "TRANSCENDPLEXITY": {
        "convivial": "Convivial emergence — resilient, self-directed, collapse-resistant",
        "institutional": "Impossible — gatekeeping prevents autonomous coherence",
    },
    "COLLAPSEPLEXITY": {
        "convivial": "Temporary setback — resilience enables rapid recovery",
        "institutional": "Hidden curriculum trap — dependency blocks recovery",
    },
}


class DeschoolingEngine:
    """
    The Deschooling Engine: Illich force modifiers for TranscendPlexity.

    Wraps the compound integration system with per-step modifiers
    that apply convivial (amplifying) or institutional (suppressing)
    forces to GCI, CP, coupling, limb alphas, and phase thresholds.

    Usage:
        engine = DeschoolingEngine(mode=DeschoolingMode.CONVIVIAL, intensity=0.7)
        modified_gci = engine.modify_gci(original_gci, step_idx, total_steps)
        modified_cp  = engine.modify_cp(original_cp, step_idx, total_steps)
        # etc.

    Or use as a confidence modifier for WormsEngine:
        worms.run(steps=500, deschooling_modifier=engine.confidence_modifier)
    """

    def __init__(
        self,
        mode: DeschoolingMode = DeschoolingMode.BASELINE,
        intensity: float = 0.5,
    ):
        self.mode = mode
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.resilience = ResilienceState()
        self._step = 0

    def set_mode(self, mode: DeschoolingMode, intensity: float = 0.5):
        """Switch modes. Resets resilience state."""
        self.mode = mode
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.resilience = ResilienceState()
        self._step = 0

    def _ramp(self, step_idx: int, total_steps: int) -> float:
        """Intensity ramp: starts gentle, increases over time."""
        t = step_idx / max(1, total_steps)
        return self.intensity * min(1.0, t * 3)

    # ─── GCI MODIFIER ──────────────────────────────────────

    def modify_gci(
        self, gci: float, step_idx: int, total_steps: int
    ) -> float:
        """Apply Illich force to GCI."""
        if self.mode == DeschoolingMode.BASELINE:
            return gci

        t = step_idx / max(1, total_steps)
        k = self._ramp(step_idx, total_steps)
        self.resilience.update(gci, t)

        if self.mode == DeschoolingMode.CONVIVIAL:
            return self._convivial_gci(gci, t, k)
        else:
            return self._institutional_gci(gci, t, k)

    def _convivial_gci(self, gci: float, t: float, k: float) -> float:
        """
        Convivial GCI: amplify positive, resilient recovery from negative.

        Learning webs create peer-matching resonance and emergent coherence.
        """
        boost = 1 + k * 1.2

        if gci < 0:
            # Resilient recovery: dampen negative, pull toward zero
            val = gci * (1 - self.resilience.recovery_strength)
            val += abs(gci) * k * 0.25  # upward bias from learning web momentum
        else:
            val = gci * boost

        # Peer-matching resonance: constructive interference
        resonance = np.sin(t * np.pi * 12) * k * 120 * t
        # Emergent coherence floor that rises over time
        floor = k * 50 * t * t

        return val + resonance + floor

    def _institutional_gci(self, gci: float, t: float, k: float) -> float:
        """
        Institutional GCI: suppress peaks, deepen collapses.

        Hidden curriculum blocks autonomous coherence.
        """
        suppress = 1 - k * 0.6
        deepen = gci * (1 + k * 0.8) if gci < 0 else gci * suppress

        # Institutional drag
        drag = -k * 150 * t * t
        # Credential crackdowns (periodic gatekeeping)
        crackdown = -abs(np.sin(t * np.pi * 5)) * k * 180 * t
        # Conformity damping: kills high-frequency innovation
        damping = -(gci - 200) * k * 0.5 if gci > 200 else 0

        return deepen + drag + crackdown + damping

    # ─── CP MODIFIER ────────────────────────────────────────

    def modify_cp(
        self, cp: float, step_idx: int, total_steps: int
    ) -> float:
        """Apply Illich force to Compounding Parallplexity."""
        if self.mode == DeschoolingMode.BASELINE:
            return cp

        t = step_idx / max(1, total_steps)
        k = self._ramp(step_idx, total_steps)

        if self.mode == DeschoolingMode.CONVIVIAL:
            # Skills exchange: learning compounds on learning
            compound = (1 + k * 1.5) ** (t * 4)
            floor = k * 0.5 * t  # minimum CP from network effects
            return max(cp * compound, floor)
        else:
            # Credential dependency: CP decays
            decay = (1 - k * 0.6) ** (t * 3)
            capped = min(cp * decay, 10 * (1 - k * 0.5))
            return max(0.0001, capped)

    # ─── COUPLING MODIFIER ──────────────────────────────────

    def modify_coupling(
        self, coupling: float, step_idx: int, total_steps: int
    ) -> float:
        """Apply Illich force to inter-limb coupling."""
        if self.mode == DeschoolingMode.BASELINE:
            return coupling

        t = step_idx / max(1, total_steps)
        k = self._ramp(step_idx, total_steps)

        if self.mode == DeschoolingMode.CONVIVIAL:
            # Peer-matching: coupling increases faster (quadratic)
            peer_boost = k * 0.3 * t * (1 + t)
            min_coupling = coupling + k * 0.05
            return max(coupling + peer_boost, min_coupling)
        else:
            # Institutional silos
            silo_effect = k * 0.25 * t * (1 + t)
            rigid_floor = 0.3 - k * 0.1
            return max(coupling - silo_effect, rigid_floor)

    # ─── LIMB ALPHA MODIFIER ────────────────────────────────

    def modify_limb_alpha(
        self,
        alpha: float,
        limb_name: str,
        step_idx: int,
        total_steps: int,
    ) -> float:
        """Apply Illich force to individual limb fractional orders."""
        if self.mode == DeschoolingMode.BASELINE:
            return alpha
        if limb_name == "Reasoning":
            return alpha  # Cantor-Golden stays fixed

        t = step_idx / max(1, total_steps)
        k = self._ramp(step_idx, total_steps)

        if self.mode == DeschoolingMode.CONVIVIAL:
            bias = CONVIVIAL_LIMB_BIAS.get(limb_name, 1.0)
            growth = k * 0.15 * t * bias

            if limb_name == "MetaCognition":
                # Adaptive oscillation + growth
                return alpha + k * 0.2 * t * np.sin(t * np.pi * 6) * 0.5 + k * 0.1 * t

            return alpha + growth
        else:
            # Standardization: force all toward 0.55 uniformity
            mean = 0.55
            force = k * 0.6 * t
            return alpha + (mean - alpha) * force

    # ─── PHASE THRESHOLD MODIFIER ───────────────────────────

    def modify_phase_thresholds(
        self, step_idx: int, total_steps: int
    ) -> Dict[str, float]:
        """
        Return modified phase thresholds based on Illich mode.

        Convivial: easier transcendplexity, harder collapse
        Institutional: near-unreachable transcendplexity, trivial collapse
        """
        base = {
            "gci_transcend": PHI_SQUARED,
            "gci_compound": 1.0,
            "gci_collapse": 0.0,
        }

        if self.mode == DeschoolingMode.BASELINE:
            return base

        if self.mode == DeschoolingMode.CONVIVIAL:
            return {
                "gci_transcend": PHI_SQUARED * 0.8,  # easier to reach
                "gci_compound": 0.5,
                "gci_collapse": -50.0,  # much harder to collapse
            }
        else:
            return {
                "gci_transcend": PHI_SQUARED * 1.5,  # almost unreachable
                "gci_compound": PHI_SQUARED * 2.0,
                "gci_collapse": 0.0,  # trivially collapses
            }

    # ─── WORMS CONFIDENCE MODIFIER ──────────────────────────

    def confidence_modifier(
        self,
        confidence: float,
        limb_name: str,
        step_idx: int,
        total_layers: int,
    ) -> float:
        """
        Modifier function compatible with WormsEngine.step().

        Adjusts per-layer confidence based on Illich mode.
        """
        if self.mode == DeschoolingMode.BASELINE:
            return confidence

        # Use a default total_steps estimate
        total_steps = 500
        t = step_idx / max(1, total_steps)
        k = self._ramp(step_idx, total_steps)

        if self.mode == DeschoolingMode.CONVIVIAL:
            bias = CONVIVIAL_LIMB_BIAS.get(limb_name, 1.0)
            boost = 1 + k * 0.3 * bias
            return confidence * boost
        else:
            # Institutional: suppress, flatten toward mediocrity
            suppress = 1 - k * 0.2
            return confidence * suppress

    # ─── ILLICH PHASE DESCRIPTION ───────────────────────────

    def get_phase_description(self, phase: str) -> str:
        """Get the Illich interpretation of the current phase."""
        if self.mode == DeschoolingMode.BASELINE:
            return phase

        mode_key = self.mode.value
        mapping = ILLICH_PHASE_MAP.get(phase, {})
        return mapping.get(mode_key, phase)

    # ─── SUMMARY ────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return current deschooling state."""
        return {
            "mode": self.mode.value,
            "intensity": self.intensity,
            "resilience": {
                "recovery_strength": round(self.resilience.recovery_strength, 4),
                "cp_floor": round(self.resilience.cp_floor, 4),
                "positive_streak": self.resilience.positive_streak,
                "negative_streak": self.resilience.negative_streak,
            },
            "thresholds": self.modify_phase_thresholds(self._step, 500),
            "limb_biases": CONVIVIAL_LIMB_BIAS,
        }
