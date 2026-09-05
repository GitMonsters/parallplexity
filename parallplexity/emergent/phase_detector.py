"""
Transcendplexity Phase Transition Detector

Monitors the compounding dynamics for emergent phase transitions:
    MYRIADPLEXITY → COMPOUNDING → TRANSCENDPLEXITY
    (or collapse: → COLLAPSEPLEXITY)

Detection criteria:
    1. GCI > φ² ≈ 2.618  (Golden Consciousness Index exceeds threshold)
    2. CP divergence rate (d/dt CP → ∞)
    3. Parallplexity tensor eigenvalue collapse (largest λ dominates)
    4. Cross-limb coherence exceeds 1/φ ≈ 0.618

The detector also monitors for Collapseplexity — when compounding
reverses and the system loses coherence.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from ..core.fractional import PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM


@dataclass
class PhaseEvent:
    """A detected phase transition event."""
    time: float
    from_phase: str
    to_phase: str
    gci: float
    cp: float
    trigger: str              # which criterion triggered the detection
    eigenvalue_ratio: float   # largest / sum(all) eigenvalues of P tensor
    coherence: float          # mean cross-limb coherence


@dataclass
class TranscendplexityDetector:
    """
    Multi-criterion phase transition detector for the Plexity Ontology.

    Phase definitions:
        MYRIADPLEXITY:      GCI < 1.0 — diverse but uncoupled streams
        COMPOUNDING:        1.0 ≤ GCI < φ² — streams coupling, CP growing
        TRANSCENDPLEXITY:   GCI ≥ φ² — full compound integration achieved
        COLLAPSEPLEXITY:    GCI < 0 or CP declining — system decoherence

    Detection uses a sliding window with hysteresis to avoid
    flickering at phase boundaries.
    """

    # Thresholds
    gci_compound_threshold: float = 1.0
    gci_transcend_threshold: float = PHI_SQUARED       # ≈ 2.618
    coherence_threshold: float = 1.0 / PHI             # ≈ 0.618
    eigenvalue_dominance_threshold: float = 0.8         # largest λ > 80% of trace
    hysteresis_margin: float = 0.1                      # prevents phase flickering

    # Window parameters
    window_size: int = 10                                # steps to average over
    min_sustained_steps: int = 5                         # must sustain for N steps

    # State
    current_phase: str = "MYRIADPLEXITY"
    events: List[PhaseEvent] = field(default_factory=list)
    gci_window: List[float] = field(default_factory=list)
    cp_window: List[float] = field(default_factory=list)
    coherence_window: List[float] = field(default_factory=list)
    _phase_counter: int = 0                              # steps in proposed new phase

    def detect(
        self,
        gci: float,
        cp: float,
        p_tensor: np.ndarray,
        t: float,
    ) -> Optional[PhaseEvent]:
        """
        Run detection on the current step's metrics.

        Returns a PhaseEvent if a transition is detected, else None.
        """
        # Update sliding windows
        self.gci_window.append(gci)
        self.cp_window.append(cp)
        coherence = self._compute_coherence(p_tensor)
        self.coherence_window.append(coherence)

        # Trim windows
        if len(self.gci_window) > self.window_size:
            self.gci_window = self.gci_window[-self.window_size:]
        if len(self.cp_window) > self.window_size:
            self.cp_window = self.cp_window[-self.window_size:]
        if len(self.coherence_window) > self.window_size:
            self.coherence_window = self.coherence_window[-self.window_size:]

        # Compute windowed metrics
        avg_gci = np.mean(self.gci_window)
        avg_coherence = np.mean(self.coherence_window)
        eigenvalue_ratio = self._eigenvalue_dominance(p_tensor)

        # Determine proposed phase
        proposed_phase, trigger = self._classify_phase(
            avg_gci, cp, avg_coherence, eigenvalue_ratio
        )

        # Hysteresis: require sustained change
        if proposed_phase != self.current_phase:
            self._phase_counter += 1
            if self._phase_counter >= self.min_sustained_steps:
                event = PhaseEvent(
                    time=t,
                    from_phase=self.current_phase,
                    to_phase=proposed_phase,
                    gci=gci,
                    cp=cp,
                    trigger=trigger,
                    eigenvalue_ratio=eigenvalue_ratio,
                    coherence=coherence,
                )
                self.current_phase = proposed_phase
                self._phase_counter = 0
                self.events.append(event)
                return event
        else:
            self._phase_counter = 0

        return None

    def _classify_phase(
        self,
        avg_gci: float,
        cp: float,
        avg_coherence: float,
        eigenvalue_ratio: float,
    ) -> Tuple[str, str]:
        """
        Classify the current state into a phase based on multiple criteria.

        Returns (phase_name, trigger_description).
        """
        # Check for Transcendplexity (highest priority)
        if avg_gci >= self.gci_transcend_threshold - self.hysteresis_margin:
            return "TRANSCENDPLEXITY", f"GCI={avg_gci:.3f} ≥ φ²={self.gci_transcend_threshold:.3f}"

        if avg_coherence >= self.coherence_threshold and eigenvalue_ratio >= self.eigenvalue_dominance_threshold:
            return "TRANSCENDPLEXITY", (
                f"coherence={avg_coherence:.3f} ≥ 1/φ "
                f"AND eigenvalue_dominance={eigenvalue_ratio:.3f} ≥ {self.eigenvalue_dominance_threshold}"
            )

        # Check for Collapseplexity
        if avg_gci < 0 and len(self.cp_window) >= 3:
            cp_trend = np.diff(self.cp_window[-3:])
            if np.all(cp_trend < 0):
                return "COLLAPSEPLEXITY", f"GCI={avg_gci:.3f} < 0 and CP declining"

        # Check for Compounding
        if avg_gci >= self.gci_compound_threshold:
            return "COMPOUNDING", f"GCI={avg_gci:.3f} ≥ {self.gci_compound_threshold}"

        # Default: Myriadplexity
        return "MYRIADPLEXITY", f"GCI={avg_gci:.3f} (sub-threshold)"

    def _compute_coherence(self, p_tensor: np.ndarray) -> float:
        """
        Compute mean cross-limb coherence from the parallplexity tensor.

        Coherence = 1 - mean(off-diagonal P_ij)
        When all streams are perfectly coupled, coherence = 1.
        """
        K = p_tensor.shape[0]
        if K < 2:
            return 0.0
        mask = ~np.eye(K, dtype=bool)
        off_diag = p_tensor[mask]
        return 1.0 - np.mean(off_diag)

    def _eigenvalue_dominance(self, p_tensor: np.ndarray) -> float:
        """
        Compute eigenvalue dominance ratio.

        When one eigenvalue dominates, the system has collapsed to
        a single coherent mode — characteristic of Transcendplexity.
        """
        eigenvalues = np.abs(np.linalg.eigvals(p_tensor))
        eigenvalues = np.sort(eigenvalues)[::-1]
        total = eigenvalues.sum()
        if total < 1e-12:
            return 0.0
        return eigenvalues[0] / total

    def summary(self) -> Dict:
        """Return a summary of all detected phase transitions."""
        return {
            "current_phase": self.current_phase,
            "total_transitions": len(self.events),
            "events": [
                {
                    "t": e.time,
                    "from": e.from_phase,
                    "to": e.to_phase,
                    "gci": e.gci,
                    "cp": e.cp,
                    "trigger": e.trigger,
                    "eigenvalue_ratio": e.eigenvalue_ratio,
                    "coherence": e.coherence,
                }
                for e in self.events
            ],
            "reached_transcendplexity": any(
                e.to_phase == "TRANSCENDPLEXITY" for e in self.events
            ),
            "thresholds": {
                "gci_compound": self.gci_compound_threshold,
                "gci_transcend": self.gci_transcend_threshold,
                "coherence": self.coherence_threshold,
                "eigenvalue_dominance": self.eigenvalue_dominance_threshold,
            },
        }
