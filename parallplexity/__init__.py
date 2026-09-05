"""
Compounding Fractional Parallplexity
=====================================

A framework for modeling emergent intelligence through fractional-order
parallel processing, compounding information dynamics, and phase
transitions in the Plexity Ontology.

Core Equation:
    CP^(α)(t) = det(I - ^C D_t^α P(t))^{-1}

When the Compounding Parallplexity CP diverges, the Golden Consciousness
Index GCI ∝ d/dt ln(CP) exceeds φ² ≈ 2.618, triggering the
Transcendplexity phase transition.

Modules:
    core        — Fractional calculus, parallplexity tensors, lattice substrate,
                  evolution operators
    emergent    — 8-limb processor, phase transition detection, emergent spacetime
    tests       — Compound integration tests
    examples    — Demo scripts and visualizations
"""

from .core.fractional import (
    CaputoDerivative,
    HistoryBuffer,
    PHI,
    PHI_SQUARED,
    CANTOR_GOLDEN_DIM,
    mittag_leffler,
    fractional_power_law_kernel,
)

from .core.parallplexity import (
    StreamState,
    ParallplexityTensor,
    CompoundingTracker,
)

from .core.evolution import FractionalEvolutionOperator

from .core.lattice import Lattice

from .emergent.eight_limb import (
    EightLimbProcessor,
    LimbConfig,
    DEFAULT_LIMBS,
)

from .emergent.phase_detector import (
    TranscendplexityDetector,
    PhaseEvent,
)

from .emergent.spacetime import EmergentSpacetime

from .worms import (
    WormsEngine,
    DeschoolingEngine,
    DeschoolingMode,
    RustyWormBridge,
    CompoundWormIntegration,
)

__version__ = "0.3.0"
__author__ = "Evan Pieser / GitMonsters"

__all__ = [
    # Constants
    "PHI", "PHI_SQUARED", "CANTOR_GOLDEN_DIM",
    # Core
    "CaputoDerivative", "HistoryBuffer",
    "mittag_leffler", "fractional_power_law_kernel",
    "ParallplexityTensor", "CompoundingTracker", "StreamState",
    "FractionalEvolutionOperator",
    "Lattice",
    # Emergent
    "EightLimbProcessor", "LimbConfig", "DEFAULT_LIMBS",
    "TranscendplexityDetector", "PhaseEvent",
    "EmergentSpacetime",
    # Worms (Python distribution)
    "WormsEngine",
    "DeschoolingEngine", "DeschoolingMode",
    "RustyWormBridge",
    "CompoundWormIntegration",
]
