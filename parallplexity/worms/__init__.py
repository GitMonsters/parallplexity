"""
Worms Engine — Python Distribution
====================================

TranscendPlexity's Python-native implementation of the 8-Layer Multiplicative
Integration architecture, mapped to the OctoTetrahedral limb system.

This is the Python counterpart to RustyWorm (Rust distribution).
The "worm" metaphor: information threads through layers like a worm
through soil — each layer compounds on the last multiplicatively.

Architecture:
    Layer 1 (Base Physics)       → Perception limb (α=0.9)
    Layer 2 (Extended Physics)   → Memory limb (α=0.3)
    Layer 3 (Cross-Domain)       → Planning limb (α=0.7)
    Layer 4 (GAIA Consciousness) → Language limb (α=0.5)
    Layer 5 (Multilingual)       → Spatial limb (α=0.6)
    Layer 6 (Collaborative)      → Reasoning limb (α=1.44)
    Layer 7 (External APIs)      → MetaCognition limb (adaptive)
    Layer 8 (Pre-Cognitive Viz)  → Action limb (α=0.95)

Key difference from additive stacking:
    Additive:        confidence = min(c1, c2, ..., c8)
    Multiplicative:  confidence = c1 × c2 × ... × c8 × amplification
    → Can EXCEED 1.0 through compound resonance

Bridges:
    11 bidirectional bridges connect non-adjacent layers,
    enabling cross-stream information flow that drives
    parallplexity reduction and compounding.
"""

from .engine import WormsEngine, LayerConfig, BridgeConfig, WORM_LAYERS, WORM_BRIDGES
from .deschooling import DeschoolingEngine, DeschoolingMode
from .rusty_bridge import RustyWormBridge, IsingEmpathy, EmotionVector
from .compound import CompoundWormIntegration

__all__ = [
    # Worms Engine
    "WormsEngine",
    "LayerConfig",
    "BridgeConfig",
    "WORM_LAYERS",
    "WORM_BRIDGES",
    # Deschooling
    "DeschoolingEngine",
    "DeschoolingMode",
    # RustyWorm Bridge
    "RustyWormBridge",
    "IsingEmpathy",
    "EmotionVector",
    # Compound Integration
    "CompoundWormIntegration",
]
