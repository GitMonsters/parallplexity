# Compounding Fractional Parallplexity (v0.3.0)

A framework for modeling emergent intelligence through fractional-order parallel
processing, compounding information dynamics, and phase transitions in the
Plexity Ontology.

```text
CP^(α)(t) = det(I - ^C D_t^α P(t))^{-1}
```

When the Compounding Parallplexity CP diverges, the Golden Consciousness Index
GCI ∝ d/dt ln(CP) exceeds φ² ≈ 2.618, triggering the Transcendplexity phase
transition.

Limb-level "will" (goal-directed disposition vectors) is assumed, "free will" is
not. Will is generative only when anchored — routed through an acceptance
operator whose target is independent of the proposing stream; unanchored will is
the collapseplexity drift (see §Volition).

## Architecture

Eight subsystems are wired into a single compound integration loop
(`parallplexity/worms/compound.py::CompoundWormIntegration.step`):

| Subsystem | Module |
|-----------|--------|
| Worms Engine (8-layer multiplicative) | `worms/engine.py` |
| RustyWorm Bridge (OCTO Braid, Ising empathy, consciousness) | `worms/rusty_bridge.py` |
| Deschooling Engine (Illich convivial/institutional) | `worms/deschooling.py` |
| Fractional calculus core (Caputo, Mittag-Leffler) | `core/fractional.py` |
| Parallplexity tensor + CompoundingTracker (CP/GCI/phase) | `core/parallplexity.py` |
| Fractional evolution operator | `core/evolution.py` |
| MetaCognition α governor | `emergent/metacognition.py` |
| Phase-transition detection | `emergent/phase_detector.py` |
| Eight-Limb OctoTetrahedral processor | `emergent/eight_limb.py` |

## Install & test

```bash
pip install -e .
python -m pytest parallplexity/tests -q        # 66 tests, 66 pass
```

## Run

```bash
# Demo: plain compound integration
python parallplexity/examples/demo_compound_integration.py

# Performance benchmark (writes benchmark_report.json)
python parallplexity/examples/benchmark_performance.py --steps 500
```

Reproducibility: pin both the lattice substrate seed and the global numpy
stream to get bit-identical runs (README/CI check `test_seeded_determinism`):

```python
import numpy as np
from parallplexity.worms.compound import CompoundWormIntegration

np.random.seed(42)
cwi = CompoundWormIntegration(seed=42)   # seed threads into Lattice
summary = cwi.run(steps=500)
```

## v0.3.0 highlights

- **Bridges-Zeroed Paradox resolved**: `bridge_configs is not None` guard fixes
  the `or` truthiness bug that made `[]` silently fall back to defaults.
  Post-fix ablation: baseline CP 6.94 vs no-bridges 2.86 (2.4× amplification),
  collapse steps 27 vs 327 (12× reduction).
- **Three-factor CP measure** (spectral × tanh(coupling) × symmetry) replaces the
  eigenvalue-only spectral CP.
- **Eight-Limb integrated into the compound loop** with per-step
  `limb_cp` / `limb_gci` / `limb_phase` / `limb_alpha_cohesion` reporting.
- **Vectorized fractional kernel**: `compute_from_history` +
  vectorized `get_weighted_history` replace Python kernel loops.
  ~3–3.7× on the memory term, 1.46–1.62× end-to-end (see paper §Performance).

See `compounding-fractional-parallplexity.pplx.md` for the full paper.

---

Evan Pieser — TranscendPlexity
March 2026