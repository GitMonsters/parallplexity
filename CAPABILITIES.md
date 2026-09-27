# Capabilities of the parallplexity package (v0.3.0)

Honest, verified description of what this package does, what it is good for,
and — equally important — what it is *not*.

## What it is

`parallplexity` is a runnable numerical simulation of a **compounding
fractional-parallplexity** model of cognition. It treats cognition as
fractional-order parallel integration: layered memory processes with
non-integer time derivatives (via Caputo), coupled into a single compounding
loop whose state is summarized by two scalar metrics — CP (compounding
parallplexity) and GCI (its log-derivative) — and classified into four
phases.

It is a *model-as-simulator*: an abstraction over biological/psychological
concepts (memory depth, empathy, metacognition, phase transitions), not a
trained network and not a claim about any real mind.

## Architecture (all wired into `CompoundWormIntegration`)

| Component | What it contributes |
|---|---|
| **8 worm layers** (engine) | Fractional Caputo evolution + Gross–Pitaevskii nonlinearity per layer; layer-specific fractional order α (memory depth); confidence scaling |
| **Eight-limb processor** | 8 parallel complex-state limbs, `ParallplexityTensor` coupling, `MetaCognitionLimb` adaptively tunes each α from the GCI error signal |
| **Lattice substrate** (16×16) | Periodic amplitude grid at α = Cantor–Golden dimension |
| **Emergent spacetime** | Curvature field, region fluctuation series, wormhole / standard-lattice signature checks |
| **RustyWorm bridge** | 8 OCTO-braid head gates modulating limbs; temperature routing (System 1/2); Ising-empathy emotion vector; prime-directive symbiotic check |
| **Compounding tracker** | CP, GCI, phase detection (MYRIAD → COMPOUNDING → TRANSCENDPLEXITY → COLLAPSE) with transition log |

One `step()` runs the full pipeline: worms → parallplexity snapshot → bridge
modulation → 8-limb cohesion coupling → compounding metrics.

## Verified engineering state

- **78 tests pass** (`python -m pytest parallplexity/tests -q`), incl.
  `test_seeded_determinism`.
- **Performance**: eight-limb pass ~139 steps/s; full compound loop ~60 steps/s.
- **Reproducible**: `np.random.seed(s)` + `CompoundWormIntegration(seed=s)`
  gives byte-identical runs (see below). Fixed upstream: the lattice substrate
  previously drew from OS-entropy (`default_rng(None)`), and per-instance state
  mutated module-level default configs — both fixed and covered by tests.
- **Reproduces its own evidence**: `evaluation_v3.json` ablate baseline CP 6.94
  vs no-bridges 2.86, no-memory 7.13, consistent with the README claims.
- Demo and benchmark write `compound_integration_data.json` /
  `benchmark_report.json` for downstream figures.

## What it is good for

1. **A fast, explorable sandbox** for fractional-order memory, compounding
   dynamics, and phase-transition dynamics in a computable toy cognition model.
   Tune `dt`, `deschooling_mode`/`deschooling_intensity`, bridges, memory, and
   layer count; watch CP/GCI/phase evolve step-by-step.
2. **An ablation testbed.** Cheap sweeps test structural hypotheses (bridges
   amplify compounding; no-memory changes the CP trajectory) before committing
   to any real-system implementation.
3. **Evidence artifacts for research/teaching.** Tests, benchmark report, eval
   json, and demo timeline are all reproducible, so figures and claims can be
   regenerated from a fixed commit.
4. **Design inspiration.** The α-governor and head-gate modulation vector are
   transferrable *schemes* for adaptive memory-depth / attention routing in
   real agent architectures (adapt to production code; not drop-in).

## What it is NOT

- **Not a trained model** and **not an AGI**. There is no learning; the
  "intelligence" is the unfolding of the specified dynamics.
- **Not externally validated.** CP/GCI/phases are internal-consistency metrics
  with no ground truth. Across 12 seeds all four final phases occur, so
  "PASS/FAIL" verdicts are initial-condition sensitive — treat them as
  *illustrations*, not measurements.
- **Not deployable** as a runtime component without significant adaptation.

## Reproducible usage

```python
import numpy as np
from parallplexity.worms.compound import CompoundWormIntegration

np.random.seed(42)
cwi = CompoundWormIntegration(seed=42)   # seed threads into Lattice
summary = cwi.run(steps=500)
```

See `README.md` for full run instructions.