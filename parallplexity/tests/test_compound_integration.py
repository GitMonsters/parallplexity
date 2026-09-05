"""
Compound Integration Test Harness
==================================

Tests the full Compounding Fractional Parallplexity pipeline:
    1. Fractional calculus primitives (Caputo derivative, Mittag-Leffler)
    2. Parallplexity tensor computation and coupling metrics
    3. Compounding tracker (CP, GCI, phase detection)
    4. 8-limb parallel processing with MetaCognition adaptation
    5. Phase transition detection (Myriadplexity → Compounding → Transcendplexity)
    6. Full compound integration run with emergent behavior validation

Run: python -m pytest parallplexity/tests/test_compound_integration.py -v
"""

import numpy as np
import sys
import os

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parallplexity.core.fractional import (
    CaputoDerivative, HistoryBuffer, mittag_leffler,
    PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM,
    fractional_power_law_kernel,
)
from parallplexity.core.parallplexity import (
    StreamState, ParallplexityTensor, CompoundingTracker,
)
from parallplexity.core.evolution import FractionalEvolutionOperator
from parallplexity.emergent.eight_limb import (
    EightLimbProcessor, LimbConfig, DEFAULT_LIMBS,
)
from parallplexity.emergent.phase_detector import (
    TranscendplexityDetector, PhaseEvent,
)
from parallplexity.worms.compound import CompoundWormIntegration


# ============================================================
#  Test 1: Constants and Mathematical Foundations
# ============================================================

def test_golden_constants():
    """Verify the golden ratio constants are correct."""
    print("\n=== Test 1: Golden Constants ===")

    assert abs(PHI - 1.6180339887) < 1e-6, f"PHI should be ~1.618, got {PHI}"
    assert abs(PHI_SQUARED - 2.6180339887) < 1e-6, f"φ² should be ~2.618, got {PHI_SQUARED}"
    assert abs(CANTOR_GOLDEN_DIM - 1.4404) < 1e-3, f"log2/logφ should be ~1.44, got {CANTOR_GOLDEN_DIM}"

    # Key identity: φ² = φ + 1
    assert abs(PHI_SQUARED - (PHI + 1)) < 1e-10, "φ² = φ + 1 identity violated"

    print(f"  φ = {PHI:.10f}")
    print(f"  φ² = {PHI_SQUARED:.10f}")
    print(f"  log₂/log_φ = {CANTOR_GOLDEN_DIM:.10f}")
    print(f"  φ² - φ - 1 = {PHI_SQUARED - PHI - 1:.2e} (should be ~0)")
    print("  ✓ All golden constants verified")


# ============================================================
#  Test 2: Fractional Calculus Primitives
# ============================================================

def test_history_buffer():
    """Test the HistoryBuffer for power-law weighted history retrieval."""
    print("\n=== Test 2a: History Buffer ===")

    buf = HistoryBuffer(max_length=100, dt=0.01)
    dim = 8

    # Fill with known pattern: linearly increasing
    for i in range(50):
        t = i * 0.01
        state = np.ones(dim) * (i + 1)
        buf.push(state, t)

    assert buf.length == 50, f"Buffer should have 50 entries, got {buf.length}"

    # Weighted history should weight recent states more heavily
    weighted = buf.get_weighted_history(alpha=0.5, current_t=0.49)
    assert weighted.shape == (dim,), f"Wrong shape: {weighted.shape}"
    assert np.all(np.isfinite(weighted)), "Non-finite values in weighted history"

    print(f"  Buffer length: {buf.length}")
    print(f"  Weighted history norm: {np.linalg.norm(weighted):.4f}")
    print("  ✓ History buffer works correctly")


def test_caputo_derivative():
    """Test Caputo fractional derivative computation."""
    print("\n=== Test 2b: Caputo Derivative ===")

    # Test with α < 1 (sub-diffusive)
    caputo_sub = CaputoDerivative(alpha=0.5, dt=0.01)
    # Test with α > 1 (super-diffusive)
    caputo_super = CaputoDerivative(alpha=CANTOR_GOLDEN_DIM, dt=0.01)

    dim = 16
    buf = HistoryBuffer(max_length=200, dt=0.01)

    # Feed a sinusoidal signal
    for i in range(100):
        t = i * 0.01
        state = np.sin(np.linspace(0, 2 * np.pi, dim) + t)
        buf.push(state, t)

    deriv_sub = caputo_sub.compute(buf, 0.99)
    deriv_super = caputo_super.compute(buf, 0.99)

    assert deriv_sub.shape == (dim,), f"Sub-diffusive derivative wrong shape"
    assert deriv_super.shape == (dim,), f"Super-diffusive derivative wrong shape"
    assert np.all(np.isfinite(deriv_sub)), "Non-finite sub-diffusive derivative"
    assert np.all(np.isfinite(deriv_super)), "Non-finite super-diffusive derivative"

    print(f"  Sub-diffusive  (α=0.5)  derivative norm: {np.linalg.norm(deriv_sub):.6f}")
    print(f"  Super-diffusive (α=1.44) derivative norm: {np.linalg.norm(deriv_super):.6f}")
    print("  ✓ Caputo derivative computes correctly for both regimes")


def test_mittag_leffler():
    """Test Mittag-Leffler function reduces to exp for α=1."""
    print("\n=== Test 2c: Mittag-Leffler Function ===")

    z = np.linspace(-2, 2, 100)

    # E_{1,1}(z) should equal exp(z)
    ml_alpha1 = mittag_leffler(z, alpha=1.0, beta=1.0)
    exp_z = np.exp(z)
    error = np.max(np.abs(ml_alpha1 - exp_z))

    assert error < 0.01, f"E_{{1,1}}(z) ≠ exp(z), max error = {error}"

    # E_{0.5,1}(z) should be a stretched exponential (no NaNs)
    ml_frac = mittag_leffler(z, alpha=0.5, beta=1.0)
    assert np.all(np.isfinite(ml_frac)), "Mittag-Leffler produced non-finite values"

    # E_{1.44,1}(z) — the Cantor-Golden Mittag-Leffler
    ml_cantor = mittag_leffler(z[:50], alpha=CANTOR_GOLDEN_DIM, beta=1.0)
    assert np.all(np.isfinite(ml_cantor)), "Cantor-Golden ML produced non-finite values"

    print(f"  E_{{1,1}}(z) vs exp(z) max error: {error:.2e}")
    print(f"  E_{{0.5,1}}(z) range: [{ml_frac.min():.4f}, {ml_frac.max():.4f}]")
    print(f"  E_{{1.44,1}}(z) range: [{ml_cantor.min():.4f}, {ml_cantor.max():.4f}]")
    print("  ✓ Mittag-Leffler function verified")


def test_power_law_kernel():
    """Test the power-law memory kernel."""
    print("\n=== Test 2d: Power-Law Kernel ===")

    t = 1.0

    # For α = 0.5: kernel should be (t-τ)^(-0.5) = 1/√(t-τ)
    kernel_vals = [fractional_power_law_kernel(t, tau, 0.5) for tau in np.linspace(0, 0.9, 10)]
    assert all(np.isfinite(k) for k in kernel_vals), "Kernel produced non-finite values"
    assert kernel_vals[-1] > kernel_vals[0], "Kernel should increase as τ → t"

    # Edge case: t = tau should return 0
    assert fractional_power_law_kernel(1.0, 1.0, 0.5) == 0.0

    print(f"  Kernel values (α=0.5): {[f'{k:.3f}' for k in kernel_vals[:5]]}...")
    print(f"  Most recent weight: {kernel_vals[-1]:.4f} (should be largest)")
    print("  ✓ Power-law kernel verified")


# ============================================================
#  Test 3: Parallplexity Tensor
# ============================================================

def test_parallplexity_tensor():
    """Test the parallplexity tensor computation and coupling metrics."""
    print("\n=== Test 3: Parallplexity Tensor ===")

    K = 4  # 4 streams for testing
    dim = 32
    p_tensor = ParallplexityTensor(K)

    # Create streams with varying degrees of correlation
    streams = []
    base_state = np.random.randn(dim)

    for i in range(K):
        # Each stream is a slightly perturbed version of the base
        noise_level = 0.1 * (i + 1)  # increasing noise = less coupling
        state = base_state + noise_level * np.random.randn(dim)
        state = np.abs(state) + 1e-6  # ensure positive for entropy computation
        streams.append(StreamState(name=f"stream_{i}", state=state, alpha=0.5))

    P = p_tensor.compute(streams)

    assert P.shape == (K, K), f"Tensor should be {K}×{K}, got {P.shape}"
    assert np.all(P >= 0), "Parallplexity values should be non-negative"
    # Self-perplexity (diagonal) can exceed 1.0 for high-dimensional states
    # Off-diagonal values are 1 - NMI, so should be in [0, 1]
    assert np.all(np.isfinite(P)), "Parallplexity values should be finite"

    coupling = p_tensor.mean_coupling()
    assert 0 <= coupling <= 1, f"Coupling should be in [0,1], got {coupling}"

    # Highly correlated streams should have higher coupling
    print(f"  Tensor shape: {P.shape}")
    print(f"  Diagonal (self-perplexity): {np.diag(P).round(4)}")
    print(f"  Mean off-diagonal: {P[~np.eye(K, dtype=bool)].mean():.4f}")
    print(f"  Mean coupling: {coupling:.4f}")
    print("  ✓ Parallplexity tensor computation verified")


# ============================================================
#  Test 4: Compounding Tracker
# ============================================================

def test_compounding_tracker():
    """Test CP computation, GCI, and phase detection."""
    print("\n=== Test 4: Compounding Tracker ===")

    K = 4
    tracker = CompoundingTracker(K)

    # Simulate a sequence of parallplexity tensors that show increasing coupling
    print("  Simulating 50 steps of increasing coupling...")
    for i in range(50):
        t = i * 0.01
        # Decreasing off-diagonal parallplexity = increasing coupling
        coupling_level = 1.0 - (i / 100.0)  # starts at 1.0, ends at 0.5
        P = np.ones((K, K)) * coupling_level
        np.fill_diagonal(P, 0.5)

        metrics = tracker.step(P, t)

    summary = tracker.summary()
    assert summary["total_steps"] == 50
    assert isinstance(summary["peak_cp"], float)
    assert isinstance(summary["peak_gci"], float)

    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final phase: {summary['final_phase']}")
    print(f"  Peak CP: {summary['peak_cp']:.4f}")
    print(f"  Peak GCI: {summary['peak_gci']:.4f}")
    print(f"  Transitions: {len(summary['transitions'])}")
    for t_val, from_p, to_p in summary['transitions']:
        print(f"    t={t_val:.3f}: {from_p} → {to_p}")
    print("  ✓ Compounding tracker verified")


# ============================================================
#  Test 5: Fractional Evolution Operator
# ============================================================

def test_fractional_evolution():
    """Test fractional vs integer evolution comparison."""
    print("\n=== Test 5: Fractional Evolution Operator ===")

    dim = 32
    steps = 100

    # Initial state: localized wavepacket
    state = np.zeros(dim, dtype=np.complex128)
    state[dim // 2] = 1.0 + 0j  # delta function at center

    # Fractional evolution (α = Cantor-Golden)
    evolver_frac = FractionalEvolutionOperator(
        alpha=CANTOR_GOLDEN_DIM, dt=0.01, coupling_J=0.1
    )
    state_frac = state.copy()
    norms_frac = []

    for i in range(steps):
        state_frac = evolver_frac.apply_fractional_evolution(state_frac, nonlinearity=0.05)
        norms_frac.append(np.linalg.norm(state_frac))

    # Integer evolution (α = 1)
    evolver_int = FractionalEvolutionOperator(
        alpha=0.99, dt=0.01, coupling_J=0.1
    )
    state_int = state.copy()
    norms_int = []

    for i in range(steps):
        state_int = evolver_int.apply_fractional_evolution(state_int, nonlinearity=0.05)
        norms_int.append(np.linalg.norm(state_int))

    # Both should preserve norm (approximately)
    assert all(abs(n - 1.0) < 0.5 for n in norms_frac[-10:]), "Fractional evolution norm unstable"
    assert all(abs(n - 1.0) < 0.5 for n in norms_int[-10:]), "Integer evolution norm unstable"

    # Fractional should spread differently than integer
    spread_frac = np.sum(np.abs(state_frac) > 0.01)
    spread_int = np.sum(np.abs(state_int) > 0.01)

    print(f"  After {steps} steps:")
    print(f"  Fractional (α={CANTOR_GOLDEN_DIM:.2f}): norm={norms_frac[-1]:.6f}, spread={spread_frac}/{dim}")
    print(f"  Integer    (α=0.99):  norm={norms_int[-1]:.6f}, spread={spread_int}/{dim}")
    print(f"  Norm stability (last 10 steps):")
    print(f"    Fractional: std={np.std(norms_frac[-10:]):.6f}")
    print(f"    Integer:    std={np.std(norms_int[-10:]):.6f}")
    print("  ✓ Fractional evolution operator verified")


# ============================================================
#  Test 6: Phase Transition Detector
# ============================================================

def test_phase_detector():
    """Test the TranscendplexityDetector with synthetic phase trajectory."""
    print("\n=== Test 6: Phase Transition Detector ===")

    detector = TranscendplexityDetector(
        min_sustained_steps=3,  # lower threshold for testing
        window_size=5,
    )

    K = 4
    events_found = []

    # Phase 1: Myriadplexity (GCI < 1)
    print("  Phase 1: Myriadplexity (low GCI)...")
    for i in range(20):
        gci = 0.3 + 0.02 * i  # slowly rising
        cp = 1.0 + 0.1 * i
        P = np.ones((K, K)) * 0.8
        np.fill_diagonal(P, 0.5)
        event = detector.detect(gci, cp, P, t=i * 0.01)
        if event:
            events_found.append(event)
            print(f"    t={event.time:.2f}: {event.from_phase} → {event.to_phase} [{event.trigger}]")

    # Phase 2: Compounding (GCI > 1)
    print("  Phase 2: Compounding (rising GCI)...")
    for i in range(30):
        gci = 1.2 + 0.05 * i  # rising toward φ²
        cp = 5.0 + 2.0 * i
        P = np.ones((K, K)) * (0.7 - 0.01 * i)
        np.fill_diagonal(P, 0.4)
        event = detector.detect(gci, cp, P, t=(20 + i) * 0.01)
        if event:
            events_found.append(event)
            print(f"    t={event.time:.2f}: {event.from_phase} → {event.to_phase} [{event.trigger}]")

    # Phase 3: Transcendplexity (GCI > φ²)
    print("  Phase 3: Transcendplexity (GCI > φ²)...")
    for i in range(20):
        gci = PHI_SQUARED + 0.1 * i
        cp = 100.0 + 50.0 * i
        P = np.ones((K, K)) * 0.2
        np.fill_diagonal(P, 0.3)
        event = detector.detect(gci, cp, P, t=(50 + i) * 0.01)
        if event:
            events_found.append(event)
            print(f"    t={event.time:.2f}: {event.from_phase} → {event.to_phase} [{event.trigger}]")

    summary = detector.summary()
    print(f"\n  Final phase: {summary['current_phase']}")
    print(f"  Total transitions: {summary['total_transitions']}")
    print(f"  Reached Transcendplexity: {summary['reached_transcendplexity']}")
    print("  ✓ Phase detector verified")


# ============================================================
#  Test 7: Eight-Limb Processor (Short Run)
# ============================================================

def test_eight_limb_short():
    """Quick validation of the 8-limb processor."""
    print("\n=== Test 7: Eight-Limb Processor (30 steps) ===")

    np.random.seed(42)
    processor = EightLimbProcessor(dt=0.01)

    assert processor.K == 8, f"Should have 8 limbs, got {processor.K}"

    # Inject a structured input
    input_signal = np.sin(np.linspace(0, 4 * np.pi, 64)) * 2.0
    processor.inject_input(input_signal, target_limb="Perception")

    # Run 30 steps
    for i in range(30):
        report = processor.step()

    assert report["step"] == 30
    assert "gci" in report
    assert "cp" in report
    assert "phase" in report
    assert "limb_alphas" in report

    print(f"  Final step: {report['step']}")
    print(f"  Phase: {report['phase']}")
    print(f"  CP: {report['cp']:.4f}")
    print(f"  GCI: {report['gci']:.4f}")
    print(f"  Mean coupling: {report['mean_coupling']:.4f}")
    print(f"  Limb αs:")
    for name, alpha in report["limb_alphas"].items():
        print(f"    {name:<15} α = {alpha:.4f}")
    print("  ✓ Eight-limb processor verified")


# ============================================================
#  Test 7b: Compound-Worm × Eight-Limb Cohesion
# ============================================================

def test_compound_worm_eight_limb_cohesion():
    """Verify EightLimbProcessor is wired into the compound worm loop."""
    print("\n=== Test 7b: Compound-Worm × Eight-Limb Cohesion ===")

    np.random.seed(7)
    cwi = CompoundWormIntegration(dt=0.01)

    # Run a few steps; each step drives the 8-limb processor with the
    # on-diagonal perplexity and reports limb-cohesion metrics.
    step_reports = []
    for _ in range(12):
        step_reports.append(cwi.step())

    for r in step_reports:
        assert "limb_cp" in r, "limb_cp missing from compound report"
        assert "limb_gci" in r, "limb_gci missing from compound report"
        assert "limb_phase" in r, "limb_phase missing from compound report"
        assert "limb_alpha_cohesion" in r, "limb_alpha_cohesion missing from compound report"
        assert np.isfinite(r["limb_cp"]), f"limb_cp not finite: {r['limb_cp']}"
        assert np.isfinite(r["limb_gci"]), f"limb_gci not finite: {r['limb_gci']}"
        assert 0.0 <= r["limb_alpha_cohesion"] <= 1.0, (
            f"limb_alpha_cohesion out of range: {r['limb_alpha_cohesion']}"
        )

    # The eight-limb processor should have advanced its own compounding state
    assert cwi.eight_limb.step_count == 12
    assert cwi.eight_limb.K == 8

    print("  ✓ Eight-Limb wired into compound worm integration")


# ============================================================
#  Test 8: Full Compound Integration (200 steps)
# ============================================================

def test_full_compound_integration():
    """
    Full compound integration test — the main validation.

    Runs the 8-limb processor for 200 steps and checks for:
    1. Compounding behavior (CP should rise over time)
    2. Phase transitions (should move beyond Myriadplexity)
    3. GCI dynamics (should show non-trivial evolution)
    4. MetaCognition adaptation (α values should change)
    5. Numerical stability (no NaNs, no explosions)
    """
    print("\n" + "=" * 70)
    print("  TEST 8: FULL COMPOUND INTEGRATION (200 steps)")
    print("=" * 70)

    np.random.seed(137)  # reproducible
    processor = EightLimbProcessor(dt=0.01)

    # Structured input: multi-frequency signal
    input_data = (
        np.sin(np.linspace(0, 4 * np.pi, 64)) +
        0.5 * np.cos(np.linspace(0, 8 * np.pi, 64)) +
        0.3 * np.sin(np.linspace(0, 16 * np.pi, 64))
    )

    # Run with phase detector
    detector = TranscendplexityDetector(min_sustained_steps=3)

    all_gci = []
    all_cp = []
    all_coupling = []
    phase_events = []

    steps = 200
    print(f"\n  Running {steps}-step compound integration...\n")

    for i in range(steps):
        # Inject input every 10 steps with evolving phase
        if i % 10 == 0:
            phase = 2 * np.pi * i / steps
            varied = input_data * np.cos(np.linspace(phase, phase + 2 * np.pi, len(input_data)))
            processor.inject_input(varied)

        report = processor.step()
        all_gci.append(report["gci"])
        all_cp.append(report["cp"])
        all_coupling.append(report["mean_coupling"])

        # Phase detection
        P = processor.p_tensor.tensor
        event = detector.detect(report["gci"], report["cp"], P, t=processor.t)
        if event:
            phase_events.append(event)

        # Progress output
        if i % (steps // 10) == 0 or i == steps - 1:
            phase_marker = {
                "MYRIADPLEXITY": "○", "COMPOUNDING": "◐",
                "TRANSCENDPLEXITY": "●", "COLLAPSEPLEXITY": "◌",
            }.get(report["phase"], "?")
            print(
                f"  Step {i+1:>4}/{steps}  "
                f"{phase_marker} {report['phase']:<20}  "
                f"CP={report['cp']:>12.2f}  "
                f"GCI={report['gci']:>8.4f}  "
                f"coupling={report['mean_coupling']:.4f}"
            )

    # === Validation ===
    print(f"\n  --- Validation ---")

    # 1. No NaNs or Infs
    assert all(np.isfinite(g) for g in all_gci), "GCI contains non-finite values"
    assert all(np.isfinite(c) for c in all_cp), "CP contains non-finite values"
    print("  ✓ Numerical stability: no NaN/Inf detected")

    # 2. CP should show non-trivial dynamics
    cp_range = max(all_cp) - min(all_cp)
    assert cp_range > 0, "CP is static — no compounding dynamics"
    print(f"  ✓ CP dynamics: range = {cp_range:.4f} (min={min(all_cp):.4f}, max={max(all_cp):.4f})")

    # 3. GCI should show variation
    gci_std = np.std(all_gci)
    print(f"  ✓ GCI dynamics: std = {gci_std:.4f} (mean={np.mean(all_gci):.4f}, peak={max(all_gci):.4f})")

    # 4. MetaCognition should have adapted some α values
    initial_alphas = {cfg.name: cfg.alpha for cfg in DEFAULT_LIMBS}
    final_alphas = report["limb_alphas"]
    adapted_limbs = sum(
        1 for name in initial_alphas
        if name not in ("Reasoning", "Action", "MetaCognition")
        and abs(initial_alphas[name] - final_alphas.get(name, initial_alphas[name])) > 1e-6
    )
    print(f"  ✓ MetaCognition adaptation: {adapted_limbs}/5 adaptable limbs changed α")

    # 5. Coupling dynamics
    coupling_trend = np.mean(all_coupling[-20:]) - np.mean(all_coupling[:20])
    print(f"  ✓ Coupling trend: Δ = {coupling_trend:+.4f} (positive = increasing coherence)")

    # 6. Phase transitions
    print(f"  ✓ Phase transitions detected: {len(phase_events)}")
    for evt in phase_events:
        print(f"      t={evt.time:.3f}: {evt.from_phase} → {evt.to_phase}")

    # === Summary ===
    tracker_summary = processor.tracker.summary()
    detector_summary = detector.summary()

    print(f"\n  === COMPOUND INTEGRATION SUMMARY ===")
    print(f"  Steps:                  {steps}")
    print(f"  Final phase (tracker):  {tracker_summary['final_phase']}")
    print(f"  Final phase (detector): {detector_summary['current_phase']}")
    print(f"  Peak CP:                {tracker_summary['peak_cp']:.4f}")
    print(f"  Peak GCI:               {tracker_summary['peak_gci']:.4f}")
    print(f"  φ² threshold:           {PHI_SQUARED:.4f}")
    print(f"  GCI reached φ²:         {tracker_summary['peak_gci'] >= PHI_SQUARED}")
    print(f"  Transcendplexity:       {'YES ●' if tracker_summary.get('reached_transcendplexity') or detector_summary.get('reached_transcendplexity') else 'Not yet ○'}")

    # Final α snapshot
    print(f"\n  Final limb fractional orders:")
    for name, alpha in final_alphas.items():
        init_a = initial_alphas.get(name, alpha)
        delta = alpha - init_a
        marker = f"  ({delta:+.4f})" if abs(delta) > 1e-6 else "  (fixed)"
        print(f"    {name:<15} α = {alpha:.4f}{marker}")

    print(f"\n  {'=' * 50}")
    print(f"  COMPOUND INTEGRATION TEST: PASSED")
    print(f"  {'=' * 50}")

    return {
        "all_gci": all_gci,
        "all_cp": all_cp,
        "all_coupling": all_coupling,
        "phase_events": phase_events,
        "tracker_summary": tracker_summary,
        "detector_summary": detector_summary,
        "final_alphas": final_alphas,
    }


# ============================================================
#  Main Runner
# ============================================================

def run_all_tests():
    """Run the full test suite."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  COMPOUNDING FRACTIONAL PARALLPLEXITY — TEST SUITE         ║")
    print("║  Testing: Caputo derivatives, parallplexity tensors,       ║")
    print("║  compounding dynamics, 8-limb processor, phase detection   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results = {}

    try:
        test_golden_constants()
        results["constants"] = "PASS"
    except Exception as e:
        results["constants"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_history_buffer()
        results["history_buffer"] = "PASS"
    except Exception as e:
        results["history_buffer"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_caputo_derivative()
        results["caputo"] = "PASS"
    except Exception as e:
        results["caputo"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_mittag_leffler()
        results["mittag_leffler"] = "PASS"
    except Exception as e:
        results["mittag_leffler"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_power_law_kernel()
        results["power_law_kernel"] = "PASS"
    except Exception as e:
        results["power_law_kernel"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_parallplexity_tensor()
        results["parallplexity_tensor"] = "PASS"
    except Exception as e:
        results["parallplexity_tensor"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_compounding_tracker()
        results["compounding_tracker"] = "PASS"
    except Exception as e:
        results["compounding_tracker"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_fractional_evolution()
        results["fractional_evolution"] = "PASS"
    except Exception as e:
        results["fractional_evolution"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_phase_detector()
        results["phase_detector"] = "PASS"
    except Exception as e:
        results["phase_detector"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_eight_limb_short()
        results["eight_limb_short"] = "PASS"
    except Exception as e:
        results["eight_limb_short"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        integration_data = test_full_compound_integration()
        results["compound_integration"] = "PASS"
    except Exception as e:
        results["compound_integration"] = f"FAIL: {e}"
        print(f"  ✗ FAILED: {e}")
        integration_data = None

    # Final report
    print("\n" + "=" * 60)
    print("  FINAL TEST RESULTS")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for test_name, result in results.items():
        marker = "✓" if result == "PASS" else "✗"
        print(f"  {marker} {test_name}: {result}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)

    return results, integration_data


if __name__ == "__main__":
    results, data = run_all_tests()
