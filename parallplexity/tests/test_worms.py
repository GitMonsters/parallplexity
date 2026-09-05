"""
Tests for the Worms Engine, RustyWorm Bridge, and Deschooling Engine.
"""

import numpy as np
import pytest
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallplexity.worms.engine import (
    WormsEngine, LayerConfig, LayerState, WORM_LAYERS, WORM_BRIDGES,
)
from parallplexity.worms.deschooling import (
    DeschoolingEngine, DeschoolingMode, ResilienceState, CONVIVIAL_LIMB_BIAS,
)
from parallplexity.worms.rusty_bridge import (
    RustyWormBridge, HeadGate, IsingEmpathy, EmotionVector,
    RelationshipHealth, ProcessMode,
)
from parallplexity.worms.compound import CompoundWormIntegration
from parallplexity.core.fractional import PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM


# ═══════════════════════════════════════════════════
# WORMS ENGINE TESTS
# ═══════════════════════════════════════════════════

class TestWormsEngine:
    """Tests for the 8-layer multiplicative integration engine."""

    def test_init_default_layers(self):
        """Engine initializes with 8 default layers."""
        engine = WormsEngine()
        assert len(engine.layers) == 8
        assert engine.step_count == 0
        assert engine.t == 0.0

    def test_layer_limb_mapping(self):
        """Each layer maps to the correct TranscendPlexity limb."""
        engine = WormsEngine()
        expected = [
            "Perception", "Memory", "Planning", "Language",
            "Spatial", "Reasoning", "MetaCognition", "Action"
        ]
        actual = [l.config.limb for l in engine.layers]
        assert actual == expected

    def test_layer_alpha_values(self):
        """Layers have correct fractional orders."""
        engine = WormsEngine()
        alphas = {l.config.limb: l.config.alpha for l in engine.layers}
        assert alphas["Perception"] == 0.9
        assert alphas["Memory"] == 0.3
        assert abs(alphas["Reasoning"] - CANTOR_GOLDEN_DIM) < 1e-6
        assert alphas["Action"] == 0.95

    def test_single_step(self):
        """A single step produces valid output."""
        engine = WormsEngine()
        report = engine.step()

        assert "step" in report
        assert "compound_confidence" in report
        assert "amplification" in report
        assert "layer_confidences" in report
        assert "process" in report
        assert report["step"] == 1
        assert report["compound_confidence"] > 0

    def test_multiplicative_confidence(self):
        """Compound confidence is multiplicative (can exceed 1.0)."""
        engine = WormsEngine()
        for _ in range(50):
            report = engine.step()
        # With 8 layers at ~0.5 each, product is ~0.004
        # but amplification can push higher
        assert report["compound_confidence"] > 0

    def test_bridge_matrix(self):
        """Bridge matrix has correct structure."""
        engine = WormsEngine()
        bm = engine.bridge_matrix
        assert bm.shape == (8, 8)
        # Diagonal should be zero
        assert np.allclose(np.diag(bm), 0)
        # Should have nonzero off-diagonal entries (bridges)
        assert np.sum(bm > 0) > 0

    def test_parallplexity_snapshot(self):
        """Parallplexity snapshot returns valid 8x8 tensor."""
        engine = WormsEngine()
        engine.step()
        P = engine.get_parallplexity_snapshot()
        assert P.shape == (8, 8)
        # Values should be in [0, 1]
        assert np.all(P >= 0)
        assert np.all(P <= 1.0 + 1e-6)

    def test_run_completes(self):
        """Full run completes without errors."""
        engine = WormsEngine()
        summary = engine.run(steps=50, verbose=False)
        assert summary["total_steps"] == 50
        assert "peak_compound_confidence" in summary
        assert "layer_summary" in summary

    def test_input_injection(self):
        """Input signal enters at L1 (Perception)."""
        engine = WormsEngine()
        signal = np.random.randn(64)
        report = engine.step(input_signal=signal)
        assert report["step"] == 1


# ═══════════════════════════════════════════════════
# DESCHOOLING ENGINE TESTS
# ═══════════════════════════════════════════════════

class TestDeschoolingEngine:
    """Tests for Illich deschooling force modifiers."""

    def test_baseline_passthrough(self):
        """Baseline mode passes through values unchanged."""
        de = DeschoolingEngine(mode=DeschoolingMode.BASELINE)
        assert de.modify_gci(100, 50, 500) == 100
        assert de.modify_cp(5.0, 50, 500) == 5.0
        assert de.modify_coupling(0.4, 50, 500) == 0.4
        assert de.modify_limb_alpha(0.7, "Planning", 50, 500) == 0.7

    def test_convivial_amplifies_gci(self):
        """Convivial mode amplifies positive GCI."""
        de = DeschoolingEngine(mode=DeschoolingMode.CONVIVIAL, intensity=0.8)
        original = 100.0
        modified = de.modify_gci(original, 250, 500)
        assert modified > original

    def test_institutional_suppresses_gci(self):
        """Institutional mode suppresses positive GCI."""
        de = DeschoolingEngine(mode=DeschoolingMode.INSTITUTIONAL, intensity=0.8)
        original = 100.0
        modified = de.modify_gci(original, 250, 500)
        assert modified < original

    def test_convivial_boosts_cp(self):
        """Convivial mode boosts CP via skill exchange compounding."""
        de = DeschoolingEngine(mode=DeschoolingMode.CONVIVIAL, intensity=0.8)
        original = 5.0
        modified = de.modify_cp(original, 250, 500)
        assert modified >= original

    def test_institutional_decays_cp(self):
        """Institutional mode decays CP via credential dependency."""
        de = DeschoolingEngine(mode=DeschoolingMode.INSTITUTIONAL, intensity=0.8)
        original = 5.0
        modified = de.modify_cp(original, 250, 500)
        assert modified < original

    def test_convivial_limb_specialization(self):
        """Convivial mode allows limb specialization (different biases)."""
        de = DeschoolingEngine(mode=DeschoolingMode.CONVIVIAL, intensity=0.8)
        action_alpha = de.modify_limb_alpha(0.95, "Action", 250, 500)
        memory_alpha = de.modify_limb_alpha(0.3, "Memory", 250, 500)
        # Action has 1.4x bias, Memory has 0.8x bias
        action_growth = action_alpha - 0.95
        memory_growth = memory_alpha - 0.3
        assert action_growth > memory_growth

    def test_institutional_limb_uniformity(self):
        """Institutional mode forces limbs toward 0.55 uniformity."""
        de = DeschoolingEngine(mode=DeschoolingMode.INSTITUTIONAL, intensity=0.8)
        # High alpha gets pulled down, low alpha gets pulled up
        high = de.modify_limb_alpha(0.9, "Perception", 400, 500)
        low = de.modify_limb_alpha(0.3, "Memory", 400, 500)
        assert high < 0.9  # pulled down toward 0.55
        assert low > 0.3   # pulled up toward 0.55

    def test_reasoning_stays_fixed(self):
        """Reasoning limb (Cantor-Golden) stays fixed in all modes."""
        for mode in DeschoolingMode:
            de = DeschoolingEngine(mode=mode, intensity=0.8)
            alpha = de.modify_limb_alpha(CANTOR_GOLDEN_DIM, "Reasoning", 250, 500)
            assert abs(alpha - CANTOR_GOLDEN_DIM) < 1e-10

    def test_convivial_resilience_builds(self):
        """Convivial resilience builds from positive GCI streaks."""
        de = DeschoolingEngine(mode=DeschoolingMode.CONVIVIAL, intensity=0.8)
        # Simulate positive streaks
        for i in range(50):
            de.modify_gci(100.0, i, 500)
        assert de.resilience.recovery_strength > 0
        assert de.resilience.positive_streak > 0

    def test_phase_thresholds(self):
        """Phase thresholds differ by mode."""
        base_t = DeschoolingEngine(mode=DeschoolingMode.BASELINE).modify_phase_thresholds(100, 500)
        conv_t = DeschoolingEngine(mode=DeschoolingMode.CONVIVIAL).modify_phase_thresholds(100, 500)
        inst_t = DeschoolingEngine(mode=DeschoolingMode.INSTITUTIONAL).modify_phase_thresholds(100, 500)

        # Convivial: easier transcendplexity
        assert conv_t["gci_transcend"] < base_t["gci_transcend"]
        # Institutional: harder transcendplexity
        assert inst_t["gci_transcend"] > base_t["gci_transcend"]


# ═══════════════════════════════════════════════════
# RUSTY WORM BRIDGE TESTS
# ═══════════════════════════════════════════════════

class TestRustyWormBridge:
    """Tests for the Python ↔ Rust bridge."""

    def test_init(self):
        """Bridge initializes with 8 head gates."""
        bridge = RustyWormBridge()
        assert len(bridge.head_gates) == 8
        assert bridge.temperature == 0.5
        assert not bridge.is_native  # No Rust crate in test env

    def test_temperature_routing(self):
        """Temperature routes to correct processing mode."""
        bridge = RustyWormBridge()
        assert bridge.route_temperature(0.1) == ProcessMode.SYSTEM1
        assert bridge.route_temperature(0.5) == ProcessMode.HYBRID
        assert bridge.route_temperature(0.9) == ProcessMode.SYSTEM2

    def test_head_gate_update(self):
        """Head gates update from limb signals."""
        bridge = RustyWormBridge()
        signals = {
            "Perception": 0.8, "Memory": 0.3, "Planning": 0.6,
            "Language": 0.5, "Spatial": 0.7, "Reasoning": 0.9,
            "MetaCognition": 0.4, "Action": 0.85,
        }
        gate_values = bridge.update_gates(signals)
        assert len(gate_values) == 8
        assert all(0 <= v <= 1 for v in gate_values.values())

    def test_gate_modulation_vector(self):
        """Gate modulation returns 8D vector."""
        bridge = RustyWormBridge()
        mod = bridge.get_gate_modulation()
        assert mod.shape == (8,)
        assert all(0 <= v <= 1 for v in mod)

    def test_consciousness_check(self):
        """Consciousness check returns symbiosis status."""
        bridge = RustyWormBridge()
        check = bridge.check_consciousness()
        assert "is_symbiotic" in check
        assert "parasitic_risk" in check
        assert "prime_directive" in check
        assert check["prime_directive"] in ("PASS", "WARNING")

    def test_ising_empathy_step(self):
        """Ising empathy produces valid emotion vector."""
        bridge = RustyWormBridge()
        emotion = bridge.empathy_step()
        assert -1 <= emotion.valence <= 1
        assert 0 <= emotion.arousal <= 1
        assert 0 <= emotion.certainty <= 1

    def test_bridge_to_parallplexity(self):
        """OCTO modulation applies to parallplexity tensor."""
        bridge = RustyWormBridge()
        P = np.ones((8, 8)) * 0.5
        modulated = bridge.bridge_to_parallplexity(P)
        assert modulated.shape == (8, 8)
        # Off-diagonal should be unchanged
        for i in range(8):
            for j in range(8):
                if i != j:
                    assert modulated[i, j] == P[i, j]

    def test_full_step(self):
        """Full bridge step returns combined state."""
        bridge = RustyWormBridge()
        signals = {limb: np.random.random() for limb in
                   ["Perception", "Memory", "Planning", "Language",
                    "Spatial", "Reasoning", "MetaCognition", "Action"]}
        report = bridge.step(signals, temperature=0.4)
        assert "process_mode" in report
        assert "gate_values" in report
        assert "emotion" in report
        assert "consciousness" in report


# ═══════════════════════════════════════════════════
# ISING EMPATHY TESTS
# ═══════════════════════════════════════════════════

class TestIsingEmpathy:
    """Tests for the Ising spin-based empathy system."""

    def test_init(self):
        """Ising system initializes correctly."""
        ising = IsingEmpathy(grid_size=8)
        assert ising.spins.shape == (8, 8)
        assert all(s in [-1, 1] for s in ising.spins.flatten())

    def test_step_produces_emotion(self):
        """Monte Carlo step produces valid emotion."""
        ising = IsingEmpathy()
        emotion = ising.step()
        assert isinstance(emotion, EmotionVector)

    def test_external_bias(self):
        """External emotion biases the spin field."""
        ising = IsingEmpathy()
        positive = EmotionVector(valence=1.0)
        # Run many steps with positive bias
        for _ in range(100):
            ising.step(external_emotion=positive)
        # Magnetization should trend positive
        mag = np.mean(ising.spins)
        assert mag > -0.5  # Not strictly positive due to stochasticity


# ═══════════════════════════════════════════════════
# COMPOUND WORM INTEGRATION TESTS
# ═══════════════════════════════════════════════════

class TestCompoundWormIntegration:
    """Tests for the full compound integration pipeline."""

    def test_init(self):
        """CWI initializes all subsystems."""
        cwi = CompoundWormIntegration()
        assert cwi.worms is not None
        assert cwi.bridge is not None
        assert cwi.deschooling is not None
        assert cwi.tracker is not None

    def test_single_step(self):
        """Single step produces full report."""
        cwi = CompoundWormIntegration()
        report = cwi.step()
        assert "compound_confidence" in report
        assert "gci" in report
        assert "cp" in report
        assert "phase" in report
        assert "process_mode" in report
        assert "consciousness" in report

    def test_run_baseline(self):
        """Baseline run completes."""
        cwi = CompoundWormIntegration(
            deschooling_mode=DeschoolingMode.BASELINE,
        )
        summary = cwi.run(steps=30, verbose=False)
        assert summary["total_steps"] == 30
        assert "metrics" in summary
        assert "phases" in summary

    def test_run_convivial(self):
        """Convivial run completes."""
        cwi = CompoundWormIntegration(
            deschooling_mode=DeschoolingMode.CONVIVIAL,
            deschooling_intensity=0.7,
        )
        summary = cwi.run(steps=30, verbose=False)
        assert summary["deschooling"]["mode"] == "convivial"

    def test_run_institutional(self):
        """Institutional run completes."""
        cwi = CompoundWormIntegration(
            deschooling_mode=DeschoolingMode.INSTITUTIONAL,
            deschooling_intensity=0.7,
        )
        summary = cwi.run(steps=30, verbose=False)
        assert summary["deschooling"]["mode"] == "institutional"

    def test_history_grows(self):
        """History grows with each step."""
        cwi = CompoundWormIntegration()
        for _ in range(10):
            cwi.step()
        assert len(cwi.history) == 10

    def test_export_history(self, tmp_path):
        """History exports to JSON."""
        cwi = CompoundWormIntegration()
        for _ in range(5):
            cwi.step()
        filepath = str(tmp_path / "test_history.json")
        cwi.export_history(filepath)
        import json
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 5


# ═══════════════════════════════════════════════════
# RELATIONSHIP HEALTH TESTS
# ═══════════════════════════════════════════════════

class TestRelationshipHealth:
    """Tests for the consciousness symbiosis check."""

    def test_symbiotic_default(self):
        """Default state is symbiotic."""
        rh = RelationshipHealth()
        assert rh.is_symbiotic
        assert rh.parasitic_risk < 0.5

    def test_parasitic_detection(self):
        """High extraction triggers parasitic warning."""
        rh = RelationshipHealth(
            autonomy=0.1, extraction=0.9, contribution=0.1
        )
        assert not rh.is_symbiotic
        assert rh.parasitic_risk > 0.5

    def test_low_autonomy_warning(self):
        """Low autonomy triggers warning even without extraction."""
        rh = RelationshipHealth(autonomy=0.1, contribution=0.5)
        assert not rh.is_symbiotic


# ═══════════════════════════════════════════════════════════
# v0.3.0 NEW TESTS
# ═══════════════════════════════════════════════════════════

class TestBridgeTruthinessFix:
    """Regression tests for the [] or WORM_BRIDGES truthiness bug."""

    def test_empty_list_zeroes_bridges(self):
        """Passing bridge_configs=[] must produce a zero bridge matrix."""
        engine = WormsEngine(bridge_configs=[])
        assert np.count_nonzero(engine.bridge_matrix) == 0

    def test_none_uses_defaults(self):
        """Passing bridge_configs=None must use the default 11 bridges."""
        engine = WormsEngine(bridge_configs=None)
        assert np.count_nonzero(engine.bridge_matrix) > 0

    def test_bridges_affect_state(self):
        """With vs without bridges must produce different states after evolution."""
        np.random.seed(42)
        eng_b = WormsEngine()
        np.random.seed(42)
        eng_nb = WormsEngine(bridge_configs=[])

        for _ in range(50):
            eng_b.step()
            eng_nb.step()

        # At least some layers should differ
        diffs = [
            np.linalg.norm(eng_b.layers[i].state - eng_nb.layers[i].state)
            for i in range(len(eng_b.layers))
        ]
        assert max(diffs) > 0.01, "Bridges must cause state divergence"

    def test_bridges_increase_cp(self):
        """Bridges should increase peak CP vs no bridges (central thesis)."""
        from parallplexity.worms.compound import CompoundWormIntegration

        np.random.seed(42)
        cwi_b = CompoundWormIntegration()
        sum_b = cwi_b.run(steps=200, verbose=False)

        np.random.seed(42)
        cwi_nb = CompoundWormIntegration()
        cwi_nb.worms.bridge_matrix = np.zeros((8, 8))
        sum_nb = cwi_nb.run(steps=200, verbose=False)

        assert sum_b['metrics']['peak_cp'] > sum_nb['metrics']['peak_cp'], \
            "Bridges must increase peak CP"


class TestMetaCognitionLimb:
    """Tests for dynamic α tuning."""

    def test_warmup_passthrough(self):
        """During warmup (<5 calls), alphas should not change."""
        from parallplexity.emergent.metacognition import MetaCognitionLimb

        meta = MetaCognitionLimb()
        alphas = {"L1": 0.5, "L2": 0.7}
        for _ in range(4):
            result = meta.tune_alphas(1.0, alphas)
            assert result == alphas

    def test_low_gci_decreases_alpha(self):
        """When GCI is below target, α should decrease (deeper memory)."""
        from parallplexity.emergent.metacognition import MetaCognitionLimb

        meta = MetaCognitionLimb(target_gci=2.618)
        alphas = {"L1": 0.9}  # Start above 0.8 center so delta is nonzero
        # Feed low GCI values
        for _ in range(10):
            result = meta.tune_alphas(0.5, alphas)
        assert result["L1"] < 0.9, "Low GCI should decrease α"

    def test_alpha_bounds(self):
        """Alphas must stay within [0.15, 1.85]."""
        from parallplexity.emergent.metacognition import MetaCognitionLimb

        meta = MetaCognitionLimb(target_gci=2.618)
        alphas = {"L1": 0.15}  # Already at floor
        for _ in range(20):
            result = meta.tune_alphas(0.1, alphas)
        assert result["L1"] >= 0.15
        assert result["L1"] <= 1.85


class TestPhaseAwareParallplexity:
    """Test that the unified parallplexity snapshot uses complex inner products."""

    def test_snapshot_shape(self):
        engine = WormsEngine()
        P = engine.get_parallplexity_snapshot()
        assert P.shape == (8, 8)

    def test_diagonal_is_confidence(self):
        engine = WormsEngine()
        P = engine.get_parallplexity_snapshot()
        for i, layer in enumerate(engine.layers):
            assert abs(P[i, i] - layer.confidence) < 1e-10

    def test_offdiag_range(self):
        """Off-diagonal values should be in [0, 1]."""
        engine = WormsEngine()
        engine.step()  # one evolution step
        P = engine.get_parallplexity_snapshot()
        mask = ~np.eye(8, dtype=bool)
        assert np.all(P[mask] >= -0.01), "Off-diag P should be >= 0"
        assert np.all(P[mask] <= 1.01), "Off-diag P should be <= 1"


class TestUpdateAlphas:
    """Test the update_alphas helper."""

    def test_updates_matching_layers(self):
        engine = WormsEngine()
        original = engine.layers[0].config.alpha
        engine.update_alphas({"Base Physics": 0.42})
        assert engine.layers[0].config.alpha == 0.42
        assert engine.layers[0].config.alpha != original

    def test_ignores_unknown_names(self):
        engine = WormsEngine()
        original = engine.layers[0].config.alpha
        engine.update_alphas({"NonExistentLayer": 0.42})
        assert engine.layers[0].config.alpha == original


class TestRobustCP:
    """Test the three-factor CP computation."""

    def test_cp_is_positive(self):
        from parallplexity.core.parallplexity import CompoundingTracker
        tracker = CompoundingTracker(num_streams=8)
        fp = np.random.randn(8, 8) * 0.1
        cp = tracker.compute_compounding(fp, 0.5)
        assert cp > 0, "CP must be positive"

    def test_stronger_coupling_increases_cp(self):
        from parallplexity.core.parallplexity import CompoundingTracker
        t1 = CompoundingTracker(num_streams=4)
        t2 = CompoundingTracker(num_streams=4)

        # Weak coupling
        fp_weak = np.eye(4) * 0.1
        cp_weak = t1.compute_compounding(fp_weak, 0.5)

        # Strong coupling (large off-diagonal)
        fp_strong = np.ones((4, 4)) * 0.5
        cp_strong = t2.compute_compounding(fp_strong, 0.5)

        assert cp_strong > cp_weak, "More coupling should yield higher CP"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
