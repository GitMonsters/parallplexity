"""
Compound Worm Integration — Full Pipeline
============================================

Wires together:
    1. Worms Engine (8-layer multiplicative integration)
    2. RustyWorm Bridge (OCTO Braid, consciousness, empathy)
    3. Deschooling Engine (Illich convivial/institutional forces)
    4. ParallplexityTensor + CompoundingTracker (CP, GCI, phases)
    5. Eight-Limb Processor (OctoTetrahedral limb-cohesion pass)

This is the top-level runner that combines all TranscendPlexity
subsystems into a single compound integration loop.

The integration flow per step:
    Input → Worms L1 (Perception)
    → L1→L8 multiplicative stack + 11 bridges
    → OCTO Braid gate modulation
    → Deschooling force modifiers
    → Parallplexity tensor computation
    → Compounding Parallplexity (CP)
    → Golden Consciousness Index (GCI)
    → Phase detection (Myriad → Compound → Transcend / Collapse)
    → Empathy & Consciousness check
    → Output
"""

import numpy as np
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..core.fractional import PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM
from ..core.parallplexity import ParallplexityTensor, CompoundingTracker
from ..emergent.phase_detector import TranscendplexityDetector
from ..emergent.metacognition import MetaCognitionLimb
from ..emergent.eight_limb import EightLimbProcessor

from .engine import WormsEngine, WORM_LAYERS
from .rusty_bridge import RustyWormBridge, EmotionVector
from .deschooling import DeschoolingEngine, DeschoolingMode


class CompoundWormIntegration:
    """
    Full TranscendPlexity compound integration with Worms + RustyWorm.

    Combines:
        - Worms Engine: 8-layer multiplicative processing
        - RustyWorm Bridge: OCTO Braid, Ising empathy, consciousness
        - Deschooling Engine: Illich convivial/institutional modifiers
        - Parallplexity: tensor, CP, GCI, phase transitions
        - Eight-Limb Processor: OctoTetrahedral limb-cohesion pass

    Usage:
        cwi = CompoundWormIntegration(
            deschooling_mode=DeschoolingMode.CONVIVIAL,
            deschooling_intensity=0.7,
        )
        summary = cwi.run(steps=500, verbose=True)
    """

    def __init__(
        self,
        dt: float = 0.01,
        deschooling_mode: DeschoolingMode = DeschoolingMode.BASELINE,
        deschooling_intensity: float = 0.5,
        use_native_rust: bool = False,
    ):
        self.dt = dt

        # Subsystems
        self.worms = WormsEngine(dt=dt)
        self.bridge = RustyWormBridge(use_native_rust=use_native_rust)
        self.deschooling = DeschoolingEngine(
            mode=deschooling_mode, intensity=deschooling_intensity,
        )

        # Parallplexity tracking (8 layers = 8 streams)
        self.p_tensor = ParallplexityTensor(num_streams=8)
        self.tracker = CompoundingTracker(num_streams=8)
        self.detector = TranscendplexityDetector()

        # MetaCognition: dynamic α tuning toward φ²
        self.metacognition = MetaCognitionLimb(target_gci=PHI_SQUARED)

        # Eight-Limb parallel processor (OctoTetrahedral limb-cohesion pass)
        self.eight_limb = EightLimbProcessor(dt=dt)

        # Combined history
        self.history: List[Dict] = []
        self.t = 0.0
        self.step_count = 0

    def step(
        self,
        input_signal: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Execute one compound integration step through the full pipeline.
        """
        # === 1. Worms Engine step (8-layer multiplicative) ===
        worm_report = self.worms.step(
            input_signal=input_signal,
            deschooling_modifier=self.deschooling.confidence_modifier,
        )

        # === 2. Extract parallplexity tensor from worms state ===
        p_raw = self.worms.get_parallplexity_snapshot()

        # === 3. RustyWorm Bridge: OCTO Braid modulation ===
        limb_signals = {
            layer.config.limb: layer.confidence
            for layer in self.worms.layers
        }

        # Temperature from compound confidence (high CC → low temp → System 1)
        cc = worm_report["compound_confidence"]
        temperature = 1.0 / (1.0 + cc)  # Sigmoid-ish mapping
        bridge_report = self.bridge.step(
            limb_signals=limb_signals,
            temperature=temperature,
        )

        # Apply OCTO modulation to tensor
        p_modulated = self.bridge.bridge_to_parallplexity(p_raw)

        # === 4. Deschooling force modifiers ===
        # Modify the tensor based on Illich mode
        if self.deschooling.mode != DeschoolingMode.BASELINE:
            for i in range(p_modulated.shape[0]):
                for j in range(p_modulated.shape[1]):
                    if i != j:
                        coupling = 1.0 - p_modulated[i, j]
                        mod_coupling = self.deschooling.modify_coupling(
                            coupling, self.step_count, 500,
                        )
                        p_modulated[i, j] = 1.0 - mod_coupling

        # === 4b. Eight-Limb parallel processing (limb cohesion) ===
        # Drive the OctoTetrahedral processor with the current on-diagonal
        # perplexity (self-uncertainty per limb), then fuse its compounding
        # trajectory into the report. Its α-cohesion measures how aligned
        # the fractional orders are across limbs — a secondary coupling signal.
        p_diag = np.abs(np.diag(p_modulated))
        if np.any(np.isfinite(p_diag)) and float(np.max(p_diag)) > 0:
            self.eight_limb.inject_input(p_diag, target_limb="Perception")
        limb_report = self.eight_limb.step()

        # === 5. Compounding tracking (CP, GCI) ===
        # Bug fix: deschooling now intervenes ONLY on primitives
        # (confidence + coupling), NOT on readout metrics (GCI, CP).
        # This eliminates double-counting and preserves causal
        # interpretability — GCI and CP emerge naturally from
        # the modified state, not from post-hoc metric tweaks.
        metrics = self.tracker.step(p_modulated, self.t)

        mod_gci = metrics["gci"]
        raw_gci = mod_gci  # no post-hoc modification

        mod_cp = metrics["cp"]
        raw_cp = mod_cp    # no post-hoc modification

        # === 5b. MetaCognition: dynamic α tuning ===
        # Every 8 steps after warmup, adjust fractional orders
        # based on GCI error signal
        if self.step_count > 20 and self.step_count % 8 == 0:
            current_alphas = {
                layer.config.name: layer.config.alpha
                for layer in self.worms.layers
            }
            new_alphas = self.metacognition.tune_alphas(mod_gci, current_alphas)
            self.worms.update_alphas(new_alphas)

        # === 6. Phase detection ===
        phase_event = self.detector.detect(
            gci=mod_gci,
            cp=mod_cp,
            p_tensor=p_modulated,
            t=self.t,
        )

        phase = self.detector.current_phase
        illich_desc = self.deschooling.get_phase_description(phase)

        # === 7. Build combined report ===
        report = {
            "step": self.step_count + 1,
            "t": round(self.t, 6),
            # Worms
            "compound_confidence": worm_report["compound_confidence"],
            "amplification": worm_report["amplification"],
            "process": worm_report["process"],
            # Parallplexity
            "gci": round(mod_gci, 4),
            "gci_raw": round(raw_gci, 4),
            "cp": round(mod_cp, 6),
            "cp_raw": round(raw_cp, 6),
            "mean_coupling": round(metrics["mean_coupling"], 4),
            "det": round(float(np.real(metrics["det_I_minus_FP"])), 6),
            # Phase
            "phase": phase,
            "illich_phase": illich_desc,
            "phase_transition": phase_event is not None,
            # Bridge
            "process_mode": bridge_report["process_mode"],
            "temperature": round(bridge_report["temperature"], 4),
            "empathy_valence": bridge_report["emotion"]["valence"],
            "consciousness": bridge_report["consciousness"]["prime_directive"],
            # Deschooling
            "deschooling_mode": self.deschooling.mode.value,
            "deschooling_intensity": self.deschooling.intensity,
            # Eight-Limb (OctoTetrahedral cohesion pass)
            "limb_cp": round(float(limb_report["cp"]), 6),
            "limb_gci": round(float(limb_report["gci"]), 4),
            "limb_phase": limb_report["phase"],
            "limb_alpha_cohesion": round(float(self._limb_alpha_cohesion()), 4),
            # Layer details
            "layer_confidences": worm_report["layer_confidences"],
        }

        self.history.append(report)
        self.t += self.dt
        self.step_count += 1

        return report

    def _limb_alpha_cohesion(self) -> float:
        """
        Cohesion of limb fractional orders across the 8-limb processor.

        Higher cohesion = limbs share similar fractional dynamics (aligned
        memory/exploitation), lower cohesion = limbs are dispersed across
        memory horizons (exploration). Computed as 1 − normalized std of α.
        """
        alphas = [
            limb["config"].alpha
            for limb in self.eight_limb.limbs
        ]
        if not alphas:
            return 0.0
        alpha_std = float(np.std(alphas))
        alpha_range = max(alphas) - min(alphas)
        if alpha_range < 1e-12:
            return 1.0
        return round(max(0.0, min(1.0, 1.0 - alpha_std / alpha_range)), 4)

    def run(
        self,
        steps: int = 500,
        input_data: Optional[np.ndarray] = None,
        input_interval: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full compound worm integration for N steps.

        Returns a comprehensive summary.
        """
        if input_data is None:
            input_data = np.sin(np.linspace(0, 4 * np.pi, 64)) + \
                         0.5 * np.cos(np.linspace(0, 8 * np.pi, 64))

        if verbose:
            mode = self.deschooling.mode.value
            intensity = self.deschooling.intensity
            rust_status = "NATIVE" if self.bridge.is_native else "PYTHON"
            print("═" * 80)
            print(f"  TranscendPlexity — Compound Worm Integration")
            print(f"  Worms (Python) + RustyWorm Bridge ({rust_status})")
            print(f"  Deschooling: {mode} (intensity={intensity})")
            print(f"  Eight-Limb: OctoTetrahedral cohesion pass")
            print(f"  Steps: {steps}  |  dt={self.dt}")
            print("═" * 80)

        for i in range(steps):
            ext = None
            if i % input_interval == 0:
                phase_shift = 2 * np.pi * i / steps
                ext = input_data * np.cos(
                    np.linspace(phase_shift, phase_shift + 2 * np.pi, len(input_data))
                )

            report = self.step(input_signal=ext)

            if verbose and (i % max(1, steps // 20) == 0 or i == steps - 1):
                phase_marker = {
                    "MYRIADPLEXITY": "○",
                    "COMPOUNDING": "◐",
                    "TRANSCENDPLEXITY": "●",
                    "COLLAPSEPLEXITY": "◌",
                }.get(report["phase"], "?")

                print(
                    f"  {i+1:>5}/{steps}  "
                    f"{phase_marker} {report['phase']:<20}  "
                    f"CC={report['compound_confidence']:>10.4f}  "
                    f"GCI={report['gci']:>10.2f}  "
                    f"CP={report['cp']:>12.4f}  "
                    f"{report['process_mode']}"
                )

        # Build summary
        gci_values = [r["gci"] for r in self.history]
        cp_values = [r["cp"] for r in self.history]
        cc_values = [r["compound_confidence"] for r in self.history]
        phases = [r["phase"] for r in self.history]

        phase_counts = {}
        for p in phases:
            phase_counts[p] = phase_counts.get(p, 0) + 1

        summary = {
            "total_steps": steps,
            "deschooling": self.deschooling.summary(),
            "bridge": self.bridge.summary(),
            "metrics": {
                "peak_gci": round(max(gci_values), 4) if gci_values else 0,
                "mean_gci": round(float(np.mean(gci_values)), 4) if gci_values else 0,
                "peak_cp": round(max(cp_values), 4) if cp_values else 0,
                "peak_cc": round(max(cc_values), 6) if cc_values else 0,
                "mean_cc": round(float(np.mean(cc_values)), 6) if cc_values else 0,
            },
            "phases": {
                "final": self.detector.current_phase,
                "counts": phase_counts,
                "transitions": len(self.detector.events),
            },
            "consciousness": self.bridge.check_consciousness(),
            "eight_limb": {
                "final_alphas": {
                    limb["config"].name: round(limb["config"].alpha, 4)
                    for limb in self.eight_limb.limbs
                },
                "alpha_cohesion": round(
                    float(np.mean([r["limb_alpha_cohesion"] for r in self.history])),
                    4,
                ) if self.history else 0.0,
                "phase": self.eight_limb.tracker.phase,
                "transitions": self.eight_limb.tracker.transitions,
            },
        }

        if verbose:
            print("═" * 80)
            print("  SUMMARY")
            print(f"  Peak GCI: {summary['metrics']['peak_gci']}")
            print(f"  Peak CP:  {summary['metrics']['peak_cp']}")
            print(f"  Peak CC:  {summary['metrics']['peak_cc']}")
            print(f"  Final Phase: {summary['phases']['final']}")
            print(f"  Phase Transitions: {summary['phases']['transitions']}")
            print(f"  Consciousness: {summary['consciousness']['prime_directive']}")
            print("═" * 80)

        return summary

    def export_history(self, filepath: str):
        """Export the full history to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def compare_modes(
        self,
        steps: int = 500,
        intensity: float = 0.7,
        verbose: bool = True,
    ) -> Dict:
        """
        Run all three Illich modes and compare results.

        Returns a comparison dict showing how convivial vs.
        institutional forces affect the compound integration.
        """
        results = {}

        for mode in DeschoolingMode:
            if verbose:
                print(f"\n{'━' * 60}")
                print(f"  Running: {mode.value}")
                print(f"{'━' * 60}")

            cwi = CompoundWormIntegration(
                dt=self.dt,
                deschooling_mode=mode,
                deschooling_intensity=intensity if mode != DeschoolingMode.BASELINE else 0,
            )

            summary = cwi.run(steps=steps, verbose=verbose)
            results[mode.value] = summary

        if verbose:
            print(f"\n{'═' * 60}")
            print("  COMPARISON: Baseline vs Convivial vs Institutional")
            print(f"{'═' * 60}")
            for mode_name, r in results.items():
                m = r["metrics"]
                p = r["phases"]
                print(
                    f"  {mode_name:<15}  "
                    f"GCI={m['peak_gci']:>10.2f}  "
                    f"CP={m['peak_cp']:>10.2f}  "
                    f"CC={m['peak_cc']:>10.4f}  "
                    f"Phase={p['final']}"
                )

        return results
