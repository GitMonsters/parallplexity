#!/usr/bin/env python3
"""
Compounding Fractional Parallplexity — Demo Runner
====================================================

Runs the full 8-limb compound integration and saves visualization data
for analysis. This is the canonical demo of the framework.

Usage:
    python parallplexity/examples/demo_compound_integration.py [steps]

Default: 500 steps
"""

import sys
import os
import json
import time
import numpy as np

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parallplexity import (
    EightLimbProcessor, LimbConfig, DEFAULT_LIMBS,
    TranscendplexityDetector, PhaseEvent,
    PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM,
)


def run_demo(steps: int = 500, seed: int = 42):
    """Run the full compound integration demo."""

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  COMPOUNDING FRACTIONAL PARALLPLEXITY — COMPOUND INTEGRATION   ║")
    print("║                                                                ║")
    print(f"║  Steps: {steps:<5}  Seed: {seed:<5}  Limbs: 8                          ║")
    print(f"║  φ = {PHI:.6f}   φ² = {PHI_SQUARED:.6f}   α_CG = {CANTOR_GOLDEN_DIM:.6f}       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    np.random.seed(seed)
    t_start = time.time()

    # Initialize processor
    processor = EightLimbProcessor(dt=0.01)
    detector = TranscendplexityDetector(min_sustained_steps=5)

    # Multi-frequency structured input
    input_data = (
        np.sin(np.linspace(0, 4 * np.pi, 64)) +
        0.5 * np.cos(np.linspace(0, 8 * np.pi, 64)) +
        0.3 * np.sin(np.linspace(0, 16 * np.pi, 64)) +
        0.2 * np.cos(np.linspace(0, 32 * np.pi, 64))
    )

    # Data collection
    timeline = {
        "steps": [],
        "gci": [],
        "cp": [],
        "coupling": [],
        "phase": [],
        "limb_alphas": [],
        "limb_norms": [],
        "det_values": [],
        "phase_events": [],
    }

    print(f"\n{'─' * 70}")
    print(f"  {'Step':>6}  {'Phase':<20}  {'CP':>12}  {'GCI':>10}  {'Coupling':>10}  {'det(I-FP)':>12}")
    print(f"{'─' * 70}")

    for i in range(steps):
        # Inject evolving input every 10 steps
        if i % 10 == 0:
            phase = 2 * np.pi * i / steps
            varied = input_data * np.cos(
                np.linspace(phase, phase + 2 * np.pi, len(input_data))
            )
            # Alternate injection limbs
            targets = ["Perception", "Language", "Spatial", "Memory"]
            target = targets[(i // 10) % len(targets)]
            processor.inject_input(varied, target_limb=target)

        report = processor.step()

        # Phase detection
        P = processor.p_tensor.tensor
        event = detector.detect(report["gci"], report["cp"], P, t=processor.t)
        if event:
            timeline["phase_events"].append({
                "step": i + 1,
                "t": event.time,
                "from": event.from_phase,
                "to": event.to_phase,
                "gci": event.gci,
                "trigger": event.trigger,
            })

        # Record data
        timeline["steps"].append(i + 1)
        timeline["gci"].append(float(report["gci"]))
        timeline["cp"].append(float(report["cp"]))
        timeline["coupling"].append(float(report["mean_coupling"]))
        timeline["phase"].append(report["phase"])
        timeline["limb_alphas"].append({k: float(v) for k, v in report["limb_alphas"].items()})
        timeline["limb_norms"].append({k: float(v) for k, v in report["limb_norms"].items()})
        det_val = report.get("det_I_minus_FP", 0)
        timeline["det_values"].append(float(np.real(det_val)))

        # Progress output
        if i % (steps // 20) == 0 or i == steps - 1:
            phase_marker = {
                "MYRIADPLEXITY": "○", "COMPOUNDING": "◐",
                "TRANSCENDPLEXITY": "●", "COLLAPSEPLEXITY": "◌",
            }.get(report["phase"], "?")
            print(
                f"  {i+1:>5}  "
                f"{phase_marker} {report['phase']:<20}  "
                f"{report['cp']:>12.4f}  "
                f"{report['gci']:>10.4f}  "
                f"{report['mean_coupling']:>10.4f}  "
                f"{report.get('det_I_minus_FP', 0):>12.6f}"
            )

    elapsed = time.time() - t_start

    # === Final Report ===
    tracker_summary = processor.tracker.summary()
    detector_summary = detector.summary()

    print(f"\n{'═' * 70}")
    print(f"  COMPOUND INTEGRATION COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Runtime:              {elapsed:.2f}s ({steps/elapsed:.0f} steps/sec)")
    print(f"  Steps:                {steps}")
    print(f"  Final phase:          {tracker_summary['final_phase']}")
    print(f"  Peak CP:              {tracker_summary['peak_cp']:.4f}")
    print(f"  Peak GCI:             {tracker_summary['peak_gci']:.4f}")
    print(f"  φ² threshold:         {PHI_SQUARED:.4f}")
    print(f"  GCI reached φ²:       {tracker_summary['peak_gci'] >= PHI_SQUARED}")
    print(f"  Transcendplexity:     {'●  YES' if tracker_summary.get('reached_transcendplexity') or detector_summary.get('reached_transcendplexity') else '○  Not yet'}")

    # Phase transitions
    print(f"\n  Phase Transitions ({len(timeline['phase_events'])}):")
    for evt in timeline["phase_events"]:
        print(f"    Step {evt['step']:>4} (t={evt['t']:.3f}): {evt['from']} → {evt['to']}")

    # Alpha evolution
    print(f"\n  Final Limb Fractional Orders:")
    initial_alphas = {cfg.name: cfg.alpha for cfg in DEFAULT_LIMBS}
    final_alphas = timeline["limb_alphas"][-1]
    for name, alpha in final_alphas.items():
        init_a = initial_alphas.get(name, alpha)
        delta = alpha - init_a
        bar_len = int(abs(delta) * 500)
        bar = ("+" * min(bar_len, 30)) if delta > 0 else ("-" * min(bar_len, 30))
        print(f"    {name:<15} α = {alpha:.4f}  (Δ = {delta:+.4f})  {bar}")

    # Compounding analysis
    gci_arr = np.array(timeline["gci"])
    cp_arr = np.array(timeline["cp"])

    print(f"\n  Compounding Dynamics:")
    print(f"    GCI  — mean: {gci_arr.mean():.4f}, std: {gci_arr.std():.4f}, "
          f"min: {gci_arr.min():.4f}, max: {gci_arr.max():.4f}")
    print(f"    CP   — mean: {cp_arr.mean():.4f}, std: {cp_arr.std():.4f}, "
          f"min: {cp_arr.min():.4f}, max: {cp_arr.max():.4f}")

    # Save visualization data
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "compound_integration_data.json"
    )

    # Convert for JSON serialization
    save_data = {
        "metadata": {
            "steps": steps,
            "seed": seed,
            "runtime_seconds": elapsed,
            "phi": float(PHI),
            "phi_squared": float(PHI_SQUARED),
            "cantor_golden_dim": float(CANTOR_GOLDEN_DIM),
        },
        "timeline": {
            "steps": timeline["steps"],
            "gci": timeline["gci"],
            "cp": timeline["cp"],
            "coupling": timeline["coupling"],
            "phase": timeline["phase"],
            "det_values": timeline["det_values"],
        },
        "limb_alphas_over_time": timeline["limb_alphas"],
        "phase_events": timeline["phase_events"],
        "tracker_summary": {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in tracker_summary.items()
            if k != "transitions"
        },
        "detector_summary": detector_summary,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Visualization data saved to: {output_path}")
    print(f"{'═' * 70}")

    return save_data


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_demo(steps=n_steps)
