"""
Performance Benchmark — Compounding Fractional Parallplexity
=============================================================

Benchmarks the compute hot paths and the creeping-slowdown behavior:

  1. History-depth scaling of the fractional evolution memory term
     (per-step cost as the Caputo history buffer fills toward max_history).
  2. EightLimbProcessor end-to-end throughput across a 1000-step run,
     measuring steps/sec in early vs late windows (quantifies the drift).
  3. Full CompoundWormIntegration pipeline (all 8 subsystem mixted) throughput.

Run:
    python parallplexity/examples/benchmark_performance.py [--steps N] [--seed S]
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from parallplexity.core.evolution import FractionalEvolutionOperator
from parallplexity.emergent.eight_limb import EightLimbProcessor
from parallplexity.worms.compound import CompoundWormIntegration


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def bench_history_scaling(max_depth: int = 300, dim: int = 64, alpha: float = 0.7):
    """Cost per evolution step vs how full the history buffer is."""
    evolver = FractionalEvolutionOperator(alpha=alpha, dt=0.01, coupling_J=0.1, max_history=max_depth)
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state /= np.linalg.norm(state)

    probe_every = 25
    rows = []
    for step in range(1, max_depth + 1):
        # 50 trials at each probe depth for stable timing
        trials = 3
        t0 = time.perf_counter()
        for _ in range(trials):
            state = evolver.apply_fractional_evolution(state, nonlinearity=0.05)
        elapsed = time.perf_counter() - t0
        if step % probe_every == 0:
            rows.append({
                "history_length": evolver.history.length,
                "dim": dim,
                "per_step_ms": round(1000.0 * elapsed / trials, 4),
                "steps_per_sec": round(trials / elapsed, 2),
            })
    return rows


def bench_dimension_scaling(dims=(64, 128, 256, 512), max_depth: int = 300):
    """Steady-state memory-term cost vs state dimensionality at max history."""
    rows = []
    for dim in dims:
        evolver = FractionalEvolutionOperator(alpha=0.7, dt=0.01, coupling_J=0.1, max_history=max_depth)
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        # Fill history to capacity first (steady state)
        for _ in range(max_depth):
            state = evolver.apply_fractional_evolution(state, nonlinearity=0.05)
        trials = 5
        t0 = time.perf_counter()
        for _ in range(trials):
            state = evolver.apply_fractional_evolution(state, nonlinearity=0.05)
        elapsed = time.perf_counter() - t0
        rows.append({
            "dim": dim,
            "history_length": evolver.history.length,
            "per_step_ms": round(1000.0 * elapsed / trials, 4),
            "steps_per_sec": round(trials / elapsed, 2),
            "per_dim_us": round(1e6 * elapsed / (trials * dim), 2),
        })
    return rows


def bench_eight_limb(steps: int):
    """Throughput of the 8-limb processor over a full run, binned early/late."""
    np.random.seed(42)
    p = EightLimbProcessor(dt=0.01)
    input_data = (
        np.sin(np.linspace(0, 4 * np.pi, 64))
        + 0.5 * np.cos(np.linspace(0, 8 * np.pi, 64))
    )

    t0 = time.perf_counter()
    for i in range(steps):
        if i % 10 == 0:
            varied = input_data * np.cos(np.linspace(0, 2 * np.pi, 64))
            p.inject_input(varied)
        report = p.step()
    total = time.perf_counter() - t0

    # Independent early vs late windows (25 warmup/drain, 25 timed).
    early_start = time.perf_counter()
    for _ in range(25):
        p.step()
    early_wall = time.perf_counter() - early_start

    late_start = time.perf_counter()
    for _ in range(25):
        p.step()
    late_wall = time.perf_counter() - late_start

    return {
        "steps": steps,
        "total_sec": round(total, 3),
        "mean_steps_per_sec": round(steps / total, 2),
        "early_steps_per_sec": round(25 / early_wall, 2),
        "late_steps_per_sec": round(25 / late_wall, 2),
        "late_vs_early_slowdown_x": round(late_wall / early_wall, 2),
        "final_phase": report["phase"],
        "final_cp": float(report["cp"]),
    }


def bench_compound(steps: int):
    """Throughput of the full 8-subsystem compound worm integration."""
    np.random.seed(137)
    cwi = CompoundWormIntegration(dt=0.01, use_native_rust=False)

    t0 = time.perf_counter()
    for _ in range(steps):
        cwi.step()
    total = time.perf_counter() - t0

    return {
        "steps": steps,
        "total_sec": round(total, 3),
        "mean_steps_per_sec": round(steps / total, 2),
        "final_cp": round(cwi.history[-1]["cp"], 6),
        "final_limb_cohesion": cwi.history[-1]["limb_alpha_cohesion"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 78)
    print("  PARALLPLEXITY PERFORMANCE BENCHMARK")
    print(f"  steps={args.steps}  seed={args.seed}")
    print("=" * 78)

    # 1. History scaling
    print("\n[1] HISTORY-DEPTH SCALING — fractional memory term (α=0.7, dim=64)")
    print(f"    {'hist_len':>8} {'per_step_ms':>12} {'steps/sec':>10}")
    scaling = bench_history_scaling()
    for row in scaling:
        print(f"    {row['history_length']:>8} {row['per_step_ms']:>12.4f} {row['steps_per_sec']:>10.2f}")

    # 1b. Dimension scaling at full history
    print("\n[1b] DIMENSION SCALING at capacity (hist=300)")
    print(f"    {'dim':>5} {'per_step_ms':>12} {'steps/sec':>10} {'us/dim':>8}")
    dim_scale = bench_dimension_scaling()
    for row in dim_scale:
        print(f"    {row['dim']:>5} {row['per_step_ms']:>12.4f} {row['steps_per_sec']:>10.2f} {row['per_dim_us']:>8.2f}")

    # 2. Eight-limb throughput
    print(f"\n[2] EIGHT-LIMB PROCESSOR — {args.steps} steps")
    limb = bench_eight_limb(args.steps)
    for k, v in limb.items():
        print(f"    {k:<24} {v}")

    # 3. Compound worm pipeline
    print(f"\n[3] COMPOUND WORM INTEGRATION (8 subsystems) — {args.steps} steps")
    compound = bench_compound(args.steps)
    for k, v in compound.items():
        print(f"    {k:<24} {v}")

    report = {
        "history_scaling": scaling,
        "dimension_scaling": dim_scale,
        "eight_limb": limb,
        "compound_worm": compound,
    }

    out_path = "benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Report saved to: {out_path}")

    print("\n" + "=" * 78)
    print("  DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()