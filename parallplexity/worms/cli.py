#!/usr/bin/env python3
"""
TranscendPlexity — Compound Worm Integration CLI
==================================================

Runs the full Worms + RustyWorm + Deschooling compound integration
and outputs results to console and JSON.

Usage:
    python -m parallplexity.worms.cli [--steps 500] [--mode convivial] [--intensity 0.7] [--compare]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

from .compound import CompoundWormIntegration
from .deschooling import DeschoolingMode


def main():
    parser = argparse.ArgumentParser(
        description="TranscendPlexity — Compound Worm Integration"
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of integration steps")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "convivial", "institutional"],
                        help="Deschooling mode")
    parser.add_argument("--intensity", type=float, default=0.5,
                        help="Deschooling intensity (0.0 to 1.0)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all three modes and compare")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress step-by-step output")

    args = parser.parse_args()

    mode_map = {
        "baseline": DeschoolingMode.BASELINE,
        "convivial": DeschoolingMode.CONVIVIAL,
        "institutional": DeschoolingMode.INSTITUTIONAL,
    }

    if args.compare:
        cwi = CompoundWormIntegration()
        results = cwi.compare_modes(
            steps=args.steps,
            intensity=args.intensity,
            verbose=not args.quiet,
        )
        output = results
    else:
        cwi = CompoundWormIntegration(
            deschooling_mode=mode_map[args.mode],
            deschooling_intensity=args.intensity,
        )
        summary = cwi.run(
            steps=args.steps,
            verbose=not args.quiet,
        )
        output = summary

        # Export history
        if args.output:
            cwi.export_history(args.output)
            print(f"\nHistory exported to: {args.output}")

    # Save summary
    summary_path = args.output or "worm_integration_summary.json"
    if not args.output:
        with open(summary_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
