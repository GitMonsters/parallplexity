"""
TranscendPlexity Trajectory Visualization
==========================================

Plots GCI, CP, and Compound Confidence trajectories from
exported simulation data (JSON).

Usage:
    python -m parallplexity.worms.visualize full_evaluation_data.json
"""

import json
import sys
from pathlib import Path


def plot_compounding(json_path: str, output_path: str = "trajectories.png"):
    """
    Plot GCI, CP, and CC trajectories from simulation export JSON.

    Supports both the full_evaluation_data.json format (keyed by mode)
    and single-run export format (list of step dicts).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Handle both formats
    if isinstance(data, dict):
        for key, entry in data.items():
            if isinstance(entry, dict) and "timeseries" in entry:
                ts = entry["timeseries"]
                x = range(len(ts["gci"]))
                ax1.plot(x, ts["gci"], label=key, lw=1.4, alpha=0.9)
                ax2.plot(x, ts["cp"], label=key, lw=1.4, alpha=0.9)
                ax3.plot(x, ts["cc"], label=key, lw=1.4, alpha=0.9)
    elif isinstance(data, list):
        # Single run: list of step dicts
        gci = [s.get("gci", 0) for s in data]
        cp = [s.get("cp", 0) for s in data]
        cc = [s.get("compound_confidence", 0) for s in data]
        x = range(len(gci))
        ax1.plot(x, gci, label="run", lw=1.4)
        ax2.plot(x, cp, label="run", lw=1.4)
        ax3.plot(x, cc, label="run", lw=1.4)

    ax1.set_title("GCI Trajectory")
    ax2.set_title("Compounding Parallplexity (CP)")
    ax3.set_title("Compound Confidence")

    for ax in (ax1, ax2, ax3):
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.25, ls="--")
        ax.set_ylabel("Value")

    ax3.set_xlabel("Simulation Step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = sys.argv[2] if len(sys.argv) > 2 else "trajectories.png"
        plot_compounding(sys.argv[1], out)
    else:
        print("Usage: python -m parallplexity.worms.visualize <json_path> [output.png]")
