"""
Emergent Spacetime — Geometry That Emerges From Information Density
====================================================================

This is the *real* implementation of the emergent geometry that the
paper's architecture table describes (previously documented only):

    emergent/spacetime.py:  emergent geometry of compounding parallplexity.
    Curvature ∇²(info_density) measures how information varies across the
    lattice; regions of high curvature are where compounding is most
    active. The emergent metric g_μν ∝ info_density / max(info_density)
    defines the effective geometry parallel streams navigate. The
    `measure_wormhole_signature()` method detects non-local information
    correlations — the ER=EPR analog of cross-stream parallplexity
    reduction at a distance.

The class is read-only with respect to the lattice: it observes the
info-density field (and its time history) and produces geometric
readouts the compound integration can act on (e.g., boosting coupling
between distant lattice sites — a geometric shortcut).

Wormholes are measured at the resolution of the limb regions the lattice
substrate is divided into (rows × cols). Each region's info-density
fluctuations (first time-differences) form a time series. A wormhole is
a **non-adjacent** region pair whose fluctuation series is more
correlated than the expected *local* pairing (adjacent regions). This is
the ER=EPR analog: distant regions sharing information that locality
cannot explain.
"""

import numpy as np
from typing import Dict, Tuple
from ..core.lattice import Lattice


class EmergentSpacetime:
    """
    Emergent geometry derived from a Lattice's info-density field.

    Parameters:
        lattice:              the substrate this spacetime describes
        region_rows:          number of limb regions along lattice rows
        region_cols:          number of limb regions along lattice cols
        correlation_threshold: wormhole requires |corr| to exceed the local
                               baseline by at least this amount
    """

    def __init__(
        self,
        lattice: Lattice,
        region_rows: int = 2,
        region_cols: int = 4,
        correlation_threshold: float = 0.5,
    ):
        self.lattice = lattice
        self.region_rows = region_rows
        self.region_cols = region_cols
        self.correlation_threshold = correlation_threshold
        self.n_regions = region_rows * region_cols

    # ----------------------------------------------------------
    # Curvature: ∇²(info_density) with periodic boundaries
    # ----------------------------------------------------------
    def curvature(self) -> np.ndarray:
        """Discrete Laplacian of the info-density field (periodic BCs)."""
        d = self.lattice.info_density()
        lap = (
            np.roll(d, 1, axis=0) + np.roll(d, -1, axis=0)
            + np.roll(d, 1, axis=1) + np.roll(d, -1, axis=1)
            - 4.0 * d
        )
        return lap

    def mean_curvature(self) -> float:
        """Mean absolute curvature — how 'bumpy' the information substrate is."""
        return float(np.mean(np.abs(self.curvature())))

    def peak_curvature(self) -> float:
        """Peak absolute curvature — strongest compounding site."""
        return float(np.max(np.abs(self.curvature())))

    # ----------------------------------------------------------
    # Emergent metric: g_μν ∝ info_density / max(info_density)
    # ----------------------------------------------------------
    def emergent_metric(self) -> np.ndarray:
        """Effective geometry normalized to [0, 1]."""
        d = self.lattice.info_density()
        denom = float(np.max(d))
        if denom < 1e-12:
            return np.zeros_like(d)
        return d / denom

    def metric_mean(self) -> float:
        """Mean emergent metric value — overall 'height' of the geometry."""
        return float(np.mean(self.emergent_metric()))

    # ----------------------------------------------------------
    # Region structure (limb regions on the toroidal substrate)
    # ----------------------------------------------------------
    def _region_slices(self) -> Tuple[slice, slice]:
        """(row_slice, col_slice) tiling the lattice into limb regions."""
        h, w = self.lattice.height, self.lattice.width
        row_size = h // self.region_rows
        col_size = w // self.region_cols
        return slice(0, self.region_rows * row_size), slice(0, self.region_cols * col_size)

    def _region_adjacency(self) -> np.ndarray:
        """
        Boolean matrix: region pairs considered *adjacent* (local baseline).

        Adjacent means sharing a border (4-neighborhood) in the region
        grid. Diagonal is excluded (a region is not its own neighbor).
        """
        adj = np.zeros((self.n_regions, self.n_regions), dtype=bool)
        for r in range(self.region_rows):
            for c in range(self.region_cols):
                i = r * self.region_cols + c
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.region_rows and 0 <= cc < self.region_cols:
                        j = rr * self.region_cols + cc
                        adj[i, j] = True
        return adj

    def _region_fluctuations(self) -> np.ndarray:
        """
        Per-region fluctuation time series, shape (T, n_regions).

        Each region's info-density is spatially averaged, then the first
        time-difference produces a fluctuation series — the shared signal
        a region carries over time.
        """
        history = self.lattice.density_history
        row_slice, col_slice = self._region_slices()
        T = len(history)
        # (T, H, W) -> (T, region_rows, region_cols) spatial means
        region_means = np.empty((T, self.region_rows, self.region_cols))
        h, w = self.lattice.height, self.lattice.width
        rs, cs = row_slice, col_slice
        block_r, block_c = (rs.stop - rs.start) // self.region_rows, (cs.stop - cs.start) // self.region_cols
        for i in range(T):
            d = history[i][rs, cs]
            for r in range(self.region_rows):
                for c in range(self.region_cols):
                    region_means[i, r, c] = d[
                        r * block_r:(r + 1) * block_r,
                        c * block_c:(c + 1) * block_c,
                    ].mean()
        flats = region_means.reshape(T, -1)
        if T < 3:
            return np.zeros((1, self.n_regions))
        return np.diff(flats, axis=0)

    def measure_wormhole_signature(self) -> Dict:
        """
        Detect non-local information correlations between limb regions.

        Computes the correlation of each region's fluctuation series with
        every other. The global median correlation (all pairs) is the
        baseline — the level of mutual fluctuation any two regions share
        by chance. A wormhole is a **non-adjacent** region pair whose
        correlation exceeds that baseline the most: distant regions
        sharing fluctuations locality cannot explain (ER=EPR).

        Returns:
            dict with `signature` (relative excess of the strongest
            cross-region correlation over the global baseline, in [0,1]),
            `baseline` (median correlation across all pairs),
            `wormholes` (non-adjacent pairs above the neighborhood
            baseline + correlation_threshold), and `correlation_matrix`.
        """
        fluctuations = self._region_fluctuations()
        T = fluctuations.shape[0]
        if T < 3:
            return {
                "signature": 0.0,
                "baseline": 0.0,
                "wormholes": 0,
                "correlation_matrix": np.zeros((self.n_regions, self.n_regions)),
            }

        # Guard: if all regions are essentially static (variance near zero),
        # the correlation matrix is degenerate numerical noise — nothing
        # meaningful to correlate.
        region_sigma = fluctuations.std(axis=0)
        if float(np.max(region_sigma)) < 1e-12:
            return {
                "signature": 0.0,
                "baseline": 0.0,
                "wormholes": 0,
                "correlation_matrix": np.zeros((self.n_regions, self.n_regions)),
            }

        # Exclude low-variance regions (they produce 0/0 correlations)
        active = region_sigma > 1e-12
        active_idx = np.where(active)[0]
        if np.count_nonzero(active) < 4:
            return {
                "signature": 0.0,
                "baseline": 0.0,
                "wormholes": 0,
                "correlation_matrix": np.zeros((self.n_regions, self.n_regions)),
            }
        act_fluct = fluctuations[:, active]
        mu = act_fluct.mean(axis=0, keepdims=True)
        sigma = act_fluct.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        z = (act_fluct - mu) / sigma

        corr_active = (z.T @ z) / max(T - 1, 1)
        corr_active = np.clip(corr_active, -1.0, 1.0)
        abs_corr = np.abs(corr_active)

        adj = self._region_adjacency()[np.ix_(active_idx, active_idx)]
        n = len(active_idx)
        eye = np.eye(n, dtype=bool)
        offdiag = ~eye
        nonlocal_pairs = (~adj) & (~eye)

        # Baseline: median correlation across all off-diagonal region pairs.
        if np.any(offdiag):
            baseline = float(np.median(abs_corr[offdiag]))
        else:
            baseline = 0.0

        if np.any(nonlocal_pairs):
            max_nonlocal = float(np.max(abs_corr[nonlocal_pairs]))
        else:
            max_nonlocal = 0.0

        # Excess of the strongest non-adjacent pair over the global baseline.
        # A wormhole only counts when that excess beats the noise margin
        # (`correlation_threshold`): without it, chance-level distant
        # correlations would be misread as ER=EPR shortcuts.
        excess = max(0.0, max_nonlocal - baseline)
        denom = 1.0 - baseline
        if self.correlation_threshold < denom - 1e-12:
            if excess <= self.correlation_threshold:
                signature = 0.0
            else:
                signature = float(np.clip(
                    (excess - self.correlation_threshold)
                    / (denom - self.correlation_threshold),
                    0.0, 1.0,
                ))
        else:
            # Threshold consumes the whole [baseline, 1] range: only fire on
            # effectively perfect distant correlation.
            signature = 0.0 if excess <= self.correlation_threshold else 1.0

        # Wormholes: non-adjacent pairs beating the baseline by a threshold
        wormholes = int(np.sum(nonlocal_pairs &
                               (abs_corr > baseline + self.correlation_threshold)))

        return {
            "signature": round(float(np.clip(signature, 0.0, 1.0)), 6),
            "baseline": round(float(baseline), 6),
            "wormholes": wormholes,
            "correlation_matrix": corr_active,
        }

    def step(self, drive_signal: float = 0.0,
             drive_pattern: np.ndarray = None) -> Dict:
        """
        Advance lattice under fractional dynamics and report spacetime metrics.

        1. Drive the lattice (scalar coupling and/or spatial pattern)
        2. Propagate one fractional step (memory-carrying evolution)
        3. Measure curvature, metric, and wormhole signature
        """
        if drive_pattern is not None:
            self.lattice.drive_pattern(drive_pattern)
        if drive_signal != 0.0:
            self.lattice.drive(drive_signal)
        self.lattice.evolve()
        ws = self.measure_wormhole_signature()
        return {
            "mean_density": round(self.lattice.mean_density(), 6),
            "mean_curvature": round(self.mean_curvature(), 6),
            "peak_curvature": round(self.peak_curvature(), 6),
            "metric_mean": round(self.metric_mean(), 6),
            "wormhole_signature": round(ws["signature"], 6),
            "wormhole_baseline": ws["baseline"],
            "wormholes": ws["wormholes"],
        }