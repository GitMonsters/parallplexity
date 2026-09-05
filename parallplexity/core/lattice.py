"""
Lattice Substrate — The Physical Base of the Parallplexity Tensor
==================================================================

This is the *real* implementation of the lattice that the paper's
architecture table describes (previously only documented, never coded):

    core/lattice.py:  discrete substrate on which the parallplexity tensor
    P_ij(t) is defined. Each lattice site carries a complex quantum
    amplitude ψ_{x,y}; info_density = |ψ|² is the local information
    content that parallel streams operate on. Periodic boundary
    conditions mirror the torus (NGVT) topology.

The lattice is a 2-D periodic grid evolved under fractional dynamics
reusing `FractionalEvolutionOperator`, so the site amplitudes carry
power-law memory: each site's value depends on its full history weighted
by (t-τ)^(-α), not just the previous step. The info density field this
produces is what `emergent/spacetime.py` reads to derive curvature and
wormhole signatures.

Conventions
-----------
Site indexing:        (row, col) with row along axis 0, col along axis 1.
Periodic boundaries:  implemented via np.roll — index -1 wraps around,
                      so the substrate is genuinely toroidal (no edges).
Memory:               lattice records its info-density history so the
                      spacetime layer can measure time-domain, non-local
                      correlations (ER=EPR wormhole signatures).
"""

import numpy as np
from typing import List, Optional
from .fractional import CaputoDerivative, CANTOR_GOLDEN_DIM
from .evolution import FractionalEvolutionOperator


class Lattice:
    """
    Discrete 2-D periodic lattice substrate for quantum amplitudes.

    Attributes:
        width, height:  grid size (N = width × height sites)
        amplitudes:     complex amplitude ψ at each site, shape (W, H)
        info_density:   |ψ(x,y)|² — the local information content
        density_history: flattened info-density snapshots over time
        evolver:        fractional evolution operator (memory-carrying)

    Periodic boundary conditions make every site a neighbor of the
    opposite edge, mirroring the torus topology of the NGVT.
    """

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        alpha: float = CANTOR_GOLDEN_DIM,
        dt: float = 0.01,
        coupling_J: float = 0.1,
        max_history: int = 500,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.N = width * height
        self.dt = dt
        self.t = 0.0

        # Fractional evolution operator handles the memory-carrying step.
        self.evolver = FractionalEvolutionOperator(
            alpha=min(alpha, 1.99),
            dt=dt,
            coupling_J=coupling_J,
            max_history=max_history,
        )

        # Initial amplitudes: localized Gaussian bump + small noise.
        rng = np.random.default_rng(seed)
        cx, cy = (width - 1) / 2, (height - 1) / 2
        yy, xx = np.mgrid[0:height, 0:width]
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 3.0)
        noise = 0.1 * rng.standard_normal((height, width))
        psi = gauss * np.exp(1j * noise)
        psi /= np.linalg.norm(psi)
        self.amplitudes = psi

        self.density_history: List[np.ndarray] = []
        self.density_history.append(self.info_density())  # baseline

    def info_density(self) -> np.ndarray:
        """Local information content: info_density(x,y) = |ψ(x,y)|²."""
        return np.abs(self.amplitudes) ** 2

    def mean_density(self) -> float:
        return float(np.mean(self.info_density()))

    def drive(self, signal: float) -> None:
        """
        Inject a uniform external information flow into the lattice.

        Scales the amplitude envelope by a small factor around the given
        signal level. Note: uniform driving only shifts overall density
        (a global scale change, not spatial structure) — for spatially
        structured drives (per-limb regions), use `drive_pattern`.
        """
        self.amplitudes = self.amplitudes * (1.0 + 0.1 * float(np.clip(signal, -1.0, 1.0)))
        self._renormalize()

    def drive_pattern(self, pattern: np.ndarray) -> None:
        """
        Inject a spatially-structured information flow.

        `pattern` is a 2-D field over the lattice (e.g., from limb states).
        Each site's information content is scaled by (1 + g·pattern), so
        regions receiving more input grow locally — this is what lets
        spatially-separated regions develop *independent* dynamics that
        the spacetime layer can then detect synchronizing non-locally.
        """
        if pattern.shape != self.amplitudes.shape:
            raise ValueError(
                f"pattern shape {pattern.shape} != lattice {self.amplitudes.shape}"
            )
        self.amplitudes = self.amplitudes * (1.0 + 0.1 * pattern)
        self._renormalize()

    def _renormalize(self) -> None:
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-12:
            self.amplitudes /= norm

    def evolve(self, nonlinearity: float = 0.02) -> np.ndarray:
        """
        Advance the lattice one fractional time step.

        Uses the memory-carrying fractional evolution operator, so each
        site integrates its own power-law history (deep compounding at
        sites that receive repeated activation).
        """
        self.amplitudes = self.evolver.apply_fractional_evolution(
            self.amplitudes,
            nonlinearity=nonlinearity,
        )
        self.t += self.dt
        self.density_history.append(self.info_density())
        return self.amplitudes

    def compute_caputo_memory(self) -> np.ndarray:
        """
        Caputo fractional derivative of the site-amplitude history.

        Represents how strongly the lattice is currently *remembering*
        its past — the fractional memory term of the state evolution.
        """
        if self.evolver.history.length < 3:
            return np.zeros_like(self.amplitudes)
        deriv = self.evolver.caputo.compute_from_history(
            self.evolver.history.states,
            self.evolver.history.timestamps,
            self.t,
        )
        if deriv.shape[0] == self.N:
            return deriv.reshape(self.amplitudes.shape)
        return np.zeros_like(self.amplitudes)

    def reset(self) -> None:
        """Reset lattice amplitudes and history."""
        self.evolver.reset()
        self.t = 0.0
        self.density_history.clear()
        self.density_history.append(self.info_density())