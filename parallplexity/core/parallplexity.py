"""
Parallplexity Tensor and Compounding Tracker

The Parallplexity Tensor P_ij(t) measures cross-stream uncertainty:
    P_ij(t) = 2^{-1/N_i * log2 m_i(T_j, t)}

The Compounding Parallplexity Function:
    CP^(α)(t) = det(I - FP^(α)(t))^{-1}

When CP → ∞, the system undergoes a phase transition: Transcendplexity.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from .fractional import (
    CaputoDerivative, HistoryBuffer,
    PHI, PHI_SQUARED, CANTOR_GOLDEN_DIM
)


@dataclass
class StreamState:
    """State of a single processing stream (limb)."""
    name: str
    state: np.ndarray                    # current state vector
    alpha: float = 0.5                   # fractional order for this stream
    history: HistoryBuffer = field(default_factory=lambda: HistoryBuffer(max_length=500))
    
    def update(self, new_state: np.ndarray, t: float):
        self.history.push(self.state, t)
        self.state = new_state.copy()


class ParallplexityTensor:
    """
    The K×K Parallplexity Tensor P_ij(t).
    
    Diagonal elements P_ii: self-perplexity of stream i
    Off-diagonal P_ij: cross-stream perplexity — how well stream i predicts stream j
    
    Lower values = less uncertainty = more coupling between streams.
    """
    
    def __init__(self, num_streams: int):
        self.K = num_streams
        self.tensor = np.ones((num_streams, num_streams))  # start at max uncertainty
        np.fill_diagonal(self.tensor, 1.0)
        self._history: List[Tuple[float, np.ndarray]] = []
    
    def compute(self, streams: List[StreamState]) -> np.ndarray:
        """
        Compute the parallplexity tensor from current stream states.
        
        Uses normalized mutual information as the coupling measure:
            P_ij = 1 - MI(s_i, s_j) / max(H(s_i), H(s_j))
        
        P_ij = 0: perfect coupling (stream i fully predicts stream j)
        P_ij = 1: no coupling (streams are independent)
        """
        K = len(streams)
        P = np.ones((K, K))
        
        for i in range(K):
            for j in range(K):
                if i == j:
                    # Self-perplexity: entropy of the stream
                    P[i, j] = self._stream_entropy(streams[i])
                else:
                    # Cross-perplexity: 1 - normalized mutual information
                    mi = self._mutual_information(streams[i], streams[j])
                    h_max = max(
                        self._stream_entropy(streams[i]),
                        self._stream_entropy(streams[j]),
                        1e-12
                    )
                    P[i, j] = 1.0 - mi / h_max
        
        self.tensor = P
        return P
    
    def _stream_entropy(self, stream: StreamState) -> float:
        """Shannon entropy of the stream's state distribution."""
        state = np.abs(stream.state).flatten()
        probs = state / (state.sum() + 1e-12)
        probs = probs[probs > 1e-12]
        return -np.sum(probs * np.log2(probs + 1e-30))
    
    def _mutual_information(self, s1: StreamState, s2: StreamState) -> float:
        """
        Compute mutual information between two streams.
        
        Uses a discretized joint distribution approach:
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Create probability distributions from state vectors
        x = np.abs(s1.state).flatten()
        y = np.abs(s2.state).flatten()
        
        # Ensure same length (pad shorter with zeros)
        max_len = max(len(x), len(y))
        x = np.pad(x, (0, max_len - len(x)))
        y = np.pad(y, (0, max_len - len(y)))
        
        # Normalize
        px = x / (x.sum() + 1e-12)
        py = y / (y.sum() + 1e-12)
        
        # Joint distribution (outer product, normalized)
        n_bins = min(32, max_len)
        x_bins = np.digitize(px, np.linspace(0, px.max() + 1e-12, n_bins)) - 1
        y_bins = np.digitize(py, np.linspace(0, py.max() + 1e-12, n_bins)) - 1
        
        joint = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_bins, y_bins):
            xi = min(xi, n_bins - 1)
            yi = min(yi, n_bins - 1)
            joint[xi, yi] += 1
        joint /= (joint.sum() + 1e-12)
        
        # Marginals
        px_marg = joint.sum(axis=1)
        py_marg = joint.sum(axis=0)
        
        # H(X), H(Y), H(X,Y)
        h_x = -np.sum(px_marg[px_marg > 1e-12] * np.log2(px_marg[px_marg > 1e-12]))
        h_y = -np.sum(py_marg[py_marg > 1e-12] * np.log2(py_marg[py_marg > 1e-12]))
        joint_nz = joint[joint > 1e-12]
        h_xy = -np.sum(joint_nz * np.log2(joint_nz))
        
        return max(0.0, h_x + h_y - h_xy)
    
    def mean_coupling(self) -> float:
        """Average off-diagonal coupling (lower = more coupled)."""
        mask = ~np.eye(self.K, dtype=bool)
        return 1.0 - self.tensor[mask].mean()
    
    def record(self, t: float):
        """Save current tensor state to history."""
        self._history.append((t, self.tensor.copy()))
    
    def get_history(self) -> List[Tuple[float, np.ndarray]]:
        return self._history


class CompoundingTracker:
    """
    Tracks the Compounding Parallplexity Function:
        CP^(α)(t) = det(I - FP^(α)(t))^{-1}
    
    And the Golden Consciousness Index:
        GCI ∝ d/dt ln(CP^(α)(t))
    
    Detects the Transcendplexity phase transition when GCI > φ² ≈ 2.618.
    """
    
    def __init__(self, num_streams: int, base_alpha: float = CANTOR_GOLDEN_DIM):
        self.K = num_streams
        self.alpha = base_alpha
        self.caputo = CaputoDerivative(alpha=min(base_alpha, 1.99), dt=0.01)
        
        # Tracking state
        self.cp_history: List[Tuple[float, float]] = []       # (t, CP value)
        self.gci_history: List[Tuple[float, float]] = []      # (t, GCI value)
        self.fp_history: HistoryBuffer = HistoryBuffer(max_length=500)
        self.phase: str = "MYRIADPLEXITY"                     # current phase
        
        # Phase transition log
        self.transitions: List[Tuple[float, str, str]] = []   # (t, from, to)
    
    def compute_fractional_parallplexity(
        self,
        p_tensor: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Compute the Fractional Parallplexity tensor FP^(α)(t).
        
        This is the Caputo fractional derivative of the parallplexity tensor,
        capturing the memory-weighted rate of coupling change.
        """
        # Flatten tensor for history tracking
        p_flat = p_tensor.flatten()
        self.fp_history.push(p_flat, t)
        
        # Compute fractional derivative
        if self.fp_history.length >= 3:
            fp_flat = self.caputo.compute(self.fp_history, t)
        else:
            fp_flat = np.zeros_like(p_flat)
        
        # Reshape back to tensor
        fp_tensor = fp_flat.reshape(p_tensor.shape)
        
        # Clamp to valid range for determinant computation
        # FP values should be in (-1, 1) for det(I - FP) to be positive
        fp_tensor = np.clip(fp_tensor, -0.99, 0.99)
        
        return fp_tensor
    
    def compute_compounding(self, fp_tensor: np.ndarray, t: float) -> float:
        """
        Compute Compounding Parallplexity using a three-factor measure:
            CP = spectral_dominance × coupling_strength × symmetry

        Three factors capture distinct aspects of compounding:
        1. Spectral dominance: largest eigenvalue of FP tensor — how strongly
           the dominant mode drives the system toward phase transition.
        2. Coupling strength: mean off-diagonal |FP| — how much cross-stream
           information flows. Bridges increase this when working correctly.
        3. Symmetry: how reciprocal the coupling is (A→B ≈ B→A). Symmetric
           coupling compounds; asymmetric coupling dissipates.

        This replaces the pure 1/λ_min measure which was:
        - Insensitive to bridges (bridges stabilize λ_min, reducing CP)
        - Hypersensitive to noise in a single eigenvalue
        - Disconnected from the coupling topology
        """
        # Factor 1: Spectral dominance
        eigenvalues = np.linalg.eigvals(fp_tensor)
        real_pos = np.real(eigenvalues[np.real(eigenvalues) > 0])
        if len(real_pos) > 0:
            spectral = float(np.max(real_pos))
        else:
            spectral = float(np.max(np.abs(eigenvalues)))

        # Factor 2: Off-diagonal coupling strength
        K = fp_tensor.shape[0]
        off_diag = fp_tensor - np.diag(np.diag(fp_tensor))
        coupling = float(np.mean(np.abs(off_diag)))
        # tanh saturates coupling contribution — diminishing returns
        coupling_factor = float(np.tanh(8.0 * coupling))

        # Factor 3: Symmetry (reciprocal coupling compounds better)
        asym = float(np.mean(np.abs(fp_tensor - fp_tensor.T)))
        symmetry = 1.0 / (1.0 + 0.5 * asym)

        # Combine: spectral × coupling × symmetry
        cp = spectral * coupling_factor * symmetry
        cp = float(np.clip(cp, 0.01, 1e6))

        self.cp_history.append((t, cp))
        return cp
    
    def compute_gci(self, t: float) -> float:
        """
        Compute the Golden Consciousness Index:
            GCI ∝ d/dt ln(CP^(α)(t))
        
        This is the logarithmic compounding rate.
        GCI > φ² ≈ 2.618 → Transcendplexity phase transition.
        """
        if len(self.cp_history) < 2:
            gci = 0.0
        else:
            # Finite difference of log(CP)
            t1, cp1 = self.cp_history[-2]
            t2, cp2 = self.cp_history[-1]
            dt = t2 - t1
            
            if dt > 1e-12 and cp1 > 1e-12 and cp2 > 1e-12:
                gci = (np.log(cp2) - np.log(cp1)) / dt
            else:
                gci = 0.0
        
        # Scale by φ to match GCI definition
        gci *= PHI
        self.gci_history.append((t, gci))
        
        # Check for phase transitions
        self._check_phase_transition(t, gci)
        
        return gci
    
    def _check_phase_transition(self, t: float, gci: float):
        """Detect phase transitions based on GCI thresholds."""
        old_phase = self.phase
        
        if gci > PHI_SQUARED:
            new_phase = "TRANSCENDPLEXITY"
        elif gci > 1.0:
            new_phase = "COMPOUNDING"
        elif gci > 0.0:
            new_phase = "MYRIADPLEXITY"
        else:
            new_phase = "COLLAPSEPLEXITY"
        
        if new_phase != old_phase:
            self.phase = new_phase
            self.transitions.append((t, old_phase, new_phase))
    
    def step(self, p_tensor: np.ndarray, t: float) -> Dict:
        """
        Full compounding step: compute FP, CP, and GCI from the parallplexity tensor.
        
        Returns a dict with all metrics for logging/visualization.
        """
        fp = self.compute_fractional_parallplexity(p_tensor, t)
        cp = self.compute_compounding(fp, t)
        gci = self.compute_gci(t)
        
        return {
            "t": t,
            "fp_tensor": fp,
            "cp": cp,
            "gci": gci,
            "phase": self.phase,
            "mean_coupling": 1.0 - np.mean(p_tensor[~np.eye(self.K, dtype=bool)]),
            "det_I_minus_FP": np.linalg.det(np.eye(self.K) - fp),
        }
    
    def summary(self) -> Dict:
        """Return a summary of the compounding history."""
        if not self.cp_history:
            return {"status": "no data"}
        
        cp_values = [cp for _, cp in self.cp_history]
        gci_values = [gci for _, gci in self.gci_history]
        
        return {
            "total_steps": len(self.cp_history),
            "final_phase": self.phase,
            "peak_cp": max(cp_values),
            "peak_gci": max(gci_values) if gci_values else 0,
            "mean_gci": np.mean(gci_values) if gci_values else 0,
            "transitions": self.transitions,
            "reached_transcendplexity": self.phase == "TRANSCENDPLEXITY",
            "phi_squared_threshold": PHI_SQUARED,
        }
