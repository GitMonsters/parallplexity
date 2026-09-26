"""
OctoTetrahedral 8-Limb Parallel Processor

Each limb is a parallel processing stream with its own fractional order α_i.
The MetaCognition limb dynamically adjusts other limbs' α values based on
the compounding parallplexity state.

Limb fractional orders from the paper:
    Perception:    α ≈ 0.9 (nearly memoryless, reacts to immediate input)
    Memory:        α ≈ 0.3 (deep memory, integrates long history)
    Planning:      α ≈ 0.7 (moderate memory, exploration/exploitation balance)
    Language:      α ≈ 0.5 (median memory, bridges immediate and historical)
    Spatial:       α ≈ 0.6 (spatial structures have moderate persistence)
    Reasoning:     α ≈ 1.44 (Cantor-Golden fractal order — primary compounder)
    MetaCognition: adaptive (governs other limbs)
    Action:        α → 1.0 (nearly integer-order, crisp outputs)
"""

import numpy as np
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.fractional import CANTOR_GOLDEN_DIM, PHI, PHI_SQUARED, HistoryBuffer
from ..core.parallplexity import StreamState, ParallplexityTensor, CompoundingTracker
from ..core.evolution import FractionalEvolutionOperator


@dataclass
class LimbConfig:
    """Configuration for a single cognitive limb."""
    name: str
    alpha: float                # fractional order
    state_dim: int = 64         # dimensionality of this limb's state
    coupling_strength: float = 0.1
    nonlinearity: float = 0.05  # Gross-Pitaevskii self-interaction strength


# Default 8-limb configuration from the paper
DEFAULT_LIMBS = [
    LimbConfig("Perception",    alpha=0.9,              state_dim=64,  nonlinearity=0.01),
    LimbConfig("Memory",        alpha=0.3,              state_dim=128, nonlinearity=0.02),
    LimbConfig("Planning",      alpha=0.7,              state_dim=64,  nonlinearity=0.05),
    LimbConfig("Language",      alpha=0.5,              state_dim=64,  nonlinearity=0.03),
    LimbConfig("Spatial",       alpha=0.6,              state_dim=64,  nonlinearity=0.04),
    LimbConfig("Reasoning",     alpha=CANTOR_GOLDEN_DIM,state_dim=128, nonlinearity=0.08),
    LimbConfig("MetaCognition", alpha=0.5,              state_dim=64,  nonlinearity=0.02),
    LimbConfig("Action",        alpha=0.95,             state_dim=64,  nonlinearity=0.01),
]


class EightLimbProcessor:
    """
    The OctoTetrahedral 8-Limb Parallel Processor.
    
    8 specialized limbs process in parallel, each with its own fractional order.
    The parallplexity tensor tracks inter-limb coupling.
    The compounding tracker detects phase transitions.
    MetaCognition governs α adaptation.
    """
    
    def __init__(
        self,
        limb_configs: Optional[List[LimbConfig]] = None,
        dt: float = 0.01,
    ):
        configs = limb_configs or DEFAULT_LIMBS
        self.K = len(configs)
        self.dt = dt
        self.t = 0.0
        self.step_count = 0

        # Copy the module defaults: _metacognition_adapt mutates limb config
        # alphas in place, so sharing DEFAULT_LIMBS LimbConfigs across
        # instances let one run retune the defaults for all later runs.
        configs = [copy.deepcopy(c) for c in configs]
        
        # Initialize limbs
        self.limbs: List[Dict] = []
        self.streams: List[StreamState] = []
        
        for cfg in configs:
            # Random initial state (complex amplitudes on a 1D "lattice")
            init_state = (
                np.random.randn(cfg.state_dim) +
                1j * np.random.randn(cfg.state_dim)
            )
            init_state /= np.linalg.norm(init_state)  # normalize
            
            stream = StreamState(
                name=cfg.name,
                state=init_state,
                alpha=cfg.alpha,
            )
            
            evolver = FractionalEvolutionOperator(
                alpha=min(cfg.alpha, 1.99),
                dt=dt,
                coupling_J=cfg.coupling_strength,
                max_history=300,
            )
            
            self.limbs.append({
                "config": cfg,
                "evolver": evolver,
                "stream": stream,
            })
            self.streams.append(stream)
        
        # Parallplexity tensor and compounding tracker
        self.p_tensor = ParallplexityTensor(self.K)
        self.tracker = CompoundingTracker(self.K)
        
        # Quantum coupling matrix g_ij (initialized to uniform weak coupling)
        self.coupling_matrix = np.ones((self.K, self.K)) * 0.1
        np.fill_diagonal(self.coupling_matrix, 0.0)
        
        # Step log
        self.log: List[Dict] = []
    
    def inject_input(self, data: np.ndarray, target_limb: str = "Perception"):
        """
        Inject external input into a specific limb.
        Typically enters through Perception, then propagates.
        """
        for limb in self.limbs:
            if limb["config"].name == target_limb:
                stream = limb["stream"]
                # Blend input with current state
                dim = len(stream.state)
                input_resized = np.interp(
                    np.linspace(0, 1, dim),
                    np.linspace(0, 1, len(data)),
                    np.abs(data)
                ) * np.exp(1j * np.angle(stream.state))
                
                blend = 0.7 * stream.state + 0.3 * input_resized
                blend /= np.linalg.norm(blend) + 1e-12
                stream.state = blend
                break
    
    def step(self) -> Dict:
        """
        Execute one parallel processing step across all 8 limbs.
        
        1. Each limb evolves under its own fractional dynamics
        2. Inter-limb coupling is applied
        3. Parallplexity tensor is computed
        4. Compounding metrics are tracked
        5. MetaCognition adjusts α values
        """
        # === 1. Parallel fractional evolution ===
        new_states = []
        for limb in self.limbs:
            cfg = limb["config"]
            evolver = limb["evolver"]
            stream = limb["stream"]
            
            new_state = evolver.apply_fractional_evolution(
                stream.state,
                nonlinearity=cfg.nonlinearity,
            )
            new_states.append(new_state)
        
        # === 2. Inter-limb coupling ===
        coupled_states = self._apply_coupling(new_states)
        
        # === 3. Update stream states ===
        for i, (limb, new_state) in enumerate(zip(self.limbs, coupled_states)):
            limb["stream"].update(new_state, self.t)
        
        # === 4. Compute parallplexity tensor ===
        p = self.p_tensor.compute(self.streams)
        self.p_tensor.record(self.t)
        
        # === 5. Compounding tracking ===
        metrics = self.tracker.step(p, self.t)
        
        # === 6. MetaCognition α adaptation ===
        self._metacognition_adapt(metrics)
        
        self.t += self.dt
        self.step_count += 1
        
        # Build step report
        report = {
            **metrics,
            "step": self.step_count,
            "limb_alphas": {
                limb["config"].name: limb["config"].alpha
                for limb in self.limbs
            },
            "limb_norms": {
                limb["config"].name: float(np.linalg.norm(limb["stream"].state))
                for limb in self.limbs
            },
        }
        self.log.append(report)
        
        return report
    
    def _apply_coupling(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply inter-limb coupling via the quantum coupling matrix g_ij.
        
        Each limb receives a weighted blend of other limbs' states,
        creating the cross-stream information flow that drives
        parallplexity reduction.
        """
        coupled = []
        for i in range(self.K):
            s_i = states[i].copy()
            coupling_contribution = np.zeros_like(s_i)
            
            for j in range(self.K):
                if i == j:
                    continue
                g_ij = self.coupling_matrix[i, j]
                
                # Resize state j to match state i if needed
                s_j = states[j]
                if len(s_j) != len(s_i):
                    s_j_resized = np.interp(
                        np.linspace(0, 1, len(s_i)),
                        np.linspace(0, 1, len(s_j)),
                        np.abs(s_j)
                    ) * np.exp(1j * np.linspace(0, 2*np.pi, len(s_i)))
                else:
                    s_j_resized = s_j
                
                coupling_contribution += g_ij * s_j_resized
            
            # Blend: (1 - total_coupling) * self + coupling_contribution
            total_g = self.coupling_matrix[i, :].sum()
            weight_self = max(0.5, 1.0 - total_g)
            
            coupled_state = weight_self * s_i + (1 - weight_self) * coupling_contribution
            norm = np.linalg.norm(coupled_state)
            if norm > 1e-12:
                coupled_state *= np.linalg.norm(s_i) / norm
            
            coupled.append(coupled_state)
        
        return coupled
    
    def _metacognition_adapt(self, metrics: Dict):
        """
        MetaCognition limb adjusts α_i values based on compounding state.
        
        Rules:
        - If GCI is rising (compounding accelerating): increase α (sharpen, exploit)
        - If GCI is falling (compounding stalling): decrease α (broaden, explore)
        - Reasoning limb always stays at Cantor-Golden dimension
        - Action limb stays near 1.0
        """
        gci = metrics.get("gci", 0)
        
        if len(self.tracker.gci_history) < 5:
            return  # not enough data
        
        # GCI trend: compare recent to older
        recent_gci = np.mean([g for _, g in self.tracker.gci_history[-3:]])
        older_gci = np.mean([g for _, g in self.tracker.gci_history[-6:-3]])
        
        gci_delta = recent_gci - older_gci
        
        # Adaptation rate — scaled by tanh to prevent runaway
        adapt_rate = 0.005 * np.tanh(gci_delta / 10.0)
        
        for limb in self.limbs:
            name = limb["config"].name
            
            # Skip fixed-alpha limbs
            if name == "Reasoning":
                continue  # always Cantor-Golden
            if name == "Action":
                continue  # always near 1.0
            if name == "MetaCognition":
                continue  # self-referential — doesn't adapt itself
            
            current_alpha = limb["config"].alpha
            
            if adapt_rate > 0:
                # Compounding accelerating: increase α (exploit)
                new_alpha = min(1.6, current_alpha + abs(adapt_rate))
            else:
                # Compounding stalling: decrease α (explore)
                new_alpha = max(0.15, current_alpha - abs(adapt_rate))
            
            limb["config"].alpha = new_alpha
            # Update the evolver's alpha too
            limb["evolver"].alpha = min(new_alpha, 1.99)
            limb["evolver"].caputo.alpha = min(new_alpha, 1.99)
    
    def run(self, steps: int, input_data: Optional[np.ndarray] = None,
            input_interval: int = 10, verbose: bool = True) -> Dict:
        """
        Run the full compound integration for N steps.
        
        Optionally injects input data periodically to drive the system.
        Returns final compounding summary.
        """
        if input_data is None:
            # Generate a structured test signal
            input_data = np.sin(np.linspace(0, 4*np.pi, 64)) + \
                         0.5 * np.cos(np.linspace(0, 8*np.pi, 64))
        
        for i in range(steps):
            # Periodic input injection
            if i % input_interval == 0:
                # Vary the input slightly each injection
                phase_shift = 2 * np.pi * i / steps
                varied_input = input_data * np.cos(
                    np.linspace(phase_shift, phase_shift + 2*np.pi, len(input_data))
                )
                self.inject_input(varied_input)
            
            report = self.step()
            
            if verbose and (i % max(1, steps // 20) == 0 or i == steps - 1):
                phase_marker = {
                    "MYRIADPLEXITY": "○",
                    "COMPOUNDING": "◐",
                    "TRANSCENDPLEXITY": "●",
                    "COLLAPSEPLEXITY": "◌",
                }.get(report["phase"], "?")
                
                print(
                    f"  Step {i+1:>5}/{steps}  "
                    f"{phase_marker} {report['phase']:<20}  "
                    f"CP={report['cp']:>12.2f}  "
                    f"GCI={report['gci']:>8.4f}  "
                    f"coupling={report['mean_coupling']:.4f}"
                )
        
        return self.tracker.summary()
