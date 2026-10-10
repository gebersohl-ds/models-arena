"""Pure NumPy 2-layer MLP policy for Neuroevolution (Genetic Algorithm) with Brain Topology & Causal Attribution."""

import numpy as np
from src.agents.base import BaseAgent


SENSORY_CLUSTER_SPECS: list[tuple[str, list[int]]] = [
    ("Self Pos", [0, 1]),
    ("Self Vel", [2, 3]),
    ("Spin/Aim", [4, 5, 6, 11]),
    ("Opp Dist", [7, 8]),
    ("Opp Close", [9, 10]),
    ("Resources", [12, 13]),
    ("Front Lidar", [14, 15, 28, 29, 30, 31, 44, 45]),
    ("Void Lidar", list(range(16, 28)) + list(range(32, 44))),
]


def compute_sensory_clusters(obs: np.ndarray) -> list[tuple[str, float]]:
    """Aggregate a 46-dim observation vector into 8 normalized [-1, 1] sensory cluster activations."""
    x = np.asarray(obs, dtype=np.float32)
    if len(x) < 46:
        x = np.pad(x, (0, 46 - len(x)))

    r_norm = float(np.clip(np.hypot(x[0], x[1]), 0.0, 1.0))
    v_norm = float(np.clip(np.hypot(x[2], x[3]), 0.0, 1.0))
    spin_aim = float(np.clip(0.5 * x[6] + 0.5 * x[11], -1.0, 1.0))
    opp_dist = float(np.clip(1.0 - np.hypot(x[7], x[8]) * 2.0, -1.0, 1.0))

    dist_raw = max(0.05, float(np.hypot(x[7], x[8])))
    closing = float(np.clip(-(x[7] * x[9] + x[8] * x[10]) / dist_raw, -1.0, 1.0))
    resources = float(np.clip(x[12] - 0.5 * x[13], -1.0, 1.0))

    # Front rays: indices 0, 1, 15 in the 16-ray array
    dists = x[14:30]
    types = x[30:46]
    front_indices = [0, 1, 15]
    front_prox = max(float(1.0 - dists[i]) * (1.0 if types[i] >= 0.4 else 0.4) for i in front_indices)
    front_val = float(np.clip(front_prox, 0.0, 1.0))

    # Void edge proximity: rays where surface type == -1.0
    void_masks = [1.0 - float(dists[i]) for i in range(16) if types[i] < -0.5]
    void_val = float(np.clip(max(void_masks) if void_masks else (1.0 - float(np.min(dists))), 0.0, 1.0))

    return [
        ("Self Pos", r_norm),
        ("Self Vel", v_norm),
        ("Spin/Aim", spin_aim),
        ("Opp Dist", opp_dist),
        ("Opp Close", closing),
        ("Resources", resources),
        ("Front Lidar", front_val),
        ("Void Lidar", void_val),
    ]


def synthesize_tactical_brain_snapshot(
    obs: np.ndarray | None,
    act: np.ndarray | None,
    intent: str = "TACTICAL CORTEX",
) -> dict:
    """Build an interpretable cognitive graph & causal attribution snapshot for rule/tactical agents."""
    x = np.zeros(46, dtype=np.float32) if obs is None else np.asarray(obs, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32) if act is None else np.asarray(act, dtype=np.float32)
    clusters = compute_sensory_clusters(x)
    c_vals = np.array([v for _, v in clusters], dtype=np.float32)  # (8,)

    # Deterministic structured projection (8 -> 16 -> 16 -> 3) for visual cortex topology
    rng = np.random.RandomState(31415)
    cluster_w1 = rng.uniform(-0.85, 0.85, size=(8, 16)).astype(np.float32)
    w2 = rng.uniform(-0.75, 0.75, size=(16, 16)).astype(np.float32)
    w3 = rng.uniform(-0.85, 0.85, size=(16, 3)).astype(np.float32)

    h1 = np.tanh(c_vals @ cluster_w1)
    h2 = np.tanh(h1 @ w2 + 0.35 * (out @ w3.T))

    # Compute top 3 drivers from active sensory clusters and action channels
    actuator_names = ["THR", "STR", "ABL"]
    dom_act_idx = int(np.argmax(np.abs(out))) if np.max(np.abs(out)) > 0.05 else 0
    raw_scores: list[tuple[str, float, str]] = []
    total = 0.0
    for idx, (c_name, val) in enumerate(clusters):
        score = abs(float(val)) + 0.08
        # Map cluster semantics to primary actuator
        if c_name in ("Spin/Aim", "Front Lidar"):
            target_act = "STR" if abs(float(out[1])) >= 0.15 else actuator_names[dom_act_idx]
        elif c_name in ("Resources", "Opp Close"):
            target_act = "ABL" if abs(float(out[2])) >= 0.35 else "THR"
        else:
            target_act = "THR" if abs(float(out[0])) >= 0.15 else actuator_names[dom_act_idx]
        raw_scores.append((c_name, score, target_act))
        total += score

    denom = max(1e-6, total)
    drivers = [(n, s / denom, a) for n, s, a in raw_scores]
    drivers.sort(key=lambda item: item[1], reverse=True)

    return {
        "arch": "TACTICAL CORTEX",
        "clusters": clusters,
        "h1": h1,
        "h2": h2,
        "out": out.copy(),
        "cluster_w1": cluster_w1,
        "w2": w2,
        "w3": w3,
        "top_drivers": drivers[:3],
        "intent": intent,
    }



class GAMLP(BaseAgent):
    """Flat NumPy MLP (46 -> 32 -> 32 -> 3) for Genetic Algorithm neuroevolution."""

    NUM_PARAMS = 46 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3  # 2,659

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dims: tuple[int, int] = (32, 32),
        output_dim: int = 3,
        weights: np.ndarray | None = None,
        stochasticity: float = 0.0,
    ):
        self.input_dim = input_dim
        self.h1, self.h2 = hidden_dims
        self.output_dim = output_dim
        self.stochasticity = float(stochasticity)

        # Calculate parameter partition sizes
        self.w1_size = self.input_dim * self.h1
        self.b1_size = self.h1
        self.w2_size = self.h1 * self.h2
        self.b2_size = self.h2
        self.w3_size = self.h2 * self.output_dim
        self.b3_size = self.output_dim

        self.total_params = (
            self.w1_size + self.b1_size + self.w2_size + self.b2_size + self.w3_size + self.b3_size
        )

        if weights is not None:
            self.set_weights(weights)
        else:
            self.init_weights()

        self.last_obs: np.ndarray = np.zeros(self.input_dim, dtype=np.float32)
        self.last_h1: np.ndarray = np.zeros(self.h1, dtype=np.float32)
        self.last_h2: np.ndarray = np.zeros(self.h2, dtype=np.float32)
        self.last_hidden: np.ndarray = self.last_h2
        self.last_out: np.ndarray = np.zeros(self.output_dim, dtype=np.float32)
        self.last_intent: str = "NEURAL MLP"
        self.target_vec: np.ndarray | None = None

    def init_weights(self) -> None:
        """Xavier / Glorot initialization of flat parameter vector."""
        w1 = np.random.randn(self.input_dim, self.h1) * np.sqrt(2.0 / (self.input_dim + self.h1))
        b1 = np.zeros(self.h1)
        w2 = np.random.randn(self.h1, self.h2) * np.sqrt(2.0 / (self.h1 + self.h2))
        b2 = np.zeros(self.h2)
        w3 = np.random.randn(self.h2, self.output_dim) * np.sqrt(2.0 / (self.h2 + self.output_dim))
        b3 = np.zeros(self.output_dim)

        self.w1 = w1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.w2 = w2.astype(np.float32)
        self.b2 = b2.astype(np.float32)
        self.w3 = w3.astype(np.float32)
        self.b3 = b3.astype(np.float32)

    def get_weights(self) -> np.ndarray:
        """Flatten all network parameters into a single 1D numpy array."""
        return np.concatenate(
            [
                self.w1.ravel(),
                self.b1,
                self.w2.ravel(),
                self.b2,
                self.w3.ravel(),
                self.b3,
            ]
        ).astype(np.float32)

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """Unflatten 1D parameter array into network matrices."""
        flat = np.asarray(flat_weights, dtype=np.float32)
        idx = 0

        self.w1 = flat[idx : idx + self.w1_size].reshape(self.input_dim, self.h1)
        idx += self.w1_size
        self.b1 = flat[idx : idx + self.b1_size]
        idx += self.b1_size

        self.w2 = flat[idx : idx + self.w2_size].reshape(self.h1, self.h2)
        idx += self.w2_size
        self.b2 = flat[idx : idx + self.b2_size]
        idx += self.b2_size

        self.w3 = flat[idx : idx + self.w3_size].reshape(self.h2, self.output_dim)
        idx += self.w3_size
        self.b3 = flat[idx : idx + self.b3_size]

    def act(self, obs: np.ndarray, temperature: float | None = None) -> np.ndarray:
        """Forward pass for a single 46-dimensional observation vector."""
        x = np.asarray(obs, dtype=np.float32)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        out = np.tanh(h2 @ self.w3 + self.b3)

        self.last_obs = x.copy()
        self.last_h1 = h1.copy()
        self.last_h2 = h2.copy()
        self.last_hidden = self.last_h2

        effective_noise = temperature if temperature is not None else self.stochasticity
        if effective_noise > 0.0:
            noise = np.random.normal(0.0, effective_noise, size=out.shape).astype(np.float32)
            out = np.clip(out + noise, -1.0, 1.0)

        self.last_out = out.copy()

        # Derive visual intent label & approximate heading target vector for X-Ray HUD
        if out[2] > 0.5:
            self.last_intent = "MLP: DASH SURGE"
        elif out[2] < -0.5:
            self.last_intent = "MLP: ANCHOR BRACE"
        elif out[0] > 0.5:
            self.last_intent = "MLP: FULL DRIVE"
        elif out[0] < -0.2:
            self.last_intent = "MLP: REVERSE BRAKE"
        else:
            self.last_intent = "MLP: MICRO PIVOT"

        if len(x) >= 9:
            cos_th, sin_th = float(x[4]), float(x[5])
            heading = np.arctan2(sin_th, cos_th)
            aim_ang = heading + float(out[1]) * 0.8
            mag = max(1.5, abs(float(out[0])) * 4.0)
            self.target_vec = np.array([np.cos(aim_ang) * mag, np.sin(aim_ang) * mag], dtype=np.float32)

        return out.astype(np.float32)

    def get_sensory_importance(self) -> dict[str, float]:
        """Compute normalized synaptic importance across the 4 sensory observation slices."""
        row_norms = np.mean(np.abs(self.w1), axis=1)
        kin = float(np.mean(row_norms[0:7]))
        opp = float(np.mean(row_norms[7:12]))
        res = float(np.mean(row_norms[12:14]))
        lidar = float(np.mean(row_norms[14:46]))
        total = max(1e-6, kin + opp + res + lidar)
        return {
            "Self Kinematics": kin / total,
            "Opponent Rel": opp / total,
            "Resources": res / total,
            "16-Ray Lidar": lidar / total,
        }

    def get_input_salience(self) -> np.ndarray:
        """Return normalized 46-element vector of input synaptic magnitudes in [0, 1]."""
        row_norms = np.mean(np.abs(self.w1), axis=1)
        max_v = max(1e-6, float(np.max(row_norms)))
        return (row_norms / max_v).astype(np.float32)

    def get_causal_attribution(self, obs: np.ndarray | None = None) -> list[tuple[str, float, str]]:
        """Compute Top-3 sensory drivers explaining WHICH inputs drove WHICH output actuator.

        Uses pathway sensitivity |x_i| * sum_j(|W1_ij| * |h1_j| * sum_k(|W2_jk| * |h2_k| * |W3_km|))
        aggregated across the 8 human-readable sensory clusters.
        Returns list of (cluster_name, normalized_share, primary_actuator_name).
        """
        if obs is not None:
            x = np.asarray(obs, dtype=np.float32)
            h1 = np.tanh(x @ self.w1 + self.b1)
            h2 = np.tanh(h1 @ self.w2 + self.b2)
        else:
            x = self.last_obs
            h1 = self.last_h1
            h2 = self.last_h2

        # Propagate active magnitudes backward from 3 outputs -> H2 -> H1 -> 46 inputs
        # h2_to_out: (32, 3)
        h2_contrib = np.abs(h2)[:, None] * np.abs(self.w3) + 1e-4
        # h1_to_out: (32, 3)
        h1_contrib = np.abs(h1)[:, None] * (np.abs(self.w2) @ h2_contrib)
        # in_to_out: (46, 3)
        in_mag = np.maximum(np.abs(x), 0.05)[:, None]
        in_to_out = in_mag * (np.abs(self.w1) @ h1_contrib)

        actuator_names = ["THR", "STR", "ABL"]
        cluster_drivers: list[tuple[str, float, str]] = []
        total_drive = 0.0

        raw_entries: list[tuple[str, float, str]] = []
        for c_name, indices in SENSORY_CLUSTER_SPECS:
            valid_idx = [i for i in indices if i < len(in_to_out)]
            if not valid_idx:
                continue
            c_vec = np.mean(in_to_out[valid_idx], axis=0)  # (3,)
            c_score = float(np.sum(c_vec))
            dom_act = actuator_names[int(np.argmax(c_vec))]
            raw_entries.append((c_name, c_score, dom_act))
            total_drive += c_score

        denom = max(1e-6, total_drive)
        for c_name, c_score, dom_act in raw_entries:
            cluster_drivers.append((c_name, c_score / denom, dom_act))

        cluster_drivers.sort(key=lambda item: item[1], reverse=True)
        return cluster_drivers[:3]

    def get_brain_snapshot(self, obs: np.ndarray | None = None) -> dict:
        """Return complete neural activation & synaptic pathway snapshot for Brain View rendering."""
        if obs is not None:
            self.act(obs, temperature=0.0)

        clusters = compute_sensory_clusters(self.last_obs)
        # Aggregate W1 (46, 32) into (8, 32) cluster-to-H1 effective weights
        cluster_w1 = np.zeros((len(SENSORY_CLUSTER_SPECS), self.h1), dtype=np.float32)
        for c_idx, (_, indices) in enumerate(SENSORY_CLUSTER_SPECS):
            valid_idx = [i for i in indices if i < self.input_dim]
            if valid_idx:
                cluster_w1[c_idx] = np.mean(self.w1[valid_idx], axis=0)

        return {
            "arch": f"{self.input_dim}->{self.h1}->{self.h2}->{self.output_dim}",
            "clusters": clusters,
            "h1": self.last_h1.copy(),
            "h2": self.last_h2.copy(),
            "out": self.last_out.copy(),
            "cluster_w1": cluster_w1,
            "w2": self.w2,
            "w3": self.w3,
            "top_drivers": self.get_causal_attribution(),
            "intent": self.last_intent,
        }

    def batch_act(self, obs_batch: np.ndarray) -> np.ndarray:
        """Vectorized forward pass for a batch of observations (N, 46) -> (N, 3)."""
        x = np.asarray(obs_batch, dtype=np.float32)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        out = np.tanh(h2 @ self.w3 + self.b3)
        return out.astype(np.float32)

    def crossover(self, other: "GAMLP", crossover_rate: float = 0.5) -> "GAMLP":
        """Two-parent synaptic crossover combining hidden feature detectors from both parents."""
        child = self.clone()
        if not isinstance(other, GAMLP) or other.total_params != self.total_params:
            return child

        # Neuron-aligned crossover on Layer 1 (preserves incoming receptor fields per hidden neuron)
        h1_mask = np.random.rand(self.h1) < crossover_rate
        child.w1[:, h1_mask] = other.w1[:, h1_mask]
        child.b1[h1_mask] = other.b1[h1_mask]

        # Neuron-aligned crossover on Layer 2
        h2_mask = np.random.rand(self.h2) < crossover_rate
        child.w2[:, h2_mask] = other.w2[:, h2_mask]
        child.b2[h2_mask] = other.b2[h2_mask]

        # Uniform synaptic crossover on Output Layer 3
        w3_mask = np.random.rand(*self.w3.shape) < crossover_rate
        child.w3[w3_mask] = other.w3[w3_mask]
        b3_mask = np.random.rand(self.output_dim) < crossover_rate
        child.b3[b3_mask] = other.b3[b3_mask]

        return child

    def mutate(self, mutation_rate: float = 0.05, sigma: float = 0.1) -> "GAMLP":
        """Apply Gaussian parameter perturbation to create a mutated offspring."""
        w = self.get_weights()
        mask = np.random.rand(len(w)) < mutation_rate
        noise = np.random.randn(len(w)).astype(np.float32) * sigma
        w[mask] += noise[mask]

        offspring = GAMLP(self.input_dim, (self.h1, self.h2), self.output_dim, stochasticity=self.stochasticity)
        offspring.set_weights(w)
        return offspring

    def clone(self) -> "GAMLP":
        """Create an identical copy of this agent."""
        c = GAMLP(self.input_dim, (self.h1, self.h2), self.output_dim, stochasticity=self.stochasticity)
        c.set_weights(self.get_weights())
        return c

    def save(self, filepath: str) -> None:
        """Save weights to a .npy file."""
        np.save(filepath, self.get_weights())

    @classmethod
    def load(cls, filepath: str) -> "GAMLP":
        """Load weights from a .npy file."""
        weights = np.load(filepath)
        agent = cls()
        agent.set_weights(weights)
        return agent
