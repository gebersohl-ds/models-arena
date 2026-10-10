"""Pure NumPy 2-layer MLP policy for Neuroevolution (Genetic Algorithm)."""

import numpy as np
from src.agents.base import BaseAgent


class GAMLP(BaseAgent):
    """Flat NumPy MLP (46 -> 32 -> 32 -> 3) for Genetic Algorithm neuroevolution."""

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

        self.last_hidden: np.ndarray = np.zeros(self.h2, dtype=np.float32)
        self.last_intent: str = "NEURAL MLP"
        self.target_vec: np.ndarray | None = None

    def init_weights(self) -> None:
        """Xavier / Glorot initialization of flat parameter vector."""
        # Xavier std = sqrt(2 / (fan_in + fan_out))
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

        self.last_hidden = h2.copy()

        effective_noise = temperature if temperature is not None else self.stochasticity
        if effective_noise > 0.0:
            noise = np.random.normal(0.0, effective_noise, size=out.shape).astype(np.float32)
            out = np.clip(out + noise, -1.0, 1.0)

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

    def batch_act(self, obs_batch: np.ndarray) -> np.ndarray:
        """Vectorized forward pass for a batch of observations (N, 46) -> (N, 3)."""
        x = np.asarray(obs_batch, dtype=np.float32)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        out = np.tanh(h2 @ self.w3 + self.b3)
        return out.astype(np.float32)

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
