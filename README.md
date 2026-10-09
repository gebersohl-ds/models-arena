# Kinetic Sumo (1v1 2D Physics-Driven Hover Arena Duel)

An open-source, modular, CPU-optimized multi-agent reinforcement learning (MARL) framework and competitive robotics simulation. Agents pilot high-inertia hovering discs armed with reinforced bumper shields, competing to knock opponents out of a dynamically shrinking ring.

Built exclusively for low-spec CPU machines with **pure NumPy vectorized physics achieving >100,000 steps/sec per core**, full PettingZoo `ParallelEnv` compliance, and support for competing AI paradigms: **Genetic Algorithm Neuroevolution**, **CleanRL Self-Play PPO**, and a deterministic **PD Heuristic Baseline**.

---

## 1. System Architecture

```text
kinetic-sumo/
├── pyproject.toml
├── README.md
├── src/
│   ├── engine/             # Headless pure NumPy 2D physics & raycasting
│   │   ├── entities.py     # Disc agent kinematics, mass, moment of inertia, bumper arcs
│   │   ├── physics.py      # Elastic collisions, drag, boundaries, angular damping
│   │   └── raycast.py      # Vectorized 16-ray Lidar intersection engine
│   ├── env/                # PettingZoo ParallelEnv & high-speed vector environments
│   │   ├── arena_env.py    # PettingZoo ParallelEnv wrapper, stepping, resets
│   │   ├── vector_env.py   # Pure NumPy batch simulator (>100k steps/sec)
│   │   └── rewards.py      # PBRS, kinetic exchange, anti-spin & anti-jitter penalties
│   ├── maps.py             # Geometric arena presets (Open, Sanctuary, Pinball, Atolls)
│   ├── agents/             # Policies & model architectures
│   │   ├── base.py         # Abstract Agent interface
│   │   ├── heuristic.py    # Rule-based PD steering & intercept state machine bot
│   │   ├── ga_mlp.py       # Flat NumPy MLP (46 -> 32 -> 32 -> 3) for Neuroevolution
│   │   └── ppo_policy.py   # Actor-Critic MLP policy (CleanRL style)
│   ├── training/           # Self-play training loops
│   │   ├── train_ga.py     # Multiprocessing genetic algorithm & mutation loop
│   │   ├── train_ppo.py    # Self-play PPO with GAE-lambda rollout buffer
│   │   └── league.py       # Prioritized Fictitious Self-Play (PFSP) historical league
│   ├── eval/               # Evaluation & ratings
│   │   └── tournament.py   # Automated round-robin Elo ladder generator (K=32)
│   └── ui/                 # Rendering and human play
│       ├── renderer.py     # Pygame-CE renderer (sparks, thrust trails, lidar overlay)
│       └── play.py         # Human vs AI / AI vs AI interactive launcher
└── tests/                  # Pytest suite (30 unit & compliance tests)
```

---

## 2. Gameplay & Combat Mechanics

### Continuous Action Space: $a = [\text{thrust}, \text{steer}, \text{ability}] \in [-1.0, 1.0]^3$
- **`thrust` $\in [-1.0, 1.0]$**: Forward throttle ($[0, 1] \times 120\,\text{N}$) or reverse brake ($[-1, 0] \times 60\,\text{N}$).
- **`steer` $\in [-1.0, 1.0]$**: Rotational torque mapped to angular velocity.
- **`ability` $\in [-1.0, 1.0]$**:
  - $> 0.5$ triggers **Dash**: $2\times$ forward speed burst, steering locked for $0.25\,\text{s}$, costs $40\%$ energy.
  - $< -0.5$ triggers **Anchor**: $4\times$ ground friction, absorbs incoming impacts, zero forward drive.

### Combat Rules
- **Front $120^\circ$ Reinforced Bumper**: Standard elastic collision recoil ($e = 0.8$).
- **Flank / Rear Hit**: Hitting an opponent's flank or rear outside their $120^\circ$ bumper delivers **$2.5\times$ impulse** to the target and **cancels attacker recoil**.
- **Non-Linear Rotational Damping**: $\tau_{\text{drag}} = -\mu \omega |\omega|$ prevents infinite spinning.
- **Ring Shrink**: Outer circle shrinks from $10.0\,\text{m}$ to $4.0\,\text{m}$ between $t = 20\,\text{s}$ and $t = 45\,\text{s}$. Ring-out equals instant elimination.

---

## 3. Sensory Pipeline (Observation Shape: 46,)

Every agent receives a compact, Markovian 1D `float32` vector of shape `(46,)`:
1. **Self Kinematics (7)**: $[x/R_0, y/R_0, v_x/V_{\max}, v_y/V_{\max}, \cos\theta, \sin\theta, \omega/\omega_{\max}]$
2. **Opponent Relative (5)**: $[\Delta x/(2R_0), \Delta y/(2R_0), \Delta v_x/(2V_{\max}), \Delta v_y/(2V_{\max}), (\theta_{\text{opp}} - \theta)/\pi]$
3. **Resources (2)**: $[\text{energy}/E_{\max}, \text{cooldown}/CD_{\max}]$
4. **1D Lidar Array (32)**: 16 radial rays reporting normalized distance $[d_0 \dots d_{15}]$ and hit surface types $[t_0 \dots t_{15}]$:
   - `-1.0`: Void edge (death boundary)
   - `0.0`: None (clear space within max range)
   - `0.5`: Arena obstacle / bumper wall
   - `1.0`: Opponent disc

---

## 4. Reward Architecture & Anti-Exploitation

$$R_t = R_{\text{terminal}} + (\gamma \Phi(s_{t+1}) - \Phi(s_t)) + R_{\text{contact}} - R_{\text{penalties}}$$

- **Terminal ($R_{\text{terminal}}$)**: Win $= +1.0$, Loss $= -1.0$, Timeout Draw $= -1.0$ (eliminates cold-war stalling).
- **PBRS Potential ($\Phi(s)$)**: $0.4 \times \text{Center\_Safety} + 0.6 \times \text{Closing\_Velocity\_Vector}$.
- **Contact ($R_{\text{contact}}$)**: $+0.4 \times (J_{\text{opp}} / V_{\max}) - 0.1 \times (J_{\text{self}} / V_{\max}) + 0.3 \times \text{flank\_bonus}$.
- **Anti-Exploitation Penalties**:
  - Spin suppression: $-0.05 \times \max(0, |\omega| - 1.0)^2$
  - Actuator jitter: $-0.01 \times \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$
  - Inactivity penalty: $-0.002$

---

## 5. Quickstart & Installation

```bash
# Clone and install dependencies
pip install -e .
```

### Running Tests
```bash
python -m pytest -v
```

### Run Benchmarking & Elo Tournament
```bash
# Automated round-robin Elo rating ladder (K=32)
python -m src.eval.tournament --matches-per-pair 10 --map Open
```

### Training Pipelines

```bash
# 1. Train Neuroevolution GA (Genetic Algorithm)
python -m src.training.train_ga --pop-size 64 --generations 20 --map Open

# 2. Train Self-Play PPO (Prioritized Fictitious Self-Play)
python -m src.training.train_ppo --timesteps 50000 --num-envs 16 --map Open
```

### Interactive Human & AI Play (Pygame-CE)

```bash
# Human vs Heuristic Bot
python -m src.ui.play --agent0 human --agent1 heuristic --map Open

# AI vs AI (Watch PPO vs Heuristic on Pinball arena)
python -m src.ui.play --agent0 ppo --agent1 heuristic --map Pinball

# Map choices: Open, Sanctuary, Pinball, Atolls
```

**Controls**:
- `W` / `S`: Forward Drive / Reverse Brake
- `A` / `D`: Steer Counter-Clockwise / Clockwise
- `Space`: Dash Burst (2x speed, locked steering)
- `Shift`: Anchor Brace (4x friction, absorbs hits)
- `L`: Toggle 16-ray Lidar debug overlay
- `R`: Reset match
- `ESC`: Quit
