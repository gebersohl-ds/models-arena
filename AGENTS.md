# Models Arena (Kinetic Sumo) — Workspace Rules

## 1. Communication & Interactive Questions
- **Use Starbridge for Questions**: Whenever asking the user clarifying, design, or architectural questions (including during `/grill-me` interviews), use the `starbridge` CLI (`starbridge ask ... --waiting --wait` with `WaitMsBeforeAsync: 1500`) and mirror the question summary in chat.

## 2. Physics & Arena Determinism
- **100% Deterministic Maps & Mechanics**: All arena maps (`src/maps.py`), obstacle layouts, and physics simulation steps (`src/engine/physics.py`) must remain strictly deterministic with zero random events or non-deterministic hazards. This guarantees reproducible RL rollouts and fair Elo tournament evaluations.

## 3. Observation & Checkpoint Compatibility
- **Preserve 1v1 `(46,)` Observation Contract**: Maintain backward compatibility with the 46-dimensional observation space (`7` self kinematics + `5` opponent relative + `2` resources + `32` Lidar) so existing trained PPO (`.pt`) and GA (`.npy`) checkpoints continue functioning without retraining.
