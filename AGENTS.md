# Models Arena (Kinetic Sumo) — Workspace Rules

## 1. Communication & Interactive Questions
- **Use Starbridge for Questions**: Whenever asking the user clarifying, design, or architectural questions (including during `/grill-me` interviews), use the `starbridge` CLI (`starbridge ask ... --waiting --wait` with `WaitMsBeforeAsync: 1500`) and mirror the question summary in chat. Keep every `--option` string **<= 100 characters** (put longer explanations in the main question text).

## 2. Physics & Arena Determinism
- **100% Deterministic Maps & Mechanics**: All arena maps (`src/maps.py`), obstacle layouts, and physics simulation steps (`src/engine/physics.py`) must remain strictly deterministic with zero random events or non-deterministic hazards. This guarantees reproducible RL rollouts and fair Elo tournament evaluations.

## 3. Observation & Checkpoint Compatibility
- **Preserve 1v1 `(46,)` Observation Contract**: Maintain backward compatibility with the 46-dimensional observation space (`7` self kinematics + `5` opponent relative + `2` resources + `32` Lidar) so existing trained PPO (`.pt`) and GA (`.npy`) checkpoints continue functioning without retraining.

## 4. ArcadeHub UI & Hitbox Invariants
- **Zero Text Overlap & Font Wrapping (`src/ui/hub.py`)**: `test_no_text_overlaps_across_all_tabs` wraps `hub.font_*` attributes after `ArcadeHub.__init__()` and only rebinds `s.font` for sliders in `hub.sliders + hub.evo_sliders`. Custom widgets (`UIDropdown`, helper cards) must accept `self.font_small` / `self.font_tiny` at draw time, and all rendered text across all 7 tabs must have non-overlapping bounding boxes.
- **Preserve Tested Hitboxes**: Maintain tested click hitboxes in `ArcadeHub` (e.g., Analytics view toggle buttons at `(200, 175)` and `(500, 175)` in Tab 3; Reset Defaults at `(500, 525)` and Mutate Jitter at `(500, 580)` in Tab 5 `BOT LOCKER`).

## 5. QA Subagent Verification & Git Workflow
- **Parallel Subagent QA Before Merge**: For major feature additions, run parallel QA subagents (Physics/Env Determinism + UI/Telemetry) to verify `pytest` and `parallel_api_test`, regenerate `tab_screenshots/` if UI changed, and then create a feature branch, push to `origin`, merge into `main`, and push `main`.
