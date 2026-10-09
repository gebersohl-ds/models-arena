---
name: gym-env-checker
description: Validates Gymnasium single-agent and PettingZoo multi-agent RL environments against official API compliance test suites (check_env, parallel_api_test) to catch silent bugs like observation bound violations, broken truncation/termination, and non-deterministic resets.
---

# Gymnasium / PettingZoo Environment Checker Tool (`gym-env-checker`)

This skill validates custom Gymnasium and PettingZoo simulation environments against official API compliance suites before training begins, catching silent bugs like observation out of bounds, broken truncation flags, and non-deterministic resets.

---

## 1. When to Use This Skill

Activate this skill when:
1. **Authoring a New Environment:** Verifying a newly created environment before launching an expensive RL training run.
2. **Upgrading Legacy Gym to Gymnasium:** Auditing step returns (ensuring 5-tuple: `obs, reward, terminated, truncated, info`).
3. **Debugging Training Instabilities:** Checking if agents receive out-of-bounds observations, NaNs, or non-deterministic initial states.

---

## 2. Running Compliance Checks

### Using the Bundled Runner
```bash
python ~/.gemini/config/skills/gym-env-checker/scripts/check_environments.py --env path.to.module:MyEnvClass
```

On Windows PowerShell:
```powershell
python "C:\Users\xSyl0\.gemini\config\skills\gym-env-checker\scripts\check_environments.py" --env path.to.module:MyEnvClass
```

### Direct Native API Verification

#### Single-Agent (Gymnasium):
```python
from gymnasium.utils.env_checker import check_env
check_env(my_env)
```

#### Multi-Agent (PettingZoo):
```python
from pettingzoo.test import parallel_api_test
parallel_api_test(my_parallel_env, num_cycles=1000)
```

---

## 3. Reference Files & Checklists

* [Environment Compliance Checker Script](./scripts/check_environments.py)
* [Silent RL Bugs Checklist](./references/rl_env_checklist.md)
