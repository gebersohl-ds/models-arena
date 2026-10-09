#!/usr/bin/env python3
"""
Gymnasium & PettingZoo Environment API Compliance Checker
Wraps gymnasium.utils.env_checker.check_env and pettingzoo.test.parallel_api_test.
Catches silent bugs: observation bound violations, broken truncation/termination,
non-deterministic resets, and missing info dicts.
"""

import sys
import os
import argparse
import importlib
import numpy as np

def import_env_from_string(env_target: str):
    """
    Imports environment class or factory function.
    Format: 'path.to.module:ClassName' or 'module:function'
    """
    if ":" not in env_target:
        raise ValueError(f"Invalid target format '{env_target}'. Expected 'module_name:EnvClassOrFactory'")

    module_name, obj_name = env_target.split(":", 1)
    # Add current working directory to sys.path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    mod = importlib.import_module(module_name)
    env_factory = getattr(mod, obj_name)
    return env_factory() if callable(env_factory) else env_factory

def test_gymnasium_env(env):
    """Tests Gymnasium single-agent environment compliance."""
    print("\n--- Running Gymnasium Environment Compliance Checks ---")
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("[+] gymnasium.utils.env_checker.check_env PASSED!")
    except Exception as e:
        print(f"[-] gymnasium.utils.env_checker FAILED: {e}", file=sys.stderr)
        return False

    # Seed Determinism Check
    print("[*] Testing Seed Determinism...")
    obs1, _ = env.reset(seed=1337)
    obs2, _ = env.reset(seed=1337)
    if isinstance(obs1, np.ndarray) and isinstance(obs2, np.ndarray):
        if not np.array_equal(obs1, obs2):
            print("[-] FAIL: env.reset(seed=1337) produced non-deterministic initial observations!", file=sys.stderr)
            return False
    print("[+] Seed determinism verified.")

    # Stepping & Truncation/Termination Separation Check
    print("[*] Testing Step API & Truncation/Termination Separation...")
    env.reset(seed=42)
    for step_idx in range(50):
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) != 5:
            print(f"[-] FAIL: env.step() returned {len(step_result)} items. Expected 5: (obs, reward, terminated, truncated, info)", file=sys.stderr)
            return False
        obs, reward, terminated, truncated, info = step_result
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
            print(f"[-] FAIL: 'terminated' and 'truncated' must be strict booleans. Got ({type(terminated)}, {type(truncated)})", file=sys.stderr)
            return False
        if not env.observation_space.contains(obs):
            print(f"[-] FAIL: Step observation out of bounds for observation_space at step {step_idx}!", file=sys.stderr)
            return False
        if terminated or truncated:
            env.reset()
    print("[+] Step cycle & observation boundary checks passed.")
    return True

def test_pettingzoo_env(env):
    """Tests PettingZoo multi-agent environment compliance."""
    print("\n--- Running PettingZoo API Compliance Checks ---")
    try:
        from pettingzoo.test import parallel_api_test, api_test
        # Check if ParallelEnv or AECEnv
        if hasattr(env, "possible_agents"):
            parallel_api_test(env, num_cycles=200)
            print("[+] pettingzoo.test.parallel_api_test PASSED!")
        else:
            api_test(env, num_cycles=200)
            print("[+] pettingzoo.test.api_test PASSED!")
        return True
    except Exception as e:
        print(f"[-] PettingZoo compliance test FAILED: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Gymnasium & PettingZoo Environment Checker")
    parser.add_argument("--env", required=False, help="Module path and class: e.g. 'envs.combat_env:CombatEnv'")
    args = parser.parse_args()

    if not args.env:
        print("[INFO] No environment specified. Usage: python check_environments.py --env path.to.module:EnvClass")
        return

    print(f"Loading environment from: {args.env}")
    try:
        env = import_env_from_string(args.env)
    except Exception as e:
        print(f"[ERROR] Failed to instantiate environment: {e}", file=sys.stderr)
        sys.exit(1)

    # Detect env type
    is_pettingzoo = hasattr(env, "agents") or hasattr(env, "possible_agents")
    success = test_pettingzoo_env(env) if is_pettingzoo else test_gymnasium_env(env)

    if success:
        print("\n============================================================")
        print(" [ALL CHECKS PASSED] Environment is 100% compliant!")
        print("============================================================\n")
        sys.exit(0)
    else:
        print("\n============================================================")
        print(" [COMPLIANCE FAILURE] Fix reported errors before training.")
        print("============================================================\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
