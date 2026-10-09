#!/usr/bin/env python3
"""
Performance Profiler & Bottleneck Filter Tool
Executes cProfile on a target Python script/benchmark and filters functions
taking more than the specified threshold (default: 5.0%) of total execution time.
"""

import sys
import os
import cProfile
import pstats
import argparse
import runpy

def profile_target(script_path: str, script_args: list, threshold: float = 0.05, top_n: int = 25):
    if not os.path.isfile(script_path):
        print(f"[ERROR] Target script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    print(f"============================================================")
    print(f" Performance Profiler: {script_path}")
    print(f" Threshold: >= {threshold * 100:.1f}% total execution time")
    print(f"============================================================")

    # Backup sys.argv and configure for target
    old_argv = sys.argv.copy()
    sys.argv = [script_path] + script_args

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        runpy.run_path(script_path, run_name="__main__")
        profiler.disable()
    except SystemExit as e:
        profiler.disable()
        if e.code not in (0, None):
            print(f"[WARN] Target script exited with status: {e.code}", file=sys.stderr)
    except Exception as e:
        profiler.disable()
        print(f"[ERROR] Exception during profiled run: {e}", file=sys.stderr)
        raise
    finally:
        sys.argv = old_argv

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")

    total_time = stats.total_tt
    if total_time <= 0:
        print("[INFO] Total execution time was too small to measure (<0.001s).")
        return

    print(f"\nTotal CPU Execution Time: {total_time:.4f}s\n")
    print(f"{'Function':<42} {'File:Line':<35} {'Tottime (s)':>12} {'% Total':>9} {'Calls':>10} {'Per Call (ms)':>14}")
    print("-" * 126)

    bottlenecks_found = 0
    # Iterate through stats
    for func_tuple, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line_no, func_name = func_tuple
        pct = (tt / total_time) * 100.0

        if (tt / total_time) >= threshold:
            bottlenecks_found += 1
            # Clean display path
            short_file = os.path.basename(filename) if len(filename) > 30 else filename
            file_loc = f"{short_file}:{line_no}"
            func_disp = func_name[:40]
            per_call_ms = (tt / nc * 1000.0) if nc > 0 else 0.0

            print(f"{func_disp:<42} {file_loc:<35} {tt:>12.4f} {pct:>8.1f}% {nc:>10} {per_call_ms:>14.4f}")

    print("-" * 126)
    if bottlenecks_found == 0:
        print(f"[OK] No individual functions consumed >= {threshold * 100:.1f}% of total time.")
    else:
        print(f"[ALERT] Identified {bottlenecks_found} function(s) consuming >= {threshold * 100:.1f}% total time.")
        print(f"[HINT] Check raycasting, pairwise collision loops, and list comprehensions.")

def main():
    parser = argparse.ArgumentParser(description="Profile Python code and isolate >=5% bottlenecks.")
    parser.add_argument("script", help="Path to Python script to benchmark (e.g. scripts/benchmark_env.py)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Bottleneck threshold fraction (default 0.05 = 5%%)")
    parser.add_argument("--top", type=int, default=25, help="Max items to list (default 25)")
    parser.add_argument("script_args", nargs="*", help="Arguments forwarded to target script")

    args = parser.parse_args()
    profile_target(args.script, args.script_args, threshold=args.threshold, top_n=args.top)

if __name__ == "__main__":
    main()
