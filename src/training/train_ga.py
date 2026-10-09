import os
import sys
from pathlib import Path
import argparse
import time
import multiprocessing as mp
import numpy as np

# Ensure repository root is on sys.path for direct CLI execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.ga_mlp import GAMLP
from src.agents.heuristic import HeuristicAgent
from src.env.vector_env import VectorArenaEnv


def _evaluate_individual_worker(
    ind_weights: np.ndarray,
    opp_specs: list[tuple[str, np.ndarray | None]],
    num_matches: int,
    map_name: str,
    max_steps: int,
) -> float:
    """Worker function executing matches for a single individual in a separate process."""
    ind = GAMLP(weights=ind_weights)
    opponents = []
    for opp_type, opp_w in opp_specs:
        if opp_type == "heuristic":
            opponents.append(HeuristicAgent())
        else:
            opponents.append(GAMLP(weights=opp_w))

    env = VectorArenaEnv(num_envs=1, map_name=map_name)
    total_score = 0.0

    for m in range(num_matches):
        opp = opponents[m % len(opponents)]
        obs = env.reset()
        match_score = 0.0

        for _ in range(max_steps):
            act_0 = ind.act(obs[0, 0])
            act_1 = opp.act(obs[0, 1])
            actions = np.stack([[act_0, act_1]], axis=0)
            obs, rewards, dones, infos = env.step(actions)

            match_score += float(rewards[0, 0])
            if dones[0]:
                if infos[0].get("winner") == "agent_0":
                    match_score += 10.0
                elif infos[0].get("winner") == "agent_1":
                    match_score -= 10.0
                break

        total_score += match_score

    return float(total_score / num_matches)


def evaluate_population_multiprocessing(
    population: list[GAMLP],
    opponent_pool: list,
    num_matches_per_ind: int = 4,
    map_name: str = "Open",
    max_steps: int = 500,
    num_workers: int | None = None,
) -> np.ndarray:
    """Evaluate population using multiprocessing over CPU cores."""
    pop_size = len(population)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, pop_size, 8)

    # Serialize opponents for cross-process compatibility
    opp_specs = []
    for opp in opponent_pool:
        if isinstance(opp, HeuristicAgent):
            opp_specs.append(("heuristic", None))
        elif isinstance(opp, GAMLP):
            opp_specs.append(("ga", opp.get_weights()))
        else:
            opp_specs.append(("heuristic", None))

    tasks = [
        (ind.get_weights(), opp_specs, num_matches_per_ind, map_name, max_steps)
        for ind in population
    ]

    if num_workers > 1:
        try:
            with mp.Pool(processes=num_workers) as pool:
                fitnesses = pool.starmap(_evaluate_individual_worker, tasks)
            return np.array(fitnesses, dtype=np.float32)
        except Exception as e:
            # Fallback to sequential evaluation if multiprocessing fails
            print(f"[Warning] Multiprocessing fallback to sequential: {e}")

    # Sequential fallback
    fitnesses = [_evaluate_individual_worker(*task) for task in tasks]
    return np.array(fitnesses, dtype=np.float32)


# Backwards compatibility alias
evaluate_population_vectorized = evaluate_population_multiprocessing


def train_ga(
    pop_size: int = 64,
    generations: int = 20,
    elite_ratio: float = 0.25,
    mutation_rate: float = 0.08,
    mutation_sigma: float = 0.12,
    save_dir: str = "checkpoints/ga",
    map_name: str = "Open",
    num_workers: int | None = None,
) -> GAMLP:
    """Run Genetic Algorithm neuroevolution loop with truncation selection & Gaussian perturbation."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== Starting GA Neuroevolution ===")
    print(f"Population: {pop_size} | Generations: {generations} | Elite Ratio: {elite_ratio}")

    # 1. Initialize random population of GAMLP policies
    population = [GAMLP() for _ in range(pop_size)]
    heuristic = HeuristicAgent()
    opponent_pool = [heuristic]

    num_elites = max(2, int(pop_size * elite_ratio))
    best_overall_fitness = -np.inf
    best_overall_agent = population[0].clone()

    t_start = time.perf_counter()

    for gen in range(1, generations + 1):
        gen_start = time.perf_counter()

        # Evaluate population using multiprocessing
        fitnesses = evaluate_population_multiprocessing(
            population,
            opponent_pool,
            num_matches_per_ind=2,
            map_name=map_name,
            max_steps=400,
            num_workers=num_workers,
        )

        # Sort descending by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        best_fit = fitnesses[sorted_indices[0]]
        mean_fit = np.mean(fitnesses)

        # Track best
        current_best = population[sorted_indices[0]]
        if best_fit > best_overall_fitness:
            best_overall_fitness = best_fit
            best_overall_agent = current_best.clone()
            best_overall_agent.save(os.path.join(save_dir, "best_ga_agent.npy"))

        # Elitism & Truncation Selection
        elites = [population[i].clone() for i in sorted_indices[:num_elites]]

        # Periodically add current best elite into opponent pool to promote arms-race
        if gen % 5 == 0:
            opponent_pool.append(current_best.clone())
            if len(opponent_pool) > 6:
                opponent_pool.pop(1)

        # Next generation: preserve elites, fill remainder with mutated copies of elites
        next_population = [e.clone() for e in elites]
        while len(next_population) < pop_size:
            parent = np.random.choice(elites)
            child = parent.mutate(mutation_rate=mutation_rate, sigma=mutation_sigma)
            next_population.append(child)

        population = next_population
        gen_time = time.perf_counter() - gen_start

        print(
            f"Gen {gen:3d}/{generations} | Best: {best_fit:7.2f} | Mean: {mean_fit:7.2f} | "
            f"Elites: {num_elites} | Time: {gen_time:.2f}s"
        )

    total_time = time.perf_counter() - t_start
    print(f"\nGA Training Complete in {total_time:.1f}s! Best fitness: {best_overall_fitness:.2f}")
    save_path = os.path.join(save_dir, "best_ga_agent.npy")
    best_overall_agent.save(save_path)
    print(f"Saved best GA policy to {save_path}")

    return best_overall_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kinetic Sumo using GA Neuroevolution")
    parser.add_argument("--pop-size", type=int, default=64, help="Population size")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--map", type=str, default="Open", help="Arena map preset")
    parser.add_argument("--save-dir", type=str, default="checkpoints/ga", help="Output directory")
    args = parser.parse_args()

    train_ga(pop_size=args.pop_size, generations=args.generations, map_name=args.map, save_dir=args.save_dir)
