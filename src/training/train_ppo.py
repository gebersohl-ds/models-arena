"""Self-Play Proximal Policy Optimization (PPO) with Prioritized Fictitious Self-Play (PFSP)."""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.ppo_policy import PPOPolicy
from src.agents.heuristic import HeuristicAgent
from src.env.vector_env import VectorArenaEnv
from src.training.league import LeagueManager


def train_ppo(
    total_timesteps: int = 50_000,
    num_envs: int = 16,
    num_steps: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    update_epochs: int = 4,
    num_minibatches: int = 4,
    snapshot_interval: int = 5,
    save_dir: str = "checkpoints/ppo",
    map_name: str = "Open",
) -> PPOPolicy:
    """Train Kinetic Sumo policy using PPO Self-Play on CPU."""
    os.makedirs(save_dir, exist_ok=True)
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    print(f"=== Starting PPO Self-Play Training ===")
    print(f"Timesteps: {total_timesteps:,} | Iterations: {num_iterations} | Envs: {num_envs} | Steps/Env: {num_steps}")
    print(f"Batch Size: {batch_size} | Minibatch Size: {minibatch_size} | Map: {map_name}")

    # Initialize environment and league
    env = VectorArenaEnv(num_envs=num_envs, map_name=map_name, gamma=gamma)
    league = LeagueManager(checkpoint_dir=os.path.join(save_dir, "league"))

    # Learner policy (Agent 0)
    device = torch.device("cpu")
    policy = PPOPolicy(obs_dim=46, act_dim=3, hidden_dim=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    # Initial snapshot to league
    league.set_best_agent(policy)
    league.add_snapshot(policy, iteration=0)

    # Rollout storage buffers (for learning Agent 0)
    obs_buf = torch.zeros((num_steps, num_envs, 46), dtype=torch.float32)
    actions_buf = torch.zeros((num_steps, num_envs, 3), dtype=torch.float32)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32)

    # Opponent tracking per environment
    opponents: list[tuple[any, str]] = [league.sample_opponent(policy) for _ in range(num_envs)]

    # Initial reset
    obs = env.get_observations()  # (num_envs, 2, 46)

    global_step = 0
    start_time = time.perf_counter()

    for iteration in range(1, num_iterations + 1):
        policy.eval()

        for step in range(num_steps):
            global_step += num_envs

            obs_0 = torch.as_tensor(obs[:, 0], dtype=torch.float32)  # (num_envs, 46)
            obs_buf[step] = obs_0

            # 1. Action for Agent 0 (Learner)
            with torch.no_grad():
                action_0, logprob_0, _, value_0 = policy.get_action_and_value(obs_0)

            actions_buf[step] = action_0
            logprobs_buf[step] = logprob_0
            values_buf[step] = value_0.flatten()

            # 2. Action for Agent 1 (Opponents from League)
            actions_1_np = np.zeros((num_envs, 3), dtype=np.float32)
            for e_idx in range(num_envs):
                opp_agent, _ = opponents[e_idx]
                actions_1_np[e_idx] = opp_agent.act(obs[e_idx, 1])

            actions_combined = np.stack([action_0.cpu().numpy(), actions_1_np], axis=1)  # (num_envs, 2, 3)

            # 3. Environment Step
            next_obs, rewards, dones, infos = env.step(actions_combined)

            rewards_buf[step] = torch.as_tensor(rewards[:, 0], dtype=torch.float32)
            dones_buf[step] = torch.as_tensor(dones, dtype=torch.float32)

            # If environment finished, resample opponent for that env
            for e_idx in range(num_envs):
                if dones[e_idx]:
                    opponents[e_idx] = league.sample_opponent(policy)

            obs = next_obs

        # 4. GAE-Lambda Advantage Estimation
        policy.eval()
        with torch.no_grad():
            next_obs_0 = torch.as_tensor(obs[:, 0], dtype=torch.float32)
            next_value = policy.get_value(next_obs_0).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones_buf[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # 5. Flatten the rollout batch for optimization
        b_obs = obs_buf.reshape(-1, 46)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, 3)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 6. PPO Update Epochs
        policy.train()
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # 7. Checkpointing & League Snapshotting
        mean_reward = rewards_buf.mean().item()
        sps = int(global_step / (time.perf_counter() - start_time))

        if iteration % snapshot_interval == 0 or iteration == num_iterations:
            league.add_snapshot(policy, iteration=iteration)
            league.set_best_agent(policy)
            policy.save(os.path.join(save_dir, "best_ppo_policy.pt"))

        print(
            f"Iter {iteration:3d}/{num_iterations} | Step: {global_step:6d} | SPS: {sps:5d} | "
            f"Mean Rew: {mean_reward:6.3f} | Policy Loss: {pg_loss.item():.4f} | Value Loss: {v_loss.item():.4f}"
        )

    # Final save
    final_path = os.path.join(save_dir, "best_ppo_policy.pt")
    policy.save(final_path)
    print(f"\nPPO Training Complete! Policy saved to {final_path}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kinetic Sumo using Self-Play PPO")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total timesteps")
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--map", type=str, default="Open", help="Arena map preset")
    parser.add_argument("--save-dir", type=str, default="checkpoints/ppo", help="Output directory")
    args = parser.parse_args()

    train_ppo(total_timesteps=args.timesteps, num_envs=args.num_envs, map_name=args.map, save_dir=args.save_dir)
