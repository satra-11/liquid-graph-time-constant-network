#!/usr/bin/env python
"""
train.py ― minimal training script for LGTC / CfGC controllers

* This script has been adapted to work with a custom "flocking" scenario
  that uses Lidar sensors and a continuous HeuristicPolicy expert.
"""

# ---------------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import random
from pathlib import Path
import importlib

# ---------------------------------------------------------------------------
# 3rd-party libs
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import vmas
from vmas.simulator.environment import Environment

from lgtcn.models import LGTCNController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_obs(obs):
    """Convert raw VMAS observation to (batch_size, N_agents, obs_dim) ndarray."""
    if isinstance(obs, dict):
        obs = [obs[k] for k in sorted(obs.keys())]
    # NOTE: Stack along axis=1 to get the correct (batch, agent, feature) shape
    if isinstance(obs, (list, tuple)):
        obs = np.stack(obs, axis=1)
    return obs


def get_single_spaces(env: Environment):
    """Return (single_observation_space, single_action_space) regardless of API."""
    if hasattr(env, "single_observation_space"):
        return env.single_observation_space, env.single_action_space

    def unwrap(space):
        while True:
            if isinstance(space, (list, tuple)):
                space = space[0]
            elif isinstance(space, dict):
                space = next(iter(space.values()))
            elif hasattr(space, "spaces"):
                space = space[0]
            else:
                break
        return space
    return unwrap(env.observation_space), unwrap(env.action_space)


def build_adj(obs_arr: np.ndarray, sensor_range: float, device: torch.device) -> torch.Tensor:
    """Compute adjacency matrix based on pairwise distance (< sensor_range)."""
    if obs_arr.ndim == 2:
        obs_arr = obs_arr[None, :, :]

    pos = obs_arr[:, :, :2]
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    A = (dist < sensor_range).astype(np.float32)
    B, N, _ = A.shape
    A[:, np.arange(N), np.arange(N)] = 0.0
    return torch.as_tensor(A, device=device)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(n_agents: int, seed: int) -> Environment:
    """Create a single-env VMAS environment for the custom scenario."""
    # NOTE: `sensor_range` is removed as the custom scenario does not accept it.
    return vmas.make_env(
        num_envs=1,
        scenario="flocking",
        n_agents=n_agents,
        seed=seed,
    )


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Loss: behaviour cloning with continuous actions
# ---------------------------------------------------------------------------

def imitation_loss(policy_logits: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
    """
    Behaviour Cloning loss for continuous actions using Mean Squared Error.
    """
    # Squash the model's raw output (logits) to the action range [-1, 1]
    policy_action = torch.tanh(policy_logits)
    # Calculate the difference between the model's action and the expert's action
    return nn.functional.mse_loss(policy_action, expert_action)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = choose_device()
    set_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Environment ────────────────────────────────────────────────────────
    env = make_env(args.n_agents, args.seed)
    obs_space, act_space = get_single_spaces(env)
    obs_dim = obs_space.shape[-1]

    # The custom scenario uses continuous actions.
    discrete_act = False
    act_dim = act_space.shape[-1]

    # ── Expert Policy ──────────────────────────────────────────────────────
    # NOTE: Import the module directly instead of relying on dynamic introspection
    scenario_module = importlib.import_module("vmas.scenarios.flocking")
    expert_policy = scenario_module.HeuristicPolicy(env.scenario)
    print(f"Using HeuristicPolicy from {scenario_module.__name__} as expert.")


    # ── Model ──────────────────────────────────────────────────────────────
    model = LGTCNController(
        input_dim=obs_dim,
        hidden_dim=args.hidden,
        K=args.K,
        output_dim=act_dim,
        use_closed_form=args.closed_form,
    ).to(device)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    episode_returns: list[float] = []

    for epoch in range(args.epochs):
        obs_raw = env.reset()
        if isinstance(obs_raw, tuple):
            obs, _ = obs_raw
        else:
            obs = obs_raw

        done, ep_ret = False, 0.0
        x_state = None

        while not done:
            obs_arr = preprocess_obs(obs)
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            
            # NOTE: Use the hardcoded Lidar range (0.2) from the custom scenario
            S = build_adj(obs_arr, 0.2, device)

            logits, x_state_new = model(obs_t, S, x_state)

            # ── Expert action retrieval ────────────────────────────────────
            with torch.no_grad():
                # The expert policy expects a 2D tensor (N, D), but our model uses a 3D tensor (B, N, D).
                # We reshape the observation before passing it to the expert.
                batch_size, num_agents, obs_features = obs_t.shape
                obs_for_expert = obs_t.reshape(batch_size * num_agents, obs_features)
                
                expert_actions_flat = expert_policy.compute_action(obs_for_expert, u_range=1.0)
                
                # Reshape the expert's action back to match the model's output shape.
                expert_actions = expert_actions_flat.reshape(batch_size, num_agents, -1)


            loss = imitation_loss(logits, expert_actions)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            # ── Env step ──────────────────────────────────────────────────
            # The action is continuous, so we apply tanh to the model output.
            action_tensor = logits.tanh().detach()[0] # Get actions for the single batch -> shape (num_agents, action_dim)
            
            # The environment expects a list of tensors, where each tensor is for one agent
            # and has shape (num_envs, action_dim).
            action_for_env = [action_tensor[i].unsqueeze(0) for i in range(args.n_agents)]

            # NOTE: Unpack 4 values from env.step, consistent with older gym API
            obs, reward, done, _ = env.step(action_for_env)
            
            # The reward might be per-agent, so we average it
            if reward is not None and hasattr(reward, "mean"):
                ep_ret += float(reward.mean())
            x_state = x_state_new.detach()

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[{global_step:7d}] loss={loss.item():.4e}")

        episode_returns.append(ep_ret)
        print(f"Epoch {epoch:03d} ─ return={ep_ret:.3f}")

        # ── Checkpoint ─────────────────────────────────────────────────────
        if (epoch + 1) % args.ckpt_every == 0:
            path = ckpt_dir / f"model_e{epoch+1:03d}.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, path)
            print(f"Saved checkpoint → {path}")

    # ── Training finished ─────────────────────────────────────────────────
    json.dump({"returns": episode_returns, "args": vars(args)},
              open(ckpt_dir / "train_log.json", "w"), indent=2)
    print("Training finished ✔")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli():
    p = argparse.ArgumentParser(description="Train LGTC / CfGC controller on a custom VMAS scenario")
    p.add_argument("--n_agents", type=int, default=10, help="number of agents N")
    p.add_argument("--hidden", type=int, default=32, help="hidden dimension")
    p.add_argument("-K", type=int, default=2, help="graph filter order")
    p.add_argument("--closed-form", action="store_true", help="use CfGC layer instead of LGTC")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_cli())
