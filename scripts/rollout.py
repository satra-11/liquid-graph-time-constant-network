# scripts/generate_rollouts.py
import vmas
import torch

def run_one_episode(env_name, n_agents, sensor_range, policy, seed, num_envs=1):
    env = vmas.make_env(
        num_envs=num_envs,
        scenario=env_name,
        n_agents=n_agents,
        agent_radius=0.05,
        world_size=1.0,
        proximity=sensor_range,
        seed=seed,
    )
    obs, _ = env.reset()
    done, leader_err, flock_err = False, [], []

    while not done:
        # policy は [N, act_dim] を返す関数 (学習済 model か expert)
        act = policy(torch.tensor(obs))
        obs, _, terminated, truncated, info = env.step(act.numpy())
        done = terminated or truncated
        leader_err.append(info["leader_error"])
        flock_err.append(info["flocking_error"])

    return torch.tensor(leader_err).mean(), torch.tensor(flock_err).mean()

# ループして .pt に保存（後でプロット時に読むだけ）
