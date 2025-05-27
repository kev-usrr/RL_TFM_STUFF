import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy = nn.Linear(64, act_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value


def compute_returns(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo(env, total_timesteps=100_000, update_timesteps=2000,
        epochs=10, minibatch_size=64, clip_eps=0.2, gamma=0.99, lam=0.95, lr=3e-4):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = PPOActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset()
    observations, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
    ep_rewards = []
    reward_buffer = deque(maxlen=100)

    policy_loss_val, value_loss_val = None, None  # inicialización

    for timestep in range(1, total_timesteps + 1):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = model(obs_tensor)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        observations.append(obs)
        actions.append(action.item())
        logprobs.append(probs.log_prob(action).item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        obs = next_obs
        ep_rewards.append(reward)

        if done:
            reward_sum = sum(ep_rewards)
            reward_buffer.append(reward_sum)
            ep_rewards = []
            obs, _ = env.reset()

        if timestep % update_timesteps == 0:
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
            act_tensor = torch.tensor(actions)
            old_logprobs = torch.tensor(logprobs)

            with torch.no_grad():
                _, values_tensor = model(obs_tensor)

            advs, returns = compute_returns(rewards, dones, values, gamma, lam)
            advs = torch.tensor(advs, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            for _ in range(epochs):
                idxs = np.random.permutation(len(observations))
                for i in range(0, len(observations), minibatch_size):
                    batch_idx = idxs[i:i + minibatch_size]
                    batch_obs = obs_tensor[batch_idx]
                    batch_act = act_tensor[batch_idx]
                    batch_old_logprob = old_logprobs[batch_idx]
                    batch_adv = advs[batch_idx]
                    batch_ret = returns[batch_idx]

                    logits, value = model(batch_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    logprob = dist.log_prob(batch_act)
                    ratio = torch.exp(logprob - batch_old_logprob)

                    policy_loss = -torch.min(ratio * batch_adv,
                                             torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv).mean()
                    value_loss = ((value.squeeze() - batch_ret) ** 2).mean()
                    loss = policy_loss + 0.5 * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()

            avg_reward = np.mean(reward_buffer) if reward_buffer else 0
            print(f"Step {timestep} | Avg Reward (last 100 eps): {avg_reward:.2f} "
                  f"| Policy Loss: {policy_loss_val:.4f} | Value Loss: {value_loss_val:.4f}")

            # Limpiar buffer de batch
            observations, actions, logprobs, rewards, dones, values = [], [], [], [], [], []

    env.close()
    print("Entrenamiento PPO finalizado.")

