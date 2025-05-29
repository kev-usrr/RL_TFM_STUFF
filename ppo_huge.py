import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Activación Swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Bloque residual estilo ResNet
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            Swish(),
            nn.LayerNorm(size),
            nn.Linear(size, size),
            Swish(),
            nn.LayerNorm(size)
        )

    def forward(self, x):
        return x + self.block(x)

# Red Actor-Critic con bloques residuales y Swish
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_sizes = [2048, 1024, 1024, 512, 512]

        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            Swish(),
            nn.LayerNorm(hidden_sizes[0])
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            Swish(),
            nn.LayerNorm(hidden_sizes[1]),
            ResidualBlock(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            Swish(),
            nn.LayerNorm(hidden_sizes[2]),
            ResidualBlock(hidden_sizes[2]),
        )

        self.policy = nn.Linear(hidden_sizes[2], act_dim)
        self.value = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.policy(x), self.value(x)


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
        epochs=10, minibatch_size=128, clip_eps=0.2, gamma=0.99, lam=0.95, lr=3e-4,
        initial_entropy_coef=0.05, final_entropy_coef=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    entropy_decay_steps = total_timesteps

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = PPOActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1_000_000, eta_min=1e-6
    )

    obs, _ = env.reset()
    observations, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
    ep_rewards = []
    reward_buffer = deque(maxlen=100)

    policy_loss_val, value_loss_val = None, None

    for timestep in range(1, total_timesteps + 1):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
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
            obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32).to(device)
            act_tensor = torch.tensor(actions).to(device)
            old_logprobs = torch.tensor(logprobs).to(device)

            with torch.no_grad():
                _, values_tensor = model(obs_tensor)

            advs, returns = compute_returns(rewards, dones, values, gamma, lam)
            advs = torch.tensor(advs, dtype=torch.float32).to(device)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            current_entropy_coef = final_entropy_coef + (initial_entropy_coef - final_entropy_coef) * \
                                   max(0, (entropy_decay_steps - timestep) / entropy_decay_steps)

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

                    policy_loss = -torch.min(
                        ratio * batch_adv,
                        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
                    ).mean()
                    value_loss = ((value.squeeze() - batch_ret) ** 2).mean()
                    entropy = dist.entropy().mean()

                    loss = policy_loss + 0.5 * value_loss - current_entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()

            avg_reward = np.mean(reward_buffer) if reward_buffer else 0
            print(f"Step {timestep} | Avg Reward (last 100 eps): {avg_reward:.2f} "
                  f"| Policy Loss: {policy_loss_val:.4f} | Value Loss: {value_loss_val:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.6f} | Entropy: {entropy.item():.4f} "
                  f"| Entropy Coef: {current_entropy_coef:.6f}")

            observations, actions, logprobs, rewards, dones, values = [], [], [], [], [], []

        scheduler.step()

    env.close()
    print("Entrenamiento PPO finalizado.")
    return model, avg_reward