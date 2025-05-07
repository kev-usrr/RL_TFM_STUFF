# PPO desde cero con PyTorch y reward diferenciable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Política (actor) y value function (crítico) ---

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.actor(x), self.critic(x)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value.squeeze()

# --- 2. Red de agregación de reward ---

class RewardAggregator(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_agents, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, reward_vector):
        return self.model(reward_vector).squeeze()  # output: escalar

# --- 3. Entrenamiento PPO simplificado ---

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[step]
    return torch.tensor(advantages, device=rewards.device)

def ppo_update(policy, reward_model, optimizer, trajectories, clip_ratio=0.2):
    obs = torch.stack(trajectories['obs'])
    act = torch.stack(trajectories['actions'])
    logp_old = torch.stack(trajectories['logprobs'])
    returns = trajectories['returns']
    advs = trajectories['advantages']

    logits, values = policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(act)
    ratio = torch.exp(logp - logp_old)

    # PPO clipped surrogate loss
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advs
    policy_loss = -(torch.min(ratio * advs, clip_adv)).mean()
    value_loss = ((returns - values.squeeze()) ** 2).mean()

    # Total loss (policy + critic + reward net)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- 5. Entrenamiento principal ---

def train_ppo(env, n_episodes=1_000, max_timesteps=10_000, file_name='ppo.pt'):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_agents = env.num_agents

    policy = PolicyNetwork(obs_dim, act_dim)
    reward_model = RewardAggregator(num_agents)

    optimizer = optim.Adam(list(policy.parameters()) + list(reward_model.parameters()), lr=1e-3)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        # print(obs.dtype)
        trajectories = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': []}

        for t in tqdm(range(max_timesteps)):
            # obs_tensor = obs.float()
            obs_tensor = torch.from_numpy(obs).float()
            action, logprob, value = policy.get_action(obs_tensor)
            new_obs, reward_vector, done, _, _ = env.step(int(action.item()))
            reward_vector = reward_vector
            
            scalar_reward = reward_model(reward_vector)
            # print(reward_vector, scalar_reward)

            trajectories['obs'].append(obs_tensor)
            trajectories['actions'].append(action)
            trajectories['logprobs'].append(logprob)
            trajectories['rewards'].append(scalar_reward)
            trajectories['values'].append(value)

            obs = new_obs
            if done:
                break

        rewards = torch.stack(trajectories['rewards'])
        values = torch.stack(trajectories['values'])
        advantages = compute_advantages(rewards, values)
        returns = (advantages + values)

        trajectories['advantages'] = advantages.detach()
        trajectories['returns'] = returns.detach()

        ppo_update(policy, reward_model, optimizer, trajectories)

        # if episode % 100 == 0:
        print(f"Episode {episode}, Reward sum: {rewards.sum().item():.2f}")

    torch.save({
        'policy_state_dict': policy.state_dict(),
        'reward_model_state_dict': reward_model.state_dict()
    }, file_name)

