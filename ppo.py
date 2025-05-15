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
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
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
        self.attention = nn.Sequential(
            nn.Linear(num_agents, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Linear(num_agents, 1)
        
    def forward(self, reward_vector):
        weights = self.attention(reward_vector)
        weighted_rewards = weights * reward_vector
        return self.value(weighted_rewards).squeeze()

# --- 3. Entrenamiento PPO simplificado ---

def baseline_reward_fn(reward_vector):
    return reward_vector.mean()


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae.clone())
        next_value = values[step]
    return torch.stack(advantages)


def ppo_update(policy, reward_model, policy_optimizer, reward_optimizer, trajectories, clip_ratio=0.2):
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
    entropy = dist.entropy().mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    # Backpropagation con dos optimizadores
    policy_optimizer.zero_grad()
    reward_optimizer.zero_grad()
    loss.backward()

    # Monitorizar normas de gradientes
    policy_grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in policy.parameters() if p.grad is not None]))
    reward_grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in reward_model.parameters() if p.grad is not None]))
    print(f"Policy grad norm: {policy_grad_norm:.4f}, Reward grad norm: {reward_grad_norm:.4f}")
    
    # Gradient clipping opcional pero recomendado
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=0.5)
    
    policy_optimizer.step()
    reward_optimizer.step()

    return loss.item()

# --- 5. Entrenamiento principal ---

def train_ppo(env, n_episodes=1_000, max_timesteps=10_000, file_name='ppo.pt'):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_agents = env.num_agents
    best_model = None
    best_mean_rew = float('-inf')

    policy = PolicyNetwork(obs_dim, act_dim)
    reward_model = RewardAggregator(num_agents)

    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)

    alpha = 1.0
    min_alpha = 0.0
    alpha_decay = (alpha - min_alpha) / n_episodes

    for episode in range(n_episodes):
        obs, _ = env.reset()
        # print(obs.dtype)
        trajectories = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': []}

        all_rewards = []
        for t in tqdm(range(max_timesteps)):
            # obs_tensor = obs.float()
            obs_tensor = torch.from_numpy(obs).float()
            action, logprob, value = policy.get_action(obs_tensor)
            new_obs, reward_vector, done, _, _ = env.step(int(action.item()))
            all_rewards.append(reward_vector)
            
            if len(new_obs) == 0:
                new_obs = np.zeros(shape=(obs_dim, ))
            reward_vector_tensor = torch.tensor(reward_vector, dtype=torch.float32, requires_grad=True)
            learned_reward = reward_model(reward_vector_tensor)

            heuristic_reward = torch.tensor(baseline_reward_fn(reward_vector), dtype=torch.float32, requires_grad=False)
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            
            scalar_reward = alpha_tensor * learned_reward + (1 - alpha_tensor) * heuristic_reward

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

        trajectories['advantages'] = advantages
        trajectories['returns'] = returns

        loss_val = ppo_update(policy, reward_model, policy_optimizer, reward_optimizer, trajectories)
        alpha = max(alpha - alpha_decay, min_alpha)

        last_rew_mean = np.sum(np.array(all_rewards))
        if last_rew_mean > best_mean_rew:
            best_mean_rew = last_rew_mean
            print(f'Best mean reward reached: {best_mean_rew}')
            best_model = {
                'policy_state_dict': policy.state_dict(),
                'reward_model_state_dict': reward_model.state_dict()
            }

        # if episode % 100 == 0:
        print(f"Episode {episode}, Loss: {loss_val}, Rewards mean: {last_rew_mean}")

    if best_model is not None:
        torch.save(best_model, file_name)
        print(f"Mejor modelo guardado con mean reward {best_mean_rew:.4f}")

