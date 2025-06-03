import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from collections import deque
from tqdm import tqdm

# Detectar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_sizes = [2048, 1024, 512, 256, 128]  # Capas más profundas

        layers = []
        input_size = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden))  # Ayuda en estabilización
            input_size = hidden

        self.shared = nn.Sequential(*layers)

        self.policy = nn.Sequential(
            nn.Linear(hidden_sizes[-1], act_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 1)
        )

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
        epochs=10, minibatch_size=64, clip_eps=0.2, gamma=0.99, lam=0.95, lr=3e-4,
        initial_entropy_coef=0.01, final_entropy_coef=0.001,
        model=None, idx=0):

    logs = []
    entropy_decay_steps = total_timesteps

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if model is None:
        model = PPOActorCritic(obs_dim, act_dim)

    model.to(device)  # Mover modelo a GPU
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=20,
        threshold=0.01,
        min_lr=1e-6,
        verbose=True
    )

    obs, _ = env.reset()
    observations, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
    ep_rewards = []
    reward_buffer = deque(maxlen=100)
    best_avg_reward = -float('inf')
    best_model_state = None


    policy_loss_val, value_loss_val = None, None

    for timestep in tqdm(range(1, total_timesteps + 1)):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
            obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32, device=device)
            act_tensor = torch.tensor(actions, device=device)
            old_logprobs = torch.tensor(logprobs, device=device)

            with torch.no_grad():
                _, values_tensor = model(obs_tensor)

            advs, returns = compute_returns(rewards, dones, values, gamma, lam)
            advs = torch.tensor(advs, dtype=torch.float32, device=device)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
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
                    optimizer.step()

                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()

            avg_reward = np.mean(reward_buffer) if reward_buffer else 0
            # Guardar mejor modelo hasta ahora
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model_state = model.state_dict()

            scheduler.step(avg_reward)
            print(f"Step {timestep} | Avg Reward (last 100 eps): {avg_reward:.2f} "
                  f"| Policy Loss: {policy_loss_val:.4f} | Value Loss: {value_loss_val:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.6f} | Entropy: {entropy.item():.4f} "
                  f"| Entropy Coef: {current_entropy_coef:.6f}")
            logs.append({
                'timestep': timestep,
                'avg_reward': avg_reward,
                'policy_loss': policy_loss_val,
                'value_loss': value_loss_val,
                'entropy': entropy.item(),
                'entropy_coef': current_entropy_coef,
                'lr': optimizer.param_groups[0]['lr']
            })


            observations, actions, logprobs, rewards, dones, values = [], [], [], [], [], []

    env.close()
    print("Entrenamiento PPO finalizado.")
    
    # Exportar log como CSV
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"ppo_training_log_{idx}.csv", index=False)
    
    # Devolver el mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
 
    return model, avg_reward
