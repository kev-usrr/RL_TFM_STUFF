import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    """Red PPO estándar con actor y crítico"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        shared_features = self.shared_net(x)
        return self.actor(shared_features), self.critic(shared_features)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value.squeeze()

class RewardTransformer(nn.Module):
    """Red que transforma vectores de reward en escalares con mecanismo de atención"""
    def __init__(self, num_agents):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_agents, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(num_agents, 1)
        self.reward_scale = nn.Parameter(torch.ones(1))
        self.reward_bias = nn.Parameter(torch.zeros(1))

    def forward(self, rewards):
        attention_weights = self.attention(rewards)
        weighted_rewards = attention_weights * rewards
        return self.value_head(weighted_rewards).squeeze() * self.reward_scale + self.reward_bias

class PPOTrainer:
    def __init__(self, env, lr_policy=3e-4, lr_reward=1e-3, gamma=0.99, lam=0.95, clip_ratio=0.2):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.num_agents = getattr(env, 'num_agents', 1)  # Asume 1 si no hay múltiples agentes
        
        self.policy = PolicyNetwork(self.obs_dim, self.act_dim).to(device)
        self.reward_net = RewardTransformer(self.num_agents).to(device)
        
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.reward_optim = optim.Adam(self.reward_net.parameters(), lr=lr_reward)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = 0.01
        
        # Estadísticas para normalización
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_momentum = 0.9

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        last_advantage = 0
        next_value = 0
        next_done = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_value = values[-1]
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_advantage = delta + self.gamma * self.lam * next_non_terminal * last_advantage
            advantages.insert(0, last_advantage)
        
        return torch.stack(advantages)

    def normalize_rewards(self, rewards):
        """Normalización online de recompensas"""
        batch_mean = rewards.mean()
        batch_std = rewards.std()
        
        self.reward_mean = self.reward_momentum * self.reward_mean + (1 - self.reward_momentum) * batch_mean
        self.reward_std = self.reward_momentum * self.reward_std + (1 - self.reward_momentum) * batch_std
        
        return (rewards - self.reward_mean) / (self.reward_std + 1e-8)

    def update(self, trajectories):
      obs = torch.stack(trajectories['obs'])
      actions = torch.stack(trajectories['actions'])
      old_logprobs = torch.stack(trajectories['logprobs'])
      rewards = torch.stack(trajectories['rewards'])
      values = torch.stack(trajectories['values'])
      dones = torch.tensor(trajectories['dones'], dtype=torch.float32)
      
      # Normalización de rewards (sin grafo computacional)
      with torch.no_grad():
          rewards = self.normalize_rewards(rewards)
      
      # Calcular advantages (sin grafo computacional)
      with torch.no_grad():
          advantages = self.compute_advantages(rewards, values, dones)
          returns = advantages + values
      
      # Forward pass
      logits, new_values = self.policy(obs)
      dist = Categorical(logits=logits)
      new_logprobs = dist.log_prob(actions)
      entropy = dist.entropy().mean()
      
      ratio = (new_logprobs - old_logprobs).exp()
      
      # Pérdidas
      policy_loss1 = ratio * advantages
      policy_loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
      policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
      
      value_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
      
      # Pérdida total
      loss = policy_loss + value_loss - self.entropy_coef * entropy
      
      # Backpropagation - SOLUCIÓN CLAVE
      self.policy_optim.zero_grad()
      self.reward_optim.zero_grad()
      
      # Solo un backward() sin retain_graph
      loss.backward()
      
      print("\n" + "="*50)
      print("GRADIENTES EN REWARD NETWORK:")
      for name, param in self.reward_net.named_parameters():
          if param.grad is not None:
              print(f"{name:20} grad_mean: {param.grad.abs().mean().item():.5f}")
          else:
              print(f"{name:20} SIN GRADIENTES")
      
      print("\nGRADIENTES EN POLICY NETWORK:")
      for name, param in self.policy.named_parameters():
          if param.grad is not None:
              print(f"{name:20} grad_mean: {param.grad.abs().mean().item():.5f}")
      print("="*50 + "\n")
      
      # Gradient clipping
      torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
      torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 0.5)
      
      self.policy_optim.step()
      self.reward_optim.step()
      
      return loss.item()

    def train(self, episodes, max_steps=1000, save_path='ppo_reward_model.pt'):
        best_reward = float('-inf')
        alpha = 1.0
        min_alpha = 0.0
        alpha_decay = (alpha - min_alpha) / episodes
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            trajectories = {
                'obs': [], 'actions': [], 'logprobs': [],
                'rewards': [], 'values': [], 'dones': []
            }
            episode_rewards = []
            
            for _ in range(max_steps):
                obs_tensor = torch.FloatTensor(obs).to(device)
                action, logprob, value = self.policy.get_action(obs_tensor)
                
                next_obs, reward_vector, done, _, _ = self.env.step(action.cpu().numpy().item())
                
                # print(reward_vector)
                # print(reward_vector.mean())
                # Transformar vector de reward a escalar
                reward_tensor = torch.FloatTensor(reward_vector).to(device)
                scalar_reward = self.reward_net(reward_tensor)
                
                # Guardar transición
                trajectories['obs'].append(obs_tensor)
                trajectories['actions'].append(action)
                trajectories['logprobs'].append(logprob)
                trajectories['rewards'].append(scalar_reward)
                trajectories['values'].append(value)
                trajectories['dones'].append(done)
                
                obs = next_obs
                episode_rewards.append(sum(reward_vector))
                
                if done:
                    break
            
            # Actualizar redes
            loss = self.update(trajectories)
            alpha = max(alpha - alpha_decay, min_alpha)
            
            # Guardar mejor modelo
            mean_episode_reward = np.mean(episode_rewards)
            if mean_episode_reward > best_reward:
                best_reward = mean_episode_reward
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'reward_state_dict': self.reward_net.state_dict()
                }, save_path)
            
            print(f"Episode {episode}, Loss: {loss:.4f}, Reward: {mean_episode_reward:.2f}, Best: {best_reward:.2f}, Alpha:{alpha:.2f}")

# Uso:
# env = TuEntorno()
# trainer = PPOTrainer(env)
# trainer.train(episodes=1000)