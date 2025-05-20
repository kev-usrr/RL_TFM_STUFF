import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    """Red PPO estándar con actor y crítico con escala y sesgo aprendibles"""
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
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Parámetros aprendibles para escalar y desplazar la predicción del crítico
        self.critic_scale = nn.Parameter(torch.tensor(0.1))  # Escala inicial
        self.critic_bias = nn.Parameter(torch.tensor(0.0))   # Desplazamiento inicial
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # log(T) para asegurar que T > 0

    def forward(self, x):
        shared_features = self.shared_net(x)
        logits = self.actor(shared_features)
        raw_value = self.critic(shared_features).squeeze(-1)

        # Aplicar escala y sesgo aprendibles al valor del crítico
        scaled_value = raw_value * torch.clamp(self.critic_scale, 0.01, 10.0) + self.critic_bias
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        scaled_logits = logits / temperature
        
        return scaled_logits, scaled_value

    def get_action(self, obs):
        logits, value = self.forward(obs)
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        scaled_logits = logits / temperature
        dist = Categorical(logits=scaled_logits)
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
        self.num_agents = getattr(env, 'num_agents', 1)
        
        # Mover modelos al dispositivo correcto
        self.policy = PolicyNetwork(self.obs_dim, self.act_dim).to(device)
        self.reward_net = RewardTransformer(self.num_agents).to(device)
        
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.reward_optim = optim.Adam(self.reward_net.parameters(), lr=lr_reward)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = 0.1
        
        # Estadísticas para normalización
        self.reward_mean = torch.zeros(1, device=device)
        self.reward_std = torch.ones(1, device=device)
        self.reward_momentum = 0.9

        print(f'NUM AGENTS: {self.num_agents}')

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        last_advantage = 0
        next_value = 0
        next_done = 0
        
        # Convertir dones a tensor en el dispositivo correcto
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
        
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

    def update(self, trajectories):
        # Mover todos los tensores al dispositivo correcto
        obs = torch.stack(trajectories['obs']).to(device)
        actions = torch.stack(trajectories['actions']).to(device)
        old_logprobs = torch.stack(trajectories['logprobs']).to(device)
        rewards = torch.stack(trajectories['rewards']).to(device)
        values = torch.stack(trajectories['values']).to(device)
        dones = torch.tensor(trajectories['dones'], dtype=torch.float32, device=device)
        
        advantages = self.compute_advantages(rewards, values.detach(), dones)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
        
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        value_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
        
        # Pérdida total
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Backpropagation
        self.policy_optim.zero_grad()
        self.reward_optim.zero_grad()

        # scalar_reward = trajectories['rewards'][0]
        # print(f"scalar_reward.requires_grad: {scalar_reward.requires_grad}")
        # print(f"scalar_reward.grad_fn: {scalar_reward.grad_fn}")

        with torch.no_grad():
            print("\n--- DIAGNÓSTICO PPO ---")
            print(f"Policy loss       : {policy_loss.item():.6f}")
            print(f"Value loss        : {value_loss.item():.6f}")
            print(f"Total loss        : {loss.item():.6f}")
            print(f"Entropy           : {entropy.item():.6f}")
            print(f"Logits mean/std   : {logits.mean().item():.3f} / {logits.std().item():.3f}")
            print(f"Temperature       : {torch.exp(self.policy.log_temperature).item():.4f}")
            print(f"Value pred mean/std : {new_values.mean().item():.3f} / {new_values.std().item():.3f}")
            print(f"Returns mean/std  : {returns.mean().item():.3f} / {returns.std().item():.3f}")
            print(f"Advantage mean/std: {advantages.mean().item():.3f} / {advantages.std().item():.3f}")
            print(f"Reward mean/std   : {rewards.mean().item():.3f} / {rewards.std().item():.3f}")
            print("--- FIN DIAGNÓSTICO ---\n")
        
        loss.backward()
        
        # Verificación de gradientes
        print("\n" + "="*50)
        print("GRADIENTES EN REWARD NETWORK:")
        any_reward_grad = False
        for name, param in self.reward_net.named_parameters():
            if param.grad is not None:
                print(f"{name:20} grad_mean: {param.grad.abs().mean().item():.10f}")
                any_reward_grad = True
            else:
                print(f"{name:20} SIN GRADIENTES")
        
        print("\nGRADIENTES EN POLICY NETWORK:")
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                print(f"{name:20} grad_mean: {param.grad.abs().mean().item():.10f}")
            else:
              print(f'{name:20} SIN GRADIENTES')
        print("="*50 + "\n")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 5.0)
        
        self.policy_optim.step()
        self.reward_optim.step()

        # with torch.no_grad():
        #   value_error = (returns - new_values.squeeze()).abs()
        #   print(f"Policy loss: {policy_loss.item():.5f}")
        #   print(f"Value loss: {value_loss.item():.5f}")
        #   print(f"Mean |V - R|: {value_error.mean().item():.5f}")
        #   print(f"Value predictions: mean={new_values.mean().item():.3f}, std={new_values.std().item():.3f}")
        #   print(f"Returns: mean={returns.mean().item():.3f}, std={returns.std().item():.3f}")

        return loss.item()

    def train(self, episodes, max_steps=1000, save_path='ppo_reward_model.pt'):
        best_reward = float('-inf')
        alpha = 1.0
        min_alpha = 0.0
        alpha_decay = (alpha - min_alpha) / episodes
        episode_lengths = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            trajectories = {
                'obs': [], 'actions': [], 'logprobs': [],
                'rewards': [], 'values': [], 'dones': []
            }
            episode_rewards = []
            
            for t in range(max_steps):
                obs_tensor = torch.FloatTensor(obs).to(device)
                action, logprob, value = self.policy.get_action(obs_tensor)
                
                next_obs, reward_vector, done, _, _ = self.env.step(action.cpu().numpy().item())
                
                # Transformar vector de reward a escalar manteniendo el grafo
                reward_vector_tensor = torch.FloatTensor(reward_vector).to(device).requires_grad_(True)
                learned_reward = self.reward_net(reward_vector_tensor)
                
                # Calcular componentes manteniendo el grafo
                # heuristic_reward = reward_vector_tensor.mean().detach()
                # alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)
                
                # Combinación convexa
                # scalar_reward = (1 - alpha_tensor) * learned_reward + alpha_tensor * heuristic_reward
                scalar_reward = learned_reward
                
                # Guardar transición
                trajectories['obs'].append(obs_tensor)
                trajectories['actions'].append(action)
                trajectories['logprobs'].append(logprob)
                trajectories['rewards'].append(scalar_reward)
                trajectories['values'].append(value)
                trajectories['dones'].append(done)
                
                obs = next_obs
                episode_rewards.append(sum(reward_vector))
                episode_lengths.append(t)
                
                if done:
                    break
            
            # Actualizar redes
            loss = self.update(trajectories)
            alpha = max(alpha - alpha_decay, min_alpha)
            
            # Guardar mejor modelo
            mean_episode_reward = np.sum(episode_rewards)
            if mean_episode_reward >= best_reward:
                best_reward = mean_episode_reward
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'reward_state_dict': self.reward_net.state_dict()
                }, save_path)
            
            print(f"Episode {episode}, Loss: {loss:.4f}, Reward: {mean_episode_reward:.2f}, Best: {best_reward:.2f}, Alpha:{alpha:.2f}")
            print(f"Episode length: {t}, Mean episode length: {np.mean(episode_lengths)}")

# Uso:
# env = TuEntorno()
# trainer = PPOTrainer(env)
# trainer.train(episodes=1000)