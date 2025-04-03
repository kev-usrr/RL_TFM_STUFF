import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from pettingzoo.utils.env import AECEnv, ParallelEnv
from gymnasium import spaces

class SupervisorWrapper(gym.Env):
    metadata = {'render_modes': ['human'],
                'image_based_environments': ['cooperative_pong', 'knights_archers_zombies', 'pistonball']}

    def __init__(self, pz_env: AECEnv):
        super().__init__()
        self.env = pz_env
        self.env.reset()

        # Con esto controlamos si el entorno está basado en imágenes y por tanto, si vamos a usar una ResNet-18 pre-entrenada
        # en ImageNet para obtener los embeddings. No me gusta mucho este código la verdad jajaja
        self.image_based = any([x in self.env.metadata.get('name', '').lower() for x in self.metadata['image_based_environments']])
        if 'knights_archers_zombies' in self.env.metadata.get('name', '').lower():
          self.image_based = not(self.env.vector_state)

        # Cargar ResNet-18 preentrenada
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Número de agentes del environment
        self.num_agents = len(self.env.possible_agents)
        # Espacios de acción de cada agente, en principio serán los mismos para todos excepto en algunos casos específicos
        self.action_spaces = {agent:self.env.action_space(agent) for agent in self.env.possible_agents}
        # Espacios de observación de cada agente.
        self.observation_spaces = {agent:self.env.observation_space(agent) for agent in self.env.possible_agents}
        # Aquí el tema está en que parece que gymnasium necesita conocer a priori las dimensiones y los valores posibles de las observaciones
        # Si usamos embeddings de ResNet vamos a obtener vectores de 512 dimensiones pero los valores en principio no tienen ninguna cota.
        # Para solventar esto lo que hago es una normalización L2 del vector que deja los valores en el intervalo [-1, 1].
        # En principio, este tipo de normalización conserva la relación de distancias entre vectores de embeddings, que entiendo que puede
        # ser importante para algunos algoritmos. 
        if self.image_based:
          self.observation_spaces = {agent:spaces.Box(low=-1, high=1, shape=(512, ), dtype=np.float32) for agent in self.env.possible_agents}

        # Vector de acción conjunta que ampliamos en cada timestep
        self.joint_action = []
        self.current_agent_idx = 0 # Esto es para llevar un control sobre que agente le "toca"

        # Aquí vamos a intentar que el espacio de acciones sea dinámico.
        # Es decir, cada timestep, cambiaremos el action space al del siguiente agente.
        self.action_space = self.action_spaces[self.env.possible_agents[self.current_agent_idx]]

        # Espacios de observación
        self.observation_space = self.observation_spaces[self.env.possible_agents[self.current_agent_idx]]
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents * np.prod(obs_shape),), dtype=np.float32)

    def step(self, action):
        # La acción que tomamos se la asignamos al siguiente agente.
        self.joint_action.append(action)
        self.current_agent_idx += 1

        # print(self.joint_action)
        # print(self.action_space)

        # Si todavía no hemos asignado todas las acciones...
        if len(self.joint_action) < self.num_agents:
            # Cambiamos el espacio de acciones y observaciones al del siguiente agente
            self._update_spaces()
            # Devolvemos las observaciones de todos los agentes
            # Reward 0, Terminación False, Truncado False.
            return self._get_observations(), 0, False, False, {}

        # actions_map = {agent:self.joint_action[i] for agent,i in zip(self.env.agents, range(self.num_agents))}
        # observations, rewards, terminations, truncations, infos = self.env.step(actions_map)
        rewards, terminations, truncations = [], [], []
        for action in self.joint_action:
          observation, reward, termination, truncation, info = self.env.last()
          self.env.step(action if not termination and not truncation else None) # Pass None if terminated or truncated
          rewards.append(reward)
          terminations.append(termination)
          truncations.append(truncation)

        # Reseteamos el buffer
        self.joint_action = []
        self.current_agent_idx = 0
        # Cambiamos el espacio de acciones y observaciones al del siguiente agente
        self._update_spaces()

        # Acumulamos el reward sumando sobre el reward de todos los agentes
        # tot_reward = sum(rewards.values())
        tot_reward = sum(rewards)

        # Determinamos si debemos parar la ejecución
        # done = any(terminations.values()) or any(truncations.values())
        done = any(terminations) or any(truncations)
        return self._get_observations(), tot_reward, done, False, {}


    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.joint_action = []
        self.current_agent_idx = 0
        return self._get_observations(), {}

    
    def _update_spaces(self):
      self.action_space = self.action_spaces[self.env.possible_agents[self.current_agent_idx]]
      self.observation_space = self.observation_spaces[self.env.possible_agents[self.current_agent_idx]]
    
    
    def _get_observations(self):
        #obs = np.concatenate([self.env.observe(agent).flatten() for agent in self.env.possible_agents])
        if self.image_based:
          return self._get_resnet18_embedding(torch.tensor(self.env.observe(self.env.possible_agents[self.current_agent_idx]), dtype=torch.float32).permute(2, 0, 1))
        return self.env.observe(self.env.possible_agents[self.current_agent_idx]).flatten()

    
    def _get_resnet18_embedding(self, image_tensor):
      # Asegurar que la imagen tiene la forma correcta (1, C, H, W)
      if image_tensor.ndim == 3:
          image_tensor = image_tensor.unsqueeze(0)
      
      transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      image_tensor = transform(image_tensor).to(self.device)
      
      with torch.no_grad():
          embedding = self.model(image_tensor)
      # Antes de devolver el tensor, lo normalizo con norm. L2 y lo paso a Numpy.
      return torch.nn.functional.normalize(embedding.view(-1), p=2, dim=0).cpu().numpy()