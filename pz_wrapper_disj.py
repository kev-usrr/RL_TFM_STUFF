import gymnasium as gym
import numpy as np
import numbers

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from pettingzoo.utils.env import AECEnv, ParallelEnv
from gymnasium import spaces

import matplotlib.pyplot as plt

class SupervisorWrapper(gym.Env):
    metadata = {'render_modes': ['human'],
                'image_based_environments': ['cooperative_pong', 'knights_archers_zombies', 'pistonball', 'entombed_cooperative'],
                'disjoint_actions': ['speaker_listener']}

    def __init__(self, pz_env: AECEnv, 
                 aggregation_method='sum',
                 alpha=0.5, 
                 tau=1.0,
                 max_reward=0.0,
                 min_reward=0.0):
        super().__init__()
        self.env = pz_env
        self.env.reset()

        # Con esto controlamos si el entorno está basado en imágenes y por tanto, si vamos a usar una ResNet-18 pre-entrenada
        # en ImageNet para obtener los embeddings. No me gusta mucho este código la verdad jajaja
        self.image_based = any([x in self.env.metadata.get('name', '').lower() for x in self.metadata['image_based_environments']])
        if 'knights_archers_zombies' in self.env.metadata.get('name', '').lower():
          self.image_based = not(self.env.vector_state)
        # Con esto vamos a controlar el único caso en el que los agentes tiene espacios de acción disjuntos.   
        self.disjoint_actions = any([name in self.env.metadata.get('name', '').lower() for name in self.metadata['disjoint_actions']])

        # Cargar ResNet-18 preentrenada
        if self.image_based:
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self.model = models.resnet18(weights='DEFAULT')
          self.model = nn.Sequential(*list(self.model.children())[:-1])
          self.model.eval()
          self.model.to(self.device)

        # Número de agentes del environment
        self.num_agents = len(self.env.possible_agents)
        # Espacios de acción de cada agente, en principio serán los mismos para todos excepto en algunos casos específicos
        self.action_spaces = {agent:self.env.action_space(agent) for agent in self.env.possible_agents}

        # Con este bucle vamos a calcular las posiciones del vector de observaciones que van a ocupar
        # las acciones
        aux_num = 0
        for action_space in self.action_spaces.values():
          if type(action_space) == spaces.Discrete:
            aux_num += 1
          else:
            aux_num += action_space.shape[0]
        self.action_append = aux_num

        # Espacios de observación de cada agente. En este caso, vamos a crear un espacio de observación extendido.
        # Es decir, el supervisor no solo "verá" la observación de cada agente, sinó también todas las acciones que
        # ha asignado hasta el momento (que serán siempre las últimas posiciones del vector de observaciones)
        self.observation_spaces = {agent:self.env.observation_space(agent) for agent in self.env.possible_agents}
        self.max_obs_len = max([self.__get_flatten_shape(self.observation_spaces[agent].shape) for agent in self.env.possible_agents])
        self.extended_observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs_len + self.action_append,), dtype=np.float32)
            for agent in self.env.possible_agents
        }
        self.observation_spaces = self.extended_observation_spaces

        # Si la observación se basa en imágenes hacemos lo mismo. Al vector de embeddings le vamos a concatenar otro vector
        # con las acciones asignadas (o no) hasta el momento.
        if self.image_based:
          self.observation_spaces = {agent:spaces.Box(low=-np.inf, high=np.inf, shape=(512 + self.action_append, ), dtype=np.float32) for agent in self.env.possible_agents}

        # Vector de acción conjunta que ampliamos en cada timestep
        self.joint_action = []
        self.current_agent_idx = 0 # Esto es para llevar un control sobre que agente le "toca"

        if self.disjoint_actions:
          aux = 1
          self.last_ag_n = 0
          
          for action_space in self.action_spaces.values():
            aux *= action_space.n
            self.last_ag_n = action_space.n
          
          self.action_spaces = {agent:spaces.Discrete(aux) for agent in self.env.possible_agents}
          self.discrete_dim  = aux

        # Aquí vamos a intentar que el espacio de acciones sea dinámico.
        # Es decir, cada timestep, cambiaremos el action space al del siguiente agente.
        self.action_space = self.action_spaces[self.env.possible_agents[self.current_agent_idx]]

        # Espacios de observación que también será dinámico. En cada timestep pasamos al del siguiente agente
        self.observation_space = self.observation_spaces[self.env.possible_agents[self.current_agent_idx]]

        print(f'{self.env.metadata.get("name", "")}')
        print(f'ACTION SPACES: {self.action_spaces}')
        print(f'OBSERVATION SPACES: {self.observation_spaces}\n')

        self.alpha = alpha
        self.tau   = tau
        self.ema_alpha = 0.1  # cuánto actualiza el promedio exponencial
        self.min_r = None
        self.max_r = None
        self.max_reward = max_reward
        self.min_reward = min_reward

        if aggregation_method == 'sum':
          self.agg_func = self.__reward_sum
        elif aggregation_method == 'sum_max':
          self.agg_func = self.__reward_sum_max
        elif aggregation_method == 'sum_minmax':
          self.agg_func = self.__reward_sum_minmax
        elif aggregation_method == 'expmean':
          self.agg_func = self.__reward_expmean
        elif aggregation_method == 'min':
          self.agg_func = self.__reward_min
        elif aggregation_method == 'softmin':
          self.agg_func = self.__reward_softmin
        elif aggregation_method == 'softmin_max':
          self.agg_func = self.__reward_softmin_max
        elif aggregation_method == 'softmin_minmax':
          self.agg_func = self.__reward_softmin_minmax


    def __reward_sum(self, rewards):
      return sum(rewards)

    
    def __reward_sum_minmax(self, rewards):
      reward = sum(rewards)
      
      # Paso 2: actualización de estadísticas exponenciales
      if self.min_r is None or self.max_r is None:
          self.min_r = reward
          self.max_r = reward
      else:
          self.min_r = min(self.min_r, reward)
          self.max_r = max(self.max_r, reward)

          # Opción más suave (EMA):
          self.min_r = (1 - self.ema_alpha) * self.min_r + self.ema_alpha * reward if reward < self.min_r else self.min_r
          self.max_r = (1 - self.ema_alpha) * self.max_r + self.ema_alpha * reward if reward > self.max_r else self.max_r

      # Paso 3: normalización en [0, 1]
      norm_range = self.max_reward - self.min_r + 1e-8  # evita división por cero
      normalized_reward = (reward - self.min_r) / norm_range

      return np.clip(normalized_reward, 0.0, 1.0)

    
    def __reward_sum_max(self, rewards):
      reward = sum(rewards)
      # Paso 2: actualización de estadísticas exponenciales
      if self.min_r is None:
        self.min_r = reward
      else:
        self.min_r = min(self.min_r, reward)
        # Opción más suave (EMA):
        self.min_r = (1 - self.ema_alpha) * self.min_r + self.ema_alpha * reward if reward < self.min_r else self.min_r

      # Paso 3: normalización en [0, 1]
      norm_range = self.max_reward - self.min_r + 1e-8  # evita división por cero
      normalized_reward = (reward - self.min_r) / norm_range

      return np.clip(normalized_reward, 0.0, 1.0)


    def __reward_expmean(self, rewards):
      return (np.exp(np.mean(rewards))) ** (1 / len(rewards))
    

    def __reward_min(self, rewards):
      return np.min(rewards)


    def __reward_softmin(self, rewards):
      rewards = np.array(rewards)

      # Paso 1: combinación softmin + media
      softmin = -np.log(np.sum(np.exp(-rewards / self.tau))) * self.tau
      mean_reward = np.mean(rewards)
      reward = self.alpha * softmin + (1 - self.alpha) * mean_reward

      # Paso 2: actualización de estadísticas exponenciales
      if self.min_r is None or self.max_r is None:
          self.min_r = reward
          self.max_r = reward
      else:
          self.min_r = min(self.min_r, reward)
          self.max_r = max(self.max_r, reward)

          # Opción más suave (EMA):
          self.min_r = (1 - self.ema_alpha) * self.min_r + self.ema_alpha * reward if reward < self.min_r else self.min_r
          self.max_r = (1 - self.ema_alpha) * self.max_r + self.ema_alpha * reward if reward > self.max_r else self.max_r

      # Paso 3: normalización en [0, 1]
      norm_range = self.max_r - self.min_r + 1e-8  # evita división por cero
      normalized_reward = (reward - self.min_r) / norm_range

      return normalized_reward

    def __reward_softmin_max(self, rewards):
      rewards = np.array(rewards)

      # Paso 1: combinación softmin + media
      softmin = -np.log(np.sum(np.exp(-rewards / self.tau))) * self.tau
      mean_reward = np.mean(rewards)
      reward = self.alpha * softmin + (1 - self.alpha) * mean_reward

      # Paso 2: actualización de estadísticas exponenciales
      if self.min_r is None:
        self.min_r = reward
      else:
        self.min_r = min(self.min_r, reward)
        # Opción más suave (EMA):
        self.min_r = (1 - self.ema_alpha) * self.min_r + self.ema_alpha * reward if reward < self.min_r else self.min_r

      # Paso 3: normalización en [0, 1]
      norm_range = self.max_reward - self.min_r + 1e-8  # evita división por cero
      normalized_reward = (reward - self.min_r) / norm_range

      return np.clip(normalized_reward, 0.0, 1.0)
    

    def __reward_softmin_minmax(self, rewards):
      rewards = np.array(rewards)

      # Paso 1: combinación softmin + media
      softmin = -np.log(np.sum(np.exp(-rewards / self.tau))) * self.tau
      mean_reward = np.mean(rewards)
      reward = self.alpha * softmin + (1 - self.alpha) * mean_reward

      # Paso 3: normalización en [0, 1]
      norm_range = self.max_reward - self.min_reward + 1e-8  # evita división por cero
      normalized_reward = (reward - self.min_reward) / norm_range

      return np.clip(normalized_reward, 0.0, 1.0)


    def __get_flatten_shape(self, tensor_shape):
      ret = 1
      for x in tensor_shape:
        ret *= x
      return ret
    
    
    def __decode_action(self, action, idx):
      if idx == 0:
        return action // self.last_ag_n
      else:
        return action % self.last_ag_n

    
    def step(self, action):
        # La acción que tomamos se la asignamos al siguiente agente.
        self.joint_action.append(action)
        self.current_agent_idx += 1

        # Si todavía no hemos asignado todas las acciones...
        if len(self.joint_action) < self.num_agents:
            # Cambiamos el espacio de acciones y observaciones al del siguiente agente
            self._update_spaces()
            # Devolvemos las observaciones de todos los agentes
            # Reward 0, Terminación False, Truncado False.
            return self._get_observations(action), 0, False, False, {}

        rewards, terminations, truncations = [], [], []
        for i, action in enumerate(self.joint_action):
          observation, reward, termination, truncation, info = self.env.last()
          # print(i, self.env.agent_selection)
          if self.disjoint_actions:
            action = self.__decode_action(action, i)
            # print(f'AGENT {i}: - SEL_ACTION: {action} - {self.joint_action}')

          self.env.step(action if not termination and not truncation else None) # Pass None if terminated or truncated
          if termination or truncation:
            self.num_agents -= 1

          rewards.append(reward)
          terminations.append(termination)
          truncations.append(truncation)

        # Reseteamos el buffer
        self.joint_action = []
        self.current_agent_idx = 0
        # Cambiamos el espacio de acciones y observaciones al del siguiente agente
        self._update_spaces()

        # Acumulamos el reward sumando sobre el reward de todos los agentes
        tot_reward = self.agg_func(rewards)

        # Determinamos si debemos parar la ejecución
        done = all(terminations) or all(truncations)
        if done:
          return [], tot_reward, done, False, {}
        return self._get_observations(), tot_reward, done, False, {}


    def reset(self, seed=None, options=None):
      self.env.reset(seed=seed, options=options)
      # Reseteamos la lista de acciones y el puntero al agente actual
      self.joint_action = []
      self.current_agent_idx = 0
      # Reseteamos el contador de agentes
      self.num_agents = len(self.env.possible_agents)
      # Actualizamos el espacio de acción y observación
      self._update_spaces()
      return self._get_observations(), {}

    def render(self):
      return self.env.render()

    def _update_spaces(self):
      if len(self.env.agents) > 0:
        self.action_space = self.action_spaces[self.env.agents[self.current_agent_idx]]
        self.observation_space = self.observation_spaces[self.env.agents[self.current_agent_idx]]


    def _add_action_info(self, action=None):
      # Le sumamos las acciones que ya hemos asignado y ponemos un 0 a todas las acciones sin asignar
      if action is None:
        aux = [0] * self.action_append
      else:
        # Buf no me gusta NADA este código
        aux = []
        for x in self.joint_action:
          if isinstance(x, (np.generic, numbers.Number)):
            aux.append(x+1)
          else:
            aux.extend(x)
        aux += [0] * (self.action_append - len(aux))
      return aux
      #  return [self.action_spaces_idx[self.env.possible_agents[i]].get(self.joint_action[i], 0) for i in range(len(self.joint_action))] + [0] * (len(self.env.possible_agents) - len(self.joint_action))


    def _get_observations(self, action=None):
        agent = self.env.agents[self.current_agent_idx]
        obs   = self.env.observe(agent)
        # print(obs)

        if self.image_based:
          return np.array(
            # Embeddings de la observación del agente
            self._get_resnet18_embedding(torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)).tolist() +
            # Le sumamos el identificador de las acciones
            self._add_action_info(action)
          )
        else:
          obs = np.pad(obs.flatten(), (0, self.max_obs_len - self.__get_flatten_shape(obs.shape)), mode='constant')
          return np.array(
            # Observaciones del agente actual
            obs.flatten().tolist() +
            # Le sumamos el identificador de las acciones
            self._add_action_info(action)
          )


    def _get_resnet18_embedding(self, image_tensor):
      # Asegurar que la imagen tiene la forma correcta (1, C, H, W)
      if image_tensor.ndim == 3:
          image_tensor = image_tensor.unsqueeze(0)

      transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      image_tensor = transform(image_tensor).to(self.device)

      with torch.no_grad():
          embedding = self.model(image_tensor)

      return embedding.view(-1).cpu().numpy()
  
    # Encaminamos todo lo demás al entorno original
    def __getattr__(self, name):
        return getattr(self.env, name)