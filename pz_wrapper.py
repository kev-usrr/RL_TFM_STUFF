import gymnasium as gym
import numpy as np 

from pettingzoo.utils.env import AECEnv, ParallelEnv
from gymnasium import spaces

class SupervisorWrapper(gym.Env):
    def __init__(self, pz_env: ParallelEnv):
        super().__init__()
        self.env = pz_env()
        self.env.reset()

        # Número de agentes del environment
        self.num_agents = len(self.env.possible_agents)
        # Espacios de acción de cada agente, en principio serán los mismos para todos excepto en algunos casos específicos
        self.action_spaces = {agent:self.env.action_space(agent) for agent in self.env.possible_agents}
        # Espacios de observación de cada agente.
        self.observation_spaces = {agent:self.env.observation_space(agent) for agent in self.env.possible_agents}

        # Vector de acción conjunta que ampliamos en cada timestep
        self.joint_action = []
        self.current_agent_idx = 0 # Esto es para llevar un control sobre que agente le "toca"

        # Aquí vamos a intentar que el espacio de acciones sea dinámico.
        # Es decir, cada timestep, cambiaremos el action space al del siguiente agente.
        self.action_space = self.action_spaces[self.env.possible_agents[self.current_agent_idx]]

        # Espacios de observación TODO
        self.observation_space = self.env.observation_space(self.env.possible_agents[self.current_agent_idx])

    def step(self, action):
        # La acción que tomamos se la asignamos al siguiente agente.
        self.joint_action.append(action)
        self.current_agent_idx += 1

        print(self.joint_action)
        print(self.action_space)

        # Si todavía no hemos asignado todas las acciones...
        if len(self.joint_action) < self.num_agents:
            # Cambiamos el espacio de acciones y observaciones al del siguiente agente
            self.action_space = self.action_spaces[self.env.possible_agents[self.current_agent_idx]]
            self.observation_space = self.observation_spaces[self.env.possible_agents[self.current_agent_idx]]
            # Devolvemos las observaciones de todos los agentes
            # Reward 0, Terminación False, Truncado False.
            return self._get_observations(), 0, False, False, {}
        
        actions_map = {agent:self.joint_action[i] for agent,i in zip(self.env.agents, range(self.num_agents))}
        observations, rewards, terminations, truncations, infos = self.env.step(actions_map)

        # Reseteamos el buffer
        self.joint_action = []
        self.current_agent_idx = 0
        # Cambiamos el espacio de acciones al del siguiente agente
        self.action_space = self.action_spaces[self.env.possible_agents[self.current_agent_idx]]

        # Acumulamos el reward sumando sobre el reward de todos los agentes
        tot_reward = sum(rewards.values())

        # Determinamos si debemos parar la ejecución
        done = any(terminations.values()) or any(truncations.values())
        return self._get_observations(), tot_reward, done, False, infos
    
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.joint_action = []
        self.current_agent_idx = 0
        return self._get_observations(), {}

    
    def _get_observations(self):
        return self.env.state