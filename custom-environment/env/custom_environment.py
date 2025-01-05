from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from itertools import product
from copy import copy

import functools
import json
import os

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    PICK_CUP_CLOSET     = 0
    FILL_CUP_MACHINE    = 1
    GIVE_TEA            = 3
    ACTIONS = [PICK_CUP_CLOSET, FILL_CUP_MACHINE, GIVE_TEA]

    def __init__(self,
                 config_file=os.path.join('problems', 'env_1.json')):
        super().__init__()

        self.config = {
            'rooms_allowed': {'1':[1,2,3], '2':[3,4,5]},
            'room_adjacent': {'1':[1,3], '2':[1,3], '3':[1,2,4,5], '4':[5,3], '5':[4,3]},
            'objects_location': {'tea_machine':[1], 'cup_storage':[4]},
            'people_location': {'1':1, '2':4},
            'initial_location': {'1':1, '2':4}
        }
        if config_file is not None:
            with open(config_file, 'r') as file:
                self.config = json.load(file)
            
        self.__num_robots = self.config.get('num_robots', 2)
        self.__num_people = self.config.get('num_people', 2)
        self.__num_rooms  = self.config.get('num_roomms', 5)
        self.__num_arms   = self.config.get('num_arms', 2)

        self.possible_agents = [f'robot_{i}' for i in range(1, self.__num_robots + 1)]
        self.rooms = [f'room_{i}' for i in range(1, self.__num_rooms + 1)]
        
        self.rooms_allowed = {}
        for robot in range(1, self.__num_robots + 1):
            rooms = self.config['rooms_allowed'].get(f'{robot}')
            self.rooms_allowed[f'robot_{robot}'] = [f'room_{i}' for i in rooms]
        
        self.rooms_adjacent = {}
        for room in range(1, self.__num_rooms + 1):
            other_rooms = self.config['room_adjacent'].get(f'{room}')
            self.rooms_adjacent[f'room_{room}'] = [f'room_{i}' for i in other_rooms]
        
        self.objects = {}
        for object, rooms in self.config['objects_location'].items():
            self.objects[object] = [f'room_{i}' for i in rooms]
        
        self.people = {}
        for person in range(1, self.__num_people + 1):
            self.people[f'person_{person}'] = f'room_{self.config['people_location'][f'{person}']}'

        self.state = {}
        for robot in range(1, self.__num_robots + 1):
            self.state[f'robot_{robot}'] = {}
            self.state[f'robot_{robot}']['location'] = self.config['initial_location'][f'{robot}']
            for arm in range(1, self.__num_arms + 1):
                self.state[f'robot_{robot}'][f'arm_{arm}'] = None
        for person in range(1, self.__num_people + 1):
            self.state[f'person_{person}'] = {'has_tea': False}


        self.action_spaces = {agent: self.ACTIONS for agent in self.possible_agents}
        self.observation_spaces = {
            agent: {
                'location': self.rooms,
                'arm_1': [None, 'cup_empty', 'cup_fill'],
                'arm_2': [None, 'cup_empty', 'cup_fill'],
            } for agent in self.possible_agents
        }

        aux = len(self.ACTIONS)
        self.ACTIONS.extend([i + aux for i, _ in enumerate(self.rooms)])
        self.rooms_map = {i + aux:room for i, room in enumerate(self.rooms)}
        
        aux = len(self.ACTIONS)
        self.ACTIONS.extend([i + aux for i, _ in enumerate(self.possible_agents)])
        self.robots_map = {i+aux:room for i, room in enumerate(self.possible_agents)}
        # print(self.ACTIONS)
        # print(self.rooms_map)
        print(self.state)
        

    def reset(self, seed=None, options=None):
        self.state = {}
        for robot in range(1, self.__num_robots + 1):
            self.state[f'robot_{robot}'] = {}
            self.state[f'robot_{robot}']['location'] = self.config['initial_location'][f'{robot}']
            for arm in range(1, self.__num_arms + 1):
                self.state[f'robot_{robot}'][f'arm_{arm}'] = None
        for person in range(1, self.__num_people + 1):
            self.state[f'person_{person}'] = {'has_tea': False}

        self.agents = copy(self.possible_agents)
        return self._observe_all(), {a: {} for a in self.agents}

    
    def step(self, actions):
        rewards = {agent:0 for agent in self.agents}
        dones   = {agent:False for agent in self.agents}
        infos   = {agent:{} for agent in self.agents}

        for agent, action in actions.items():
            success = self._apply_action(agent, action)
            rewards[agent] = self._get_reward(action, success)

        aux = all([x['has_tea'] for x in [self.state[y] for y in self.people.keys()]])
        dones = {agent:aux for agent in self.agents}

        return self._observe_all(), rewards, dones, {a:{} for a in self.agents}, infos

    
    def __give_tea_person(self, person, agent, current_location, current_arm1_state, current_arm2_state):
        if current_location == self.people[person] and (not self.state[person]['has_tea']):
            if current_arm1_state == 'cup_fill':
                self.state[person]['has_tea'] = True
                self.state[agent]['arm_1'] = None
                return True
            elif current_arm2_state == 'cup_fill':
                self.state[person]['has_tea'] = True
                self.state[agent]['arm_2'] = None
                return True
        return False


    def __give_cup_robot(self, agent, other_agent, current_arm_state, which_arm):
        if current_arm_state in ['cup_fill', 'cup_empty']:
            if self.state[other_agent]['arm_1'] is None:
                self.state[other_agent]['arm_1'] = current_arm_state
                self.state[agent][which_arm] = None
                return True
            elif self.state[other_agent]['arm_2'] is None:
                self.state[other_agent]['arm_2'] = current_arm_state
                self.state[agent][which_arm] = None
                return True
        return False
 
    
    def _apply_action(self, agent, action):
        current_location   = self.state[agent]['location']
        current_arm1_state = self.state[agent]['arm_1']
        current_arm2_state = self.state[agent]['arm_2']

        if action == self.PICK_CUP_CLOSET: # Recoger taza vacía
            if (current_location in self.objects['cup_cabinet']):
                if current_arm1_state is None:
                    self.state[agent]['arm_1'] = 'cup_empty'
                elif current_arm1_state is None:
                    self.state[agent]['arm_2'] = 'cup_empty'
        
        elif action == self.FILL_CUP_MACHINE: # Rellenar taza vacía
            if current_location in self.objects['tea_machine']:
                if current_arm1_state == 'cup_empty':
                    self.state[agent]['arm_1'] = 'cup_fill'
                elif current_arm2_state == 'cup_empty':
                    self.state[agent]['arm_2'] = 'cup_fill'
        
        elif action == self.GIVE_TEA: # Dar te lleno a una persona
            if not self.__give_tea_person('person_1', agent, current_location, current_arm1_state, current_arm2_state):
                return self.__give_tea_person('person_2', agent, current_location, current_arm1_state, current_arm2_state)
            else:
                return True
        
        elif action in self.rooms_map.keys(): # Moverse a habitacion adyacente
            param = self.rooms_map[action]
            if (current_location in self.rooms_allowed[agent]) and (param in self.rooms_adjacent[current_location]):
                self.state[agent]['location'] = param
        
        elif action == self.robots_map.keys(): # Robots se intercambian taza
            param = self.robots_map[action]
            if current_location == self.state[param]['location']: # Dos robots deben estar en la misma habitación
                if not self.__give_cup_robot(agent, param, current_arm1_state, 'arm_1'):
                    self.__give_cup_robot(agent, param, current_arm2_state, 'arm_2')
        
        return True
    
    
    def _observe_all(self):
        return {
            agent: {
                'location':self.state[agent]['location'],
                'arm_1':self.state[agent]['arm_1'],
                'arm_2':self.state[agent]['arm_2'],
            } for agent in self.agents
        }

    
    def _get_reward(self, action, success):
        if success and (action == self.GIVE_TEA):
            return 1

    
    def render(self):
        for i in range(1, self.__num_robots + 1):
            print(f'\nRobot {i}: {self.state[f'robot_{i}']}')

    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.ACTIONS))
    

if __name__ == '__main__':
    env = CustomEnvironment()