from pettingzoo import ParallelEnv
from copy import copy


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    PICK_CUP_CLOSET     = 0
    FILL_CUP_MACHINE    = 1
    MOVE_ADJ_ROOM       = 2
    GIVE_CUP_ROBOT      = 3
    GIVE_TEA            = 4
    ACTIONS = [PICK_CUP_CLOSET, FILL_CUP_MACHINE, 
               MOVE_ADJ_ROOM,
               GIVE_CUP_ROBOT,
               GIVE_TEA]

    def __init__(self):
        super().__init__()

        self.possible_agents = ['robot_1', 'robot_2']
        self.rooms_allowed = {
            'robot_1': ['room_1', 'room_2', 'corridor'],
            'robot_2': ['room_3', 'room_4', 'corridor'],
        }
        self.rooms_adjacent = {
            'room_1': ['room_2', 'corridor'],
            'room_2': ['room_1', 'corridor'],
            'room_3': ['room_4', 'corridor'],
            'room_4': ['room_1', 'corridor'],
            'corridor': ['room_1', 'room_2', 'room_3', 'room_4'],
        }

        self.state = {
            'robot_1': {'location':'room_1', 'arms':[None, None]},
            'robot_2': {'location':'room_3', 'arms':[None, None]},
            'person_1': {'has_tea': False},
            'person_2': {'has_tea': False},
        }

        self.objects = {
            'tea_machine':'room_1',
            'cup_cabinet':'room_3',
        }

        self.persons = {
            'person_1':'room_2',
            'person_2':'room_3',
        }

        self.action_spaces = {agent: self.ACTIONS for agent in self.possible_agents}
        self.observation_spaces = {
            agent: {
                'location': ['room_1', 'room_2', 'room_3', 'room_4', 'corridor'],
                'arm_1': [None, 'cup_empty', 'cup_fill'],
                'arm_2': [None, 'cup_empty', 'cup_fill'],
            } for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.state = {
            'robot_1': {'location':'room_1', 'arms':[None, None]},
            'robot_2': {'location':'room_3', 'arms':[None, None]},
            'person_1': {'has_tea': False},
            'person_2': {'has_tea': False},
        }

        self.agents = copy(self.possible_agents)
        return self._observe_all()

    def step(self, actions):
        rewards = {agent:0 for agent in self.agents}
        dones   = {agent:False for agent in self.agents}
        infos   = {agent:{} for agent in self.agents}

        for agent, action in actions.items():
            if len(action) == 2:
                action, dest = action
            else:
                dest=None
            self._apply_action(agent, action, dest=dest)
            rewards[agent] = self._get_reward(agent, action)

        aux = all([x['has_tea'] for x in [self.state[y] for y in ['person_1', 'person_2']]])
        dones = {agent:aux for agent in self.agents}

        return self._observe_all(), rewards, dones, infos

    def __give_tea_person(self, person, agent, current_location, current_arm1_state, current_arm2_state):
        if current_location == self.persons[person] and (not self.state[person]['has_tea']):
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
 
    def _apply_action(self, agent, action, param=None):
        current_location   = self.state[agent]['location']
        current_arm1_state = self.state[agent]['arm_1']
        current_arm2_state = self.state[agent]['arm_2']

        if action == self.PICK_CUP_CLOSET: # Recoger taza vacía
            if (current_location == self.objects['cup_cabinet']):
                if current_arm1_state is None:
                    self.state[agent]['arm_1'] = 'cup_empty'
                elif current_arm1_state is None:
                    self.state[agent]['arm_2'] = 'cup_empty'
        
        if action == self.FILL_CUP_MACHINE: # Rellenar taza vacía
            if current_location == self.objects['tea_machine']:
                if current_arm1_state is 'cup_empty':
                    self.state[agent]['arm_1'] = 'cup_fill'
                elif current_arm2_state is 'cup_empty':
                    self.state[agent]['arm_2'] = 'cup_fill'
        
        if action == self.GIVE_TEA: # Dar te lleno a una persona
            if not self.__give_tea_person('person_1', agent, current_location, current_arm1_state, current_arm2_state):
                self.__give_tea_person('person_2', agent, current_location, current_arm1_state, current_arm2_state)
        
        if action == self.MOVE_ADJ_ROOM: # Moverse a habitacion adyacente
            if (current_location in self.rooms_allowed[agent]) and (param in self.rooms_adjacent[current_location]):
                self.state[agent]['location'] = param
        
        if action == self.GIVE_CUP_ROBOT: # Robots se intercambian taza
            if current_location == self.state[param]['location']: # Dos robots deben estar en la misma habitación
                if not self.__give_cup_robot(agent, param, current_arm1_state, 'arm_1'):
                    self.__give_cup_robot(agent, param, current_arm2_state, 'arm_2')
    
    def _observe_all(self):
        return {
            agent: {
                'location':self.state[agent]['location'],
                'arm_1':self.state[agent]['arm_1'],
                'arm_2':self.state[agent]['arm_2'],
            } for agent in self.agents
        }

    def render(self):
        print(f'\nRobot 1: {self.state['robot_1']}')
        print(f'Robot 2: {self.state['robot_2']}\n')

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]