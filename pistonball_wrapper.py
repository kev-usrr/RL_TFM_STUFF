from gymnasium import spaces
import numpy as np
import cv2

class PistonBallWrapper:
    def __init__(self, env):
        # Asegúrate de usar todos los wrappers necesarios
        self.env = env
        self.env.reset()
        assert self.env.render_mode == 'rgb_array'

        self.metadata = self.env.metadata
        self.metadata['name'] = 'piston_vectorized'

    
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    
    def step(self, action):
        self.env.step(action)

    
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(8, ), dtype=np.float32)

    def __find_ball(self, blue_channel):
        height, width = blue_channel.shape

        for y in range(height):
            for x in range(width):
                val = blue_channel[y, x]
                if val == 65:
                    return [x, y]
                if val == 186:
                    return [-1, -1]
        return [-1, -1]  # no se encontró pelota

    def observe(self, agent):
        if agent == 'piston_0':
            piston_xs = [60, 100]
        elif agent == 'piston_19':
            piston_xs = [20, 60]
        else:
            piston_xs = [20, 60, 100]
        
        blue_channel = self.env.observe(agent)[:, :, 2]
        height = blue_channel.shape[0]

        piston_tops = []
        for x in piston_xs:
            y_top = None
            for y in reversed(range(height)):
                if blue_channel[y, x] == 186:
                    y_top = y
                    break
            piston_tops.append((x, y_top))
        
        if agent == 'piston_0':
            piston_tops = [None] + piston_tops
        elif agent == f'piston_{len(self.env.possible_agents)-1}':
            piston_tops = [(x, 0 if y is None else y) for (x, y) in piston_tops]
        
        piston_tops = [x if x is not None else (0,0) for x in piston_tops]
        piston_tops = [item for tup in piston_tops for item in tup]
        ball_pos    = self.__find_ball(blue_channel)
        return np.array(piston_tops + ball_pos)


    
    # Encaminamos todo lo demás al entorno original
    def __getattr__(self, name):
        return getattr(self.env, name)