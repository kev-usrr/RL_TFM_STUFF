from gymnasium import spaces
import numpy as np
import cv2


class CooperativePongWrapperSpeed:
    def __init__(self, env):
        # Asegúrate de usar todos los wrappers necesarios
        self.env = env
        self.env.reset()
        assert self.env.render_mode == 'rgb_array'
        self.prev_ball_coords = (0, 0)
        self.prev_speed = (0, 0)

        self.metadata = self.env.metadata
        self.metadata['name'] = 'pong_vectorized_speed'

    
    def reset(self, seed=None, options=None):
        self.prev_ball_coords = (0, 0)
        self.prev_speed = (0, 0)
        return self.env.reset(seed=seed, options=options)

    
    def step(self, action):
        self.env.step(action)

    
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(8, ), dtype=np.float32)
    
    
    def __find_paddle_y(self, img, x_pos):        
        for row in range(img.shape[0]):
            if img[row, x_pos] != 0:
                return row
        return -1

    
    def __find_ball(self, img, threshold, max_area=144):
        # Binarizar: 1 si es mayor al umbral, 0 en otro caso
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        # Encontrar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Buscar componente con área pequeña (ignoramos fondo y paletas)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= max_area:
                # Coordenadas del centro del componente
                y, x = map(int, centroids[i])
                return (y, x)
        
        return self.prev_ball_coords

    
    def observe(self, agent):
        LEFT_PADDLE_X  = 3
        RIGHT_PADDLE_X = 458

        original_obs = self.env.observe(agent)
        grey_img = original_obs[:, :, 0]

        left_paddle_y  = self.__find_paddle_y(grey_img, LEFT_PADDLE_X) + 20
        right_paddle_y = self.__find_paddle_y(grey_img, RIGHT_PADDLE_X) + 20
        ball_coords    = self.__find_ball(grey_img, 200)
        speed          = (ball_coords[0] - self.prev_ball_coords[0], ball_coords[1] - self.prev_ball_coords[1])
        
        if speed == (0, 0):
            speed = self.prev_speed
        self.prev_speed = speed
        self.prev_ball_coords = ball_coords
    
        return np.array([LEFT_PADDLE_X, left_paddle_y, RIGHT_PADDLE_X, right_paddle_y, ball_coords[0], ball_coords[1], speed[0], speed[1]])

    
    # Encaminamos todo lo demás al entorno original
    def __getattr__(self, name):
        return getattr(self.env, name)


class CooperativePongWrapper:
    def __init__(self, env):
        # Asegúrate de usar todos los wrappers necesarios
        self.env = env
        self.env.reset()
        assert self.env.render_mode == 'rgb_array'
        self.prev_ball_coords = (0, 0)
        self.prev_speed = 0

        self.metadata = self.env.metadata
        self.metadata['name'] = 'pong_vectorized'

    
    def reset(self, seed=None, options=None):
        self.prev_ball_coords = (0, 0)
        self.prev_speed = 0
        return self.env.reset(seed=seed, options=options)

    
    def step(self, action):
        self.env.step(action)

    
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7, ), dtype=np.float32)
    
    
    def __find_paddle_y(self, img, x_pos):        
        for row in range(img.shape[0]):
            if img[row, x_pos] != 0:
                return row
        return -1

    
    def __find_ball(self, img, threshold, max_area=144):
        # Binarizar: 1 si es mayor al umbral, 0 en otro caso
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        # Encontrar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Buscar componente con área pequeña (ignoramos fondo y paletas)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= max_area:
                # Coordenadas del centro del componente
                y, x = map(int, centroids[i])
                return (y, x)
        
        return self.prev_ball_coords

    
    def observe(self, agent):
        LEFT_PADDLE_X  = 3
        RIGHT_PADDLE_X = 458

        original_obs = self.env.observe(agent)
        grey_img = original_obs[:, :, 0]

        left_paddle_y  = self.__find_paddle_y(grey_img, LEFT_PADDLE_X) + 20
        right_paddle_y = self.__find_paddle_y(grey_img, RIGHT_PADDLE_X) + 20
        ball_coords    = self.__find_ball(grey_img, 200)
        speed          = np.sign(ball_coords[0] - self.prev_ball_coords[0])
        
        if speed == 0:
            speed = self.prev_speed
        self.prev_speed = speed
        self.prev_ball_coords = ball_coords
    
        return np.array([LEFT_PADDLE_X, left_paddle_y, RIGHT_PADDLE_X, right_paddle_y, ball_coords[0], ball_coords[1], speed])

    
    # Encaminamos todo lo demás al entorno original
    def __getattr__(self, name):
        return getattr(self.env, name)
