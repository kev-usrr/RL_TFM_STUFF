from gymnasium import spaces
import numpy as np
import cv2


class ZombiesWrapper:
    def __init__(self, env):
        # Asegúrate de usar todos los wrappers necesarios
        self.env = env
        self.env.reset()
        assert not(self.env.vector_state)

        self.metadata = self.env.metadata
        self.metadata['name'] = 'zombies_vectorized'

    
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    
    def step(self, action):
        try:
            self.env.step(action)
        except:
            pass

    
    def render(self):
        return self.env.render()

    
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(11, ), dtype=np.float32)
    
    
    def __detect_zombies(self, imagen_rgb):
        height, width = imagen_rgb.shape[:2]
        center = (width // 2, height // 2)
        # Convertir de RGB a HSV
        imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2HSV)

        # Definir rango de verde en HSV
        lower_green = np.array([65, 150, 100])
        upper_green = np.array([85, 255, 255]) 

        # Crear máscara
        mask = cv2.inRange(imagen_hsv, lower_green, upper_green)
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Lista para almacenar centros
        centros = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Calcular centro del rectángulo
            cx = x + w // 2
            cy = y + h // 2

            rel_x = cx - center[0]
            rel_y = cy - center[1]
            centros.append((rel_x, rel_y))

        return centros  # opcional: devuelve también la máscara

    
    def observe(self, agent):
        original_obs = self.env.observe(agent)
        aux = [coord for par in self.__detect_zombies(original_obs) for coord in par]
        aux = aux[:10] + [0] * (10 - len(aux))
        if agent.startswith('archer'):
            aux += [1]
        else:
            aux += [2]
        return np.array([aux])

    
    # Encaminamos todo lo demás al entorno original
    def __getattr__(self, name):
        return getattr(self.env, name)