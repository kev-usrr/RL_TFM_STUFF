from pettingzoo.butterfly import cooperative_pong_v5 as coop_pong
from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz
from pettingzoo.butterfly import pistonball_v6 as pb
# from pettingzoo.atari import entombed_cooperative_v3 as ec
import matplotlib.pyplot as plt
import numpy as np

def show_observation(observation, agent):
    plt.imshow(observation)
    plt.title(f"Observation for {agent}")
    plt.axis("off")
    plt.show()

def check_observation_shape():
    # env = coop_pong.env(render_mode="human")
    env = kaz.env(render_mode="human", vector_state=False)
    # env = pb.env(render_mode="human", max_cycles=10_000)
    # env = ec.env(render_mode="human")
    env.reset()

    state = env.state()
    print(f'State shape: {state.shape}')
    # print(state)
    show_observation(state, '')
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
        print(f'Agent {agent} reward: {reward}')
    
    for agent in env.agents:
        observation = env.observe(agent)
        print(f"Agent: {agent}, Observation Shape: {observation.shape}")
        # print(observation)
        show_observation(observation, agent)
    
    env.close()

if __name__ == "__main__":
    check_observation_shape()
