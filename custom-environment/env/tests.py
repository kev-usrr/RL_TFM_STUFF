from pettingzoo.test import parallel_api_test
from custom_environment import CustomEnvironment

if __name__ == '__main__':
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000)
