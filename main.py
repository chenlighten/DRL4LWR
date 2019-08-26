from configparser import ConfigParser
from environment import Env
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

config = ConfigParser()
config.read('config')

env = Env(config, usage='train')
for _ in range(100):
    s = np.random.randint(0, 9722, 10)
    a = np.random.randint(0, 9222)
    print(env.get_reward(s, a))