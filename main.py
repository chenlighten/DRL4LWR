from configparser import ConfigParser
from environment import Env
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

config = ConfigParser()
config.read('config')

env = Env(config, usage='train')