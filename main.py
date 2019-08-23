from configparser import ConfigParser
from environment import Env

config = ConfigParser()
config.read('config')

env = Env(config)