from configparser import ConfigParser
from recommender import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if __name__ == '__main__':

    config = ConfigParser()
    config.read('config')

    rec = Recommender(config)
    rec.run()
    if config['SHOW']['PLOT'] == 'T':
        rec.show()

# for _ in range(100):
#     s = np.random.randint(0, 9722, 10)
#     a = np.random.randint(0, 9222)
#     print(env.get_reward(s, a))