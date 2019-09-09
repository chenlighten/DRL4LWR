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
