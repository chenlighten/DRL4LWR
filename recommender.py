from environment import *
from ddpg import *
from rnn import *
import numpy as np
import tensorflow as tf

class Recommender():
    def __init__(self, config):
        self.max_episode = int(config['REC']['MAX_EPISODE'])
        self.step_per_episode = int(config['REC']['STEP_PER_EPS'])
        self.seq_len = int(config['ENV']['SEQ_LEN'])
        self.batch_size = int(config['DDPG']['BATCH_SIZE'])

        self.env = Env(config, usage='train')
        self.rnn = self.env.get_rnn()
        self.ddpg = DDPG(self.rnn, config)
        self.memory = Memory(config)

        if config['SHOW']['TENSOR_BOARD'] == 'T':
            writer = tf.summary.FileWriter('runtime_data/tf_logs', tf.get_default_graph())
            writer.close()

    def run(self):
        self.rewards = []
        self.avg_rewards = []
        for eps in range(self.max_episode):
            hist = []
            sv = self.rnn.get_state(hist)
            reward = 0
            for step in range(self.step_per_episode):
                av = self.ddpg.get_action(sv)
                aid = self.env.vector2id_item(av)
                av = self.env.id2vector_item(aid)
                r = self.env.get_reward(sv, av)

                reward += r

                if r >= 3.5:
                    if len(hist) >= self.seq_len:
                        hist.pop(0)
                        hist.append(aid)
                    else:
                        hist.append(aid)
                sv_ = self.rnn.get_state(hist)
                self.memory.add(sv, av, r, sv_)
                sv = sv_

                s, a, r, s_ = self.memory.get_batch(self.batch_size)
                self.ddpg.learn(s, a, r, s_)
            
            self.rewards.append(reward)
            self.avg_rewards.append(reward/self.step_per_episode)
            print('episode: %d, reward: %.1f, average reward: %.4f' \
                %(eps, reward, reward/self.step_per_episode))
