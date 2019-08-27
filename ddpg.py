import tensorflow as tf
import random
import numpy as np

class DDPG():
    def __init__(self, rnn, config):
        self.state_dim = int(config['EMB']['EMB_SIZE'])
        self.action_dim = int(config['EMB']['EMB_SIZE'])
        self.tau = float(config['DDPG']['TAU'])
        self.a_lr = float(config['DDPG']['ACTOR_LEARNING_RATE'])
        self.c_lr = float(config['DDPG']['CRITIC_LEARNING_RATE'])
        self.gamma = float(config['DDPG']['GAMMA'])
        self.actor_layers = [int(h) for h in config['DDPG']['ACTOR_LAYERS'].split(',')]
        self.critic_layers = [int(h) for h in config['DDPG']['CRITIC_LAYERS'].split(',')]
        self.a_upper = float(config['DDPG']['ACTION_UPPER'])
        self.a_lower = float(config['DDPG']['ACTION_LOWER'])
        self.exp_decay = float(config['DDPG']['EXPLORARION_DECAY'])

        self.s = tf.placeholder(tf.float32, [None, self.state_dim])
        self.s_ = tf.placeholder(tf.float32, [None, self.state_dim])
        self.r = tf.placeholder(tf.float32, [None, 1])

        self.a = self._build_actor(self.s, 'actor')
        self.a_ = self._build_actor(self.s_, 'target_actor')
        # 更新actor网络的时候self.a接actor网络, 更新Q网络的时候self.a接外部输入的action
        self.q = self._build_critic(self.s, self.a, 'critic')
        self.q_ = self._build_critic(self.s_, self.a_, 'target_critic')

        self.exploration_var = 3

        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')
        self.target_a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic')
        self.target_c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_critic')

        self.target_init = [[tf.assign(ta, a), tf.assign(tc, c)] 
            for ta, a, tc, c in zip(self.target_a_params, self.a_params, self.target_c_params, self.c_params)]
        self.target_update = [[tf.assign(ta, (1 - self.tau)*ta + self.tau*a), tf.assign(tc, (1 - self.tau)*tc + self.tau*c)]
            for ta, a, tc, c in zip(self.target_a_params, self.a_params, self.target_c_params, self.c_params)]

        actor_loss = -tf.reduce_mean(self.q)
        self.train_actor = tf.train.AdamOptimizer(self.a_lr).minimize(actor_loss, var_list=self.a_params)

        y = self.r + self.gamma*self.q_
        critic_loss = tf.reduce_mean(tf.square(y - self.q))
        self.train_critic = tf.train.AdamOptimizer(self.c_lr).minimize(critic_loss, var_list=self.c_params)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)
        
    def _build_actor(self, s, scope):
        init = tf.random_normal_initializer()
        with tf.variable_scope(scope):
            h = s
            for unit in self.actor_layers:
                h = tf.layers.dense(h, unit, tf.nn.relu, kernel_initializer=init)
            action = tf.layers.dense(h, self.action_dim, tf.nn.tanh, kernel_initializer=init)
        return action*self.a_upper

    def _build_critic(self, s, a, scope):
        # init = tf.random_normal_initializer()
        # with tf.variable_scope(scope):
        #     unit = CRITIC_HIDDNE_UNITS[0]
        #     h_s = tf.layers.dense(s, unit, kernel_initializer=init)
        #     h_a = tf.layers.dense(a, unit, kernel_initializer=init)
        #     h = h_s + h_a
        #     for unit in CRITIC_HIDDNE_UNITS[1:]:
        #         h = tf.layers.dense(h, unit, tf.nn.relu, kernel_initializer=init)
        #     q_value = tf.layers.dense(h, 1, kernel_initializer=init)
        # return q_value
        with tf.variable_scope(scope):
            w1 = tf.get_variable('w1', [self.state_dim, 32])
            w2 = tf.get_variable('w2', [self.action_dim, 32])
            b1 = tf.get_variable('b1', [1, 32])
            net = tf.nn.relu(tf.matmul(s, w1) + tf.matmul(a, w2) + b1)
            return tf.layers.dense(net, 1)

    def get_action(self, s):
        s = np.array(s)
        single = False
        if len(s.shape) == 1:
            single = True
            s = np.reshape(s, [-1, self.state_dim])
        a = self.sess.run(self.a, {self.s: s})
        if single == True:
            a = np.reshape(a, [self.action_dim])
        # a = np.clip(np.random.normal(a, self.exploration_var), A_LOWER, A_UPPER)
        a = np.clip(a, self.a_lower, self.a_upper)
        return a

    def learn(self, s, a, r, s_):
        self.exploration_var *= self.exp_decay
        self.sess.run(self.train_critic, {self.s: s, self.a: a, self.s_: s_, self.r: r})
        self.sess.run(self.train_actor, {self.s: s})
        self.sess.run(self.target_update)

class Memory():
    def __init__(self, config):
        self.memory = []
        self.memory_size = int(config['DDPG']['MEMORY_SIZE'])
        self.len = 0
    
    def add(self, s, a, r, s_):
        if self.len >= self.memory_size:
            self.memory.pop(0)
            self.len -= 1
        self.memory.append((s, a, r, s_))
        self.len += 1

    def get_batch(self, batch_size):
        if self.len < batch_size:
            batch = random.sample(self.memory, self.len)
        else:
            batch = random.sample(self.memory, batch_size)
        
        s = np.array([_[0] for _ in batch])
        a = np.array([_[1] for _ in batch])
        r = np.array([[_[2]] for _ in batch])
        s_ = np.array([_[3] for _ in batch])
        return s, a, r, s_
