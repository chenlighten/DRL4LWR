import numpy as np
import tensorflow as tf
import random

class Rnn():
    def __init__(self, train_data, item_embedding, config):
        self.input_dim = int(config['EMB']['EMB_SIZE'])
        # self.output_dim = int(config['RNN']['OUTPUT_DIM'])
        self.output_dim = self.input_dim
        self.batch_size = int(config['RNN']['BATCH_SIZE'])
        self.unit_num = int(config['ENV']['SEQ_LEN'])
        self.l2_factor = float(config['RNN']['L2_FACTOR'])
        self.max_train_step = int(config['RNN']['TRAIN_STEP'])
        self.train_data = train_data
        self.item_embedding = item_embedding
        self.max_item_num = item_embedding.shape[0]

        self.make_graph()
        
        if config['RNN']['LOAD_DATA'] == 'F':
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.train()
            saver = tf.train.Saver()
            saver.save(self.sess, './runtime_data/Model/model.ckpt')
        else:
            self.sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.sess, './runtime_data/Model/model.ckpt')

    def make_graph(self):
        tf.reset_default_graph()

        self.action_input = [tf.placeholder(tf.float32, [None, self.input_dim])
            for i in range(self.unit_num)]
        # self.rnn_state = tf.placeholder(tf.float32, [2, self.batch_size, self.output_dim])
        
        self.initial_state = tf.placeholder(tf.float32, [2, None, self.output_dim])
        self.make_unit = self.create_sru()

        self.cur_rnn_states = []
        h_c = self.initial_state
        for i in range(self.unit_num):
            h_c = self.make_unit(self.action_input[i], h_c)
            self.cur_rnn_states.append(h_c)
        
        self.W = tf.Variable(self.init_matrix([self.output_dim, self.max_item_num]))
        self.b = tf.Variable(self.init_matrix([self.max_item_num]))
        self.r_pred = [tf.matmul(self.cur_rnn_states[i][0], self.W) + self.b
            for i in range(self.unit_num)]
        self.r_real = tf.placeholder(tf.float32, [self.batch_size, self.max_item_num])
        self.l2_norm = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
        self.loss = [tf.reduce_mean((self.r_pred[i] - self.r_real) ** 2) + self.l2_factor*self.l2_norm 
            for i in range(self.unit_num)]
        self.train_op = [tf.train.AdamOptimizer().minimize(self.loss[i]) for i in range(self.unit_num)]

    def init_matrix(self, shape):
        return tf.truncated_normal(shape, stddev=0.1)
    
    def create_sru(self):
        Wf = tf.Variable(self.init_matrix([self.input_dim, self.output_dim]))
        bf = tf.Variable(self.init_matrix([self.output_dim]))
        Wr = tf.Variable(self.init_matrix([self.input_dim, self.output_dim]))
        br = tf.Variable(self.init_matrix([self.output_dim]))
        U = tf.Variable(self.init_matrix([self.input_dim, self.output_dim]))

        def unit(x, h_c):
            pre_h, pre_c = tf.unstack(h_c)
            f = tf.nn.sigmoid(tf.matmul(x, Wf) + bf)
            r = tf.nn.sigmoid(tf.matmul(x, Wr) + br)
            c = f*pre_c + (1 - f)*tf.matmul(x, U)
            h = r*tf.nn.tanh(c) + (1 - r)*x
            return tf.stack([h, c])
        
        return unit

    def train(self):
        records = {}
        for u, i, r, t in self.train_data:
            u, i = int(u), int(i)
            if u not in records.keys():
                records[u] = []
            records[u].append([i, r, t])
        for u in records.keys():
            records[u] = sorted(records[u], key=lambda x: x[2])

        for step in range(self.max_train_step):
            ground_truth = np.zeros([self.batch_size, self.max_item_num])
            action_in = np.zeros([self.unit_num, self.batch_size, self.input_dim])

            users = random.sample(list(records.keys()), self.batch_size)
            start_times = {u: random.randint(0, len(records[u]) - self.unit_num) for u in users}

            feed_dict = {}
            feed_dict[self.initial_state] = np.zeros([2, self.batch_size, self.output_dim])
            for n in range(self.unit_num):
                for u_index in range(len(users)):
                    u = users[u_index]
                    i, r = records[u][start_times[u] + n][0], records[u][start_times[u] + n][1]
                    action_in[n][u_index] = self.item_embedding[i]
                    ground_truth[u_index][i] = r
                
                feed_dict[self.action_input[n]] = action_in[n]
                feed_dict[self.r_real] = ground_truth
                self.sess.run(self.train_op[n], feed_dict)
                loss = self.sess.run(self.loss[n], feed_dict)

            print("Rnn pretraining step %d, loss %.4f"%(step, loss))
    
    def get_state(self, items):
        if len(items) > self.unit_num:
            raise RuntimeError("Items more than rnn units.")
        embedded_items = [np.reshape(self.item_embedding[i], [1, self.input_dim]) for i in items]

        feed_dict = {self.action_input[i]: embedded_items[i] for i in range(len(items))}
        feed_dict[self.initial_state] = np.zeros([2, 1, self.output_dim])
        h_c = self.sess.run(self.cur_rnn_states[len(items) - 1], feed_dict)
        state = h_c[0]
        return np.reshape(state, [-1])
