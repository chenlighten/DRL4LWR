import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from runtime import *
from rnn import *
from tqdm import tqdm
import pickle

class Env():
    def __init__(self, config, usage):
        filename = config['ENV']['RATING_FILE']
        if filename.split('.')[-1] == 'csv':
            df = pd.read_csv(config['ENV']['RATING_FILE'])
            self.ratings = df.values
        else:
            self.ratings = np.loadtxt(filename)

        self.user_num, self.item_num = self.reID()
        self.train_ratio = float(config['TRAIN']['TRAIN_RATIO'])
        self.boundary_user_id = int(self.train_ratio*self.user_num)
        if usage == 'train':
            self.user_num = self.boundary_user_id
            self.ratings = np.array(list(filter(lambda x: x[0] < self.boundary_user_id, self.ratings)))
        elif usage == 'test':
            self.user_num = self.user_num - self.boundary_user_id
            self.ratings = np.array(list(filter(lambda x: x[0] >= self.boundary_user_id, self.ratings)))
        else:
            raise RuntimeError('No such usage as %s'%usage)

        self.action_dim = int(config['EMB']['EMB_SIZE'])
        # self.state_dim = int(config['RNN']['OUTPUT_DIM'])
        self.state_dim = self.action_dim

        self.r_matrix = coo_matrix((self.ratings[:, 2], (self.ratings[:, 0].astype(int), self.ratings[:, 1].astype(int)))).toarray()
        self.t_matrix = coo_matrix((self.ratings[:, 3], (self.ratings[:, 0].astype(int), self.ratings[:, 1].astype(int)))).toarray()

        self.alpha = float(config['ENV']['ALPHA'])
        self.seq_len = int(config['ENV']['SEQ_LEN'])

        if config['ENV']['LOAD_EMB'] == 'F':
            self.item_embeddings = mf_embedding(self.ratings, self.user_num, self.item_num, config)
            np.savetxt('./runtime_data/item_embeddings.txt', self.item_embeddings)
        else:
            self.item_embeddings = np.loadtxt('./runtime_data/item_embeddings.txt')

        self.rnn = Rnn(self.ratings, self.item_embeddings, config)

        if config['ENV']['LOAD_PRED'] == 'F':
            self.construct_predictor()
            with open('./runtime_data/pred_dict', 'wb') as fr:
                pickle.dump(self.pred_dict, fr, True)
        else:
            with open('./runtime_data/pred_dict', 'rb') as fr:
                self.pred_dict = pickle.load(fr)
            

    def reID(self):
        user2ID = {}
        item2ID = {}
        user_count = 0
        item_count = 0
        for n in range(len(self.ratings)):
            u = self.ratings[n][0]
            i = self.ratings[n][1]
            if u not in user2ID.keys():
                user2ID[u] = user_count
                user_count += 1
            if i not in item2ID.keys():
                item2ID[i] = item_count
                item_count += 1
            self.ratings[n][0] = user2ID[u]
            self.ratings[n][1] = item2ID[i]
        return user_count, item_count

    def construct_predictor(self):
        print("Begins constructing predictor.")
        self.pred_dict = {}
        for n in tqdm(range(len(self.ratings))):
            a = self.item_embeddings[int(self.ratings[n][1])]
            hist = []
            m = n - 1
            while int(self.ratings[m][0]) == int(self.ratings[n][0]) and n - m <= self.seq_len:
                hist.append(int(self.ratings[m][1]))
                m -= 1
            hist.reverse()
            if len(hist) != 0:
                s = self.rnn.get_state(hist)
            else:
                s = np.zeros([self.state_dim])
            r = self.ratings[n][2]

            norm_s, norm_a = norm(s), norm(a)
            if norm_s == 0: norm_s = 1
            if norm_a == 0: norm_a = 1

            if r not in self.pred_dict.keys():
                self.pred_dict[r] = [s/norm_s, a/norm_a, 1]
            else:
                self.pred_dict[r][0] += s/norm_s
                self.pred_dict[r][1] += a/norm_a
                self.pred_dict[r][2] += 1
        print("Finished predictor construction.")
            
    def get_reward(self, s, a):
        weight = []
        for i in range(10):
            r = 0.5 + i*0.5
            norm_s, norm_a = norm(s), norm(a)
            if norm_s == 0: norm_s = 1
            if norm_a == 0: norm_a = 1
            weight.append(self.pred_dict[r][2]*(self.alpha*(np.dot(s/norm_s, self.pred_dict[r][0]) + \
                (1 - self.alpha)*(np.dot(a/norm_a, self.pred_dict[r][1])))))
        weight = np.array(weight)
        weight = weight/np.sum(weight)
        probs = np.exp(weight) / np.sum(np.exp(weight))
        return np.random.choice([0.5 + 0.5*i for i in range(10)], p=probs)

    def id2vector_item(self, id):
        return self.item_embeddings[id]

    def vector2id_item(self, vec):
        dis = np.sum((self.item_embeddings - vec)**2, -1)
        return np.argsort(dis)[0]

    def get_rnn(self):
        return self.rnn