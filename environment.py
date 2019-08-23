import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from runtime import *

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

        self.r_matrix = coo_matrix((self.ratings[:, 2], (self.ratings[:, 0].astype(int), self.ratings[:, 1].astype(int)))).toarray()
        self.t_matrix = coo_matrix((self.ratings[:, 3], (self.ratings[:, 0].astype(int), self.ratings[:, 1].astype(int)))).toarray()

        self.alpha = float(config['ENV']['ALPHA'])
        self.seq_len = int(config['ENV']['SEQ_LEN'])

        self.item_embeddings = mf_embedding(self.ratings, self.user_num, self.item_num)

    def reID(self):
        user2ID = {}
        item2ID = {}
        user_count = 0
        item_count = []
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
        pred_dict = {}
        

        