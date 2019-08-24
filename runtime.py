import numpy as np
import pandas as pd
import tensorflow as tf

def norm(x):
    return (np.sum(x**2))**0.5

def mf_embedding(ratings, user_num, item_num, config):
    np.random.shuffle(ratings)
    emb_train_ratio = float(config['EMB']['EMB_TRAIN_RATIO'])
    train_num = int(emb_train_ratio*len(ratings))
    data_train = ratings[:train_num]
    data_test = ratings[train_num:]

    emb_size = int(config['EMB']['EMB_SIZE'])
    
    user_embedding = tf.Variable(tf.truncated_normal([user_num, emb_size]))
    item_embedding = tf.Variable(tf.truncated_normal([item_num, emb_size]))
    item_bias = tf.Variable(tf.truncated_normal([item_num]))

    user_id = tf.placeholder(tf.int32, [None])
    item_id = tf.placeholder(tf.int32, [None])
    real_rating = tf.placeholder(tf.float32, [None])

    user_emb = tf.nn.embedding_lookup(user_embedding, user_id)
    item_emb = tf.nn.embedding_lookup(item_embedding, item_id)
    bias_emb = tf.nn.embedding_lookup(item_bias, item_id)
    dot_e = user_emb * item_emb
    pred_rating = tf.reduce_sum(dot_e, -1) + bias_emb
    
    l2_factor = float(config['EMB']['L2_FACTOR'])
    target_loss = tf.reduce_mean(0.5*tf.square(pred_rating - real_rating))
    loss = target_loss + l2_factor*tf.reduce_mean(tf.square(user_emb) + tf.square(item_emb))
    train_op = tf.train.AdamOptimizer().minimize(loss)

    rmse = tf.reduce_mean((pred_rating - real_rating)**2)**0.5

    max_step = int(config['EMB']['EMB_TRAIN_STEP'])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(max_step):
            np.random.shuffle(data_train)
            np.random.shuffle(data_test)
            sess.run(train_op, {user_id: data_train[:, 0].astype(int), item_id: data_train[:, 1].astype(int), real_rating: data_train[:, 2]})
            error = sess.run(rmse, {user_id: data_test[:, 0].astype(int), item_id: data_test[:, 1].astype(int), real_rating: data_test[:, 2]})
            print('mf train step %d, rmse %.4f'%(i, error))
        return np.array(sess.run(item_embedding))   