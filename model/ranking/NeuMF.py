# coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
from random import randint
import tensorflow as tf

tf.compat.v1.reset_default_graph()


class NeuMF(DeepRecommender):

    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(NeuMF, self).__init__(conf, trainingSet, testSet, fold)

    def initModel(self):
        super(NeuMF, self).initModel()
        # parameters used are consistent with default settings in the original paper
        mlp_regularizer = tf.keras.regularizers.l2(l=0.5 * (0.001))
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        with tf.compat.v1.variable_scope("latent_factors", reuse=tf.compat.v1.AUTO_REUSE):
            self.PG = tf.compat.v1.get_variable(name='PG', initializer=initializer([self.num_users, self.emb_size]))
            self.QG = tf.compat.v1.get_variable(name='QG', initializer=initializer([self.num_items, self.emb_size]))
            self.PM = tf.compat.v1.get_variable(name='PM', initializer=initializer([self.num_users, self.emb_size]),
                                                regularizer=mlp_regularizer)
            self.QM = tf.compat.v1.get_variable(name='QM', initializer=initializer([self.num_items, self.emb_size]),
                                                regularizer=mlp_regularizer)

        with tf.compat.v1.name_scope("input"):
            self.r = tf.compat.v1.placeholder(tf.float32, [None], name="rating")
            self.u_idx = tf.compat.v1.placeholder(tf.int32, [None], name="u_idx")
            self.i_idx = tf.compat.v1.placeholder(tf.int32, [None], name="i_idx")
            self.UG_embedding = tf.nn.embedding_lookup(params=self.PG, ids=self.u_idx)
            self.IG_embedding = tf.nn.embedding_lookup(params=self.QG, ids=self.i_idx)
            self.UM_embedding = tf.nn.embedding_lookup(params=self.PM, ids=self.u_idx)
            self.IM_embedding = tf.nn.embedding_lookup(params=self.QM, ids=self.i_idx)

        # Generic Matrix Factorization
        with tf.compat.v1.variable_scope("mf_output"):
            self.GMF_Layer = tf.multiply(self.UG_embedding, self.IG_embedding)
            self.h_mf = tf.compat.v1.get_variable(name='mf_out', initializer=initializer([self.emb_size]))

        # MLP
        with tf.compat.v1.variable_scope("mlp_params"):
            MLP_W1 = tf.compat.v1.get_variable(name='W1',
                                               initializer=initializer([self.emb_size * 2, self.emb_size * 5]),
                                               regularizer=mlp_regularizer)
            MLP_b1 = tf.compat.v1.get_variable(name='b1', initializer=tf.zeros(shape=[self.emb_size * 5]),
                                               regularizer=mlp_regularizer)
            self.h_out = tf.nn.relu(
                tf.add(tf.matmul(tf.concat([self.UM_embedding, self.IM_embedding], 1), MLP_W1), MLP_b1))

            MLP_W2 = tf.compat.v1.get_variable(name='W2',
                                               initializer=initializer([self.emb_size * 5, self.emb_size * 2]),
                                               regularizer=mlp_regularizer)
            MLP_b2 = tf.compat.v1.get_variable(name='b2', initializer=tf.zeros(shape=[self.emb_size * 2]),
                                               regularizer=mlp_regularizer)
            self.h_out = tf.nn.relu(tf.add(tf.matmul(self.h_out, MLP_W2), MLP_b2))

            MLP_W3 = tf.compat.v1.get_variable(name='W3', initializer=initializer([self.emb_size * 2, self.emb_size]),
                                               regularizer=mlp_regularizer)
            MLP_b3 = tf.compat.v1.get_variable(name='b3', initializer=tf.zeros(shape=[self.emb_size]),
                                               regularizer=mlp_regularizer)
            self.MLP_Layer = tf.nn.relu(tf.add(tf.matmul(self.h_out, MLP_W3), MLP_b3))
            self.h_mlp = tf.compat.v1.get_variable(name='mlp_out', initializer=initializer([self.emb_size]),
                                                   regularizer=mlp_regularizer)

        # single inference
        # GMF
        self.y_mf = tf.reduce_sum(input_tensor=tf.multiply(self.GMF_Layer, self.h_mf), axis=1)
        self.y_mf = tf.sigmoid(self.y_mf)
        self.mf_loss = self.r * tf.math.log(self.y_mf + 10e-10) + (1 - self.r) * tf.math.log(1 - self.y_mf + 10e-10)
        mf_reg = self.regU * (
                tf.nn.l2_loss(self.UG_embedding) + tf.nn.l2_loss(self.IG_embedding) + tf.nn.l2_loss(self.h_mf))
        self.mf_loss = -tf.reduce_sum(input_tensor=self.mf_loss) + mf_reg
        self.mf_optimizer = tf.compat.v1.train.AdamOptimizer(self.lRate).minimize(self.mf_loss)
        # MLP
        self.y_mlp = tf.reduce_sum(input_tensor=tf.multiply(self.MLP_Layer, self.h_mlp), axis=1)
        self.y_mlp = tf.sigmoid(self.y_mlp)
        self.mlp_loss = self.r * tf.math.log(self.y_mlp + 10e-10) + (1 - self.r) * tf.math.log(1 - self.y_mlp + 10e-10)
        self.mlp_loss = -tf.reduce_sum(input_tensor=self.mlp_loss)
        self.mlp_optimizer = tf.compat.v1.train.AdamOptimizer(self.lRate).minimize(self.mlp_loss)
        # fusion
        self.NeuMF_Layer = tf.concat([self.GMF_Layer, self.MLP_Layer], 1)
        self.h_NeuMF = tf.concat([0.5 * self.h_mf, 0.5 * self.h_mlp], 0)
        self.y_neu = tf.reduce_sum(input_tensor=tf.multiply(self.NeuMF_Layer, self.h_NeuMF), axis=1)
        self.y_neu = tf.sigmoid(self.y_neu)
        self.neu_loss = self.r * tf.math.log(self.y_neu + 10e-10) + (1 - self.r) * tf.math.log(1 - self.y_neu + 10e-10)

        self.neu_loss = -tf.reduce_sum(input_tensor=self.neu_loss) + mf_reg + self.regU * tf.nn.l2_loss(self.h_NeuMF)
        ###it seems Adam is better than SGD here...
        self.neu_optimizer = tf.compat.v1.train.AdamOptimizer(self.lRate).minimize(self.neu_loss)

    def buildModel(self):
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        print('pretraining... (GMF)')
        for epoch in range(self.maxEpoch):
            for num, batch in enumerate(self.next_batch_pointwise()):
                user_idx, item_idx, r = batch
                _, loss, y_mf = self.sess.run([self.mf_optimizer, self.mf_loss, self.y_mf],
                                              feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print('epoch:', epoch, 'batch:', num, 'loss:', loss)
        print('pretraining... (MLP)')
        for epoch in range(self.maxEpoch // 2):
            for num, batch in enumerate(self.next_batch_pointwise()):
                user_idx, item_idx, r = batch
                _, loss, y_mlp = self.sess.run([self.mlp_optimizer, self.mlp_loss, self.y_mlp],
                                               feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print('epoch:', epoch, 'batch:', num, 'loss:', loss)
        print('training... (NeuMF)')
        for epoch in range(self.maxEpoch // 5):
            for num, batch in enumerate(self.next_batch_pointwise()):
                user_idx, item_idx, r = batch
                _, loss, y_neu = self.sess.run([self.neu_optimizer, self.neu_loss, self.y_neu],
                                               feed_dict={self.u_idx: user_idx, self.i_idx: item_idx, self.r: r})
                print('epoch:', epoch, 'batch:', num, 'loss:', loss)

    def predict_mlp(self, uid):
        user_idx = [uid] * self.num_items
        y_mlp = self.sess.run([self.y_mlp], feed_dict={self.u_idx: user_idx, self.i_idx: list(range(self.num_items))})
        return y_mlp[0]

    def predict_mf(self, uid):
        user_idx = [uid] * self.num_items
        y_mf = self.sess.run([self.y_mf], feed_dict={self.u_idx: user_idx, self.i_idx: list(range(self.num_items))})
        return y_mf[0]

    def predict_neu(self, uid):
        user_idx = [uid] * self.num_items
        y_neu = self.sess.run([self.y_neu], feed_dict={self.u_idx: user_idx, self.i_idx: list(range(self.num_items))})
        return y_neu[0]

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]
            return self.predict_neu(u)
        else:
            return [self.data.globalMean] * self.num_items
