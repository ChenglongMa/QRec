#coding:utf8
from base.deepRecommender import DeepRecommender
import numpy as np
import random
from util import config
import tensorflow as tf

class APR(DeepRecommender):
    # APR：Adversarial Personalized Ranking for Recommendation
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(APR, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(APR, self).readConfiguration()
        args = config.LineConfig(self.config['APR'])
        self.eps = float(args['-eps'])
        self.regAdv = float(args['-regA'])
        self.advEpoch = int(args['-advEpoch'])

    def _create_variables(self):
        # perturbation vectors
        self.adv_U = tf.Variable(tf.zeros(shape=[self.num_users, self.emb_size]), dtype=tf.float32, trainable=False)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.num_items, self.emb_size]), dtype=tf.float32, trainable=False)
        self.neg_idx = tf.compat.v1.placeholder(tf.int32, [None], name="n_idx")
        self.V_neg_embed = tf.nn.embedding_lookup(params=self.item_embeddings, ids=self.neg_idx)
        # parameters
        self.eps = tf.constant(self.eps, dtype=tf.float32)
        self.regAdv = tf.constant(self.regAdv, dtype=tf.float32)

    def _create_inference(self):
        result = tf.subtract(tf.reduce_sum(input_tensor=tf.multiply(self.u_embedding, self.v_embedding), axis=1),
                             tf.reduce_sum(input_tensor=tf.multiply(self.u_embedding, self.V_neg_embed), axis=1))
        return result

    def _create_adv_inference(self):
        self.U_plus_delta = tf.add(self.u_embedding, tf.nn.embedding_lookup(params=self.adv_U, ids=self.u_idx))
        self.V_plus_delta = tf.add(self.v_embedding, tf.nn.embedding_lookup(params=self.adv_V, ids=self.v_idx))
        self.V_neg_plus_delta = tf.add(self.V_neg_embed, tf.nn.embedding_lookup(params=self.adv_V, ids=self.neg_idx))
        result = tf.subtract(tf.reduce_sum(input_tensor=tf.multiply(self.U_plus_delta, self.V_plus_delta), axis=1),
                             tf.reduce_sum(input_tensor=tf.multiply(self.U_plus_delta, self.V_neg_plus_delta), axis=1))
        return result

    def _create_adversarial(self):
        #get gradients of Delta
        self.grad_U, self.grad_V = tf.gradients(ys=self.loss_adv, xs=[self.adv_U, self.adv_V])

        # convert the IndexedSlice Data to Dense Tensor
        self.grad_U_dense = tf.stop_gradient(self.grad_U)
        self.grad_V_dense = tf.stop_gradient(self.grad_V)

        # normalization: new_grad = (grad / |grad|) * eps
        self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
        self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)


    def _create_loss(self):
        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.loss = tf.reduce_sum(input_tensor=tf.nn.softplus(-self._create_inference()))
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.u_embedding)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.v_embedding)))
        self.total_loss = tf.add(self.loss, self.reg_loss)
        #loss of adversarial training
        self.loss_adv = tf.multiply(self.regAdv,
                                    tf.reduce_sum(input_tensor=tf.nn.softplus(-self._create_adv_inference())))
        self.loss_adv = tf.add(self.total_loss, self.loss_adv)

    def _create_optimizer(self):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss)
        self.optimizer_adv = tf.compat.v1.train.AdamOptimizer(self.lRate)
        self.train_adv = self.optimizer.minimize(self.loss_adv)


    def initModel(self):
        super(APR, self).initModel()
        self.u_embedding = tf.nn.embedding_lookup(params=self.user_embeddings, ids=self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(params=self.item_embeddings, ids=self.v_idx)
        self._create_variables()
        self._create_loss()
        self._create_adversarial()
        self._create_optimizer()


    def next_batch(self):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)
        users = [self.data.trainingData[idx][0] for idx in batch_idx]
        items = [self.data.trainingData[idx][1] for idx in batch_idx]
        user_idx,item_idx=[],[]
        neg_item_idx = []
        for i,user in enumerate(users):
            item_j = random.randint(0,self.num_items-1)
            while self.data.id2item[item_j] in self.data.trainSet_u[user]:
                item_j = random.randint(0, self.num_items - 1)
            user_idx.append(self.data.user[user])
            item_idx.append(self.data.item[items[i]])
            neg_item_idx.append(item_j)

        return user_idx,item_idx,neg_item_idx

    def buildModel(self):
        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            # pretraining
            print('pretraining...')
            for epoch in range(self.maxEpoch // 2):
                for iteration, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    _, loss = sess.run([self.train, self.total_loss],
                                       feed_dict={self.u_idx: user_idx, self.v_idx: i_idx, self.neg_idx: j_idx})

            # start adversarial training
            print('adversarial training...')
            for epoch in range(self.maxEpoch // 2):
                for iteration, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    sess.run([self.update_U, self.update_V],
                             feed_dict={self.u_idx: user_idx, self.v_idx: j_idx, self.neg_idx: j_idx})
                    _,loss = sess.run([self.train_adv,self.loss_adv],feed_dict={self.u_idx: user_idx, self.v_idx: i_idx, self.neg_idx:j_idx})
                    print(self.foldInfo, 'training:', epoch + 1, 'batch', iteration, 'loss:', loss)
                self.P = sess.run(self.user_embeddings)
                self.Q = sess.run(self.item_embeddings)


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items


