from base.deepRecommender import DeepRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
from math import sqrt
import numpy as np
import os
from util import config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#For general comparison. We do not include the user/item features extracted from text/images

class DiffNet(SocialRecommender,DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(DiffNet, self).readConfiguration()
        args = config.LineConfig(self.config['DiffNet'])
        self.n_layers = int(args['-n_layer']) #the number of layers of the recommendation module (discriminator)

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0/len(self.social.followees[pair[0]])]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0/len(self.data.trainSet_u[pair[0]])]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix

    def initModel(self):
        super(DiffNet, self).initModel()
        S = self.buildSparseRelationMatrix()
        A = self.buildSparseRatingMatrix()
        indices = np.mat([S.row, S.col]).transpose()
        self.S = tf.SparseTensor(indices, S.data.astype(np.float32), S.shape)
        indices = np.mat([A.row, A.col]).transpose()
        self.A = tf.SparseTensor(indices, A.data.astype(np.float32), A.shape)

    def buildModel(self):
        self.weights = {}
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        for k in range(self.n_layers):
            self.weights['weights%d' % k] = tf.Variable(
                initializer([2 * self.emb_size, self.emb_size]), name='weights%d' % k)

        user_embeddings = self.user_embeddings
        for k in range(self.n_layers):
            new_user_embeddings = tf.sparse.sparse_dense_matmul(self.S, user_embeddings)
            user_embeddings = tf.matmul(tf.concat([new_user_embeddings, user_embeddings], 1),
                                        self.weights['weights%d' % k])
            user_embeddings = tf.nn.relu(user_embeddings)
            # user_embeddings = tf.math.l2_normalize(user_embeddings,axis=1)

        final_user_embeddings = user_embeddings + tf.sparse.sparse_dense_matmul(self.A, self.item_embeddings)
        self.neg_idx = tf.compat.v1.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(params=self.item_embeddings, ids=self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(params=final_user_embeddings, ids=self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(params=self.item_embeddings, ids=self.v_idx)
        self.test = tf.reduce_sum(input_tensor=tf.multiply(self.u_embedding, self.item_embeddings), axis=1)

        y = tf.reduce_sum(input_tensor=tf.multiply(self.u_embedding, self.v_embedding), axis=1) \
            - tf.reduce_sum(input_tensor=tf.multiply(self.u_embedding, self.neg_item_embedding), axis=1)
        loss = -tf.reduce_sum(input_tensor=tf.math.log(tf.sigmoid(y))) + self.regU * (
                tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.compat.v1.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', epoch + 1, 'batch', n, 'loss:', l)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u})
        else:
            return [self.data.globalMean] * self.num_items