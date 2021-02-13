# coding=utf-8
"""
some loss
"""
from __future__ import print_function
import paddle.fluid as F
import paddle.fluid.layers as L

from model.cnn import CnnModel


class Model(object):
    """model"""

    def __init__(self, args, task):
        """init model"""
        candi_tasks = ["predict_query", "predict_poi", "train", "predict",
                       "pointwise", "pairwise", "matrixwise",
                       "matrixwise_hinge"]

        if task not in candi_tasks:
            raise ValueError("task %s not in %s" % (task, candi_tasks))
        self.holder_list = []
        self.activate = "relu"
        self.emb_lr = 1.0
        self.addr = args.addr_task
        self.neg_num = args.neg_num
        self.geo_len = 30
        self.history_len = 10
        self.max_poi_src_len = 30
        self.max_query_src_len = 10
        self.batch_size = args.batch_size
        self.network = CnnModel()
        self.metrics = []

        self.build_model(task)

    def unsqueeze(self, tensor):
        """ernie_unsqueeze"""
        tensor = L.unsqueeze(tensor, axes=2)
        tensor.stop_gradient = True
        return tensor

    def embedding(self, inputs, dict_size, emb_size, name="geo_emb"):
        """ add embedding """
        inputs = L.unsqueeze(inputs, axes=-1)
        return L.embedding(
            inputs,
            size=(dict_size, emb_size),
            param_attr=F.ParamAttr(name=name))

    def build_model(self, task):
        """ build graph model"""
        self.poi_geo_ids = L.data(name="poi_geo", shape=[None, self.geo_len], dtype="int64")
        self.holder_list.append(self.poi_geo_ids)

        self.name_src_ids = L.data(
            name='name_src_ids',
            shape=[None, self.max_poi_src_len],
            dtype="int64")
        self.holder_list.append(self.name_src_ids)

        self.query_geo_ids = L.data(name="query_geo", shape=[None, self.geo_len], dtype="int64")
        self.holder_list.append(self.query_geo_ids)
        self.query_src_ids = L.data(
            name='query_src_ids',
            shape=[None, self.max_query_src_len],
            dtype="int64")
        self.holder_list.append(self.query_src_ids)
        if task == "pointwise":
            self.pointwise_loss()
        elif task == "pairwise":
            self.pairwise_loss()
        elif task == "matrixwise":
            self.matrixwise_loss()

    def pointwise_loss(self):
        """point wise model"""
        self.logits = L.reduce_sum(self.query_repr * self.poi_repr, -1)
        self.score = L.sigmoid(self.logits)
        self.loss = L.sigmoid_cross_entropy_with_logits(
            L.reshape(self.logits, [-1, 1]), L.reshape(self.labels, [-1, 1]))

        auc_label = L.cast(self.labels, dtype="int64")
        auc_label.stop_gradients = True
        _, self.batch_auc, _ = L.auc(
            L.reshape(self.score, [-1, 1]), L.reshape(auc_label, [-1, 1]))
        self.metrics = [L.reduce_mean(self.loss), self.batch_auc]
        self.loss = L.reduce_mean(self.loss)


    def loss_neg_log_of_pos(self, pos_score, neg_score_n, gama=5.0):
        """
            pos_score: batch_size x 1
            neg_score_n: batch_size x n
        """
        # n x batch_size
        neg_score_n = L.transpose(neg_score_n, [1, 0])
        # 1 x batch_size
        pos_score = L.reshape(pos_score, [1, -1])
        exp_pos_score = L.exp(pos_score * gama)
        exp_neg_score_n = L.exp(neg_score_n * gama)
        # (n+1) x batch_size
        pos_neg_score = L.concat([exp_pos_score, exp_neg_score_n], axis=0)
        # 1 x batch_size
        exp_sum = L.reduce_sum(pos_neg_score, dim=0, keep_dim=True)
        # 1 x batch_size
        loss = -1.0 * L.log(exp_pos_score / exp_sum)
        # batch_size
        loss = L.reshape(loss, [-1, 1])
        return loss

    def pairwise_hinge(self):
        """pairwise model"""
        poi_repr = L.split(self.poi_repr, 2, dim=0)
        pos_repr, neg_repr = poi_repr
        pos_pred = L.cos_sim(self.query_repr, pos_repr)
        neg_pred = L.cos_sim(self.query_repr, neg_repr)

        mode = 'hinge_loss'
        # log(1 + e-z), max(0, 1 - z)
        if 'hinge_loss' == mode:
            theta_z = L.relu(1 + neg_pred - pos_pred)
        elif 'logistic_loss' == mode:
            theta_z = L.log(1 + L.exp(neg_pred - pos_pred))
        self.loss = L.reduce_mean(theta_z)
        pos_cnt = L.reduce_sum(L.cast(L.greater_than(pos_pred, neg_pred), dtype="float32"))
        neg_cnt = L.reduce_sum(L.cast(L.less_than(pos_pred, neg_pred), dtype="float32"))
        self.order = pos_cnt / (1e-5 + neg_cnt)
        self.metrics = [self.loss, self.order]

    def pairwise_loss(self):
        """pairwise model"""
        # TODO: for neg_num neg poi, split num should be (neg_num + 1) on dim 0
        poi_repr = L.split(self.poi_repr, [1 * self.batch_size, self.neg_num * self.batch_size], dim=0)
        pos_repr, neg_repr = poi_repr
        # size [-1 x emb_size]
        # size [-1*n x emb_size]
        prefix_expand = L.reshape(L.expand(self.query_repr, [1, self.neg_num]), [-1, self.hidden_size])
        # size [-1*n x 1]
        neg_pred_n = self.safe_cosine_sim(neg_repr, prefix_expand)
        # size [-1 x 1]
        pos_pred = self.safe_cosine_sim(pos_repr, self.query_repr)
        cost = self.loss_neg_log_of_pos(pos_pred, L.reshape(neg_pred_n, [-1, self.neg_num]), 15)
        self.loss = L.mean(x=cost)
        # size [-1 x 1]
        neg_avg = L.reduce_mean(L.reshape(neg_pred_n, [-1, self.neg_num]), dim=1, keep_dim=True)
        pos_cnt = L.reduce_sum(L.cast(L.greater_than(pos_pred, neg_avg), dtype="float32"))
        neg_cnt = L.reduce_sum(L.cast(L.less_than(pos_pred, neg_avg), dtype="float32"))
        # equal to positive and negative order
        self.order = pos_cnt / (1e-5 + neg_cnt)
        self.metrics = [self.loss, self.order]

    def matrixwise_loss(self):
        """listwise model"""
        self.logits = L.matmul(
            self.query_repr, self.poi_repr, transpose_y=True)
        self.score = L.softmax(self.logits)
        self.loss = L.softmax_with_cross_entropy(self.logits, self.labels)
        self.loss = L.reduce_mean(self.loss)
        self.acc = L.accuracy(L.softmax(self.logits), self.labels)
        self.metrics = [self.loss, self.acc]

    def safe_cosine_sim(self, x, y):
        """
            fluid.layers.cos_sim maybe nan
            avoid nan
        """
        l2x = L.l2_normalize(x, axis=-1)
        l2y = L.l2_normalize(y, axis=-1)
        cos = L.reduce_sum(l2x * l2y, dim=1, keep_dim=True)
        return cos

    def get_poi_vec(self, inputs, tag="pos"):
        """
        get poi vec
        """
        # poi_geo_emb = self.embedding(self.poi_geo_ids, self.geo_size,
        #                              self.geo_emb_size, "poi_geo_emb")
        poi_geo_emb = L.reshape(L.cast(x=inputs[tag + "_geo_ids"], dtype="float32"), [-1, self.geo_len])

        name_pool = self.network.cnn_net(inputs[tag + "_name_ids"],
                                         "name_emb",
                                         self.vocab_size,
                                         self.emb_size,
                                         hid_dim=self.hidden_size,
                                         emb_lr=self.emb_lr,
                                         )

        if self.addr:
            addr_pool = self.network.cnn_net(inputs[tag + "_addr_ids"],
                                             "addr_emb",
                                             self.vocab_size,
                                             self.emb_size,
                                             hid_dim=self.hidden_size,
                                             emb_lr=self.emb_lr,
                                             )
            # size [64, 64, 30] todo: -> [64, 30, 30]
            poi_pool = L.concat([name_pool, addr_pool, poi_geo_emb], axis=1)
            # todo: add fc only for name and addr
        else:
            poi_pool = L.concat([name_pool, poi_geo_emb], axis=1)
        poi_vec = L.fc(input=poi_pool, size=self.hidden_size,
                       param_attr=F.ParamAttr(name='poi_fc_weight'),
                       bias_attr=F.ParamAttr(name='poi_fc_bias'))

        return poi_vec

    def get_query_vec(self, inputs):
        """
        get query & user vec
        """
        query_geo_emb = L.reshape(L.cast(x=inputs["query_geo_ids"], dtype="float32"), [-1, self.geo_len])

        query_pool = self.network.cnn_net(inputs["query_ids"],
                                          "query_emb",
                                          self.vocab_size,
                                          self.emb_size,
                                          hid_dim=self.hidden_size,
                                          emb_lr=self.emb_lr,
                                          )

        history_pool = self.network.cnn_net(inputs["history_ids"],
                                            "history_emb",
                                            self.history_vocab_size,
                                            self.emb_size,
                                            hid_dim=self.hidden_size // 2,
                                            emb_lr=self.emb_lr,
                                            )
        # concate size: [64, 32, 32] 
        context_pool = L.concat([query_pool, history_pool, query_geo_emb], axis=1)
        context_vec = L.fc(input=context_pool, size=self.hidden_size,
                           param_attr=F.ParamAttr(name='q_fc_weight'),
                           bias_attr=F.ParamAttr(name='q_fc_bias'))

        return context_vec
