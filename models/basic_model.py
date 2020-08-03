from models.layers import *


class BasicModel:

    def __init__(self, configs, partitioner=None):
        self.partitioner = partitioner
        self.act_number = configs['act_num']
        self.embed_dim = configs['embed_dim']
        self.feat_base_dim = configs['feat_dim']

        self.UL = configs['UL']
        self.SL = configs['SL']
        self.US = configs['US']
        self.SS = configs['SS']
        self.h = configs['h']
        self.use_temporal = configs['use_temporal']
        self.l2_norm = configs['l2_norm']
        self.seq_model = configs['seq_model']
        self.combine = configs['combine']

        self.MLP_layers = [64, 16]

    def get_probs(self, name_scope, batch_size, encoded_info, keep_prob=1, is_training=True):
        with tf.variable_scope(name_or_scope=name_scope, reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            user_feats, user_seq_infos, item_feats, item_seq_infos = encoded_info

            user_embed = dense(user_feats, self.embed_dim, name_scope + "user_embed", tf.nn.leaky_relu, self.l2_norm,
                                keep_prob, is_training)

            item_embed = dense(item_feats, self.embed_dim, name_scope + "item_embed", tf.nn.leaky_relu, self.l2_norm,
                                keep_prob, is_training)

            outs = tf.concat([user_embed, item_embed], axis=-1)

            for n in self.MLP_layers:
                outs = dense(outs, n, name_scope + "MLP_{}".format(n), tf.nn.leaky_relu, self.l2_norm, keep_prob,
                              is_training)
            outs = dense(outs, 1, name_scope + "MLP_{}".format(n + 1), None, self.l2_norm, keep_prob,
                          is_training)
        return outs
