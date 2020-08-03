from models.attentions import *
from models.basic_model import BasicModel
from models.layers import *


class THIGE(BasicModel):

    def __init__(self, configss, partitioner=None):
        super(THIGE, self).__init__(configss, partitioner)

    def get_probs(self, name_scope, batch_size, encoded_info, keep_prob=1, is_training=True):

        with tf.variable_scope(name_or_scope=name_scope, reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            user_feats, user_seq_infos, item_feats, item_seq_infos = encoded_info
            user_base_embed = dense(user_feats, self.embed_dim, name_scope + "user_base_embed", tf.nn.leaky_relu,
                                    self.l2_norm, keep_prob, is_training)

            item_base_embed = dense(item_feats, self.embed_dim, name_scope + "item_base_embed", tf.nn.leaky_relu,
                                    self.l2_norm, keep_prob, is_training)

            user_interest = self.generate_interest_v2(name_scope + "user_interest", user_seq_infos, user_base_embed,
                                                      self.UL, self.US,
                                                      batch_size, keep_prob, is_training)
            item_interest = self.generate_interest_v2(name_scope + "item_interest", item_seq_infos, item_base_embed,
                                                      self.SL, self.SS,
                                                      batch_size, keep_prob, is_training)

            if user_interest is not None:
                user_embed = dense(tf.concat([user_base_embed, user_interest], axis=1), self.embed_dim,
                                   name_scope + "user_embed", tf.nn.leaky_relu, self.l2_norm, keep_prob, is_training)
            else:
                user_embed = user_base_embed

            if item_interest is not None:
                item_embed = dense(tf.concat([item_base_embed, item_interest], axis=1), self.embed_dim,
                                   name_scope + "item_embed", tf.nn.leaky_relu, self.l2_norm, keep_prob, is_training)
            else:
                item_embed = item_base_embed

            outs = tf.concat([user_embed, item_embed], axis=-1)

            for n in self.MLP_layers:
                outs = dense(outs, n, name_scope + "MLP_{}".format(n), tf.nn.leaky_relu, self.l2_norm, keep_prob,
                             is_training)

            outs = dense(outs, 1, name_scope + "MLP_{}".format(n + 1), None, self.l2_norm, keep_prob,
                         is_training)
        return outs

    def generate_interest_v2(self, name_scope, seqs, base_embed, long_length, short_length, batch_size, keep_prob=1,
                             is_training=True):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            interest_info = {}
            if short_length > 0:
                short_info = self.filter_indices(seqs, short_length)
                short_base_embeds = dense(short_info[1], self.embed_dim, name_scope + 'base_feat_embed',
                                          tf.nn.leaky_relu, self.l2_norm, keep_prob, is_training)
                short_embeds = dense(short_base_embeds + short_info[2] + short_info[4] * self.use_temporal,
                                     self.embed_dim, name_scope + 'short_feats', tf.nn.leaky_relu, self.l2_norm,
                                     keep_prob, is_training)

                short_mask, short_interest, short_states = self.generate_short_interest(name_scope + "short_interest",
                                                                                        short_embeds, short_info[0],
                                                                                        short_length, batch_size)

                interest_info['short_mask'] = short_mask
                interest_info['short_interest'] = short_interest
                interest_info['short_states'] = short_states

                # final_interest = self.short_interest_aggregator(name_scope + "att_interest", short_mask, short_interest,
                #                                                 short_length, 'mean')
            if long_length > 0:
                long_info = self.filter_indices(seqs, long_length)
                long_base_embeds = dense(long_info[1], self.embed_dim, name_scope + 'base_feat_embed', tf.nn.leaky_relu,
                                         self.l2_norm, keep_prob, is_training)
                long_embeds = dense(long_base_embeds + long_info[4] * self.use_temporal, self.embed_dim, 'long_feats',
                                    tf.nn.leaky_relu, self.l2_norm, keep_prob, is_training)

                long_mask, long_interest = self.generate_long_interest(name_scope + "long_interest", long_embeds,
                                                                       long_info[0], long_info[3], batch_size,
                                                                       keep_prob, is_training)
                att_long_interest = 1
                # utilize attention or utilize mean_aggregation
                interest_info['long_mask'] = long_mask
                interest_info['long_interest'] = long_interest

                # final_interest = self.long_interest_aggregator(name_scope + "long_interest_atted", long_mask,
                #                                                long_interest, self.h, 'mean')

            if short_length > 0 and long_length > 0:

                final_interest = self.combine_short_long_interest(name_scope + "combine_att_interest", interest_info,
                                                                  base_embed, self.h, combination=self.combine,
                                                                  keep_prob=keep_prob, is_training=is_training)

            elif short_length > 0 and long_length <= 0:
                final_interest = self.short_interest_aggregator(name_scope + "short_att_interest", short_mask,
                                                                short_interest,
                                                                short_length, 'mean')
            elif short_length <= 0 and long_length > 0:
                final_interest = self.long_interest_aggregator(name_scope + "long_interest_atted", long_mask,
                                                               long_interest, self.h, 'mean')
            else:
                final_interest = None

        return final_interest

    def generate_short_interest(self, name, short_embeds, short_indices, short_length, batch_size):
        segments = short_indices[:, 0] * short_length + short_indices[:, 1]
        new_embeds = tf.reshape(tf.unsorted_segment_sum(short_embeds, segments, batch_size * short_length),
                                [batch_size, short_length, self.embed_dim])

        new_mask0 = tf.reshape(tf.unsorted_segment_sum(tf.ones(shape=[tf.size(segments), 1], dtype=tf.int32), segments,
                                                       batch_size * short_length),
                               [batch_size, short_length])
        new_mask = tf.cast(new_mask0, tf.bool)
        lengths = tf.reduce_sum(new_mask0, axis=1)

        if self.seq_model == 'GRU':
            outs, states = dynamic_gru(new_embeds, lengths, name + "GRU_layer1", self.embed_dim)
        elif self.seq_model == "LSTM":
            outs, states = dynamic_LSTM(new_embeds, lengths, name + "LSTM_layer1", self.embed_dim)
        else:
            outs = new_embeds
            states = None
        return new_mask, outs, states

    def generate_long_interest(self, name, long_embeds, long_indices, act_ids, batch_size, keep_prob, is_training):
        parttens_embedx = tf.dynamic_partition(long_embeds, act_ids, num_partitions=self.act_number)
        parttens_idx = tf.dynamic_partition(long_indices, act_ids, num_partitions=self.act_number)
        num_lengths = tf.unsorted_segment_sum(tf.ones_like(act_ids), act_ids, self.act_number)

        embeds = []
        for i in range(self.act_number):
            type_i_embed = tf.cond(
                num_lengths[i] > 0,
                true_fn=lambda: dense(
                    tf.unsorted_segment_mean(parttens_embedx[i], parttens_idx[i][:, 0], batch_size), self.embed_dim,
                    'long_embed_type__{}_{}'.format(name, i), tf.nn.leaky_relu, self.l2_norm, keep_prob, is_training),
                false_fn=lambda: tf.zeros([batch_size, self.embed_dim], dtype=tf.float32)
            )
            embeds.append(type_i_embed)
        embeds = tf.transpose(tf.reshape(embeds, [self.act_number, batch_size, self.embed_dim]), [1, 0, 2])
        new_segs = long_indices[:, 0] * self.act_number + long_indices[:, 1]
        mask = tf.reshape(tf.unsorted_segment_sum(tf.ones(shape=[tf.size(new_segs), 1], dtype=tf.int32), new_segs,
                                                  batch_size * self.act_number) > 0, [batch_size, self.act_number, 1])

        return mask, embeds

    def combine_short_long_interest(self, name_scope, interest_info, base_embed, multihead=8, combination='co_att',
                                    is_training=True, keep_prob=1.0):
        long_interest = multihead_attentions(interest_info['long_interest'], interest_info['long_interest'],
                                             interest_info['long_interest'], interest_info['long_mask'],
                                             self.embed_dim // 2, self.embed_dim, tf.nn.leaky_relu, multihead,
                                             scope=name_scope + "multi_heads_{}".format(multihead),
                                             norm_rate=self.l2_norm)

        if combination == 'co_att':
            atted_long_interest, atted_short_interest = co_attention(long_interest, interest_info['short_interest'],
                                                                     interest_info['long_mask'],
                                                                     interest_info['short_mask'], self.l2_norm,
                                                                     self.embed_dim // 2,
                                                                     name_scope + "co_att_long_short", self.embed_dim)
            new_interest = tf.reduce_sum(tf.concat([atted_long_interest, atted_short_interest], axis=-1), axis=1)
        elif combination == 'guide_att':
            long_length = tf.clip_by_value(tf.reduce_sum(tf.cast(interest_info['long_mask'], tf.float32), axis=1), 1,
                                           self.act_number)

            atted_long_interest = tf.reduce_sum(long_interest, axis=1) / tf.reshape(long_length, [-1, 1])

            # guide_keys = dense(atted_long_interest, self.embed_dim, name_scope + 'guide_keys', activation=tf.nn.tanh,
            #                    norm_rate=self.l2_norm, keep_prob=keep_prob, is_training=is_training)
            #
            atted_short_interest = guide_attention(interest_info['short_interest'],
                                                   tf.concat([atted_long_interest, base_embed], axis=1),
                                                   interest_info['short_interest'], interest_info['short_mask'],
                                                   self.embed_dim * 2, self.embed_dim, tf.nn.leaky_relu, multihead,
                                                   name_scope + "guide_atts", self.l2_norm)
            #
            new_interest = tf.concat([atted_long_interest, atted_short_interest], axis=-1)

        elif combination == 'no_att':
            long_length = tf.clip_by_value(tf.reduce_sum(tf.cast(interest_info['long_mask'], tf.float32), axis=1), 1,
                                           self.act_number)
            atted_long_interest = tf.reduce_sum(long_interest, axis=1) / tf.reshape(long_length, [-1, 1])

            short_length = tf.clip_by_value(tf.reduce_sum(tf.cast(interest_info['short_mask'], tf.float32), axis=1), 1,
                                            10000)
            atted_short_interest = tf.reduce_sum(interest_info['short_interest'], axis=1) / tf.reshape(short_length,
                                                                                                       [-1, 1])

            dense_short_interest = dense(atted_short_interest, self.embed_dim, name_scope + 'dense_short',
                                         tf.nn.leaky_relu, self.l2_norm, keep_prob, is_training)
            # new_interest = tf.concat([atted_long_interest, atted_short_interest], axis=-1)
            new_interest = tf.concat([atted_long_interest, dense_short_interest], axis=-1)

        else:
            new_interest = None

        return new_interest

    def long_interest_aggregator(self, name_scope, long_mask, long_interest, multihead=1, aggregator='mean'):

        length = tf.clip_by_value(tf.reduce_sum(tf.cast(long_mask, tf.float32), axis=1), 1, self.act_number)
        if multihead > 0:
            att_interest = multihead_attentions(long_interest, long_interest, long_interest, long_mask,
                                                self.embed_dim // 2,
                                                self.embed_dim, tf.nn.leaky_relu, multihead,
                                                scope=name_scope + "multi_heads_{}".format(multihead),
                                                norm_rate=self.l2_norm)
        else:
            att_interest = long_interest

        interest_emb = tf.reduce_sum(att_interest, axis=1) / tf.reshape(length, [-1, 1])
        # mean aggregation, can be modified by max, min pooling
        return interest_emb

    def short_interest_aggregator(self, name_scope, short_mask, short_interest, short_length, aggregator='mean'):

        length = tf.clip_by_value(tf.reduce_sum(tf.cast(short_mask, tf.float32), axis=1), 1, short_length)

        if aggregator == 'mean':
            interest_emb = tf.reduce_sum(short_interest, axis=1) / tf.reshape(length, [-1, 1])
        else:
            interest_emb = tf.reduce_sum(short_interest, axis=1) / tf.reshape(length, [-1, 1])
            # you can utilize other aggregation like max_pooling
        return interest_emb

    def filter_indices(self, seqs, sampled_nums):
        indices, base_feat, act_embed, act_ids, temporal_embed = seqs
        node_max_length = tf.gather(tf.segment_max(indices[:, 1], indices[:, 0]), indices[:, 0])
        mask = sampled_nums - (node_max_length - indices[:, 1])  # reverse id
        segment = tf.reshape(tf.where(mask > 0), (-1,))
        new_base_feat = tf.gather(base_feat, segment)
        new_act_embed = tf.gather(act_embed, segment)
        new_act_ids = tf.gather(act_ids, segment)
        new_temporal_embed = tf.gather(temporal_embed, segment)
        new_indices = tf.gather(indices, segment)

        node_min_length = tf.gather(tf.segment_max(new_indices[:, 1], new_indices[:, 0]), new_indices[:, 0])
        new_indices_final = tf.concat(
            [tf.reshape(new_indices[:, 0], [-1, 1]), tf.reshape(new_indices[:, 1] - node_min_length, [-1, 1])],
            axis=1)
        return new_indices_final, new_base_feat, new_act_embed, new_act_ids, new_temporal_embed
