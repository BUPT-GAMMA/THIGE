import numpy as np
import tensorflow as tf


class Encoders:
    def __init__(self, configs, partitioner=None):
        self.partitioner = partitioner
        self.opts = configs['opts']
        self.temporal_str = configs['temporal_str']
        self.embeddings = {}
        self.opts_base = []
        self.tempoal_feature_columns = {}
        for i in range(len(self.opts)):
            info = self.decode_opts(self.opts[i])
            self.opts_base.append(info)

    def decode_opts(self, opt, scope='encoder_variables'):
        def_vals = []
        def_names = []
        def_infos = []
        def_rows = []

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            for i in range(len(opt)):
                name, val, info, num_dims, num_rows = opt[i]
                if info in ['id', 'composed_|', 'hash_id', 'segment_feats']:
                    lims = np.sqrt(3 / (num_dims * num_rows))
                    initializer = tf.random_uniform_initializer(-lims, lims)
                    self.embeddings.setdefault(name,
                                               tf.get_variable(name, [num_rows, num_dims], initializer=initializer))
                elif info in ['temporal_statics']:
                    self.embeddings.setdefault(name, tf.constant(get_continuous_embedding(num_dims), name=name,
                                                                 shape=[1, num_dims]))
                elif info == 'temporal_buckets':
                    boundaries, num_bucket = get_feature_column(self.temporal_str)
                    self.tempoal_feature_columns[name] = [tf.feature_column.bucketized_column(
                        tf.feature_column.numeric_column(name, default_value=[0.0], dtype=tf.float32), boundaries)]
                    lims = np.sqrt(3 / (num_dims * num_bucket))
                    initializer = tf.random_uniform_initializer(-lims, lims)
                    self.embeddings.setdefault(name,
                                               tf.get_variable(name, [num_bucket, num_dims], initializer=initializer))
                    num_rows = num_bucket

                def_vals.append(val)
                def_names.append(name)
                def_infos.append(info)
                def_rows.append(num_rows)
        return def_vals, def_names, def_infos, def_rows

    def decode_inputs(self, input_info, delim1=',', delim2=':', delim3="|"):
        user_feats, _, _, _ = self.decode_features(input_info['user_feat'], self.opts_base[0])
        user_seq_infos = self.decode_sequences(input_info['user_seqs'], self.opts_base[1])
        item_feats, _, _, _ = self.decode_features(input_info['item_feat'], self.opts_base[2])
        item_seq_infos = self.decode_sequences(input_info['item_seqs'], self.opts_base[3])
        return user_feats, user_seq_infos, item_feats, item_seq_infos

    def decode_features(self, feats, opts, delim1=',', delim2=':', delim3="|", is_seqs=0):
        feat_csv = tf.decode_csv(feats, opts[0], delim2)
        my_feats = []
        for i in range(len(opts) - is_seqs * 2):
            info = opts[2][i]
            name = opts[1][i]
            rows = opts[3][i]
            if info == 'id':
                my_feats.append(self.get_id_embedding(name, feat_csv[i]))
            elif info == 'hash_id':
                my_feats.append(self.get_hash_id_embedding(name, feat_csv[i], rows))
            elif info == 'composed_|':
                my_feats.append(self.get_compose_embedding(name, feat_csv[i]))
            elif info == 'no_id':
                continue
            else:
                my_feats.append(tf.reshape(feat_csv[i], [-1, 1]))
        my_feat_concat = tf.concat(my_feats, axis=-1)

        act_embed = None
        act_ids = None
        temporal_embed = None

        if is_seqs > 0:
            temporal_embed = self.get_temporal_embedding(opts[1][-1], feat_csv[-1], opts[2][-1])
            act_embed = self.get_act_embeedding(opts[1][-2], feat_csv[-2])
            act_ids = feat_csv[-2]
        return my_feat_concat, act_embed, act_ids, temporal_embed

    def decode_sequences(self, feats, opts, delim1=',', delim2=':', delim3="|"):
        sparse_feats = tf.string_split(feats, delim1)
        obj_feats = sparse_feats.values
        indices = sparse_feats.indices
        base_feat, act_embed, act_ids, temporal_embed = self.decode_features(obj_feats, opts, is_seqs=1)
        return indices, base_feat, act_embed, act_ids, temporal_embed

    def get_id_embedding(self, name, feat):
        return tf.nn.embedding_lookup(self.embeddings[name], feat)

    def get_hash_id_embedding(self, name, feat, num_row):
        hash_id = tf.string_to_hash_bucket(feat, num_row)
        return tf.nn.embedding_lookup(self.embeddings[name], hash_id)

    def get_compose_embedding(self, name, feat, delim="|"):
        splited_info = tf.string_split(feat, delim)
        segments = splited_info.indices
        idx = tf.string_to_number(splited_info.values, tf.int32)
        embeds = tf.segment_mean(tf.nn.embedding_lookup(self.embeddings[name], idx), segments[:, 0])
        return embeds

    def get_temporal_embedding(self, name, feat, info):
        # color_data = {'color': ["R", "B", 'G', 'A', 'A']}  # 4行样本
        # builder = _LazyBuilder(color_data)
        # color_column = feature_column.categorical_column_with_vocabulary_list(
        #     'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
        # )
        # color_column_tensor = color_column._get_sparse_tensors(builder)
        #
        # # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
        # color_column_identy = feature_column.indicator_column(color_column)
        # color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])

        if info == 'temporal_buckets':
            time_feat_id = tf.cast(
                tf.feature_column.input_layer({name: feat}, self.tempoal_feature_columns[name], trainable=False)[0],
                tf.int32)
            # print(self.embeddings[name], time_feat_id)
            # exit()
            return tf.nn.embedding_lookup(self.embeddings[name], time_feat_id)
        elif info == 'temporal_statics':
            return tf.cos(tf.matmul(tf.reshape(feat, [-1, 1]), self.embeddings[name]))
        elif info == 'temporal_val':
            return feat

    def get_act_embeedding(self, name, feat):
        act_embedding = tf.nn.embedding_lookup(self.embeddings[name], feat)
        return act_embedding


def get_continuous_embedding(dimension):
    v = (1 / 10 ** np.linspace(0, 10, dimension))
    return v


def get_feature_column(config_str):
    hours, days, months, years = [int(x) for x in config_str.split(",")]
    boundaries = []
    all_ranges = []
    if hours > 0:
        all_ranges.append([hours, 3600, 0])
    if days > 0:
        all_ranges.append([days, 3600 * 24, 0])

    if months > 0:
        all_ranges.append([months, 3600 * 24 * 30, 0])

    if years > 0:
        all_ranges.append([years, 3600 * 24 * 365, 0])

    for j in range(len(all_ranges)):
        lengths, d, last = all_ranges[j]
        if j == 0:
            rg = list(range(0, lengths * d, d))
            all_ranges[j][-1] = rg[-1]
        else:
            rg = list(range(all_ranges[j - 1][-1] + d, all_ranges[j - 1][-1] + lengths * d, d))

            all_ranges[j][-1] = rg[-1]
        boundaries += rg
    num_bucket = len(boundaries) + 1
    return boundaries, num_bucket
