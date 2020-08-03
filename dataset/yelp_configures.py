# number of users, number of items
# user features, item_features[cate? temporal? type? numberical?]
# length of long memory, length of short memory


def generate_feat_setting(params):
    embed_dims = params['embed_dim'].value
    feat_dims = params['feat_dim'].value
    # print(embed_dims, feat_dims)

    # both act_dim and temporal_dim are the same as temporal_dim
    act_dims = embed_dims
    temporal_dims = embed_dims

    info_col_vals = [['user_info', ''], ['user_seqs', ''], ['item_info', ''], ['item_seqs', ''], ['label', -1.0]]
    num_users = 1636564
    num_items = 192125
    num_cities = 1202
    num_states = 36
    num_cates = 2468
    num_acts = 2
    user_feat_opts = [
        ['uid', -1, 'no_id', embed_dims, num_users], ['stamps', 0.0, 'val', 1, None],
        ['num_friends', 0.0, 'val', 1, None], ['useful', 0.0, 'val', 1, None],
        ['funny', 0.0, 'val', 1, None], ['cool', 0.0, 'val', 1, None], ['fans', 0.0, 'val', 1, None],
        ['average_stars', 0.0, 'val', 1, None], ['compliment_hot', 0.0, 'val', 1, None],
        ['compliment_more', 0.0, 'val', 1, None], ['compliment_profile', 0.0, 'val', 1, None],
        ['compliment_cute', 0.0, 'val', 1, None], ['compliment_list', 0.0, 'val', 1, None],
        ['compliment_note', 0.0, 'val', 1, None], ['compliment_plain', 0.0, 'val', 1, None],
        ['compliment_cool', 0.0, 'val', 1, None], ['compliment_funny', 0.0, 'val', 1, None],
        ['compliment_writer', 0.0, 'val', 1, None], ['compliment_photos', 0.0, 'val', 1, None]
    ]
    # name, defval, feat_type, dim, id_matrix_dim or None
    item_feat_opts = [
        ['sid', -1, 'no_id', embed_dims, num_items], ['item_city_id', -1, 'id', feat_dims, num_cities],
        ['item_state_id', -1, 'id', feat_dims, num_states], ['item_cate_ids', '', 'composed_|', feat_dims, num_cates],
        ['item_stars', 0.0, 'val', 1, None], ['item_reviews', 0.0, 'val', 1, None],
        ['item_is_open', 0.0, 'val', 1, None]
    ]

    user_seq_opts = item_feat_opts + [['edge_type', 1, 'segment_feats', act_dims, num_acts],
                                      ['timespan', -1, 'temporal_buckets', temporal_dims, None]]

    item_seq_opts = user_feat_opts + [['edge_type', 1, 'segment_feats', act_dims, num_acts],
                                      ['timespan', -1, 'temporal_buckets', temporal_dims, None]]
    # ['timespan', '', 'temporal_statics', temporal_dims, None]]
    # ['timespan', '', 'temporal_buckets', temporal_dims, None]]

    data_figure = {
        'info_col_vals': info_col_vals,
        'opts': [user_feat_opts, user_seq_opts, item_feat_opts, item_seq_opts],
        'act_num': num_acts,
        'max_length': 200 * num_acts,
        'temporal_str': '-1,7,-1,-1',
        'train_labels': 662472,
        'test_labels': 329064
    }
    return data_figure
