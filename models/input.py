import tensorflow as tf


def decode_input_config(config):
    col_names = []
    col_vals = []
    for i in range(len(config)):
        col_name = config[i][0]
        col_val = config[i][1]
        col_names.append(col_name)
        col_vals.append([col_val])
    return col_names, col_vals


def input_fn(table, configs, batch_size, epochs, shuffle=True, slice_id=None, slice_count=None):
    cols = decode_input_config(configs['info_col_vals'])
    col_names = cols[0]
    col_vals = cols[1]

    if slice_id is not None and slice_count is not None:
        dataset = tf.data.TableRecordDataset(table,
                                             record_defaults=tuple(map(lambda x: x[0], col_vals)),
                                             selected_cols=",".join(col_names),
                                             slice_count=slice_count,
                                             slice_id=slice_id
                                             )
    else:
        dataset = tf.data.TextLineDataset(table).map(
            lambda record: tf.decode_csv(record, record_defaults=col_vals, field_delim=';'))

    capacity = 64 * batch_size
    dataset = dataset.repeat(epochs)
    if shuffle:
        dataset = dataset.shuffle(capacity)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    train_iterator = dataset.make_one_shot_iterator()
    infos = train_iterator.get_next()
    
    label = tf.reshape(infos[4], [-1, 1])

    user_feat = infos[0]
    user_seqs = infos[1]
    item_feat = infos[2]
    item_seqs = infos[3]
    feat_fn = {
        'user_feat': user_feat,
        'user_seqs': user_seqs,
        'item_feat': item_feat,
        'item_seqs': item_seqs
    }
    return feat_fn, label
