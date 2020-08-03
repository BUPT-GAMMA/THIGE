import tensorflow as tf
from tensorflow.contrib import layers


def multihead_attentions(queries, keys, values, masks, num_units, num_output_units, activation_fn, num_heads=8,
                         scope="multi_heads", norm_rate=0.01):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = layers.fully_connected(queries,
                                   num_units,
                                   activation_fn=None,
                                   scope="Q",
                                   weights_regularizer=layers.l2_regularizer(norm_rate)
                                   )
        K = layers.fully_connected(keys,
                                   num_units,
                                   activation_fn=None,
                                   scope="K",
                                   weights_regularizer=layers.l2_regularizer(norm_rate)
                                   )
        V = layers.fully_connected(values,
                                   num_output_units,
                                   activation_fn=activation_fn,
                                   scope="V",
                                   weights_regularizer=layers.l2_regularizer(norm_rate)
                                   )

        dimension_2 = queries.get_shape()[1]

        Q_mh = tf.transpose(tf.reshape(Q, [-1, dimension_2, num_heads, num_units // num_heads]), [0, 2, 1, 3])
        K_mh_t = tf.transpose(tf.reshape(K, [-1, dimension_2, num_heads, num_units // num_heads]), [0, 2, 3, 1])
        V_mh = tf.transpose(tf.reshape(V, [-1, dimension_2, num_heads, num_output_units // num_heads]), [0, 2, 1, 3])

        sqrt_n = (num_units // num_heads) ** 0.5

        QK_t = tf.matmul(Q_mh, K_mh_t) / sqrt_n

        key_masks = tf.tile(tf.reshape(masks, [-1, 1, 1, dimension_2]), [1, num_heads, dimension_2, 1])
        paddings = tf.fill(tf.shape(QK_t), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        atts = tf.where(key_masks, QK_t, paddings)
        norm_atts = tf.nn.softmax(atts, axis=-1)
        outputs_dense = tf.matmul(norm_atts, V_mh)
        query_mask = tf.tile(tf.reshape(masks, [-1, 1, dimension_2, 1]),
                             [1, num_heads, 1, num_output_units // num_heads])
        paddings2 = tf.fill(tf.shape(outputs_dense), tf.constant(0, dtype=tf.float32))
        output_sparse = tf.where(query_mask, outputs_dense, paddings2)  # [batch_size, head, time_size, out_dim]
        outputs = tf.reshape(tf.transpose(output_sparse, [0, 2, 1, 3]), [-1, dimension_2, num_output_units])
        return outputs


def guide_attention(queries, keys, values, masks, num_units, num_output_units, activation_fn, num_heads=8,
                    scope="multi_heads", norm_rate=0.01):
    # queries: [batch_size, time_size, embed_dim]
    # keys: [batch_size, embed_dim]
    # mask: [batch_size, time_size
    # Q, K, V
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = layers.fully_connected(queries,
                                   num_units,
                                   activation_fn=None,
                                   scope="Q",
                                   weights_regularizer=layers.l2_regularizer(norm_rate)
                                   )

        V = layers.fully_connected(values,
                                   num_output_units,
                                   activation_fn=activation_fn,
                                   scope="V",
                                   weights_regularizer=layers.l2_regularizer(norm_rate)
                                   )
        att = tf.reduce_sum(tf.multiply(Q, tf.expand_dims(keys, 1)), axis=-1)
        paddings = tf.fill(tf.shape(att), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        paddings2 = tf.fill(tf.shape(att), tf.constant(0, dtype=tf.float32))

        atts = tf.where(masks, att, paddings)
        norm_atts = tf.expand_dims(tf.where(masks, tf.nn.softmax(atts, axis=-1), paddings2), 1)
        outputs_dense = tf.reduce_sum(tf.matmul(norm_atts, V), axis=1)

        return outputs_dense


def guide_attention_vn(queries, keys, values, masks, base_embed, num_units, num_output_units, activation_fn,
                       num_heads=8, scope="multi_heads", norm_rate=0.01):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        att = tf.reduce_sum(tf.matmul(queries, tf.expand_dims(keys, 2)), axis=-1)
        paddings = tf.fill(tf.shape(att), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        paddings2 = tf.fill(tf.shape(att), tf.constant(0, dtype=tf.float32))

        atts = tf.where(masks, att, paddings)
        norm_atts = tf.expand_dims(tf.where(masks, tf.nn.softmax(atts, axis=-1), paddings2), 1)
        outputs_dense = tf.reduce_sum(tf.matmul(norm_atts, values), axis=1)

        return outputs_dense


def guide_attention_v2(queries, keys, values, masks, base_embed, num_units, num_output_units, activation_fn,
                       num_heads=8,
                       scope="multi_heads", norm_rate=0.01):
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        X1 = layers.fully_connected(queries,
                                    1,
                                    activation_fn=None,
                                    scope="X1",
                                    weights_regularizer=layers.l2_regularizer(norm_rate)
                                    )
        X2 = layers.fully_connected(tf.concat([keys, base_embed], axis=1),
                                    1,
                                    activation_fn=None,
                                    scope="X2",
                                    weights_regularizer=layers.l2_regularizer(norm_rate)
                                    )

        att = X1 + tf.expand_dims(X2, 1)
        paddings = tf.fill(tf.shape(att), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        paddings2 = tf.fill(tf.shape(att), tf.constant(0, dtype=tf.float32))

        atts = tf.where(masks, att, paddings)
        norm_atts = tf.expand_dims(tf.where(masks, tf.nn.softmax(atts, axis=-1), paddings2), 1)
        outputs_dense = tf.reduce_sum(tf.matmul(norm_atts, values), axis=1)

        return outputs_dense


def guide_attention_v3(queries, keys, values, masks, base_embed, num_units, num_output_units, activation_fn,
                       num_heads=8,
                       scope="multi_heads", norm_rate=0.01):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        X1 = layers.fully_connected(queries,
                                    num_units,
                                    activation_fn=None,
                                    scope="X1",
                                    weights_regularizer=layers.l2_regularizer(norm_rate)
                                    )
        X2 = layers.fully_connected(base_embed,
                                    num_units,
                                    activation_fn=None,
                                    scope="X2",
                                    weights_regularizer=layers.l2_regularizer(norm_rate)
                                    )
        att = tf.reduce_sum(tf.multiply(X1, tf.expand_dims(X2, 1)), axis=-1)
        paddings = tf.fill(tf.shape(att), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        paddings2 = tf.fill(tf.shape(att), tf.constant(0, dtype=tf.float32))

        atts = tf.where(masks, att, paddings)
        norm_atts = tf.expand_dims(tf.where(masks, tf.nn.softmax(atts, axis=-1), paddings2), 1)
        outputs_dense = tf.reduce_sum(tf.matmul(norm_atts, values), axis=1)

        return outputs_dense



def co_attention(Q, V, Q_mask, V_mask, norm_rate, k, name, dim_Q):

    QW = layers.fully_connected(Q, dim_Q, activation_fn=None, scope="QW_{}".format(name),
                                weights_regularizer=layers.l2_regularizer(norm_rate))  # [batch_size, action_types, dim]
    C = tf.nn.tanh(tf.matmul(QW, tf.transpose(V, [0, 2, 1])))  # [batch_size, action_types, time_size]

    Q2 = layers.fully_connected(Q, k, activation_fn=None, scope="Q2_{}".format(name),
                                weights_regularizer=layers.l2_regularizer(norm_rate))
    V2 = layers.fully_connected(V, k, activation_fn=None, scope="V2_{}".format(name),
                                weights_regularizer=layers.l2_regularizer(norm_rate))
    HQ = tf.nn.tanh(Q2 + tf.matmul(C, V2))
    HV = tf.nn.tanh(V2 + tf.matmul(tf.transpose(C, [0, 2, 1]), Q2))

    att1 = tf.reshape(
        layers.fully_connected(HQ, 1, activation_fn=None, scope="att_q_{}".format(name),
                               weights_regularizer=layers.l2_regularizer(norm_rate)),
        Q_mask.get_shape())
    att2 = tf.reshape(
        layers.fully_connected(HV, 1, activation_fn=None, scope="att_v_{}".format(name),
                               weights_regularizer=layers.l2_regularizer(norm_rate)),
        V_mask.get_shape())

    pad_v = tf.constant(-2 ** 32 + 1, dtype=tf.float32)
    paddings1 = tf.fill(tf.shape(Q_mask), pad_v)
    paddings10 = tf.zeros_like(Q_mask, dtype=tf.float32)
    norm_att1 = tf.where(Q_mask, tf.nn.softmax(tf.where(Q_mask, att1, paddings1), axis=-1), paddings10)
    paddings2 = tf.fill(tf.shape(V_mask), pad_v)
    paddings20 = tf.zeros_like(V_mask, dtype=tf.float32)

    norm_att2 = tf.where(V_mask, tf.nn.softmax(tf.where(V_mask, att2, paddings2), axis=-1), paddings20)
    new_Q = tf.matmul(tf.transpose(norm_att1, [0, 2, 1]), Q)
    new_V = tf.matmul(tf.transpose(tf.expand_dims(norm_att2, 2), [0, 2, 1]), V)

    return new_Q, new_V


def feedforward(inputs,
                input_dim,
                num_units,
                activation_fn=None,
                scope="feedforward",
                reuse=None,
                norm_rate=0.001,
                variables_collections=None,
                outputs_collections=None):
    with tf.variable_scope(scope, reuse=reuse):
        regular = layers.l2_regularizer(norm_rate)
        if len(num_units) > 0:
            outputs = layers.fully_connected(inputs,
                                             num_units[0],
                                             activation_fn=activation_fn,
                                             variables_collections=variables_collections,
                                             outputs_collections=outputs_collections,
                                             weights_regularizer=regular)
        else:
            outputs = inputs

        for i in range(1, len(num_units)):
            outputs = layers.fully_connected(outputs,
                                             num_units[i],
                                             activation_fn=None,
                                             variables_collections=variables_collections,
                                             outputs_collections=outputs_collections,
                                             weights_regularizer=regular)

        outputs = layers.fully_connected(outputs,
                                         input_dim,
                                         activation_fn=None,
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections,
                                         weights_regularizer=regular)
    outputs += inputs
    return outputs
