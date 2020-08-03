import tensorflow as tf


def dense(inputs, out_dim, name, activation=tf.nn.leaky_relu, norm_rate=0.001, keep_prob=1.0, is_training=True):
    regulars = tf.contrib.layers.l2_regularizer(norm_rate) if norm_rate > 0 else None
    layers = tf.layers.dense(inputs, out_dim, name=name, reuse=tf.AUTO_REUSE, activation=activation,
                             kernel_regularizer=regulars)
    if is_training:
        layers = tf.layers.dropout(layers, rate=1 - keep_prob)
    return layers


def dense2(inputs, out_dim, name, activation=tf.nn.leaky_relu, norm_rate=0.001, keep_prob=1.0, is_training=True):
    layers = tf.layers.dense(inputs, out_dim, name=name, reuse=tf.AUTO_REUSE, activation=activation, use_bias=True)
    layers = tf.layers.batch_normalization(layers, training=is_training, reuse=tf.AUTO_REUSE, name=name)
    return layers

def dense3(inputs, out_dim, name, activation=tf.nn.leaky_relu, norm_rate=0.001, keep_prob=1.0, is_training=True):
    regulars = tf.contrib.layers.l2_regularizer(norm_rate) if norm_rate > 0 else None
    layers = tf.layers.dense(inputs, out_dim, name=name, reuse=tf.AUTO_REUSE, activation=activation,
                             kernel_regularizer=regulars)
    return layers



def dynamic_gru(matrix, lengths, name, num_units=128):
    cell = tf.nn.rnn_cell.GRUCell(num_units, name=name, reuse=tf.AUTO_REUSE)
    outputs, last_state = tf.nn.dynamic_rnn(cell, matrix, sequence_length=lengths, dtype=tf.float32,
                                            scope="dynamic_{}".format(name))
    return outputs, last_state


def dynamic_LSTM(matrix, lengths, name, num_units=128):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, name=name, reuse=tf.AUTO_REUSE)
    outputs, last_state = tf.nn.dynamic_rnn(cell, matrix, sequence_length=lengths, dtype=tf.float32,
                                            scope="dynamic_{}".format(name))
    return outputs, last_state

#
# def batch_normalization_dense(inputs, out_dim, name, activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, norm_rate=0.001,
#                               keep_prob=1.0, is_training=True):
#     regularizer = tf.contrib.layers.l2_regularizer(norm_rate) if norm_rate > 0 else None
#     outputs = tf.layers.dense(inputs, out_dim, name=name, reuse=reuse, activation=None, kernel_regularizer=regularizer)
#
#     batch_normalization = tf.layers.batch_normalization(outputs, training=is_training)
#     layers = activation(batch_normalization)
#     if is_training:
#         layers = tf.layers.dropout(layers, rate=1 - keep_prob)
#     return layers
#
