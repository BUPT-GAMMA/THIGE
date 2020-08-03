import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from dataset.yelp_configures import generate_feat_setting as yelp_gfs
from models.basic_model import BasicModel
from models.input import input_fn
from models.model import Model as MD
from models.thige import THIGE
from utils.fundmental import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

flags = tf.flags
FLAGS = flags.FLAGS

# >>>>>>>>>>>>>>>   alibaba distributed system configures.
flags.DEFINE_string("tables", "", "tables info")
flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_string("outputs", "", "output table")
flags.DEFINE_string("mode", 'train', "train/test")
flags.DEFINE_integer('ver', 1, 'ver')
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", None, "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer('GPU_cnt', 4, "how many gpu to use.")
flags.DEFINE_integer('CPU_cnt', 2, "how many cpu to use.")

# >>>>>>>>>>>>>>>>    device setting
flags.DEFINE_string('use_gpu', '0', 'use gpu')

# >>>>>>>>>>>>>>>>    data setting
flags.DEFINE_string("dataset", "alibaba", "dataset: alibaba/yelp/taobao")
flags.DEFINE_integer('shuffle', 1, 'shuffle')
# >>>>>>>>>>>>>>>>    run setting

flags.DEFINE_string('model_name', 'THIGE', 'model name')

flags.DEFINE_integer('remove_path', 0, 'remove_path')
flags.DEFINE_integer('seed', 0, 'seed')
flags.DEFINE_integer('epochs', 10, "number of epochs")
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_string('learning_algo', 'adam', 'learning_algo')
flags.DEFINE_float('l2_norm', 0.001, 'l2 normalization')
flags.DEFINE_float('keep_prob', 0.5, '1-dropout')

flags.DEFINE_integer('UL', 400, "user long-term")
flags.DEFINE_integer('SL', 400, "item long-term")
flags.DEFINE_integer('US', 10, "user short-term")
flags.DEFINE_integer('SS', 10, "item short-term")
flags.DEFINE_integer('use_temporal', 0, 'use_temporal')
flags.DEFINE_string('seq_model', 'GRU', 'sequential model')

flags.DEFINE_integer('train_size', 128, "train_size")
flags.DEFINE_integer('test_size', 128, "test_size")
flags.DEFINE_integer('valid_size', 128, "valid_size")

flags.DEFINE_integer('heads', 1, 'heads')
flags.DEFINE_integer('embed_dim', 128, 'embed_dim')
flags.DEFINE_integer('feat_dim', 16, 'feat_dim')
flags.DEFINE_string('combine_att', 'guide_att', 'combine att mechaine')
# >>>>>>>>>>>>>>>>    GPU initialization
if FLAGS.use_gpu == '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.use_gpu

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

conf_proto = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)


def main(_):
    params = OrderedDict(sorted(FLAGS.__flags.items()))
    if FLAGS.ps_hosts is not None:
        ps_num = len(FLAGS.ps_hosts.split(","))
        if ps_num > 0:
            params['is_distributed'] = True
            params['ps_num'] = ps_num
    else:
        params['is_distributed'] = False

    if FLAGS.dataset == 'yelp':
        params["input_config"] = yelp_gfs(params)
    else:
        print('error')
        exit()

    if FLAGS.model_name == 'basic':
        interest_model = BasicModel
    elif FLAGS.model_name == 'thige':
        interest_model = THIGE
    elif FLAGS.model_name == 'short':
        interest_model = SHORT
    elif FLAGS.model_name == 'long':
        interest_model = LONG
    elif FLAGS.model_name == 'dien':
        interest_model = DIEN
    else:
        print("please select the interest_model: basic_model, thige, and so on")
        interest_model = None
        exit()

    path_flags = [FLAGS.dataset, FLAGS.model_name, FLAGS.combine_att, FLAGS.UL, FLAGS.SL, FLAGS.US, FLAGS.SS,
                  FLAGS.heads, FLAGS.epochs, FLAGS.seed]

    path = create_buckets(FLAGS.buckets, path_flags, remove_path=FLAGS.remove_path)
    if FLAGS.mode == 'train':
        train_model = MD(params, interest_model=interest_model)
        train(train_model, path, params)
    else:
        test_model = MD(params, interest_model=interest_model)
        test(test_model, path, params)


def train(train_model, path, params):
    tables = FLAGS.tables.split(",")
    train_batch = FLAGS.train_size
    valid_batch = FLAGS.valid_size
    input_config = params["input_config"]
    input_train, label_train = input_fn(tables[0], input_config, train_batch, FLAGS.epochs, shuffle=FLAGS.shuffle)
    input_infos = {'train_feats': input_train}
    label_infos = {'train_labels': label_train}

    if len(tables) == 2:
        input_valid, label_valid = input_fn(tables[1], input_config, valid_batch, FLAGS.epochs, shuffle=False)
        input_infos['valid_feats'] = input_valid
        label_infos['valid_labels'] = label_valid

    global_step = tf.train.get_or_create_global_step()
    opts, saver = train_model.get_model(input_infos, label_infos, 'train', global_step, len(tables))

    hooks = [tf.train.StopAtStepHook(last_step=8000000000)]
    scaffold = tf.train.Scaffold(saver=saver, init_op=tf.global_variables_initializer())

    with tf.train.MonitoredTrainingSession(scaffold=scaffold, hooks=hooks, checkpoint_dir=path,
                                           config=tf.ConfigProto(allow_soft_placement=True,
                                                                 log_device_placement=False)) as mon_sess:
        train_writer = tf.summary.FileWriter(path + "logs/", mon_sess.graph)
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            step = 0
            while not mon_sess.should_stop():
                step += 1
                _, loss, pred, pr_auc, roc_auc, valid_loss, pr_auc_v, roc_auc_v, train_summary = mon_sess.run(opts)
                if step % 100 == 0:
                    train_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    print("step: %d, loss_train: %.4f, auc: %.4f, loss_test: %.4f, auc: %.4f %s" % (
                        step, loss, roc_auc, valid_loss, roc_auc_v, train_time))
            train_writer.add_summary(train_summary, step)
        finally:
            end_time = time.strftime('%Y-%m-%d %H:%M:%S')
            print(start_time, end_time, FLAGS.epochs, train_batch)
            train_writer.close()


def test(test_model, path, params):
    tables = FLAGS.tables.split(",")
    test_batch = FLAGS.test_size
    input_config = params["input_config"]
    input_test, label_test = input_fn(tables[0], input_config, test_batch, epochs=1, shuffle=False)
    input_infos = {'test_feats': input_test}
    label_infos = {'test_labels': label_test}

    global_step = tf.train.get_or_create_global_step()
    opts, saver = test_model.get_model(input_infos, label_infos, 'test', global_step, len(tables))

    hooks = [tf.train.StopAtStepHook(last_step=8000000000)]

    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=path,
                                           config=tf.ConfigProto(allow_soft_placement=True,
                                                                 log_device_placement=False)) as mon_sess:
        writer = open(path + FLAGS.outputs, "w")
        try:
            step = 0
            while not mon_sess.should_stop():
                pred, lb, auc_t, _ = mon_sess.run(opts)
                str_pred = ",".join(np.array(pred, np.str).reshape(-1))
                str_lb = ",".join(np.array(lb, np.str).reshape(-1))
                line = "{}\t{}\t{}\t{}\n".format(step, str_pred, str_lb, auc_t)
                writer.write(line)
        finally:
            writer.close()


if __name__ == "__main__":
    tf.app.run(main)
