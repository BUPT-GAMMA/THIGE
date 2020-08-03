from models.encoder import *

class Model:
    def __init__(self, params, interest_model):
        self.params = params
        self.partitioner = None
        self.saver = None
        self.partitioner = tf.min_max_variable_partitioner(max_partitions=params['ps_num']) if params[
            'is_distributed'] else None

        self.input_config = params["input_config"]
        self.learning_rate = params['learning_rate'].value
        self.learning_algo = params['learning_algo'].value

        self.keep_prob = params["keep_prob"].value
        self.train_size = params["train_size"].value
        self.valid_size = params["valid_size"].value
        self.test_size = params["test_size"].value

        if self.learning_algo == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.learning_algo == "asyncadam":
            self.optimizer = tf.train.AdamAsyncOptimizer(learning_rate=self.learning_rate)
        elif self.learning_algo == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.learning_algo == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            print("Error: No optimizer")
            exit()

        model_config = {
            'act_num': self.input_config["act_num"],
            'embed_dim': self.params["embed_dim"].value,
            'feat_dim': self.params["feat_dim"].value,
            'UL': params["UL"].value,
            'US': params["US"].value,
            'SL': params["SL"].value,
            'SS': params["SS"].value,
            'h': params['heads'].value,
            'use_temporal': params['use_temporal'].value,
            'l2_norm': params['l2_norm'].value,
            'seq_model': params['seq_model'].value,
            'combine': params['combine_att'].value
        }

        self.encoders = Encoders(self.input_config, self.partitioner)
        self.input_model = interest_model(model_config, self.partitioner)

    def get_model(self, feature_fn, label_fn, mode='train', global_step=None, valid_flag=1):
        opts = []
        if mode == 'train':
            train_input = feature_fn["train_feats"]
            train_label = label_fn["train_labels"]

            train_pred, train_loss = self.calculate_loss(train_input, train_label, mode="train",
                                                         batch_size=self.train_size)
            loss_opt = self.optimizer.minimize(train_loss, global_step, name="loss_{}".format("train"))
            auc, opt_auc, pr_auc, pr_opt_auc = self.metrics(train_pred, train_label)
            names_summary = ["roc_auc", "loss", "pr_auc"]
            values_train = [auc, train_loss, pr_auc]
            self.summary(names_summary, values_train, mode)
            if valid_flag == 2:
                valid_input = feature_fn["valid_feats"]
                valid_label = label_fn["valid_labels"]

                valid_pred, valid_loss = self.calculate_loss(valid_input, valid_label, mode="valid",
                                                             batch_size=self.valid_size)
                auc_v, opt_auc_v, pr_auc_v, pr_opt_auc_v = self.metrics(valid_pred, valid_label)
                values_train = [auc_v, valid_loss, pr_auc_v]
                self.summary(names_summary, values_train, 'valid')
            else:
                valid_loss = train_loss
                pr_opt_auc_v = pr_opt_auc
                opt_auc_v = opt_auc
            summary = tf.summary.merge_all()
            opts = [loss_opt, train_loss, train_pred, pr_opt_auc, opt_auc, valid_loss, pr_opt_auc_v, opt_auc_v, summary]

        elif mode == 'test':
            test_input = feature_fn["test_feats"]
            test_label = label_fn["test_labels"]
            test_pred, test_loss = self.calculate_loss(test_input, test_label, mode="test", batch_size=self.test_size)
            auc, opt_auc, pr_auc, pr_opt_auc = self.metrics(test_pred, test_label)
            opts = [test_pred, test_label, auc, opt_auc]

        saver = self.init_saver()
        return opts, saver

    def calculate_loss(self, train_input, train_label, mode="train", batch_size=128):
        if mode != "train":
            keep_prob = 1.0
            is_training = False
        else:
            keep_prob = self.keep_prob
            is_training = True

        encoded_info = self.encoders.decode_inputs(train_input)
        pred_v = self.input_model.get_probs("model_construction", batch_size, encoded_info, keep_prob, is_training)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label, logits=pred_v))

        return pred_v, loss

    def init_saver(self):
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        bn_moving_vars += [g for g in g_list if 'global_step' in g.name]
        saver = tf.train.Saver(var_list=var_list + bn_moving_vars, max_to_keep=1)
        return saver

    def metrics(self, pred, label):
        auc, opt_auc = tf.metrics.auc(label, tf.nn.sigmoid(pred), curve='ROC')
        pr_auc, pr_opt_auc = tf.metrics.auc(label, tf.nn.sigmoid(pred), curve='PR')
        return auc, opt_auc, pr_auc, pr_opt_auc

    def summary(self, names, values, suffix):
        for i, name in enumerate(names):
            tf.summary.scalar('{}_{}'.format(suffix, name), values[i])
