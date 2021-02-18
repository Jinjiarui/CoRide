import tensorflow as tf
import numpy as np


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name, n_group=1, sess=None):
        self.name_set = set()
        self.n_group = n_group
        self.gra = None

        if sess is None:
            self.gra = tf.Graph()
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess_config.gpu_options.allow_growth = True

            with self.gra.as_default():
                self.sess = tf.Session(graph=self.gra, config=sess_config)
                self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
                self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
            self.train_writer = tf.summary.FileWriter(log_dir + '/' + log_name, graph=tf.get_default_graph())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """
        if self.gra is not None:
            with self.gra.as_default():
                self._register(name_list)
        else:
            self._register(name_list)

    def _register(self, name_list):
        for name in name_list:
            if name in self.name_set:
                raise Exception("You cannot define different operations with same name: `{}`".format(name))
            self.name_set.add(name)
            setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                 for i in range(self.n_group)])
            setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                         for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)

        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i], feed_dict={
                        getattr(self, key)[i]: value[i]}), global_step=step)
            else:
                self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0], feed_dict={
                        getattr(self, key)[0]: value}), global_step=step)


def get_target_updates(eval_vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(eval_vars) == len(target_vars)
    for eval_var, target_var in zip(eval_vars, target_vars):
        init_updates.append(tf.assign(target_var, eval_var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * eval_var))
    return init_updates, soft_updates
    # assert len(init_updates) == len(eval_vars)
    # assert len(soft_updates) == len(eval_vars)


def kl_loss(p, q):
    p = tf.distributions.Categorical(probs=p)
    q = tf.distributions.Categorical(probs=q)
    return tf.distributions.kl_divergence(p, q)
