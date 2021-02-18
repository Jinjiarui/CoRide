import os
import os.path as osp
import tensorflow as tf
import numpy as np
import shutil
import collections

from functools import reduce
from collections import namedtuple
import tensorflow.contrib.rnn as rnn

Red = namedtuple('Red', ['grid_id', 'node_id'])


class Net(object):
    __slots__ = ('_x', '_out_dim', '_active_func', '_scope_name')

    def __init__(self, x, out_dim, active_func, name):
        self._x = x
        self._out_dim = out_dim
        self._active_func = active_func

        with tf.variable_scope(name):
            self._scope_name = tf.get_variable_scope().name
            self._construct()

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope_name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope_name)

    def _construct(self):
        raise NotImplementedError


class BaseModel(object):
    def __init__(self, sess, observation_space, action_space, name, batch_size):
        """ Init BaseModel with given session, and space definition

        Parameters
        ----------
        sess : tf.Session
        observation_space
        feature_space
        action_space
        name
        batch_size
        """
        self.obs_space = observation_space
        self.act_space = action_space
        self.n_act = reduce(lambda x, y: x * y, tuple([self.act_space]))
        self.name = name
        self.sess = sess
        self.batch_size = batch_size

        # others
        self.global_scope = None

    @property
    def gpu_config(self):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        return gpu_config

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.global_scope)

    def train(self, **kwargs):
        raise NotImplementedError

    def save(self, step, model_dir):
        """Saving model

        Parameters

        """
        model_dir = osp.join(model_dir, self.name)
        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        assert self.sess is not None

        print('[INFO] Saving model ...')
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.global_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, osp.join(model_dir, self.name), global_step=step)
        print('[INFO] Model saved at: `{}`'.format(save_path))

    def load(self, step, model_dir):
        assert self.sess is not None

        save_path = osp.join(model_dir, self.name, self.name + '-' + str(step))
        print('[INFO] Restoring model ...')
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.global_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, save_path)
        print('[INFO] Model resotred from: `{}`'.format(save_path))


class CNNEmbedding(object):
    @staticmethod
    def new_instance(obs_input, feat_input, fake_name="", reuse=False):
        with tf.variable_scope(fake_name + "embedding", reuse=reuse):
            h_obs = tf.layers.conv2d(obs_input, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv1')
            h_obs = tf.layers.conv2d(h_obs, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2')
            h_obs = tf.reshape(h_obs, [-1, np.prod([v.value for v in h_obs.shape[1:]])])
            h_obs = tf.layers.dense(h_obs, units=256, activation=tf.nn.relu, name='dense-view')
            if feat_input is not None:
                h_feat = tf.layers.dense(feat_input, units=64, activation=tf.nn.relu, name='feat-emb')
                return tf.concat([h_obs, h_feat], axis=1)
            else:
                return h_obs


class LinearEmbedding(object):
    @staticmethod
    def new_instance(obs_input, feat_input, fake_name="", reuse=False):
        with tf.variable_scope(fake_name + "embedding", reuse=reuse):
            h_obs = tf.layers.dense(obs_input, units=256, activation=tf.nn.relu, name='dense-view')
            h_obs = tf.layers.dense(h_obs, units=128, activation=tf.nn.relu, name='dense-view-2')
            if feat_input is not None:
                h_feat = tf.layers.dense(feat_input, units=64, activation=tf.nn.relu, name='feat-emb')
                return tf.concat([h_obs, h_feat], axis=1)
            else:
                return h_obs


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name, n_group=1, sess=None):
        self.name_set = set()
        self.n_group = n_group
        self.gra = None

        if sess is not None:
            self.sess = sess
            if os.path.exists(os.path.join(log_dir, log_name)):
                shutil.rmtree(os.path.join(log_dir, log_name))

            self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
        else:
            self.gra = tf.Graph()
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess_config.gpu_options.allow_growth = True
            with self.gra.as_default():
                self.sess = tf.Session(graph=self.gra, config=sess_config)
                self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
                self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        if self.gra is not None:
            with self.gra.as_default():
                for name in name_list:
                    if name in self.name_set:
                        raise Exception("You cannot define different operations with same name: `{}`".format(name))
                    self.name_set.add(name)
                    setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                        for i in range(self.n_group)])
                    setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                                for i in range(self.n_group)])
        else:
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

def dRNN(inputs, previous_hidden, rate=1, cell_type='RNN', hidden_dims=32):
    """
    :param cell: tensorflow rnn cell
    :param inputs: (batch_size, input_dims)
    :param rate: rate for dilations in orginal WaveNet paper
    :param scope: variable scope
    :return:
    """
    # select cell
    if cell_type == 'RNN':
        cell = rnn.BasicRNNCell(hidden_dims)
    elif cell_type == 'LSTM':
        cell = rnn.BasicLSTMCell(hidden_dims)
    else:
        cell = rnn.GRUCell(hidden_dims)

    n_steps = len(inputs)
    assert rate < 0 or rate >= n_steps
    if (n_steps % rate) != 0:
        #  zero padding
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1
        for i_pad in range(dilated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dilated_n_steps = n_steps // rate

    # divide rate = dilated_n_steps
    # reshape tensor with length of each tensor is dilated_n_step
    dilated_inputs = [tf.concat(inputs[i * rate : (i + 1) * rate], axis=0) for i in range(dilated_n_steps)]
    dilated_outputs, dilated_hidden = cell(dilated_inputs, previous_hidden)
    # dilated_outputs, dilated_hidden = rnn.static_rnn(cell, dilated_inputs, dtype=tf.float32, scope=scope)

    # reshape output back
    splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
    unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]

    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]

    return outputs, dilated_hidden



def Mask(inputs, seq_len, mode='mul'):
    """
    :param inputs: Tensor:(batch_size, seq_len, input_size)
    :param seq_len: (batch_size, )
    :param mode: mul(before MLP), add(before softmax)
    :return:
    """
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), dtype=tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        elif mode == 'add':
            return inputs - (1 - mask) * 1e12
        else:
            print('Mask Mode Error')



def Dense(inputs, output_size, bias=True, seq_len=None):
    """
    MLP
    :param inputs: Tensor: (batch_size, input_size)
    outputs: (batch_size, output_size)
    :return:
    """
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, output_size], minval=-0.05, maxval=0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([output_size], minval=-0.05, maxval=0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:, -1], [output_size]], 0))
    if seq_len != None:
        outputs = Mask(outputs, seq_len, mode='mul')
    return outputs


def Attention(Q, K, V, nb_head, batch_num, size_per_head, T=0.2):
    "Multi-head Attention: https://kexue.fm/archives/4765"
    # map Q(target), K(source), V(cooperation), A(importance)
    Q.set_shape([batch_num, size_per_head])
    K.set_shape([batch_num, size_per_head])
    V.set_shape([batch_num, size_per_head])
    with tf.variable_scope('source'):
        res = []
        for _ in range(nb_head):
            H0 = tf.layers.dense(Q, units=size_per_head, activation=tf.nn.relu6)
            H0 = tf.expand_dims(H0, -1)
            res.append(H0)
        Q = tf.concat(res, axis=-1)
    with tf.variable_scope('target'):
        res = []
        for _ in range(nb_head):
            H1 = tf.layers.dense(K, units=size_per_head, activation=tf.nn.relu6)
            H1 = tf.expand_dims(H1, -1)
            res.append(H1)
        K = tf.concat(res, axis=-1)
    with tf.variable_scope('cooperation'):
        res = []
        for _ in range(nb_head):
            H2 = tf.layers.dense(V, units=size_per_head, activation=tf.nn.relu6)
            H2 = tf.expand_dims(H2, -1)
            res.append(H2)
        V = tf.concat(res, axis=-1)
    # calculate product, then mask, softmax
    Q_list = tf.unstack(Q, axis=-1)
    K_list = tf.unstack(K, axis=-1)
    A_list = []
    for _Q, _K in zip(Q_list, K_list):
        H = tf.matmul(_Q, _K, transpose_b=True) / tf.sqrt(float(size_per_head))
        A_list.append(H)
    # A = tf.concat(A_list, axis=-1)

    # res = []
    # for _h in range(nb_head):
    #     H = tf.matmul(Q[:,:,_h], K[:,:,_h], transpose_b=True) / tf.sqrt(float(size_per_head))
    #     H = tf.expand_dims(H, -1)
    #     res.append(H)
    # A = tf.concat(res, axis=-1)

    new_A_list = []
    for _A in A_list:
        _A_list = tf.unstack(_A, axis=0)
        _new_A_list = []
        for __A in _A_list:
            _value = tf.exp(__A / T)
            __A = _value / tf.reduce_sum(_value)
            _new_A_list.append(__A)
        _new_A = tf.stack(_new_A_list, axis=0)
        new_A_list.append(_new_A)
    A = tf.stack(new_A_list, axis=-1)

    # for _head in range(nb_head):
    #     for _batch in range(batch_num):
    #         _value = tf.exp(A[_batch,:,_head] / T)
    #         _importance = _value / tf.reduce_sum(_value, axis=0)
    #         A[_batch,:,_head] = _importance

    # calculate mean in nb_head dimension
    # O = tf.Variable(tf.zeros([batch_num, size_per_head, nb_head]))
    # for _head in range(nb_head):
    #     O[:, :, _head] = tf.matmul(A[:, :, _head], V[:, :, _head])
    V_list = tf.unstack(V, axis=-1)
    O_list = []
    for _A, _V in zip(new_A_list, V_list):
        _O = tf.matmul(_A, _V)
        O_list.append(_O)
    O = tf.stack(O_list, axis=-1)
    O = tf.reduce_mean(O, axis=-1)
    O = tf.layers.dense(O, units=size_per_head, activation=tf.nn.relu6)
    # O = tf.transpose(O, [0, 2, 1, 3])
    # O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    # O = Mask(O, Q_len, mode='mul')
    return O



class CommNet(object):
    def __init__(self, num_agents=7, dim=5, learning_rate=0.003, vector_len=32, tf_device='/cpu:*',
                 batch_size=64, scope_name='Default'):
        super(CommNet, self).__init__()
        self._dim = dim
        self._num_agent = num_agents
        self._lr = learning_rate
        self._vector_len = vector_len
        self.batch_size = batch_size
        self.scope_name = scope_name


        #  ============ DEFINE NETWORK ==========
        with tf.variable_scope(self.scope_name):
            with tf.device(tf_device):
                self.i_hidden_ph = tf.placeholder(tf.float32, shape=(None, self._num_agent, self._dim))
                self.mask_ph = tf.placeholder(tf.float32, shape=(self._dim, self._dim))
                self.c_meta_ph = tf.placeholder(tf.float32, shape=(None, self._dim, self._vector_len))

                self._build_networks()


        # ========= BUFFER =======
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_cell(self, c, h, h_meta, name='Cell'):
        with tf.variable_scope(name):
            self.h = tf.get_variable('w_h', shape=(self._vector_len, self._vector_len), initializer=tf.random_normal_initializer())
            self.c = tf.get_variable('w_c', shape=(self._vector_len, self._vector_len), initializer=tf.random_normal_initializer())
            self.h_meta = tf.get_variable('w_h_meta', shape=(self._vector_len, self._vector_len), initializer=tf.random_normal_initializer())

        dense_h = tf.einsum('ijk, kl->ijl', h, self.h)
        dense_c = tf.einsum('ijk, kl->ijl', c, self.c)
        dense_h_meta = tf.einsum('ijk, kl->ijl', h_meta, self.h_meta)
        # dense_h = tf.matmul(h, self.h)
        # dense_c = tf.matmul(c, self.c)
        # dense_h_meta = tf.matmul(h_meta, self.h_meta)
        dense = dense_c + dense_h + dense_h_meta

        return tf.nn.relu6(dense)

    def _mean(self, h, mask):
        print('Mask')
        print(mask.shape)
        return tf.einsum('ij, kjl->kil', mask, h) / (self._dim - 1)

    def _get_network_type(self):
        return collections.namedtuple('ValueNetwork', 'value')

    def _network_template(self, i_hidden, mask, c_meta):
        # build look-up table
        print(i_hidden.shape)
        hidden = tf.cast(i_hidden, tf.int32)
        input_one_hot = tf.one_hot(hidden, self._num_agent)



        # build CommNet
        self.look_up = tf.get_variable('look_up_table', shape=(self._num_agent, self._vector_len), initializer=tf.random_normal_initializer())
        self.dense_weight = tf.get_variable('dense_w', shape=(self._vector_len, self._dim))
        h0 = tf.einsum('ijk, kl->ijl', input_one_hot, self.look_up)
        # h0 = tf.matmul(input_one_hot, self.look_up)
        # h0 = i_hidden
        h1 = self._build_cell(c_meta, h0, h0, 'step_first')
        c1 = self._mean(h1, mask)
        h2 = self._build_cell(c1, h1, h0, 'step_second')
        out = tf.einsum('ijk, kl->ijl', h2, self.dense_weight)
        # out = tf.matmul(h2, self.dense_weight)
        value = tf.nn.softmax(out)

        return self._get_network_type()(value)

    def _build_networks(self):
        with tf.variable_scope(self.scope_name):
            self.comm_net = tf.make_template('CommNet', self._network_template)
            self.comm_tf = self.comm_net(self.i_hidden_ph, self.mask_ph, self.c_meta_ph)


    def act(self, hidden):
        mask_data = np.ones(shape=(self._dim, self._dim), dtype=np.float32)
        mask_data[np.arange(self._dim), np.arange(self._dim)] = 0.0
        c_meta_data = np.zeros((self.batch_size, self._dim, self._vector_len))
        feed_dict = {
            self.i_hidden_ph: hidden,
            self.mask_ph: mask_data,
            self.c_meta_ph: c_meta_data
        }
        o_hidden = self.sess.run(self.comm_tf.value, feed_dict=feed_dict)
        print('hidden shape')
        print(o_hidden.shape)
        # hidden layer will be updated step by step
        return o_hidden


class SimpleCommNet(object):
    def __init__(self, num_agents=7, dim=5, learning_rate=0.003, vector_len=32, tf_device='/cpu:*',
                 batch_size=64, scope_name='Default'):
        super(SimpleCommNet, self).__init__()
        self._dim = dim
        self._num_agent = num_agents
        self._lr = learning_rate
        self._vector_len = vector_len
        self.batch_size = batch_size
        self.scope_name = scope_name


        #  ============ DEFINE NETWORK ==========
        with tf.variable_scope(self.scope_name):
            with tf.device(tf_device):
                self.i_hidden_ph = tf.placeholder(tf.float32, shape=(None, self._num_agent, self._dim))
                self._build_networks()


        # ========= BUFFER =======
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _get_network_type(self):
        return collections.namedtuple('ValueNetwork', 'value')

    def _network_template(self, i_hidden):
        emb = tf.layers.dense(i_hidden, units=32, activation=tf.nn.relu6)
        mean = tf.reduce_mean(i_hidden, axis=1, keep_dims=True)
        mean = tf.tile(mean, [1, self._num_agent, 1])
        mean = tf.concat([mean, emb], axis=-1)
        hidden = tf.layers.dense(mean, units=self._dim, activation=tf.nn.relu6)

        return self._get_network_type()(hidden)

    def _build_networks(self):
        with tf.variable_scope(self.scope_name):
            self.comm_net = tf.make_template('CommNet', self._network_template)
            self.comm_tf = self.comm_net(self.i_hidden_ph)


    def act(self, hidden):
        feed_dict = {
            self.i_hidden_ph: hidden
        }
        o_hidden = self.sess.run(self.comm_tf.value, feed_dict=feed_dict)
        # hidden layer will be updated step by step
        return o_hidden


