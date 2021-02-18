import numpy as np
import random
import tensorflow as tf
import collections

from algo.base import BaseModel
from algo.il.replay_buffer import WorkerBuffer


class IL(BaseModel):
    def __init__(self, sess, obs_space, act_space, learning_rate=1e-4, gamma=0.97, tau=0.99, name='IL',
                 batch_size=64, update_interval=5, memory_size=2**17, tf_device='/cpu:*', temperature=0.2):
        super(IL, self).__init__(sess, obs_space, act_space, name, batch_size)

        self.update_interval = update_interval
        self.gamma = gamma
        self._tau = tau
        self._lr = learning_rate
        self._T = temperature

        # controller = Controller(self.n_act)

        # ===== DEFINE NETWORK ======
        with tf.device(tf_device):
            self.state_ph = tf.placeholder(tf.float32, (None,) + self.obs_space, name='State')
            self.label_input = tf.placeholder(tf.float32, (None, 1), name='Label')
            self.action_ph = tf.placeholder(tf.float32, (None,) + self.act_space, name='Action')

            self._replay = WorkerBuffer(memory_size)
            self._build_networks()
            self._train_op = self._build_train_op()
            self._sync_qt_ops, self._soft_sync_qt_ops = self._build_sync_op()

        # ===== BUFFER =====
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self._sync_qt_ops)

    def _get_network_type(self):
        return collections.namedtuple('ValueNetwork', 'value')

    def _network_template(self, state, action):
        state_emb = tf.layers.dense(state, units=32, activation=tf.nn.relu6)
        act_emb = tf.layers.dense(action, units=32, activation=tf.nn.relu6)
        emb = tf.concat([state_emb, act_emb], axis=1)
        emb = tf.layers.dense(emb, units=32, activation=tf.nn.relu6)
        q_value = tf.layers.dense(emb, units=1, activation=tf.nn.relu6)

        return self._get_network_type()(q_value)

    def _build_networks(self):
        self.eval_net = tf.make_template('Eval', self._network_template)
        self.target_net = tf.make_template('Target', self._network_template)
        self.eval_tf = self.eval_net(self.state_ph, self.action_ph)
        self.target_tf = self.target_net(self.state_ph, self.action_ph)

    def _build_train_op(self):
        self._loss = tf.reduce_mean(tf.square(self.eval_tf.value - self.label_input))

        optimizer = tf.train.AdamOptimizer(self._lr)
        return optimizer.minimize(self._loss)

    def _build_sync_op(self):
        sync_qt_ops, soft_sync_qt_ops = [], []
        trainable_eval = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Eval')
        trainable_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
        for (w_eval, w_target) in zip(trainable_eval, trainable_target):
            sync_qt_ops.append(w_target.assign(w_eval, use_locking=True))
            soft_sync_qt_ops.append(w_target.assign((1. - self._tau) * w_eval + self._tau * w_target, use_locking=True))

        return sync_qt_ops, soft_sync_qt_ops

    def store_transitions(self, transitions):
        # shuffle worker_transition
        random.Random(4).shuffle(transitions)
        self._replay.append(transitions)

    def act(self, states, actions, action_ids, global_order_list):
        """ Give a list of reranked actions
        :param states: list, [states[grid], ...]
        :param actions: list, [orders[grid], ...]
        :return: list, [orders_index[grid], ...]
        """
        # red_ids = np.concatenate(red_ids)
        # states = states.reshape(4, -1)
        _states = []
        for i, _actions in enumerate(actions):
            _states.append(np.tile(states[i], [len(_actions), 1]))

        split = map(lambda x: len(x), _states)
        split = np.cumsum(list(split))[:-1]

        # tile  jk

        feed_dict = {
            self.state_ph: np.vstack(_states),
            self.action_ph: np.vstack(actions)
        }

        profits = self.sess.run(self.eval_tf.value, feed_dict=feed_dict)
        profits = profits.reshape((-1,))

        # split with the original shape
        profits = np.split(profits, split)
        # order_index_pair = []

        for i, _values in enumerate(profits):
            _values = np.exp(_values / self._T)
            _policy = _values / np.sum(_values)
            temp = action_ids[i]
            index = np.random.choice(len(_policy), len(_values), replace=False, p=_policy)
            action_ids[i] = list(map(lambda j: temp[j], index))

        return action_ids

    def cal_target_value(self, state_n_next, reward_n, done_n, feature_n_next=None):
        # repeat state, mess with feature shape
        assert len(state_n_next) == len(feature_n_next)

        split_table = [len(feature_n_next[i]) for i in range(len(feature_n_next))]
        split_arr = np.cumsum(split_table)[:-1]

        for i, e in enumerate(split_table):
            state_n_next[i] = np.tile(state_n_next[i], [e, 1])

        state_n_next = np.vstack(state_n_next)
        feature_n_next = np.vstack(feature_n_next)

        feed_dict = {
            self.state_ph: state_n_next,
            self.action_ph: feature_n_next
        }
        target_value = self.sess.run(self.target_tf.value, feed_dict=feed_dict)

        # split target_value, then calculate average target value
        target_value = np.array_split(target_value, split_arr)
        target_value = np.array(list(map(np.mean, target_value)))
        target_value = target_value.reshape((-1,))

        return np.where(done_n == False, reward_n + self.gamma * target_value, reward_n)

    def train_worker(self, print_interval=50):
        total_num = len(self._replay)
        new_add = self._replay.once_new_add
        batch_num = int((new_add + self.batch_size - 1) * 5 / self.batch_size)
        loss_record = 0.0

        print('total-length: {0:<4d} new-add: {1:<4d} batch-num: {2:<4d}'.format(total_num, new_add, batch_num))

        for i in range(batch_num):
            batch = self._replay.sample(self.batch_size)
            target_value = self.cal_target_value(batch.next_state, batch.reward, batch.done, batch.next_actions)

            feed_dict = {self.state_ph: np.vstack(batch.state), self.action_ph: np.vstack(batch.action),
                         self.label_input: target_value.reshape((-1, 1))}

            # border_values, candidate_values = self.cal_candidate(batch.state, batch.actions)
            loss, value, _ = self.sess.run([self._loss, self.eval_tf.value, self._train_op], feed_dict=feed_dict)

            loss_record += loss

            if i % print_interval == 0:
                print('--- batch #{0:<4d} loss: [{1:<.6f}] value: [{2:<.6f}] target-value: [{3:<.6f}]'.format(
                    i, loss, np.mean(value), np.mean(target_value)))

            if i % self.update_interval == 0:
                self.sess.run(self._soft_sync_qt_ops)

        return loss_record / batch_num

    def train(self, print_interval=500):
        print('\n>>>TRAINING WORKERS ...')
        loss = self.train_worker(print_interval)
        print('Mean-batch-loss [{0:<.8f}]'.format(loss))
        return {'worker-loss': loss}
