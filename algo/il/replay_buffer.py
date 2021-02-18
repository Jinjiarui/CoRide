import numpy as np

from collections import namedtuple

from tools.replay_buffers import LoopBuffer


Transition = namedtuple('Transition', 'state, action, actions, next_state, next_actions, reward, grad, done')


def extract_episode(dict_loop_buffer: dict):
    data = []
    for transitions in dict_loop_buffer.values():
        episode = transitions.episode()
        if episode is not None:
            data.extend(transitions.episode())
    return data


class Episode(object):
    def __init__(self):
        self.state = []
        self.action = []
        self.actions = []
        self.reward = []
        self.kl_gradient = []

    def append(self, state, actions, action, reward):
        self.state.append(state)
        self.actions.append(actions)
        self.action.append(action)
        self.reward.append(reward)

    def append_kl(self, kl_graident):
        self.kl_gradient.append(kl_graident)

    def episode(self):
        # return tuples: {state, action, actions, next_state, next_actions, reward, kl_gradient, done}
        length = len(self.state)
        if length == 1:
            return None
        if len(self.kl_gradient) == length - 1:
            grad = self.kl_gradient[-1]
            self.kl_gradient.append(grad)
            # print('Short grad!')
        elif len(self.kl_gradient) < length - 1:
            print('WORINGFEFSEFSFEFSEFEWFSEFESF')
        data = []
        for i in range(length):
            done = True if i == length - 1 else False
            next_i = (i + 1) % length
            meta = Transition(self.state[i], self.action[i], self.actions[i], self.state[next_i],
                              self.actions[next_i], self.reward[i], self.kl_gradient[i], done)
            data.append(meta)
        return data


class WorkerBuffer(object):
    def __init__(self, max_len, use_priority=False):
        self._state = LoopBuffer(max_len)
        self._action = LoopBuffer(max_len)
        self._actions = LoopBuffer(max_len)
        self._reward = LoopBuffer(max_len)
        self._next_state = LoopBuffer(max_len)
        self._next_actions = LoopBuffer(max_len)
        self._done = LoopBuffer(max_len)
        self._kl_grad = LoopBuffer(max_len)
        self._new_add = 0

        self._tuple = namedtuple('Buffer', 'state, action, actions, reward, next_state, next_actions, done, kl_grad')
    
    def __len__(self):
        return len(self._state)

    @property
    def once_new_add(self):
        new_add = self._new_add
        self._new_add = 0
        return new_add
    
    def append(self, data):
        for transition in data:
            # assert isinstance(transition, Transition)
            self._state.append(transition.state)
            self._action.append(transition.action)
            self._actions.append(transition.actions)
            self._reward.append(transition.reward)
            self._next_state.append(transition.next_state)
            self._next_actions.append(transition.next_actions)
            self._kl_grad.append(transition.grad)
            self._done.append(transition.done)
            self._new_add += 1
    
    def sample(self, batch_size):
        idx_arr = np.random.choice(len(self._state), batch_size)
        return self._tuple(
            state=self._state.sample(idx_arr),
            action=self._action.sample(idx_arr),
            actions=self._actions.sample(idx_arr),
            reward=self._reward.sample(idx_arr),
            next_state=self._next_state.sample(idx_arr),
            next_actions=self._next_actions.sample(idx_arr),
            done=self._done.sample(idx_arr),
            kl_grad=self._kl_grad.sample(idx_arr)
        )
