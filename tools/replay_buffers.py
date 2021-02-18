import numpy as np
from collections import namedtuple


MEMORY_KEYS = ['state', 'feature', 'mess', 'next_state', 'next_features', 'next_mess', 'reward', 'done']
Memory = namedtuple('Memory', MEMORY_KEYS)


class Buffer(object):
    __slots__ = (
        '_custom_tuple', '_capacity', '_size', '_data', '_flag'
    )

    def __init__(self, custom_tuple, capacity: int):
        self._custom_tuple = custom_tuple
        self._capacity = capacity
        self._data = np.array([None] * capacity)
        self._size = 0
        self._flag = 0

    def __len__(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity

    def append(self, **kwargs):
        raise NotImplementedError

    def sample(self, **kwargs):
        raise NotImplementedError


class LoopBuffer(object):
    def __init__(self, capacity):
        self._data = np.array([None for _ in range(capacity)])
        self._capacity = capacity
        self._size = 0
        self._flag = -1
    
    def __len__(self):
        return self._size

    def append(self, memory):
        self._flag += 1

        if self._flag == self._capacity:
            self._flag = 0
    
        self._data[self._flag] = memory
        self._size = min(self._size + 1, self._capacity)

    def sample(self, idx_arr=None, n=None):
        idx_arr = np.random.choice(self._size, n) if idx_arr is None else idx_arr
        buffers = self._data[idx_arr].tolist()
        return buffers
