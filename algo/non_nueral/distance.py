import numpy as np
from algo.base import Red


class Nearest(object):
    def __init__(self, **kwargs):
        pass

    def act(self, states, actions, action_ids, global_order_list):
        """Order dispatching with nearest matching"""
        for i in range(len(action_ids)):
            # np.random.shuffle(action_ids[i])
            temp = actions[i]
            index = list(np.arange(len(temp)))
            for _i in index:
                if temp[_i][2] > 0:
                    index.remove(_i)
                    index.insert(0, _i)
            temp, new_ids = action_ids[i], []
            for _ii in index:
                new_ids.append(temp[_ii])
            action_ids[i] = new_ids
        return action_ids

    def store_transitions(self, args):
        pass
        
    def train(self):
        pass