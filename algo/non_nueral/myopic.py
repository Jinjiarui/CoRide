import numpy as np
from algo.base import Red


class Myopic(object):
    def __init__(self, **kwargs):
        pass

    def act(self, states, actions, action_ids, global_order_list):
        """Sorted order with fee decending"""
        for i in range(len(action_ids)):
            temp = actions[i]
            index = sorted(np.arange(len(temp)), key=lambda k: temp[k][3], reverse=True)

            temp, new_ids = action_ids[i], []
            for v in index:
                new_ids.append(temp[v])

            action_ids[i] = new_ids
        return action_ids

    def store_transitions(self, args):
        pass

    def train(self):
        pass
