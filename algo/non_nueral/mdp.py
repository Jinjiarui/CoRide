""" This is a RL algorithm with KM matching for order dispatching
need to build global message for training and implementing:
"""
import numpy as np

from itertools import groupby
from scipy.optimize import linear_sum_assignment


class MdpAgent(object):
    """
    Policy iteration:
    :param:
    order_id, driver_id: start 0
    global_order_list: orders: [[order_id, start, end, self._p * 0.1, self._t],..]
    :return:: selected action
    """
    __slots__ = ('global_order_list', 'discount', 'city_time', '_value_state',
                 'time_num', 'node_num', '_n_state', 'global_driver_list')

    def __init__(self, time_num, node_num, discount=0.9):
        self.discount = discount  # discount for future value
        #  state[-1] = city_time,
        self._value_state = np.zeros([time_num + 1, node_num + 1])
        self._n_state = np.zeros([time_num + 1, node_num + 1])
        self.time_num = time_num

    # def initial_value(self):
        """Deprecated"""
        # self._value_state = np.zeros([self.time_num, self.node_num])
        # self._n_state = np.zeros([self.time_num, self.node_num])

    def step_update_value_from_value(self, city_time, global_order_list):
        """Compute Q-value of action in state"""
        for _order in global_order_list:
            if city_time + _order[4] > self.time_num:
                city_time = self.time_num - _order[4]
            self._n_state[city_time, _order[1]] += 1
            self._value_state[city_time, _order[1]] += \
                (1 / self._n_state[city_time, _order[1]]) * \
                (pow(self.discount, _order[4]) * self._value_state[city_time + _order[4], _order[2]] + _order[3] -
                 self._value_state[city_time, _order[1]])


    # def step_generate_action(self):
    """Deprecated"""
    #     km_array = np.zeros([len(self.global_order_list), len(self.global_driver_list)])
    #     for _order_index in range(km_array.shape[0]):
    #         for _driver_index in range(km_array.shape[1]):
    #             km_array[_order_index, _driver_index] = \
    #                 self._value_state[self.city_time, self.global_driver_list[_driver_index][1]] \
    #                 - self._value_state[self.city_time, self.global_order_list[_order_index][2]]
    #
    #     select_driver, select_order = linear_sum_assignment(km_array)
    #     return select_driver, select_order

    def act(self, city_time, global_order_list):
        self.step_update_value_from_value(city_time, global_order_list)
        # create look-up table
        _order_value = []
        for _order in global_order_list:
            # calculate Q-value: end_p - begin_p as potential and + _p as current
            q_value = self._value_state[city_time, _order[2]] - self._value_state[city_time, _order[1]] + _order[3]
            _order_value.append((_order[0], q_value))
        orders = sorted(_order_value, key=lambda order: order[1], reverse=True)
        orders= [order[0] for order in orders]
        orders = sorted(orders, key=lambda order: order[0], reverse=True)
        orders = groupby(orders, key=lambda order: order[0])
        action_ids = [list(group) for key, group in orders]

        return action_ids

    def store_transitions(self):
        pass

    def train(self):
        pass












