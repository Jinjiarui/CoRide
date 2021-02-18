import numpy as np
import collections
from copy import copy
from abc import ABCMeta, abstractmethod
from Simulator.simulator.utilities import *


class Distribution(object):
    ''' Define the distribution from which sample the orders'''
    # __metaclass__ = ABCMeta  # python 2.7
    @abstractmethod
    def sample(self):
        pass


class PoissonDistribution(Distribution):

    def __init__(self, lam):
        self._lambda = lam

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.poisson(self._lambda, 1)[0]


class GaussianDistribution(Distribution):

    def __init__(self, args):
        mu, sigma = args
        self.mu = mu        # mean
        self.sigma = sigma  # standard deviation

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.normal(self.mu, self.sigma, 1)[0]


class Node(object):
    __slots__ = ('neighbors', '_index', 'orders', 'orders_samples', 'drivers',
                 'order_num', 'idle_driver_num', 'offline_driver_num'
                 'order_generator', 'offline_driver_num', 'order_generator',
                 'n_side', 'layers_neighbors', 'layers_neighbors_id', '_last_state', '_last_order_list',
                 'fake_order_num', 'reward', 'fleet_order_num', 'fleet_driver_num', 'orders_samples_num',
                 '_entropy')

    def __init__(self, index, fleet_help=False):
        # private
        self._index = index   # unique node index.

        # public
        self.neighbors = []  # a list of nodes that neighboring the Nodes
        self.orders = []     # a list of orders
        self.orders_samples = []  # a list of orders includes a single fleet management: len(orders_samples) < len(orders)
        self.orders_samples_num = 0  # real_order_num + 1(fake_order) + 6(fleet_order)
        self.drivers = {}    # a dictionary of driver objects contained in this node
        self.order_num = 0
        self.idle_driver_num = 0  # number of idle drivers in this node
        self.offline_driver_num = 0
        self.fake_order_num = 0
        self.fleet_order_num = 0
        self.fleet_driver_num = 0  # number of fleet drivers in this node
        self.order_generator = None

        self.n_side = 0      # the topology is a n-sided map
        # layer 1 indices: layers_neighbors[0] = [[1,1], [0, 1], ...],
        self.layers_neighbors = []
        # layer 2 indices layers_neighbors[1]
        self.layers_neighbors_id = []  # layer 1: layers_neighbors_id[0] = [2, 1,.]
        self._last_state = None
        self._last_order_list = None
        self.reward = 0
        self._entropy = 0.  # build information for policy training


    def get_index(self):
        return self._index

    @staticmethod
    def state_space():
        # state[-1] = self.city_time / self._time_limit
        # state[0] = self._index
        # state[1] = self.idle_driver_num
        # state[2] = self.order_num
        # state[3] = self.fleet_driver_num
        # latent state = state[2] / state[1]
        return tuple([5])

    @property
    def last_state(self):
        return self._last_state

    @property
    def last_order_list(self):
        return self._last_order_list

    @property
    def entropy(self):
        return self._entropy

    def record_history(self):
        """Record the state and order list of last time"""
        self._last_state = self.get_state()
        self._last_order_list = []
        for order in self.orders:
            self._last_order_list.append(order.get_state())
        if len(self._last_order_list) == 0:
            self._last_order_list.append([self._index, self._index, 0, 1, -1])

    def initial_fleet_drivers(self, fleet_rate, fleet_help):
        self.fleet_driver_num = 0
        for _driver in self.drivers.values():
            if not fleet_help:
                break
            assert isinstance(_driver, Driver)
            if np.random.random() < fleet_rate:
                _driver.set_service_type(True)
                self.fleet_driver_num += 1


    def set_entropy(self):
        """Build entropy for emergency"""
        real_order_num = self.order_num - self.fleet_order_num - self.fake_order_num
        _rho = self.idle_driver_num / max(1, real_order_num)
        # self._entropy = np.sum(_rho * np.log(_rho))
        self._entropy = _rho

    def get_info_price_distribution(self, l_max):
        price_dist = [0 for _ in range(l_max)]
        # initialize price_dist
        temp_price_dist = dict()
        for _i in range(l_max):
            temp_price_dist[_i] = []
        for _order in self.orders:
            for _i in range(l_max):
                if _order.get_price() >= 1.5 * _i and  _order.get_price() < 1.5 * (_i + 1):
                    temp_price_dist[_i].append(_order.get_price())
                if _order.get_price() >= 1.5 * l_max:
                    temp_price_dist[l_max-1].append(_order.get_price())
        for _i in range(l_max):
            price_dist[_i] = len(temp_price_dist[_i])
        # price_dist = price_dist / np.sum(price_dist)
        return price_dist

    def get_info_time_distribution(self, l_max):
        time_dist = [0 for _ in range(l_max)]
        for _i in range(l_max):
            _num = 0 # record number of order
            for _order in self.orders:
                if _order.get_duration() == _i:
                    _num += 1
            time_dist[_i] = _num
        # time_dist = time_dist / np.sum(time_dist)
        return time_dist

    def get_info_end_layer_distribution(self, layer_neighborhood, l_max):
        end_layer_dist = [0 for _ in range(l_max)]
        for _i in range(l_max):
            _num = 0  # record number of orders layerwisely
            for _order in self.orders:
                if _order.get_end_position_id() in layer_neighborhood[self._index][_i]:
                    _num += 1
            end_layer_dist[_i] = _num
        # end_layer_dist = end_layer_dist / np.sum(end_layer_dist)
        return end_layer_dist

    def get_entropy(self):
        self.set_entropy()
        return self._entropy

    def get_info_template(self):
        # node_id, idle_driver_num, real_order_num, city_time, fleet_driver_num
        return collections.namedtuple('Status', 'states')

    def get_state(self):
        # state[-1] = self.city_time / self._time_limit
        state = np.zeros(5)
        state[0] = self._index
        state[1] = self.idle_driver_num
        state[2] = self.order_num
        state[3] = self.fleet_driver_num
        # latent state = state[2] / state[1]
        return state


    def get_orders_state(self, global_entropy):
        """Return list of order states and list of order ids of one node"""
        order_index, order_states, order_feature = [], [], []
        feature_space = None
        # node_id = self._index
        # order_num = 0
        for i, _order in enumerate(self.orders):
            if not _order.served:
                order_states.append(_order.get_state())
                order_index.append((self._index, i))
                order_feature.append(_order.get_feature(global_entropy))
                if feature_space == None:
                    feature_space = len(_order.get_feature(global_entropy))
                # order_num += 1
        # print("---> Node {} has order {}".format(node_id, order_num))
        return order_index, order_states, order_feature, feature_space


    def get_sample_orders_state(self):
        """Return list of order states, ids, feature(entropy)"""
        order_index, order_states, order_feature = [], [], []
        for i, _order in enumerate(self.orders_samples):
            if not _order.served:
                order_states.append(_order.get_state())
                order_index.append((self._index, i))
                order_feature.append(_order.get_feature())
        return order_index, order_states, order_feature


    def clean_node(self):
        self.orders = []
        self.order_num = 0
        self.drivers = {}
        self.idle_driver_num = 0
        self.offline_driver_num = 0
        self.fake_order_num = 0
        self.fleet_driver_num = 0
        self.fleet_order_num = 0
        self.orders_samples_num = 0

    def get_layers_neighbors(self, l_max, M, N, env):

        x, y = ids_1dto2d(self.get_node_index(), M, N)
        self.layers_neighbors = get_layers_neighbors(x, y, l_max, M, N)
        for layer_neighbors in self.layers_neighbors:
            temp = []
            for item in layer_neighbors:
                x, y = item
                node_id = ids_2dto1d(x, y, M, N)
                if env.nodes[node_id] is not None:
                    temp.append(node_id)
            self.layers_neighbors_id.append(temp)

    def get_node_index(self):
        return self._index

    def get_driver_numbers(self):
        return self.idle_driver_num

    def get_fleet_driver_numbers(self):
        return self.fleet_driver_num

    def get_order_numbers_loop(self):
        temp_order_number = 0
        temp_fleet_order_number = 0
        temp_fake_order_number = 0
        temp_real_order_number = 0
        for _order in self.orders:
            assert isinstance(_order, Order)
            temp_order_number += 1
            if _order.get_service_type() == -1:
                temp_real_order_number += 1
            elif _order.get_service_type() == 0:
                temp_fake_order_number += 1
            elif _order.get_service_type() > 0:
                temp_fleet_order_number += 1
            else:
                pass
        return temp_order_number, temp_real_order_number, temp_fake_order_number, temp_fleet_order_number

    def get_driver_numbers_loop(self):
        temp_idle_driver_number = 0
        temp_fleet_driver_number = 0
        for _driver in self.drivers.values():
            if _driver.onservice is False and _driver.online is True:
                temp_idle_driver_number += 1
                if _driver.get_service_type():
                    temp_fleet_driver_number += 1
        return temp_idle_driver_number, temp_fleet_driver_number

    def get_idle_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is True:
                temp_idle_driver += 1
        return temp_idle_driver

    def get_off_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is False:
                temp_idle_driver += 1
        return temp_idle_driver

    def order_distribution(self, distribution, dis_paras):

        if distribution == 'Poisson':
            self.order_generator = PoissonDistribution(dis_paras)
        elif distribution == 'Gaussian':
            self.order_generator = GaussianDistribution(dis_paras)
        else:
            pass

    def generate_order_random(self, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        num_order_t = self.order_generator.sample(seed)
        self.order_num += num_order_t
        self.orders_samples_num += num_order_t

        for ii in np.arange(num_order_t):
            price = np.random.normal(50, 5, 1)[0]
            price = 10 if price < 0 else price

            current_node_id = self.get_node_index()
            destination_node = [kk for kk in np.arange(
                len(nodes)) if kk != current_node_id]
            self.orders.append(Order(nodes[current_node_id],
                                     nodes[np.random.choice(
                                         destination_node, 1)[0]],
                                     city_time,
                                     # city_time + np.random.choice(5, 1)[0]+1,
                                     np.random.choice(2, 1)[0] + 1,  # duration
                                     price, 1))
            self.orders_samples.append(Order(nodes[current_node_id],
                                     nodes[np.random.choice(
                                         destination_node, 1)[0]],
                                     city_time,
                                     # city_time + np.random.choice(5, 1)[0]+1,
                                     np.random.choice(2, 1)[0] + 1,  # duration
                                     price, 1))
        return


    def update_order_fleet(self, order_id):
        self.orders[order_id].set_service_type(-1)

    def update_driver_fleet(self, driver_id):
        self.drivers[driver_id].set_service_type(True)

    def clean_order_fleet_fake(self):
        """Clean order fleet fake order at each time before dispatching"""
        # Update sample_order before get value
        # samples_clean_order_index = []
        # for idx, o in enumerate(self.orders_samples):
        #     assert isinstance(o, Order)
        #     if o.get_service_type() >= 0:
        #         # service_type > 0 present neighbor nodes for fleet control
        #         samples_clean_order_index.append(idx)
        #         self.orders_samples_num -= 1
        # if len(samples_clean_order_index) != 0:
        #     # remove fleet and fake order
        #     self.orders_samples = [i for j, i in enumerate(
        #         self.orders_samples) if j not in samples_clean_order_index]

        clean_order_index = []
        for idx, o in enumerate(self.orders):
            assert isinstance(o, Order)
            if o.get_service_type() >= 0:
                # service_type > 0 present neighbor nodes for
                clean_order_index.append(idx)
                self.order_num -= 1
                if o.get_service_type() == 0:
                    self.fake_order_num -= 1
                else:
                    self.fleet_order_num -= 1
        if len(clean_order_index) != 0:
            # remove fleet and fake order
            self.orders = [i for j, i in enumerate(
                self.orders) if j not in clean_order_index]

        assert self.fake_order_num == 0
        assert self.fleet_order_num == 0
        assert self.order_num >= 0
        assert self.orders_samples_num >= 0


    def generate_order_sample(self, city_time, nodes, fleet_price=0):
        """Update sample order"""
        self.orders_samples_num = 0
        self.orders_samples = []
        for _order in self.orders:
            # add real orders
            if _order.get_service_type() == -1:
                self.orders_samples.append(_order)

        # add fleet orders
        for ii in self.layers_neighbors_id[0]:
            self.orders_samples.append(Order(self, nodes[ii], city_time, 1, fleet_price, 0, service_type=ii + 1))
        # add fake order
        self.orders_samples.append(Order(self, self, city_time, 1, 0, 0, service_type=0))

        self.orders_samples_num = len(self.orders_samples)


    def generate_order_fleet(self, city_time, nodes, fleet_price=0):
        """
        Generate orders for fleet management: fleet to neighbors
        """
        # for ii in range(get_layers_neighbors(i, j, l_max, M, N)[-1]):
        for ii in self.layers_neighbors_id[0]:
            # self.orders_samples_num += 1
            # self.orders_samples.append(Order(self, nodes[ii], city_time, 1, fleet_price, 0, service_type=ii + 1))
            for jj in range(self.get_fleet_driver_numbers()):
                self.orders.append(Order(self, nodes[ii], city_time, 1, fleet_price, 0, service_type=ii + 1))
                self.fleet_order_num += 1
                self.order_num += 1


    def generate_order_fake(self, city_time):
        pass
        # self.orders.append(Order(self, self, city_time, 1, 0, 0, service_type=0))
        # self.order_num += 1
        # self.fake_order_num += 1

    def generate_order_real(self, l_max, order_time_dist, order_price_dist, city_time, nodes, seed):
        """Generate new orders at each time step
        """
        # self.orders_samples_num = 0
        num_order_t = self.order_generator.sample(seed)
        self.order_num += num_order_t
        # self.orders_samples_num += num_order_t
        self.fake_order_num = 0
        # self._last_order_list = copy(self.orders)

        for ii in np.arange(num_order_t):

            if l_max == 1:
                duration = 1
            else:
                duration = np.random.choice(
                    np.arange(1, l_max + 1), p=order_time_dist)
            price_mean, price_std = order_price_dist[duration - 1]
            price = np.random.normal(price_mean, price_std, 1)[0]
            price = price if price > 0 else price_mean

            current_node_id = self.get_node_index()
            destination_node = []
            for jj in np.arange(duration):
                for kk in self.layers_neighbors_id[jj]:
                    if nodes[kk] is not None:
                        destination_node.append(kk)
            self.orders.append(Order(nodes[current_node_id],
                                     nodes[np.random.choice(
                                         destination_node, 1)[0]],
                                     city_time,
                                     duration,
                                     price, 1, service_type=-1))
            # self.orders_samples.append(Order(nodes[current_node_id],
            #                          nodes[np.random.choice(
            #                              destination_node, 1)[0]],
            #                          city_time,
            #                          duration,
            #                          price, 1, service_type=-1))
        # fake orders for idle drivers
        # self.orders_samples_num += 1  # single fake order
        # self.orders_samples.append(Order(self, self, city_time, 1, 0, 0, service_type=0))

        # fake order for nodes
        self.orders.append(Order(self, self, city_time, 1, 0, 0, service_type=0))
        self.order_num += 1
        self.fake_order_num += 1

        # fake order for drivers
        for _ in range(self.get_driver_numbers()):
            self.orders.append(Order(self, self, city_time, 1, 0, 0, service_type=0))
            # self.orders.append(Order(nodes[current_node_id], nodes[current_node_id], city_time, 1, 0, 0))
            self.order_num += 1
            self.fake_order_num += 1
        return

    def add_order_real(self, city_time, destination_node, duration, price):
        # current_node_id = self.get_node_index()
        self.orders.append(Order(self,
                                 destination_node,
                                 city_time,
                                 duration,
                                 price, 0))
        self.order_num += 1

    def set_neighbors(self, nodes_list):
        self.neighbors = nodes_list
        self.n_side = len(nodes_list)

    def set_dest(self, dest, drivers):
        for _driver in drivers:
            _driver.set_dest(dest)

    def remove_idle_driver_random(self):
        """Randomly remove one idle driver from current grid"""
        removed_driver_id = "NA"
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is True:
                self.remove_driver(key)
                removed_driver_id = key
            if removed_driver_id != "NA":
                break
        assert removed_driver_id != "NA"
        return removed_driver_id

    def set_idle_driver_offline_random(self):
        """Randomly set one idle driver offline"""
        if len(self.drivers) == 0:
            return
        removed_driver_id = "NA"
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is True:
                item.set_offline()
                removed_driver_id = key
            if removed_driver_id != "NA":
                break
        assert removed_driver_id != "NA"
        return removed_driver_id

    def set_offline_driver_online(self):

        online_driver_id = "NA"
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is False:
                item.set_online()
                online_driver_id = key
            if online_driver_id != "NA":
                break
        assert online_driver_id != "NA"
        return online_driver_id

    def get_driver_random(self):
        """Randomly get one driver"""
        assert self.idle_driver_num > 0
        get_driver_id = 0
        for key in self.drivers.keys():
            get_driver_id = key
            break
        return self.drivers[get_driver_id]

    def remove_driver(self, driver_id):
        assert isinstance(self.drivers[driver_id], Driver)
        if self.drivers[driver_id].get_service_type():
            self.fleet_driver_num -= 1
        removed_driver = self.drivers.pop(driver_id, None)
        self.idle_driver_num -= 1
        if removed_driver is None:
            raise ValueError(
                'Nodes.remove_driver: Remove a driver that is not in this node')

        return removed_driver

    def add_driver(self, driver_id, driver):
        self.drivers[driver_id] = driver
        self.idle_driver_num += 1
        if driver.get_service_type():
            self.fleet_driver_num += 1

    def remove_unfinished_order(self, city_time):
        """Remove unfinished orders which out of the service limit or completed"""
        un_finished_order_index = []
        for idx, o in enumerate(self.orders):
            # order un served
            if o.get_wait_time() + o.get_begin_time() < city_time:
                un_finished_order_index.append(idx)
                self.order_num -= 1
                if o.get_service_type() == 0:
                    self.fake_order_num -= 1
                elif o.get_service_type() > 0:
                    self.fleet_order_num -= 1

            # order completed
            if o.get_assigned_time() + o.get_duration() == city_time and o.get_assigned_time() != -1:
                un_finished_order_index.append(idx)
                self.order_num -= 1
                if o.get_service_type() == 0:
                    self.fake_order_num -= 1
                elif o.get_service_type() > 0:
                    self.fleet_order_num -= 1

        if len(un_finished_order_index) != 0:
            # remove unfinished orders
            self.orders = [i for j, i in enumerate(
                self.orders) if j not in un_finished_order_index]
            self.order_num = len(self.orders)

    def remove_unfinished_order_sample(self, city_time):
        """Remove unfinished orders which out of the service limit or completed"""
        un_finished_order_sample_index = []
        for idx, o in enumerate(self.orders_samples):
            # order un served
            if o.get_wait_time() + o.get_begin_time() < city_time:
                un_finished_order_sample_index.append(idx)
                self.orders_samples_num -= 1

            # order completed
            if o.get_assigned_time() + o.get_duration() == city_time and o.get_assigned_time() != -1:
                un_finished_order_sample_index.append(idx)
                self.orders_samples_num -= 1

        if len(un_finished_order_sample_index) != 0:
            # remove unfinished orders
            self.orders_samples = [i for j, i in enumerate(
                self.orders) if j not in un_finished_order_sample_index]
            self.orders_samples_num = len(self.orders_samples)



    # def simple_order_assign(self, city_time, city):
    #     """Deprecated"""
    #     reward = 0
    #     num_assigned_order = min(self.order_num, self.idle_driver_num)
    #     served_order_index = []
    #     for idx in np.arange(num_assigned_order):
    #         order_to_serve = self.orders[idx]
    #         order_to_serve.set_assigned_time(city_time)
    #         self.order_num -= 1
    #         reward += order_to_serve.get_price()
    #         served_order_index.append(idx)
    #         for key, assigned_driver in self.drivers.items():
    #             if assigned_driver.onservice is False and assigned_driver.online is True:
    #                 assigned_driver.take_order(order_to_serve)
    #                 removed_driver = self.drivers.pop(
    #                     assigned_driver.get_driver_id(), None)
    #                 assert removed_driver is not None
    #                 city.n_drivers -= 1
    #                 break
    #
    #     all_order_num = len(self.orders)
    #     finished_order_num = len(served_order_index)
    #
    #     # remove served orders
    #     self.orders = [i for j, i in enumerate(
    #         self.orders) if j not in served_order_index]
    #     assert self.order_num == len(self.orders)
    #
    #     return reward, all_order_num, finished_order_num

    def simple_order_assign_real(self, orders, city_time, city, fleet_help=False):
        rewards = []
        # num_assigned_order = min(self.order_num, self.idle_driver_num)
        served_orders = []
        served_fleet_orders = []
        serve_driver_idx = []
        removed_drivers = []


        # record the original state of node
        assert self.idle_driver_num >= self.fleet_driver_num
        ori_idle = self.idle_driver_num
        ori_order_num = self.order_num
        ori_fake_order_num = self.fake_order_num
        ori_fleet_order_num = self.fleet_order_num
        ori_fleet = self.fleet_driver_num

        dest = [0.] * (city.M * city.N)

        assert self.order_num == len(self.orders)
        # assign fleet_order to fleet_driver first
        for _order in orders:
            assert isinstance(_order, Order)
            if self.idle_driver_num <= 0:
                break
            if not fleet_help:
                break
            for key, assigned_driver in self.drivers.items():
                # select fleet_driver
                if self.idle_driver_num <= 0:
                    break
                if not assigned_driver.onservice and assigned_driver.online and assigned_driver.get_service_type():
                    assert _order.get_begin_position() is assigned_driver.node
                    # select fleet_order
                    if _order.get_service_type() > 0:
                        _order.set_assigned_time(city_time)
                        self.order_num -= 1
                        self.fleet_order_num -= 1
                        rewards.append(_order.get_price())
                        served_fleet_orders.append(_order)
                        if _order.get_end_position() is not None:
                            assigned_driver.take_order(_order)
                            self.fleet_driver_num -= 1
                            removed = self.drivers.pop(assigned_driver.get_driver_id(), None)
                            assert removed is not None
                            removed_drivers.append(removed)
                        else:
                            assigned_driver.set_offline()
                            city.n_offline_drivers += 1
                        city.n_drivers -= 1
                        # self.idle_driver_num -= 1 has done in take order
                        # self.idle_driver_num -= 1
                        city.n_fleet_drivers -= 1
                        serve_driver_idx.append([self._index, assigned_driver.get_driver_id()])
                        break


        # assign rest of order normally
        if fleet_help:
            self.orders = [i for j, i in enumerate(self.orders) if i not in served_fleet_orders]
            rest_orders = [i for j, i in enumerate(orders) if i not in served_fleet_orders]
            assert self.order_num == len(self.orders)
        else:
            rest_orders = orders

        for _order in rest_orders:
            assert isinstance(_order, Order)
            if self.idle_driver_num <= 0:
                break
            for key, assigned_driver in self.drivers.items():
                if self.idle_driver_num <= 0:
                    break
                if _order.get_service_type() > 0:
                    break
                if not assigned_driver.onservice and assigned_driver.online:
                    assert _order.get_begin_position() is assigned_driver.node
                    _order.set_assigned_time(city_time)
                    self.order_num -= 1
                    if _order.get_service_type() == 0:
                        self.fake_order_num -= 1
                    if _order.get_service_type() > 0:
                        self.fleet_order_num -= 1
                    rewards.append(_order.get_price())
                    served_orders.append(_order)
                    if _order.get_end_position() is not None:
                        if assigned_driver.get_service_type():
                            self.fleet_driver_num -= 1
                        assigned_driver.take_order(_order)
                        # policy[_order.get_end_position().get_node_index()] += 1
                        dest[_order.get_end_position().get_node_index()] += 1
                        removed = self.drivers.pop(assigned_driver.get_driver_id(), None)
                        assert removed is not None
                        removed_drivers.append(removed)
                    else:
                        assigned_driver.set_offline()
                        city.n_offline_drivers += 1
                    # assigned_driver.set_info(ori_idle_driver_num, ori_order_num)
                    # self.idle_driver_num -= 1 has done in take order
                    # self.idle_driver_num -= 1
                    city.n_drivers -= 1
                    serve_driver_idx.append([self._index, assigned_driver.get_driver_id()])
                    break
        # for idx in np.arange(num_assigned_order):
        #     order_to_serve = self.orders[idx]
        #     order_to_serve.set_assigned_time(city_time)
        #     self.order_num -= 1
        #     reward += order_to_serve.get_price()
        #     served_order_index.append(idx)
        #     for key, assigned_driver in self.drivers.items():
        #         if assigned_driver.onservice is False and assigned_driver.online is True:
        #             if order_to_serve.get_end_position() is not None:
        #                 assigned_driver.take_order(order_to_serve)
        #                 removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
        #                 assert removed_driver is not None
        #             else:
        #                 assigned_driver.set_offline()  # order destination is not in target region
        #             city.n_drivers -= 1
        #             break

        # all_order_num = len(self.orders)
        # finished_order_num = len(served_order_index)

        # remove served orders

        self.orders = [i for j, i in enumerate(self.orders) if i not in served_orders]
        self.set_dest(dest, removed_drivers)

        assert self.order_num == len(self.orders)

        real_finished = ori_order_num - ori_fake_order_num - ori_fleet_order_num - self.order_num + self.fake_order_num + self.fleet_order_num
        fake_finished = ori_fake_order_num - self.fake_order_num
        fleet_finished = ori_fleet_order_num - self.fleet_order_num
        ori_real_order = ori_order_num - ori_fake_order_num - ori_fleet_order_num

        # Update number of drivers: All cars have been dispatched: All zeros
        # ori_idle = self.idle_driver_num
        # ori_fleet = self.fleet_driver_num


        return rewards, ori_real_order, real_finished, serve_driver_idx, fake_finished, fleet_finished, ori_fake_order_num, ori_fleet_order_num, ori_idle, ori_fleet

    def simple_order_assign_broadcast_update(self, city, neighbor_node_reward):

        assert self.idle_driver_num == 0
        reward = 0
        num_finished_orders = 0
        for neighbor_node in self.neighbors:
            if neighbor_node is not None and neighbor_node.idle_driver_num > 0:
                num_assigned_order = min(
                    self.order_num, neighbor_node.idle_driver_num)
                rr = self.utility_assign_orders_neighbor(
                    city, neighbor_node, num_assigned_order)
                reward += rr
                neighbor_node_reward[neighbor_node.get_node_index()] += rr
                num_finished_orders += num_assigned_order
            if self.order_num == 0:
                break

        assert self.order_num == len(self.orders)
        return reward, num_finished_orders

    def utility_assign_orders_neighbor(self, city, neighbor_node, num_assigned_order):

        served_order_index = []
        reward = 0
        curr_city_time = city.city_time
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(curr_city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)
            for key, assigned_driver in neighbor_node.drivers.items():
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    if order_to_serve.get_end_position() is not None:
                        assigned_driver.take_order(order_to_serve)
                        removed_driver = neighbor_node.drivers.pop(
                            assigned_driver.get_driver_id(), None)
                        assert removed_driver is not None
                    else:
                        assigned_driver.set_offline()
                    city.n_drivers -= 1
                    break

        # remove served orders
        self.orders = [i for j, i in enumerate(
            self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)

        return reward


class Driver(object):
    __slots__ = ("online", "onservice", 'order', 'node', 'city_time', '_driver_id', '_reward',
                 '_last_order_state', '_last_reward', '_time_limit', '_last_state', '_last_order_list',
                 '_service_type', '_dest')

    def __init__(self, driver_id, time_limit, service_type=False):
        self.online = True
        self.onservice = False
        self.order = None     # the order this driver is serving
        self.node = None      # the node that contain this driver.
        self.city_time = 0  # track the current system time

        # private
        self._driver_id = driver_id  # unique driver id.
        self._reward = 0.
        self._dest = None  #  record destination for policy training
        self._last_order_state = None
        self._last_reward = 0.
        self._time_limit = time_limit  # state[-1] = self.city_time / self._time_limit
        self._last_state = None  # node state at last time
        self._last_order_list = None  # node orders at last time
        self._service_type = service_type  #  True for Joint Fleet Management and Order Dispatching, False for Order Dispatching

    @property
    def last_reward(self):
        return self._last_reward

    @property
    def last_order_state(self):
        return self._last_order_state

    @property
    def last_state(self):
        return self._last_state

    @property
    def last_order_list(self):
        return self._last_order_list

    @property
    def dest(self):
        return self._dest

    def get_service_type(self):
        return self._service_type

    def set_service_type(self, service_type):
        self._service_type = service_type

    def get_info_template(self):
        return collections.namedtuple('Status', 'state, order_list, order, reward, dest, node_index')

    def get_state(self):
        # state = np.zeros(3)
        # state[0] = self.node.idle_driver_num
        # state[1] = self.node.order_num
        state = self.node.get_state()
        state[-1] = self.city_time / self._time_limit
        return state

    def set_position(self, node):
        self.node = node


    def set_order_start(self, order):
        self.order = order

    def set_order_finish(self):
        self.order = None
        self.onservice = False

    def get_driver_id(self):
        return self._driver_id

    def get_node_id(self):
        return self.node.get_node_index()

    def update_city_time(self):
        self.city_time += 1

    def set_city_time(self, city_time):
        self.city_time = city_time

    def set_offline(self):
        """
        nodewise: idle, offline driver_num
        :return:
        """
        assert self.onservice is False and self.online is True
        self.online = False
        self.node.idle_driver_num -= 1
        self.node.offline_driver_num += 1
        if self.get_service_type():
            self.node.fleet_driver_num -= 1
        else:
            pass

    def set_offline_for_start_dispatch(self):

        assert self.onservice is False
        self.online = False

    def set_online(self):
        assert self.onservice is False
        self.online = True
        self.node.idle_driver_num += 1
        self.node.offline_driver_num -= 1
        if self.get_service_type():
            self.node.fleet_driver_num += 1

    def set_online_for_finish_dispatch(self):

        self.online = True
        assert self.onservice is False

    def set_dest(self, dest):
        self._dest = np.array(dest) / np.clip(np.sum(dest), 1, np.inf)


    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
        """
        assert self.online is True
        assert self.node is order.get_begin_position()
        self._last_order_state = order.get_state()
        self.set_order_start(order)
        self.onservice = True
        self.node.idle_driver_num -= 1
        self._reward = order.get_price()
        self._last_reward = self._reward
        self._last_state = self.node.get_state()

        self._last_order_list = []
        for order in self.node.orders:
            self._last_order_list.append(order.get_state())
        if len(self._last_order_list) == 0:
            self._last_order_list.append([self.node.get_node_index(), self.node.get_node_index(), 0, 1])

    def status_control_eachtime(self, city):
        """Update the status of driver

        Note
        ----
            1) If the driver finished its order serving
            2) then switch the status from onservice to False
            3) then update the node which the driver in
            4) Add the number of drivers in the city
        """
        assert self.city_time == city.city_time
        if self.onservice is True:
            assert self.online is True
            order_end_time = self.order.get_assigned_time() + self.order.get_duration()
            if self.city_time == order_end_time:
                self.set_position(self.order.get_end_position())
                self.set_order_finish()
                self.node.add_driver(self._driver_id, self)
                city.n_drivers += 1
                if self.get_service_type():
                    city.n_fleet_drivers += 1
            elif self.city_time < order_end_time:
                pass
            else:
                raise ValueError(
                    'Driver: status_control_eachtime(): order end time less than city time')


class Order(object):
    __slots__ = ('_begin_p', '_end_p', '_begin_t',
                 '_t', '_p', '_waiting_time', '_assigned_time', '_served', '_service_type')

    def __init__(self, begin_position, end_position, begin_time, duration, price, wait_time=1, service_type=-1):
        self._begin_p = begin_position  # node
        self._end_p = end_position      # node
        self._begin_t = begin_time
        # self._end_t = end_time
        self._t = duration              # the duration of order.
        self._p = price
        self._waiting_time = wait_time  # a order can last for "wait_time" to be taken
        self._assigned_time = -1
        self._served = False
        self._service_type = service_type  # 1+ for fleet management and represents neighbor node, 0 for staying, -1 for order dispatching

    @staticmethod
    def state_space():
        return tuple([5])

    @property
    def served(self):
        return self._served

    def get_service_type(self):
        return self._service_type

    def set_service_type(self, service_type):
        self._service_type = service_type

    def get_begin_position(self):
        return self._begin_p

    def get_begin_position_id(self):
        return self._begin_p.get_node_index()

    def get_end_position(self):
        return self._end_p

    def get_end_position_id(self):
        return self._end_p.get_node_index()

    def get_begin_time(self):
        return self._begin_t

    def set_assigned_time(self, city_time):
        self._served = True
        self._assigned_time = city_time

    def get_assigned_time(self):
        return self._assigned_time

    # def get_end_time(self):
    #     return self._end_t

    def get_duration(self):
        return self._t

    def get_price(self):
        return self._p * 0.1

    def get_wait_time(self):
        return self._waiting_time

    def get_state(self):
        start = self._begin_p.get_node_index()
        end = -1 if self._end_p is None else self._end_p.get_node_index()
        return [start, end, self._p * 0.1, self._t, self._service_type]

    def get_begin_position_entropy(self):
        start_entropy = self._begin_p.get_entropy()
        return start_entropy

    def get_end_position_entropy(self):
        if self._end_p is not None:
            end_entropy = self._end_p.get_entropy()
        else:
            end_entropy = 0
        return end_entropy

    def get_feature(self, global_entropy):
        """Feature for order ranking"""
        start_entropy = self.get_begin_position_entropy()
        start_entropy = abs(global_entropy - start_entropy)
        end_entropy = self.get_end_position_entropy()
        end_entropy = abs(global_entropy - end_entropy)
        # enlarge end_entropy - start_entropy
        # enlarge price
        # minify duration
        # prior real_order
        return [end_entropy - start_entropy, self._p * 0.1, -self._t, -self._service_type]
