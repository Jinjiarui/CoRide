import os
import sys
import random
import time
import numpy as np
import logging
sys.path.append("../")

from Simulator.simulator.objects import *
from Simulator.simulator.utilities import *
# from algorithm import *

# current_time = time.strftime("%Y%m%d_%H-%M")
# log_dir = "/nfs/private/linkaixiang_i/data/dispatch_simulator/experiments/"+current_time + "/"
# mkdir_p(log_dir)
# logging.basicConfig(filename=log_dir +'logger_env.log', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(logger_ch)
RANDOM_SEED = 0  # unit test use this random seed.


class CityReal:
    '''A real city is consists of M*N grids '''

    def __init__(self, mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat, order_time_dist, order_price_dist,
                 l_max, M, N, n_side, time_limit, probability=1.0 / 28, real_orders="", onoff_driver_location_mat="",
                 global_flag="global", time_interval=10, fleet_rate=0.8, fleet_help=False):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        the city at each time
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :param fleet_help: trigger for joint order dispatching and fleet management
        :return:
        """
        # City.__init__(self, M, N, n_side, time_interval)
        self.M = M  # row numbers
        self.N = N  # column numbers
        # a list of nodes: node id start from 0
        self.nodes = [Node(i) for i in range(M * N)]
        # driver[driver_id] = driver_instance, driver_id start from 0
        self.drivers = {}
        # total idle number of drivers. online and not on service.
        self.n_drivers = 0
        self.n_offline_drivers = 0  # total number of offline drivers.
        self.n_fleet_drivers = 0  # total number of fleet drivers
        self.construct_map_simulation(M, N, n_side)
        self.city_time = 0
        # self.idle_driver_distribution = np.zeros((M, N))
        self.n_intervals = 1440 // time_interval
        self.n_nodes = self.M * self.N
        self.n_side = n_side
        self.order_response_rate = 0.
        self.fake_response_rate = 0.
        self.fleet_response_rate = 0.
        self.ori_idle = 0  # number of idle drivers in real node
        self.ori_fleet = 0  # number of fleet drivers in real node
        self.ori_order_num = 0  # number of order
        self.gmv = 0
        self.fleet_rate = fleet_rate  # rate of  driver conduct fleet management setting
        self.fleet_help = fleet_help  # trigger for joint order dispatching and fleet management

        self.RANDOM_SEED = RANDOM_SEED
        self.state_space = None  # state space
        self.feature_space = None  # feature space

        # Start from 1. The max number of layers an order can across.
        self.l_max = l_max
        assert l_max <= M - 1 and l_max <= N - 1
        assert 1 <= l_max <= 9   # Ignore orders less than 10 minutes and larger than 1.5 hours

        self.target_grids = []
        self.n_valid_grids = 0  # num of valid grid
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        self.construct_node_real(mapped_matrix_int)
        self.mapped_matrix_int = mapped_matrix_int

        self.construct_map_real(n_side)
        self.order_num_dist = order_num_dist
        self.distribution_name = "Poisson"
        self.idle_driver_dist_time = idle_driver_dist_time
        self.idle_driver_location_mat = idle_driver_location_mat

        self.order_time_dist = order_time_dist[:l_max] / np.sum(order_time_dist[:l_max])
        self.order_price_dist = order_price_dist
        self._current_dist = None  # order and driver distribution for policy training
        self._entropy = 0  # entropy for policy training
        self._global_entropy = 0  # global entropy

        target_node_ids = []
        target_grids_sorted = np.sort(
            mapped_matrix_int[np.where(mapped_matrix_int >= 0)])
        for item in target_grids_sorted:
            x, y = np.where(mapped_matrix_int == item)
            target_node_ids.append(ids_2dto1d(x, y, M, N)[0])
        self.target_node_ids = target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.

        self.node_mapping = {}
        self.layer_neighborhood = dict()
        self.construct_mapping()
        self.construct_neighborhood_layerwise()

        # serve_type will be added in data preprocessing
        self.real_orders = real_orders  # 4 weeks' data
        # [[92, 300, 143, 2, 13.2],...] origin grid, destination grid, start time, end time, price.

        self.p = probability   # sample probability
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2017, 9, 1, 0), datetime(2017, 9, 2, 0),
                                         timedelta(minutes=time_interval))]
        self.day_orders = []  # one day's order.

        self.onoff_driver_location_mat = onoff_driver_location_mat

        # States
        self.all_grids_on_number = 0  # current online # drivers.
        self.all_grids_off_number = 0

        self.out_grid_in_orders = np.zeros(
            (self.n_intervals, len(self.target_grids)))
        self.global_flag = global_flag
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]
        self._time_limit = time_limit

    @property
    def real_node(self):
        res = []
        for _node in self.nodes:
            if _node is not None:
                res.append(_node)
        return res

    def get_state_space_node(self):
        if self.state_space == None:
            self.state_space = 5 + self.l_max * 3
        return self.state_space

    def get_action_space_node(self):
        if self.state_space == None:
            self.state_space = 5 + self.l_max * 3
        return self.state_space

    def get_state_space(self):
        return Node.state_space()

    def get_action_space(self):
        return Node.state_space()

    def get_feature_space(self):
        if self.feature_space == None:
            self.feature_space = 4
        return self.feature_space


    def get_node_ids(self):
        vaild_nodes = []
        for _node in self.nodes:
            if _node is not None:
                assert isinstance(_node, Node)
                vaild_nodes.append(_node.get_node_index())
        return vaild_nodes


    def construct_map_simulation(self, M, N, n):
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon
        """
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:
                i, j = ids_1dto2d(idx, M, N)
                current_node.set_neighbors(
                    get_neighbor_list(i, j, M, N, n, self.nodes))

    def construct_mapping(self):
        """
        :return:
        """
        target_grid_id = self.mapped_matrix_int[np.where(
            self.mapped_matrix_int > 0)]
        for g_id, n_id in zip(target_grid_id, self.target_grids):
            self.node_mapping[g_id] = n_id

    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_in will be initialized.
        """
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)

        target_ids = []  # start from 0.
        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id] = Node(node_id)  # node id start from 0.
            target_ids.append(node_id)

        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id].get_layers_neighbors(
                self.l_max, self.M, self.N, self)

        self.target_grids = target_ids
        self.n_valid_grids = len(target_ids)

    def construct_map_real(self, n_side):
        """Build node connection.
        """
        for idx, current_node in enumerate(self.nodes):
            i, j = ids_1dto2d(idx, self.M, self.N)
            if current_node is not None:
                current_node.set_neighbors(get_neighbor_list(
                    i, j, self.M, self.N, n_side, self.nodes))

    def initial_order_random(self, distribution_all, dis_paras_all):
        """Initialize order distribution
        :param distribution: 'Poisson', 'Gaussian'
        :param dis_paras:     lambda,    mu, sigma
        """
        for idx, node in enumerate(self.nodes):
            if node is not None:
                node.order_distribution(
                    distribution_all[idx], dis_paras_all[idx])

    def initial_driver_fleet(self):
        """Initialize drivers for fleet management"""
        for _node in self.nodes:
            if _node is not None:
                assert isinstance(_node, Node)
                _node.initial_fleet_drivers(self.fleet_rate, self.fleet_help)

    def construct_neighborhood_layerwise(self):
        vaild_node_index = []
        # get neighbor node layerwise
        for _row in range(self.M):
            for _column in range(self.N):
                if self.mapped_matrix_int[_row][_column] != -100:
                    vaild_node_index.append(self.mapped_matrix_int[_row][_column])
                    i, j = ids_1dto2d(self.mapped_matrix_int[_row][_column], self.M, self.N)
                    # build layer neighbors according to get_layers_neighbors list
                    # initialize layer_neighborhood
                    _layer_neighbors = dict()
                    for _layer in range(self.l_max):
                        _layer_neighbors[_layer] = []
                    for _layer in range(self.l_max):
                        for _node in get_layers_neighbors(i, j, self.l_max, self.M, self.N)[_layer]:
                            _index = ids_2dto1d(_node[0], _node[1], self.M, self.N)
                            _layer_neighbors[_layer].append(_index)
                    self.layer_neighborhood[self.mapped_matrix_int[_row][_column]] = _layer_neighbors
        # drop invalid nodes
        for _layers in self.layer_neighborhood.values():
            for _layer in _layers.values():
                for _node in _layer:
                    if _node not in vaild_node_index:
                        _layer.remove(_node)



    def step_get_global_observation(self):
        assert self.fleet_help == False
        """Return the state of global grid world: Only for Order Dispatching
        global_orders=[[order_id, begin_p, end_p, _t],...]
        """
        global_orders = []
        for _node in self.nodes:
            if _node is not None:
                if _node.order_num > 0:
                    for _order_idx, _order in enumerate(_node.orders):
                        # combine node_idx and node.order_idx to build unique id
                        _order_id = (_node.get_node_index(), _order_idx)
                        global_orders.append([
                            _order_id, _order.get_begin_position_id(), _order.get_end_position_id(),
                            _order.get_price(), _order.get_duration()
                        ])
        return global_orders

    def step_get_node_observation(self):
        """Return the state of each grid, includes the """

        # next_state = np.zeros((4, self.M, self.N))
        # for _node in self.nodes:
        #     if _node is not None:
        #         row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
        #         next_state[0, row_id, column_id] = _node.get_node_index()
        #         next_state[1, row_id, column_id] = _node.idle_driver_num
        #         next_state[2, row_id, column_id] = _node.order_num - _node.fake_order_num
        #         next_state[3, row_id, column_id] = self.city_time / self._time_limit
        #
        # return next_state

        # calculate global entropy
        self.step_set_global_entropy()

        next_state = []
        for _node in self.nodes:
            if _node is not None:
                if _node.order_num > 0:
                    # row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                    _state = [_node.get_node_index(),  # for split
                              _node.idle_driver_num,
                              _node.order_num - _node.fake_order_num - _node.fleet_order_num,
                              _node.get_entropy(),
                              abs(_node.get_entropy() - self._global_entropy),
                            # _node.fleet_driver_num
                              ]
                    _state.extend(_node.get_info_price_distribution(self.l_max))
                    _state.extend(_node.get_info_time_distribution(self.l_max))
                    _state.extend(_node.get_info_end_layer_distribution(self.layer_neighborhood, self.l_max))
                    next_state.append(np.array(_state))

                    # set observe space
                    if self.state_space == None:
                        self.state_space = len(_state)
                else:
                    _state = [_node.get_node_index(),
                              _node.idle_driver_num,
                              0,
                              _node.get_entropy(),
                              abs(_node.get_entropy() - self._global_entropy),
                            # _node.fleet_driver_num
                              ]
                    _zeros = [0 for _ in range(self.l_max)]
                    _state.extend(_zeros)
                    _state.extend(_zeros)
                    _state.extend(_zeros)
                    next_state.append(np.array(_state))

        return next_state

    def step_get_observation(self):
        next_state = []
        for _node in self.nodes:
            if _node is not None:
                if _node.order_num > 0:
                    next_state.append(np.array([
                        _node.get_index(),
                        _node.idle_driver_num,
                        _node.order_num - _node.fake_order_num - _node.fleet_order_num,
                        self.city_time / self._time_limit,
                        _node.fleet_driver_num
                    ]))

                else:
                    next_state.append(np.array([
                        _node.get_index(),
                        _node.idle_driver_num,
                        0,
                        self.city_time / self._time_limit,
                        _node.fleet_driver_num
                    ]))
        return next_state


    def get_agent_observation(self):
        next_state, driver_id = [], []
        for _node in self.nodes:
            node_id = 1 if _node is None else _node._index
            driver_num = 0
            if _node is not None:
                states, ids = [], []
                for key, _driver in _node.drivers.items():
                    states.append(_driver.get_observation())
                    driver_num += 1
                    # ids.append(key)
                if len(states) < 1:
                    states.append(_node.get_state())
                next_state.append(states)
            # driver_id.append(ids)
            # print('-------> Node {} has driver {}'.format(node_id, driver_num))

        return next_state

    def get_order_list(self):
        self.step_set_global_entropy()
        order_idx, order_state, order_feature = [], [], []
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                assert isinstance(_node, Node)
                idx, states, feature, feature_space = _node.get_orders_state(self._global_entropy)
                if len(states) == 0:
                    continue
                order_state.append(states)
                order_idx.append(idx)
                order_feature.append(feature)
                if self.feature_space == None:
                    self.feature_space = feature_space
        return order_idx, order_state, order_feature


    def get_sample_order_list(self):
        order_idx, order_state, order_feature = [], [], []
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                assert isinstance(_node, Node)
                idx, states, feature = _node.get_sample_orders_state()
                if len(states) == 0:
                    continue
                order_state.append(states)
                order_idx.append(idx)
                order_feature.append(feature)
        return order_idx, order_state, order_feature

    def get_num_idle_drivers(self):
        """ Compute idle drivers
        :return:
        """
        temp_n_idle_drivers = 0
        for _node in self.nodes:
            if _node is not None:
                temp_n_idle_drivers += _node.idle_driver_num
        return temp_n_idle_drivers

    def get_dist_of_idle_driver(self):
        """ Get the distribution of idle drivers"""
        n_idle_drivers = np.zeros((len(self.nodes), 1))
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                n_idle_drivers[i][0] = _node.idle_driver_num
        return np.clip(n_idle_drivers / np.maximum(np.sum(n_idle_drivers), 1), 1e-5, 1.)

    def get_dist_of_idle_n_driver(self):
        n_idle_drivers = np.zeros((len(self.nodes), 1))
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                n_idle_drivers[i][0] = _node.idle_driver_num
        return np.clip(n_idle_drivers, 1., np.inf)

    def get_dist_of_orders(self):
        """Get the distribution of orders"""
        n_orders = np.zeros((len(self.nodes), 1))
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                n_orders[i][0] = _node.order_num - _node.fake_order_num - _node.fleet_order_num
        return np.clip(n_orders / np.maximum(np.sum(n_orders), 1), 1e-5, 1.)

    def get_all_driver_ids(self):
        driver_ids = []
        for _node in self.nodes:
            assert isinstance(_node, Node)
            driver_ids.extend(list(_node.drivers.keys()))
        return driver_ids

    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        next_state = np.zeros((self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(
                    _node.get_node_index(), self.M, self.N)
                next_state[row_id, column_id] = _node.get_idle_driver_numbers_loop()

        return next_state

    def check_nodewise(self):
        for _node in self.nodes:
            if _node is not None:
                assert isinstance(_node, Node)
                order_num, real_order_num, fake_order_num, fleet_order_num = _node.get_order_numbers_loop()
                idle_driver_num, fleet_driver_num = _node.get_driver_numbers_loop()
                assert order_num == _node.order_num
                assert fake_order_num == _node.fake_order_num
                assert fleet_order_num == _node.fleet_order_num
                # assert _node.orders_samples_num == real_order_num + idle_driver_num * 7
                assert idle_driver_num == _node.idle_driver_num
                assert fleet_driver_num == _node.fleet_driver_num




    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset(self):
        """ Return initial observation: get order distribution and idle driver distribution
        """

        _M = self.M
        _N = self.N
        # clean orders and drivers
        # driver[driver_id] = driver_instance  , driver_id start from 0
        self.city_time = 0
        self.drivers = {}
        # total idle number of drivers. online and not on service.
        self.n_drivers = 0
        self.n_fleet_drivers = 0
        self.n_offline_drivers = 0  # total number of offline drivers.
        for node in self.nodes:
            if node is not None:
                node.clean_node()

        # assert self.city_time == 0
        # initialization drivers according to the distribution at time 0
        # num_idle_driver = self.utility_get_n_idle_drivers_real()

        #self.step_driver_online_offline_control(num_idle_driver)

        self.step_driver_online_offline_nodewise()

        # initialize drivers for fleet management
        if self.fleet_help:
            self.initial_driver_fleet()

        # generate orders at first time step
        distribution_name = [self.distribution_name] * (_M * _N)
        distribution_param_dictionary = self.order_num_dist[self.city_time]
        distribution_param = [0] * (_M * _N)
        for key, value in distribution_param_dictionary.items():
            if self.distribution_name == 'Gaussian':
                mu, sigma = value
                distribution_param[key] = mu, sigma
            elif self.distribution_name == 'Poisson':
                mu = value[0]
                distribution_param[key] = mu
            else:
                print("Wrong distribution")

        self.initial_order_random(distribution_name, distribution_param)
        # clean orders for fleet control
        self.step_clean_fleet_fake_orders()
        # regenerate orders for real and fleet control
        # generate fake order node-wisely
        self.step_generate_order_fake()
        self.step_generate_order_real()
        if self.fleet_help:
            self.step_generate_order_fleet()

        # Generate and update sample_order
        self.step_generate_order_sample()

        states_node = self.step_get_node_observation()
        states = self.step_get_observation()# get states and drivers ids of each grid
        global_orders_states = self.step_get_global_observation() # get global states and driver for km

        order_idx, order_list, order_feature = self.get_order_list()  # get order list of each grid
        # sample_order_idx, sample_order_list, sample_order_feature = self.get_sample_order_list()

        return states_node, states, order_list, order_idx, order_feature, global_orders_states


    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        """ 1. bootstrap oneday's order data.
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime
        :return:
        """
        if city_time != "":
            self.city_time = city_time

        # clean orders and drivers
        # driver[driver_id] = driver_instance  , driver_id start from 0
        self.drivers = {}
        # total idle number of drivers. online and not on service.
        self.n_drivers = 0
        self.n_offline_drivers = 0  # total number of offline drivers.
        for node in self.nodes:
            if node is not None:
                node.clean_node()

        # Generate one day's order by sampling real_orders
        if generate_order == 1:
            self.step_generate_order_real()
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])  # Init orders of current time step

        # Init current driver distribution
        if self.global_flag == "global":
            num_idle_driver = self.utility_get_n_idle_drivers_real()
            num_idle_driver = int(num_idle_driver * ratio)
        else:
            num_idle_driver = self.utility_get_n_idle_drivers_nodewise()
        self.step_driver_online_offline_control_new(num_idle_driver)

        # initialize orders and drivers for fleet control
        self.initial_driver_fleet()

        states = self.step_get_node_observation()  # get states and drivers ids of each grid
        global_orders_states, global_drivers_states = self.step_get_global_observation()  # get global states and driver for km

        order_idx, order_list, order_feature = self.get_order_list()  # get order list of each grid

        return states, order_list, order_idx, order_feature, global_orders_states, global_drivers_states

    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        count = 0  # offline driver num
        offline_drivers = []   # record offline driver id
        for key, _driver in self.drivers.items():
            if _driver.online is False:
                count += 1
                offline_drivers.append(_driver.get_driver_id())
        return offline_drivers

    def utility_get_n_idle_drivers_nodewise(self):
        """ compute idle drivers.
        :return:
        """
        time = self.city_time % self.n_intervals
        idle_driver_num = np.sum(self.idle_driver_location_mat[time])
        return int(idle_driver_num)

    def utility_add_driver_real_new(self, num_added_driver):
        """
        :param num_added_driver: number of drivers to add
        :return:
        """
        # curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution = self.get_dist_of_idle_driver()
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time %
                                                                 self.n_intervals, :]

        idle_diff = idle_driver_distribution.astype(
            int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        n_total_drivers = len(self.drivers.keys())
        for ii, node_id in enumerate(node_ids):
            added_driver_id = n_total_drivers + ii
            self.drivers[added_driver_id] = Driver(added_driver_id, self._time_limit)
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_driver(
                added_driver_id, self.drivers[added_driver_id])

        self.n_drivers += num_added_driver

    def utility_add_driver_real_new_offlinefirst(self, num_added_driver):
        # curr_idle_driver_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        # curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution = self.get_dist_of_idle_driver()
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index]) for index in
                                                         self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time %
                                                                 self.n_intervals, :]

        idle_diff = idle_driver_distribution.astype(
            int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        if float(np.sum(idle_diff)) == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):

            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:

                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id, self._time_limit)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(
                    added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1

    def utility_add_driver_real_nodewise(self, node_id, num_added_driver):
        while num_added_driver > 0:
            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:

                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id, self._time_limit)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(
                    added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1
            num_added_driver -= 1

    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):
        while n_drivers_to_off > 0:
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break

    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):
        """
        :param n_drivers_to_off: similar with added above
        :return:
        """
        # curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution = self.get_dist_of_idle_driver()
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time %
                                                                 self.n_intervals, :]

        # diff of curr idle driver distribution and history
        idle_diff = curr_idle_driver_distribution_resort - \
            idle_driver_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_drivers_can_be_off = int(
            np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:

            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1

    def utility_bootstrap_oneday_order(self):
        # self.p is possibility for binomial
        num_all_orders = len(self.real_orders)
        index_sampled_orders = np.where(
            np.random.binomial(1, self.p, num_all_orders) == 1)
        #one_day_orders = self.real_orders[index_sampled_orders]
        one_day_orders = []
        for i in list(index_sampled_orders):
            for ii in i:
                one_day_orders.append(self.real_orders[ii])

        self.out_grid_in_orders = np.zeros(
            (self.n_intervals, len(self.target_grids)))

        day_orders = [[] for _ in np.arange(self.n_intervals)]
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2, serve_type]
            start_time = int(iorder[2])
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys():
                continue
            start_node = self.node_mapping.get(iorder[0], -100)
            end_node = self.node_mapping.get(iorder[1], -100)
            duration = int(iorder[3])
            price = iorder[4]

            # select out of grids orders
            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) %
                                        self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append(
                [start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders

    def step_driver_status_control(self):
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        for key, _driver in self.drivers.items():
            _driver.status_control_eachtime(self)
        moment = self.city_time % self.n_intervals
        orders_to_on_drivers = self.out_grid_in_orders[moment, :]
        for idx, item in enumerate(orders_to_on_drivers):
            if item != 0:
                node_id = self.target_grids[idx]
                self.utility_add_driver_real_nodewise(node_id, int(item))

    def step_driver_online_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        moment = self.city_time % self.n_intervals
        # curr_onoff_distribution = self.onoff_driver_location_mat[moment]
        cur_idle_num = self.idle_driver_location_mat[moment]

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        new_drivers = 0
        offline_drivers = 0
        for idx, target_node_id in enumerate(self.target_node_ids):
            # curr_mu = curr_onoff_distribution[idx, 0]
            # curr_sigma = curr_onoff_distribution[idx, 1]
            # on_off_number = np.round(np.random.normal(
            #     curr_mu, curr_sigma, 1)[0]).astype(int)
            on_off_number = cur_idle_num[idx] - self.nodes[target_node_id].idle_driver_num

            if on_off_number > 0:
                # print('Some drivers online ...')
                new_drivers += on_off_number
                self.utility_add_driver_real_nodewise(
                    target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                # print('Some drivers offline ...')
                offline_drivers += on_off_number
                self.utility_set_drivers_offline_real_nodewise(
                    target_node_id, abs(on_off_number))
            else:
                pass

        # print('>>> Pure drivers add:', new_drivers - offline_drivers)

    def step_driver_online_offline_control_new(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """
        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)

        if n_idle_drivers > self.n_drivers:

            self.utility_add_driver_real_new_offlinefirst(
                n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(
                self.n_drivers - n_idle_drivers)
        else:
            pass

    def step_driver_online_offline_control(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """
        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)
        if n_idle_drivers > self.n_drivers:
            # bring drivers online.
            while self.n_drivers < n_idle_drivers:
                if self.n_offline_drivers > 0:
                    for ii in np.arange(self.n_offline_drivers):
                        self.drivers[offline_drivers[ii]].set_online()
                        self.n_drivers += 1
                        self.n_offline_drivers -= 1
                        if self.n_drivers == n_idle_drivers:
                            break

                self.utility_add_driver_real_new(
                    n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(
                self.n_drivers - n_idle_drivers)
        else:
            pass

    def utility_get_n_idle_drivers_real(self):
        """ control the number of idle drivers in simulator;
        :return: transform float into int
        """
        time = self.city_time % self.n_intervals
        mean, std = self.idle_driver_dist_time[time]
        # initial idle driver by design
        # mean, std = self.idle_driver_dist_time[2]
        np.random.seed(self.city_time)
        return np.round(np.random.normal(mean, std, 1)[0]).astype(int)

    def utility_set_neighbor_weight(self, weights):
        self.weights_layers_neighbors = weights



    def step_clean_fleet_fake_orders(self):
        for node in self.nodes:
            if node is not None:
                # clean fleet and fake order generated at last time
                node.clean_order_fleet_fake()


    def step_generate_order_fake(self):
        for _node in self.nodes:
            if _node is not None:
                _node.generate_order_fake(self.city_time)

    def step_generate_order_real(self):
        # generate order at t + 1
        for node in self.nodes:
            if node is not None:
                # node_id = node.get_node_index()
                # generate orders start from each node
                random_seed = node.get_node_index() + self.city_time
                node.generate_order_real(self.l_max, self.order_time_dist, self.order_price_dist,
                                         self.city_time, self.nodes, random_seed)

    def step_generate_order_fleet(self):
        # generate fleet order
        for node in self.nodes:
            if node is not None:
                node.generate_order_fleet(self.city_time, self.nodes)

    def step_generate_order_sample(self):
        # update sample order
        for _node in self.nodes:
            if _node is not None:
                _node.generate_order_sample(self.city_time, self.nodes)


    def step_bootstrap_order_real(self, day_orders_t):
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.nodes[start_node_id]

            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                end_node = None
            start_node.add_order_real(
                self.city_time, end_node, iorder[3], iorder[4])

    def step_assign_order(self):
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        fake_order_num = 0
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(
                    self.city_time, self)
                reward += reward_node
                all_order_num += all_order_num_node
                fake_order_num = node.fake_order_num
                finished_order_num += finished_order_num_node
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num - fake_order_num)
        else:
            self.order_response_rate = -1
        return reward

    def step_assign_order_broadcast_neighbor_reward_update(self, dispatch_action, neighbor_help=False):
        """ Consider the orders whose destination or origin is not in the target region
        """
        # node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        rewards = []
        serve_driver_ids = []
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        all_fake_order_num = 0
        all_fleet_order_num = 0
        all_idle = 0
        all_fleet = 0
        fake_fin = 0
        fleet_fin = 0
        self.gmv = 0

        for _orders in dispatch_action:
            node = _orders[0].get_begin_position()
            # Interpretation:
            # 1) node_rewards: list of reward
            # 2) all_order_num_node: the number of real order
            # 3) finished_order_num_node: the number of finished real order
            # 4) servers: id of drivers who served orders
            # 5) fake_fin_node: the number of finished fake order
            # 6) all_fake_node: the number of all fake order in the certain node
            # 7) idle_num_node: the number of idle drivers in the certain node
            # 8) fleet_fin_node: the number of finished fleet order
            # 9) all_fleet_node: the number of all fleet order in the certain node
            # 10) fleet_num_node: the number of fleet drivers in the certain node
            node_rewards, all_order_num_node, finished_order_num_node, servers, fake_fin_node, fleet_fin_node, all_fake_node, all_fleet_node, idle_num_node, fleet_num_node = node.simple_order_assign_real(_orders, self.city_time, self, self.fleet_help)
            self.gmv += np.sum(node_rewards)
            all_order_num += all_order_num_node
            finished_order_num += finished_order_num_node
            fake_fin += fake_fin_node
            fleet_fin += fleet_fin_node
            all_fake_order_num += all_fake_node
            all_fleet_order_num += all_fleet_node
            all_idle += idle_num_node
            all_fleet += fleet_num_node
            # node_reward[node.get_node_index()] += reward_node
            if len(servers) > 0:
                rewards.extend(node_rewards)
                serve_driver_ids.extend(servers)
        # Neighbor order dispatching, if you wanna use, trigger it on
        # Second round broadcast
        if neighbor_help:
            for node in self.nodes:
                if node is not None:
                    if node.order_num != 0:
                        reward_node_broadcast, finished_order_num_node_broadcast \
                            = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                        reward += reward_node_broadcast
                        finished_order_num += finished_order_num_node_broadcast

        if all_order_num != 0:
            self.order_response_rate = finished_order_num / max(1., float(all_order_num))
            self.fake_response_rate = fake_fin / max(1., float(all_fake_order_num))
            self.fleet_response_rate = fleet_fin / max(1., float(all_fleet_order_num))
            self.ori_idle = all_idle
            self.ori_order_num = all_order_num
            self.ori_fleet = all_fleet
        else:
            self.order_response_rate = -1

        # return reward, [node_reward, neighbor_reward]
        return serve_driver_ids, rewards

    def step_assign_fake_orders(self, city, city_time):
        """Assign fake orders which are Order(departure, departure, time, 0, 0)s"""
        servers = []
        for _node in self.nodes:
            if _node is not None:
                serve_driver_idx = []
                for key, assigned_driver in self.drivers.items():
                    if not assigned_driver.onservice and assigned_driver.online:
                        fake_order = Order(_node, _node, city_time, 1, 0, 0)
                        assigned_driver.take_order(fake_order)
                        fake_order.set_assigned_time(city_time)
                        city.n_drivers -= 1
                        serve_driver_idx.append(assigned_driver.get_driver_id())
                if len(serve_driver_idx) > 0:
                    servers.extend(serve_driver_idx)
        return servers, [0] * len(servers)

    def step_remove_unfinished_orders(self):
        """Remove unserved orders and complished orders"""
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)
                node.remove_unfinished_order_sample(self.city_time)

    def step_pre_order_assign(self, next_state):
        remain_drivers = next_state[0] - next_state[1]
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.array([remain_drivers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        for node in self.nodes:
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_drivers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a - b, 0)
                            remain_drivers_1d[neighbor_id] = max(b - a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                            remain_orders_1d.reshape(self.M, self.N)])
        return context

    def step_dispatch_invalid(self, dispatch_actions):
        """ This is a method for taxi dispatching
        :param dispatch_actions: list, element consists of {start_node_id, end_node_id, num_of_drivers}
        :return: a list contains some tuples (end_node_id, remove_driver_id) denote the vehicles which be dispatched
            to another grid.
        """
        save_remove_id = []
        for action in dispatch_actions:

            start_node_id, end_node_id, num_of_drivers = action
            if self.nodes[start_node_id] is None or num_of_drivers == 0:
                continue  # not a feasible action

            if self.nodes[start_node_id].get_driver_numbers() < num_of_drivers:
                num_of_drivers = self.nodes[start_node_id].get_driver_numbers()

            if end_node_id < 0:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] is None:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:
                raise ValueError('City:step(): not a feasible dispatch')

            for _ in np.arange(num_of_drivers):
                # t = 1 dispatch start, idle driver decrease
                remove_driver_id = self.nodes[start_node_id].remove_idle_driver_random()
                save_remove_id.append((end_node_id, remove_driver_id))
                self.drivers[remove_driver_id].set_position(None)
                self.drivers[remove_driver_id].set_offline_for_start_dispatch()
                self.n_drivers -= 1

        return save_remove_id

    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_driver_id in save_remove_id:
            self.drivers[arrive_driver_id].set_position(
                self.nodes[destination_node_id])
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            self.nodes[destination_node_id].add_driver(
                arrive_driver_id, self.drivers[arrive_driver_id])
            self.n_drivers += 1

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of drivers
        for driver_id, driver in self.drivers.items():
            driver.set_city_time(self.city_time)

    def step_set_distribution(self):
        order_dist = self.get_dist_of_orders()
        # driver_dist = self.get_dist_of_idle_driver()
        driver_dist = self.get_dist_of_idle_n_driver()

        #return np.concatenate([order_dist, driver_dist], axis=1)
        self._current_dist = np.concatenate([order_dist, driver_dist], axis=1)

    def step_get_distribution(self):
        return self._current_dist

    def step_get_entropy(self):
        return self._entropy

    def step_set_global_entropy(self):
        assert self.n_drivers >= 0
        _rho = self.ori_idle / max(1, self.ori_order_num)
        # global_entropy = np.sum(_rho * np.log(_rho))
        self._global_entropy = _rho

    def step_set_entropy(self):
        """Get global_entropy - node_entropy and sum"""
        self._entropy = 0.
        self.step_set_global_entropy()
        node_num = 0
        for _node in self.nodes:
            if _node is not None:
                node_num += 1
                _node.set_entropy()
                self._entropy += abs(self._global_entropy - _node.get_entropy())
        self._entropy = self._entropy / node_num


    def get_orders_by_id(self, id_pairs: list):
        orders = []
        for _order_id_pairs in id_pairs:
            tmp = []
            for _order_id_pair in _order_id_pairs:
                node = self.nodes[_order_id_pair[0]]
                assert isinstance(node, Node)
                tmp.append(node.orders[_order_id_pair[1]])
            orders.append(tmp)
        return orders


    def get_info_manager_node(self, central_id):
        self.step_set_global_entropy()
        i, j = ids_1dto2d(central_id, self.M, self.N)
        neighbors_id = get_layers_neighbors(i, j, 1, self.M, self.N)[0]
        neighbors_id = [ids_2dto1d(id[0], id[1], self.M, self.N) for id in neighbors_id]
        neighbors_id.append(central_id)
        idle_driver_num, real_order_num, node_entropy, entropy, num = 0, 0, 0, 0, 0
        zeros_dist = np.zeros([self.l_max])
        price_dist, time_dist, end_layer_dist = zeros_dist, zeros_dist, zeros_dist
        for node_id in neighbors_id:
            node = self.nodes[node_id]
            assert isinstance(node, Node)
            num += 1
            idle_driver_num += node.idle_driver_num
            real_order_num += node.order_num - node.fake_order_num - node.fleet_order_num
            node_entropy += node.get_entropy()
            entropy += abs(node.get_entropy() - self._global_entropy)
            price_dist += np.array(node.get_info_price_distribution(self.l_max))
            time_dist += np.array(node.get_info_time_distribution(self.l_max))
            end_layer_dist += np.array(node.get_info_end_layer_distribution(self.layer_neighborhood, self.l_max))
        d_info = [central_id,
                  idle_driver_num,
                  real_order_num,
                  node_entropy / num,
                  entropy / num
                  ]
        d_info.extend(list(price_dist / num))
        d_info.extend(list(time_dist / num))
        d_info.extend(list(end_layer_dist / num))

        return d_info


    def get_info_worker_node(self, nodes_id):
        node = self.nodes[nodes_id]
        assert isinstance(node, Node)

        d_info = [nodes_id,
                  node.idle_driver_num,
                  node.order_num - node.fake_order_num - node.fleet_order_num,
                  node.get_entropy(),
                  abs(node.get_entropy() - self._global_entropy)
                  ]
        d_info.extend(node.get_info_price_distribution(self.l_max))
        d_info.extend(node.get_info_time_distribution(self.l_max))
        d_info.extend(node.get_info_end_layer_distribution(self.layer_neighborhood, self.l_max))
        return d_info

    def get_info_of_driver(self, driver_id):
        driver = self.drivers[driver_id]
        assert isinstance(driver, Driver)
        # assert driver.onservice == True
        # assert driver.online == True

        # get node id
        node = driver.node
        # include node_index and dist_dispatching
        assert driver.last_order_state in driver.last_order_list
        d_info = driver.get_info_template()(driver.last_state, driver.last_order_list, driver.last_order_state,
                                            driver.last_reward, driver.dest, node.get_node_index())
        return d_info

    def record_history(self):
        for _node in self.nodes:
            if _node is not None:
                _node.record_history()

    # def record_policy(self):
    #     # check each driver's target
    #     # calculate policy should ignore the offline car
    #     for _node in self.nodes:
    #         if _node is None:
    #             continue
    #         policy = [0] * (self.M * self.N)
    #         for _driver in _node.drivers.values():
    #             if _driver.online and _driver.order is not None:
    #                 end_point = _driver.order.get_end_position()
    #                 if end_point is not None:
    #                     policy[end_point.get_node_index()] += 1
    #         _node.set_policy(policy)

    def step(self, dispatch_actions, generate_order=1):
        """Step method for environment update"""

        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # save_remove_id = self.step_dispatch_invalid(dispatch_actions)
        # When the drivers go to invalid grid, set them offline.

        # reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()
        # record history node-wisely
        self.record_history()

        # get current distribution and entropy
        self.step_set_distribution()
        self.step_set_entropy()

        # 1. Do dispatching and fleet management
        #self.check_nodewise()
        serve_driver_id, rewards = self.step_assign_order_broadcast_neighbor_reward_update(dispatch_actions)

        # 2. Increase city time
        #self.check_nodewise()
        self.step_increase_city_time()

        '''**************************** T = 2 ****************************'''
        # 3. Update status of all driver and drivers finish order become available again
        # but the vehicle distribution will not be updated
        self.step_driver_status_control()
        # 4. Generate real order
        # clean fleet and fake orders generated at last time node-wisely

        # 5. Unable add drivers after initialization
        #self.check_nodewise()
        self.step_driver_online_offline_nodewise()

        # add driver to each node

        #self.check_nodewise()
        self.step_clean_fleet_fake_orders()

        if generate_order == 1:
            #self.check_nodewise()
            self.step_generate_order_real()
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        #self.check_nodewise()
        self.step_remove_unfinished_orders()
        # 6. Generate fleet and fake order
        if self.fleet_help:
            self.step_generate_order_fleet()
        # Generate and Update sample_order
        self.step_generate_order_sample()

        # 7. Move to next states
        order_idx, order_list, order_feature = self.get_order_list()
        next_states_node = self.step_get_node_observation()
        next_states = self.step_get_observation()
        # self.check_nodewise()

        # sample_order_idx, sample_order_list, sample_order_feature = self.get_sample_order_list()
        # return np.array(serve_driver_id, dtype=np.int32), next_states, order_list, order_idx, sample_order_idx, sample_order_list, sample_order_feature

        global_orders_states = self.step_get_global_observation()
        return np.array(serve_driver_id, dtype=np.int32)[:, 1], next_states_node, next_states, order_list, order_idx, order_feature, global_orders_states
