import itertools
import numpy as np
import collections

from tools.distribution import GaussianDistribution, RandomDistribution, Normal, LimitLayers


class Entity(object):
    def __init__(self, grid, index):
        # state
        self.grid = grid
        self.index = index


class Experience(object):
    def __init__(self, name_list):
        self.name_list = name_list
        for e in name_list:
            setattr(self, e, [])

    def append(self, **kwargs):
        for key, value in kwargs.items():
            getattr(self, key).append(value)

    def pop_all(self):
        res = dict(zip(self.name_list, [[]
                                        for _ in range(len(self.name_list))]))
        for e in self.name_list:
            res[e] = getattr(self, e)
        return res


class Grid(object):
    def __init__(self, grid_id, map_vector, world):
        self.blues = dict()
        self.reds = []
        self.grid_id = grid_id
        self.grid_policy = []
        self.world = world

        self._red_num = 0
        self._blue_num = 0
        self._last_state = None
        self._last_red_list = None
        self._fake_red_num = 0  # the reds which will not bring rewards
        self._map_dim = map_vector
        self._id_emb = np.zeros(map_vector)

        self._id_emb[grid_id] = 1.

    @property
    def index(self):
        return self.grid_id

    @staticmethod
    def state_space(map_dim):
        """The state space has 4 dimension, in detail:
        {grid_id[map_dim], red_num, blue_num, destination_dist[map_dim]}"""
        return tuple([3 + 1 * map_dim])

    @property
    def state(self):
        red_dist = np.zeros(self._map_dim)
        for red in self.reds:
            red_dist[red.destination.index] += 1.
        red_dist /= max(1., self.n_reds)
        return np.concatenate([red_dist, np.array([self.n_reds, self.n_blues, self.grid_id])])

    @property
    def last_state(self):
        return self._last_state

    @property
    def last_red_list(self):
        return self._last_red_list

    @property
    def local_reward(self):
        """Local reward is the expectation of response rate"""
        return 0.

    @property
    def n_blues(self):
        # if self._blue_num != len(self.blues):
        #     print(self._blue_num, len(self.blues), self.blues)
        return self._blue_num
        # return len(self.blues)

    @property
    def n_reds(self):
        return self._red_num

    @property
    def n_fake_reds(self):
        return self._fake_red_num

    def set_policy(self, policy):
        for _blue in self.blues.values():
            _blue.set_policy(policy)

    def record_history(self):
        self._last_state = self.state
        self._last_red_list = []
        for red in self.reds:
            self._last_red_list.append(red.state)

    def add_blue(self, blue):
        self.blues[blue.index] = blue
        self._blue_num += 1

    def add_red(self, red):
        self.reds.append(red)
        self._red_num += 1

    def add_fake_reds(self):
        node_num = self._map_dim
        units = 1
        for i in range(self.n_blues):
            node_id = np.random.choice(node_num, 1)[0]
            # self.reds.append(Red(self, self.world.nodes[node_id], self._red_num, price=0.))
            self.reds.append(Red(self, self, self._red_num, price=0.))
            self._fake_red_num += units
            self._red_num += units

    def get_reds_state(self):
        """Return list of red states and list of red ids"""
        red_index, red_states = [], []
        grid_id = self.grid_id
        for i, _red in enumerate(self.reds):
            if not _red.occupied:
                red_states.append(_red.state)
                red_index.append((grid_id, i))
        return red_index, red_states

    def get_red_list(self):
        red_list = []
        for red in self.reds:
            red_list.append(red.state)
        return red_list

    def assign(self, reds, time):
        gmv = 0.
        ori_fake_red = self._fake_red_num
        ori_all_red = self._red_num
        blue_idx = []
        served_reds = []

        for _red in reds:
            if self.n_blues <= 0:
                break
            self._blue_num -= 1

            if _red.fee == 0:
                self._fake_red_num -= 1

            self._red_num -= 1
            # self._blue_num -= 1
            served_reds.append(_red)
            gmv += _red.fee

            for key, blue in self.blues.items():
                if not blue.assigned:
                    blue.take_red(_red)
                    blue_idx.append(blue.index)
                    break

        self.reds = [_red for _red in self.reds if _red not in served_reds]
        assert self._red_num == len(self.reds)
        real_finished = ori_all_red - ori_fake_red - self._red_num + self._fake_red_num
        return gmv, ori_all_red - ori_fake_red, real_finished, blue_idx, ori_fake_red - self._fake_red_num, ori_fake_red

    def clean_red(self):
        self.reds = []
        self._red_num = 0
        self._fake_red_num = 0

    def reset(self):
        self.reds = []
        self.blues = {}
        self._blue_num = 0
        self._red_num = 0
        self._fake_red_num = 0


class Red(Entity):
    color = [229 / 255, 132 / 255, 129 / 255]
    name = 'red'

    def __init__(self, source, destination, index=1, price=None):
        super().__init__(source, index)
        self._source = source
        self._destination = destination
        if price is None:
            self._fee = float(abs(destination.index - source.index)) + 0.001  # this bias is designed for add fake reds
        else:
            self._fee = price
        self._occupied = False
        self.grid_id = source.index

    def __repr__(self):
        return '<Red: node_id ({}) index ({})>'.format(self._source.index, self.index)

    @staticmethod
    def state_space():
        return tuple([3])

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    @property
    def fee(self):
        return self._fee

    @property
    def occupied(self):
        return self._occupied

    @property
    def state(self):
        """Include {destination, duration}"""
        return [self.source.index, self.destination.index, self._fee * 0.1]

    def set_taken(self):
        assert self._occupied == False
        self._occupied = True

    def set_fee(self, fee):
        self._fee = fee


class Blue(Entity):
    color = [96 / 255, 198 / 255, 248 / 255]
    name = 'agent'

    def __init__(self, grid, index=0):
        super().__init__(grid, index)
        self.reward = 0.
        self.red = None
        self.node = grid
        self.time = 0
        self.node_id = index
        self.grid_id = grid.index

        self._last_reward = 0.
        self._last_red_state = None
        self._policy = None
        self._assigned = False
        self._last_state = None
        self._last_red_list = None

    @property
    def assigned(self):
        return self._assigned

    @property
    def last_reward(self):
        return self._last_reward

    @property
    def last_red_state(self):
        return self._last_red_state

    @property
    def last_state(self):
        return self._last_state

    @property
    def last_red_list(self):
        return self._last_red_list

    @property
    def policy(self):
        return self._policy

    def set_policy(self, policy):
        self._policy = np.array(policy) / np.clip(np.sum(policy), 1, np.inf)

    def get_info_template(self):
        return collections.namedtuple('Status', 'state, reds, red, reward, dist_dispatching, node_index')

    def get_observation(self):
        """Return state of current grid"""
        assert isinstance(self.node, Grid)
        state = self.node.get_state()
        return state

    def set_position(self, node: Grid):
        self.node = node

    def set_time(self, time):
        self.time = time

    def take_red(self, red: Red):
        self.red = red
        self.reward = 0.1 * red.fee
        self.red.set_taken()  # occupy this red spot
        self._assigned = True
        self._last_red_state = red.state
        self._last_reward = 0.1 * red.fee
        assert isinstance(self.node, Grid)
        self._last_state = self.node.state
        self._last_red_list = self.node.get_red_list()
        assert len(self._last_red_list) >= 1

    def update_state(self):
        """Update blue state"""
        self.node = self.red.destination
        self.grid_id = self.node.index
        self.red = None
        self.reward = 0.
        self._assigned = False
        self.time += 1


class World(object):
    def __init__(self, width, height, time_limit, blue_max, red_max):
        # === PRIVATE PARAMETERS ===
        self._time = 0
        self._time_limit = time_limit

        # === ENTITIES ===
        self.width = width
        self.height = height
        self.map_size = width * height
        self.blue_max = blue_max
        self.red_max = red_max

        self.nodes = None
        self.reds = None
        self.blues = None

        # === OTHERS ===
        self.dist_blue = None
        self.dist_red = None
        self.dist_distance = None

        self.nodes = [Grid(i, self.map_size, self) for i in range(self.map_size)]

    def reset(self):
        self._time = 0
        self.dist_blue = GaussianDistribution(self.width, self.height, n_peak=1, max_range=self.blue_max, sigma=np.array([[1., 0.], [0., 1.]]))
        self.dist_red = GaussianDistribution(self.width, self.height, n_peak=1, max_range=self.red_max, sigma=np.array([[1., 0.], [0., 1.]]) * 0.5)
        self.dist_distance = LimitLayers(self.width, self.height, 3)

        for _node in self.nodes:
            _node.reset()

        self.blues = dict()
        self.reds = []

        self.generate_blues()

        self.generate_new_reds()

        states = self.node_states()
        red_idx, red_list = self.red_list()

        return states, red_list, red_idx

    @property
    def n_reds(self):
        res = 0
        for i in range(self.n_grid):
            res += self.nodes[i].n_reds - self.nodes[i].n_fake_reds
        return max(1, res)

    @property
    def n_blues(self):
        res = 0
        for i in range(self.n_grid):
            res += self.nodes[i].n_blues
        return max(1, res)

    @property
    def time(self):
        return self._time

    @property
    def n_grid(self):
        return len(self.nodes)

    @property
    def is_done(self):
        return self._time == self._time_limit - 1

    @property
    def state_space(self):
        return Grid.state_space(self.map_size)

    @property
    def action_space(self):
        return Red.state_space()

    @property
    def entities(self):
        # get all blues and reds
        entities = self.all_blues
        entities.extend(self.all_reds)
        return entities

    def generate_new_reds(self):
        """Generate new reds with ditributution of reds
        """

        n_list = self.dist_red.sample(time=2 * self.time)
        dest = self.dist_distance.sample(n=n_list, focus_grid=np.arange(self.n_grid))
        self.reds = []

        for i in range(self.n_grid):
            self.nodes[i].clean_red()
            for j in range(n_list[i]):
                red = Red(source=self.nodes[i], destination=self.nodes[dest[i][j]], index=j)
                self.nodes[i].add_red(red)
                self.reds.append(red)

        for i in range(self.n_grid):
            # generate fake orders
            self.nodes[i].add_fake_reds()

    def generate_blues(self):
        """Generate blues with current blue dist, the blue dist is
        related with the position and time, a Gaussian with (x, y, t)
        """
        n_list = self.dist_blue.sample(time=2 * self.time + 1)

        for i in range(self.n_grid):
            for j in range(n_list[i]):
                node_id = '{0}_{1}_{2}'.format(self.time, i, j)
                blue = Blue(grid=self.nodes[i], index=node_id)
                self.blues[node_id] = blue
                self.nodes[i].add_blue(blue)

    def record_history(self):
        for _node in self.nodes:
            _node.record_history()

    def record_policy(self):
        for _node in self.nodes:
            policy = [0] * (self.width * self.height)
            for _blue in _node.blues.values():
                if _blue.red is not None:
                    end_point = _blue.red.destination
                    if end_point is not None:
                        policy[end_point.index] += 1
            _node.set_policy(policy)

    def node_states(self):
        """Ignore the nodes which have no any reds in it (fake and real)"""
        res = []
        for i, _node in enumerate(self.nodes):
            if _node.n_reds == 0:
                continue
            res.append(_node.state)

        return np.array(res)

    def red_list(self):
        red_idx, red_state = [], []
        for i, _node in enumerate(self.nodes):
            assert isinstance(_node, Grid)
            if _node.n_reds == 0:
                continue
            idx, states = _node.get_reds_state()
            red_state.append(states)
            red_idx.append(idx)
        return red_idx, red_state

    def migrate_blues(self):
        for _node in self.nodes:
            non_key = []
            for key, _blue in _node.blues.items():
                if _blue.assigned:
                    non_key.append(key)
                    _node.blues[key] = None
                    target_node = _blue.red.destination
                    target_node.add_blue(_blue)
                    _blue.update_state()

            for key in non_key:
                if _node.blues[key] is None:
                    _node.blues.pop(key)

    def step(self):
        """Update state of agent (blue)"""
        self._time += 1
        self.migrate_blues()
        self.generate_new_reds()

        states = self.node_states()
        red_idx, red_list = self.red_list()

        return states, red_idx, red_list

    def get_reds_by_id(self, id_pairs):
        reds = []
        for _red_id_pairs in id_pairs:
            tmp = []
            for _red_id_pair in _red_id_pairs:
                node = self.nodes[_red_id_pair[0]]
                tmp.append(node.reds[_red_id_pair[1]])
            reds.append(tmp)
        return reds

    def get_distribution(self):
        red_dist = self.get_red_dist()
        blue_dist = self.get_blue_n_dist()

        return np.concatenate([red_dist, blue_dist], axis=1)

    def get_red_dist(self):
        n_reds = np.zeros((len(self.nodes), 1))
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                n_reds[i][0] = _node.n_reds - _node.n_fake_reds
        return np.clip(n_reds / np.maximum(np.sum(n_reds), 1), 1e-5, 1.)

    def get_blue_dist(self):
        n_blues = np.zeros((len(self.nodes), 1))
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                n_blues[i][0] = _node.n_blues
        return np.clip(n_blues / np.maximum(np.sum(n_blues), 1), 1e-5, 1.)
        # return np.clip(n_blues, 1., np.inf)

    def get_blue_n_dist(self):
        n_blues = np.zeros((len(self.nodes), 1))
        for i, _node in enumerate(self.nodes):
            if _node is not None:
                n_blues[i][0] = _node.n_blues
        # return np.clip(n_blues / np.maximum(np.sum(n_blues), 1), 1e-5, 1.)
        return np.clip(n_blues, 1., np.inf)

    def get_grid_state(self, grid_id):
        grid_raw_state = self.nodes[grid_id].state
        grid_raw_state[0] /= self.n_blues
        grid_raw_state[1] /= self.n_reds

        grid_one_hot = np.zeros(self.n_grid)
        grid_one_hot[grid_id] = 1.

        return np.concatenate([grid_raw_state, grid_one_hot, self.time_one_hot])

    def get_red_states_emb(self, grid_id):
        state_emb = []
        for red in self.reds[grid_id].values():
            tmp = red.state
            state = np.concatenate(
                [Grid.id_emb(self.n_grid, tmp[i]) for i in range(len(tmp))])
            state_emb.append(state)
        return state_emb

    def pos_of_grid(self, grid_id, w, h, unit_size, grid_size):
        width = grid_id // w
        height = (grid_id - width * w) * grid_size
        width *= grid_size

        width_range = (width + unit_size, width + grid_size - unit_size)
        height_range = (height + unit_size, height + grid_size - unit_size)

        x = np.random.uniform(*width_range)
        y = np.random.uniform(*height_range)

        return (x, y)
