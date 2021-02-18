import os.path as osp
import numpy as np

from copy import copy

from Simulator.imitation_world.core import World, Blue, Red
from tools.distribution import GaussianDistribution, RandomDistribution, Normal, LimitLayers
from settings import BASE_DIR


class Scenario(object):
    def __init__(self, width, height, time_limit):
        self.dist_blue = None
        self.dist_red = None
        self.dist_distance = None
        self.time_limit = time_limit
        self.height = height
        self.width = width
        self.last_blue_n = None

        self._global_state_record = [None for _ in range(self.time_limit)]
        # self._global_mess = [None for _ in range(self.time_limit)]
        self._red_dist_record = [None for _ in range(self.time_limit)]
        self._blue_dist_record = [None for _ in range(self.time_limit)]

    def make_world(self, blue_max, red_max):
        world = World(self.width, self.height, self.time_limit)
        self.reset_world(world, red_max, blue_max)
        return world

    def reset_world(self, world: World, red_max, blue_max):
        """Generate reds"""
        world.init()

        self.dist_blue = GaussianDistribution(self.width, self.height, n_peak=1, max_range=blue_max, sigma=np.array([[1., 0.], [0., 1.]]))
        self.dist_red = GaussianDistribution(self.width, self.height, n_peak=1, max_range=red_max, sigma=np.array([[1., 0.], [0., 1.]]) * 0.5)
        self.dist_distance = LimitLayers(self.width, self.height, 3)

        # === GENARATE BLUE & RED ===
        self.generate_blues(world)
        self.generate_reds(world)

        for i in range(world.n_grid):
            red_states = world.get_reds_state(i)
            for blue in world.blues[i].values():
                blue.orders = red_states

    def generate_reds(self, world: World):
        """Generate new reds with ditributution of reds
        """

        n_list = self.dist_red.sample(time=2 * world.time)
        dest = self.dist_distance.sample(n=n_list)

        for i in range(world.n_grid):
            # world.reds[i]['-1'] = Red(node_id='-1', grid_id=i, source=0, destination=0)
            world.grids[i].n_actived_red = 0  # reset red num
            world.reds[i] = {}
            for j in range(n_list[i]):
                node_id = '{0}_{1}_{2}'.format(world.time, i, j)
                red = Red(node_id=node_id, grid_id=i, source=i, destination=dest[i][j])
                red.fee /= self.width * self.height
                world.reds[i][node_id] = red
            world.grids[i].reds = n_list[i]

    def generate_blues(self, world: World):
        """Generate blues with current blue dist, the blue dist is
        related with the position and time, a Gaussian with (x, y, t)
        """
        n_list = self.dist_blue.sample(time=2 * world.time + 1)

        for i in range(world.n_grid):
            for j in range(n_list[i]):
                node_id = '{0}_{1}_{2}'.format(world.time, i, j)
                blue = Blue(node_id=node_id, grid_id=i)
                blue.state = world.get_grid_state(i)
                world.blues[i][node_id] = blue
            world.grids[i].blues += n_list[i]

    def dispatch_red(self, grid_id, red_id, blue: Blue, world: World, red_grid=None):
        world.grids[grid_id].blues -= 1
        red = None

        if red_grid is None:
            red = world.reds[grid_id].get(red_id, None)
            world.grids[grid_id].n_actived_red += 1
            world.grids[grid_id].reds -= 1
        else:
            red = world.reds[red_grid].get(red_id, None)
            world.grids[red_grid].n_actived_red += 1
            world.grids[red_grid].reds -= 1

        assert isinstance(red, Red)
        blue.to_grid = red.destination
        blue.matched_red = red

        # update dispatched info: to destination
        world.grids[grid_id].grid_policy[world.time][red.destination] += 1.
        world.blue_buffer[grid_id].append(blue)

    def neighbor_dispatch(self, grid_id, node_id, world: World):
        search_grid = [-self.width - 1, -self.width, -self.width + 1, -1, 1, self.width - 1, self.width + 1]
        np.random.shuffle(search_grid)
        for i in search_grid:
            nei = grid_id + i

            if 0 <= nei < self.width * self.height and world.grids[nei].n_blue > world.grids[nei].n_red:
                blue = world.have_ava_blue(nei)
                self.dispatch_red(nei, node_id, blue, world, red_grid=grid_id)
                break

    def match(self, reds_status, world: World):
        """Match red with blues, return blues buffer"""
        # group red_status
        self.last_blue_n = np.zeros(world.n_grid)
        for i in range(world.n_grid):
            self.last_blue_n[i] = world.grids[i].n_blue

        for i, ele in enumerate(reds_status):
            grid_id = ele.grid_id
            node_id = ele.node_id

            ava_blue = world.have_ava_blue(grid_id)  # pop blue from grid[grid_id]
            if ava_blue is not None:
                self.dispatch_red(grid_id, node_id, ava_blue, world)
            else:
                pass

    def reward(self, blue: Blue, world: World):
        """Set reward for each blue"""
        fee_factor = 0.5
        blue.reward = fee_factor * blue.matched_red.fee if blue.matched_red is not None else 0.
        blue.reward += world.grids[blue.to_grid].local_reward

        return blue.reward

    def done(self, world: World):
        return world.is_done

    def blue_observation(self, blue: Blue, world: World):
        """Get blue observation, include the grid"""
        return world.get_grid_state(blue.grid_id)

    def red_info(self, world: World):
        red_state = [None] * world.n_grid
        red_id = [None] * world.n_grid
        for i in range(world.n_grid):
            red_state[i] = np.array([v.state for v in world.reds[i].values()])
            red_id[i] = np.array([v.node_id for v in world.reds[i].values()])
        return red_state, red_id

    def global_state_feature(self, world: World):
        """Get observation and """
        red_dist = world.get_red_distribution()
        blue_dist = world.get_blue_distribution_emb()
        self._global_state_record[world.time] = (blue_dist, red_dist)
        self._red_dist_record[world.time] = world.get_red_distribution()
        return (blue_dist, red_dist)

    def grid_state(self, world: World):
        """For red, one to one"""
        state_list = []
        for i in range(world.n_grid):
            state = world.get_grid_state(i)
            state_list.append(np.tile(state, [world.grids[i].n_red, 1]))
        return np.vstack(state_list)

    def grid_and_red_ids(self, world: World):
        grid_ids, red_ids, red_states = [], [], []
        for i, grid in enumerate(world.grids):
            grid_ids.extend([i] * grid.n_red)
            # red_ids.extend(list(world.reds[i].keys()))
            red_ids.append(list(world.reds[i].keys()))
            red_states.extend(world.get_red_states_emb(i))

        return np.array(grid_ids, dtype=np.int32), red_ids, np.array(red_states)

    def get_global_transition(self):
        print('\nEXTRACT GLOBAL TRANSITION ...')
        feature = [e[1] for e in self._global_state_record]
        state = [e[0] for e in self._global_state_record]

        res = {
            'state': np.vstack(state),
            'feature': np.vstack(feature),
            'label': np.vstack(self._red_dist_record)
        }

        self._global_state_record = [None for _ in range(self.time_limit)]
        # self._global_mess = [None for _ in range(self.time_limit)]
        self._red_dist_record = [None for _ in range(self.time_limit)]
        self._blue_dist_record = [None for _ in range(self.time_limit)]

        return res

    def record_kl_loss(self, world: World):
        grid_policy_matrix = np.zeros((world.n_grid, world.n_grid))  # grid_from x grid_to

        blue_dist = np.clip(world.get_blue_distribution(), 0.0001, 1.)
        red_dist = np.clip(world.get_red_distribution(), 0.0001, 1.)
        policy_sum = 0
        for i in range(world.n_grid):
            policy = world.grids[i].grid_policy[world.time - 1]
            # print('policy:', policy)
            policy_sum += np.sum(policy)
            # grid_policy_matrix[i] = np.clip(policy / np.maximum(1., np.sum(policy)), 0.0001, 1.)
            grid_policy_matrix[i] = policy

        n_dispatch = np.clip(grid_policy_matrix.T / np.maximum(1, np.sum(grid_policy_matrix, axis=1)), 0., 1.)
        # print('n_dispatch:', n_dispatch)

        grid_base = -1. * red_dist / blue_dist
        grid_base = grid_base.reshape((-1, 1)) * n_dispatch
        for blue in world.all_blues:
            blue.kl_loss = grid_base[blue.grid_id, blue.to_grid]

    def info(self, world: World):
        """Get information of world,
        currently: response-rate, fee-rate, kl of blue and red"""
        fee = 0.
        # response_rate_arr = []
        response_rate = 0.0
        dispatched_blue = 0.0
        dispatched_red = 0.0

        for i, blue_arr in enumerate(world.blue_buffer):
            for blue in blue_arr:
                fee += blue.matched_red.fee
            reds = world.grids[i].n_actived_red + world.grids[i].n_red
            dispatched_red += reds
            dispatched_blue += len(blue_arr)
            # response_rate_arr.append(len(blue_arr) / max(reds, 1))

        # get total fee
        total_fee = 0.
        for red in world.all_reds:
            total_fee += red.fee

        return {
            # 'response-rate': sum(response_rate_arr) / max(1, len(response_rate_arr)),
            'response-rate': dispatched_blue / dispatched_red,
            'fee-rate': fee / max(total_fee, 1.),
        }
