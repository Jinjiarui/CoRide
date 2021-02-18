import gym
import random
import numpy as np

from gym import spaces
from Simulator.imitation_world.scenario import Scenario
from Simulator.imitation_world.core import World, Blue, Red, Grid


class MultiAgentEnv(gym.Env):
    """Environment for all agents in the multi-agent world, currently
    code assumes that no agents will be created/destroyed at runtime!"""

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world: World):
        """Initialize the multi-agent environment, attributions
        1) discrete action space
        2) discrete action input
        3) if world has the `discrete_action` attribution, force convert action to discrete
        4) if `world.collaborative` is True, share_reward (use group reward function)

        :param world: World, the world instance
        :param reset_callback: func, world reset callback of scenario
        :param reward_callback: func, reward callback of scenario
        :param observation_callback: func, observation callback of scenario
        :param info_callback: func, default is None
        :param done_callback: func, default is None
        """

        self.world = world
        # self.scenario = scenario
        self.agents = self.world.blues
        # set required vectorized gym env property
        # self.n = len(self.agents)
        # scenario callbacks
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        # self.discrete_action_input = discrete_action_input
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        # self.blue_max = max_blue
        # self.red_max = max_red

        # configure spaces
        # blue = list(self.world.blues[0].values())[0]
        # obs_dim = len(observation_callback(blue, self.world))
        # global_obs_dim = self.global_state_callback(world)[0].shape[1]
        # global_feature_dim = self.global_state_callback(world)[1].shape[1]
        # feature_dim = len(world.get_red_states_emb(0)[0])

        # self._agent_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        # self._global_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(global_obs_dim,), dtype=np.float32)
        # self._agent_action_space = spaces.Discrete(1)
        # self._agent_feature_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(feature_dim,), dtype=np.float32)
        # self._global_feature_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(global_feature_dim,), dtype=np.float32)

        self.viewer = None
        self.viewer2 = None
        self.blue_render_geoms = None
        self.blue_render_geoms_xform = None
        self.blue_id_map = None

        self.old_pos_set = dict()
        self.new_pos_set = None

        self._gmv = 0
        self._response_rate = 0.
        self._fake_res_rate = 0.

    @property
    def response_rate(self):
        return self._response_rate

    @property
    def fake_response_rate(self):
        return self._fake_res_rate

    @property
    def gmv(self):
        return self._gmv

    @property
    def n_idle_blues(self):
        nodes = self.world.nodes
        n_blues = 0
        for _node in nodes:
            n_blues += _node.n_blues
        return max(n_blues, 1)

    @property
    def action_space(self):
        return self.world.action_space

    @property
    def agent_observation_space(self):
        return self.world.state_space

    @property
    def global_observation_space(self):
        return self._global_observation_space.shape

    @property
    def global_feature_space(self):
        return self._global_feature_space.shape

    def get_distribution(self):
        return self.world.get_distribution()

    def _get_agent_observation(self):
        """Get observation of all agents"""
        obs_n = []
        for agent in self.world.blues.values():
            obs_n.append(agent.state)
        return obs_n

    def _get_agent_reward(self):
        reward_n = []
        for agent in self.world.blues.values():
            reward_n.append(agent.reward)
        return reward_n

    def reset(self):
        """Reset world"""
        return self.world.reset()

    def get_info_of_blue(self, blue_id):
        blue = self.world.blues[blue_id]

        node = blue.node
        # include node_index and dist_dispatching
        d_info = blue.get_info_template()(blue.last_state, blue.last_red_list,
                                          blue.last_red_state, blue.last_reward, blue.policy, node.index)
        return d_info

    def match_callback(self, dispatch_actions):
        """Match blue and reds"""
        serve_blue_ids = []
        # First round broadcast
        all_red_num = 0
        finished_red_num = 0
        fake_num = 0
        fake_finish = 0

        self._gmv = 0

        for _reds in dispatch_actions:
            node = _reds[0].source
            gmv, all_red_num_node, finished_red_num_node, blue_idx, f_fin, f_num = node.assign(_reds, self.world.time)

            self._gmv += gmv
            all_red_num += all_red_num_node
            finished_red_num += finished_red_num_node
            fake_finish += f_fin
            fake_num += f_num
            # node_reward[node.get_node_index()] += reward_node
            if len(blue_idx) > 0:
                serve_blue_ids.extend(blue_idx)

        if all_red_num != 0:
            self._response_rate = finished_red_num / max(float(all_red_num), 1)
            self._fake_res_rate = fake_finish / max(float(fake_finish), 1)
        else:
            self._response_rate = -1

        # return reward, [node_reward, neighbor_reward]
        return serve_blue_ids

    def get_reds_by_id(self, id_pairs):
        return self.world.get_reds_by_id(id_pairs)

    def step(self, dispatch_actions):
        """ Environment step method
        :return:
        """
        self.world.record_history()

        # 1. Dispatching orders and return the ids of serving blues
        blue_id = self.match_callback(dispatch_actions)

        # record policy
        self.world.record_policy()
        next_states, next_red_idx, next_red_list = self.world.step()

        return blue_id, next_states, next_red_list, next_red_idx

    def render_ready(self):
        """Get the former pos of blue"""
        from Simulator.imitation_world import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.world.width, self.world.height)

        VIEW_BOUNDER = 2

        # scale unit size and grid size
        unit_size = rendering.UNIT_SIZE / self.viewer.width * VIEW_BOUNDER
        grid_size = rendering.GRID_SIZE / self.viewer.width * VIEW_BOUNDER

        # ===== add blue if none
        if self.blue_render_geoms is None:
            self.blue_render_geoms = []
            self.blue_render_geoms_xform = []
            self.blue_id_map = {}

            for i, entity in enumerate(self.world.blues.values()):
                self.blue_id_map[entity.node_id] = i
                geom = rendering.make_circle(unit_size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                self.blue_render_geoms.append(geom)
                self.blue_render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []

            for geom in self.blue_render_geoms:
                self.viewer.add_geom(geom)

            # ========== draw horizon lines
            for i in range(1, self.world.height):
                start = (grid_size * i, 0)
                end = (start[0], VIEW_BOUNDER)
                # geom = self.viewer.draw_line(start, end)
                geom = rendering.Line(start, end)
                self.viewer.add_geom(geom)

            # draw vertical lines
            for i in range(1, self.world.width):
                start = (0, grid_size * i)
                end = (VIEW_BOUNDER, start[1])
                # geom = self.viewer.draw_line(start, end)
                geom = rendering.Line(start, end)
                self.viewer.add_geom(geom)

            # set blue pos and record the old pos set
            for entity in self.world.blues.values():
                blue_id = entity.node_id
                idx = self.blue_id_map[blue_id]
                pos = self.world.pos_of_grid(entity.grid_id, self.world.width, self.world.height, unit_size,
                                             grid_size)
                self.blue_render_geoms_xform[idx].set_translation(*pos)
                self.old_pos_set[blue_id] = pos

    def render(self, mode='fill'):
        """Render environment"""

        from Simulator.imitation_world import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.world.width, self.world.height)

        VIEW_BOUNDER = 2

        # scale unit size and grid size
        unit_size = rendering.UNIT_SIZE / self.viewer.width * VIEW_BOUNDER
        grid_size = rendering.GRID_SIZE / self.viewer.width * VIEW_BOUNDER

        step = 0
        dist_red = self.render_red(rendering, unit_size, grid_size, style=mode)

        # pos_set_arr = self.get_new_pos_arr(0.08, unit_size, grid_size)
        dist_blue = self.render_blue(rendering, unit_size, grid_size, style=mode)

        result = {'dist_red': dist_red, 'dist_blue': dist_blue}

        if mode == 'fill':
            self.viewer.set_bounds(0, VIEW_BOUNDER, 0, VIEW_BOUNDER)
            self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
            self.viewer2.set_bounds(0, VIEW_BOUNDER, 0, VIEW_BOUNDER)
            self.viewer2.render(return_rgb_array=(mode == 'rgb_array'))

            self.viewer2.clear_one_time_geom()
        self.viewer.clear_one_time_geom()

        return result

    def get_new_pos_arr(self, delta, unit_size, grid_size):
        soft_update_arr = dict()

        for entity in self.world.blues.values():
            blue_id = entity.node_id
            pos = self.world.pos_of_grid(entity.grid_id, self.world.width, self.world.height, unit_size, grid_size)
            old_pos = self.old_pos_set[blue_id]

            distance = np.sqrt((pos[0] - old_pos[0])**2 + (pos[1] - old_pos[1])**2)
            time_slot = max(distance // delta, 1)

            self.old_pos_set[blue_id] = pos

            pos_step = ((pos[0] - old_pos[0]) / time_slot, (pos[1] - old_pos[1]) / time_slot)

            x = np.arange(old_pos[0], pos[0], pos_step[0])
            y = np.arange(old_pos[1], pos[1], pos_step[1])

            soft_update_arr[blue_id] = list(zip(x, y))

        return soft_update_arr

    def render_blue(self, rendering, unit_size, grid_size, style='fill'):
        VIEW_BOUNDER = 2

        if style == 'fill':
            if self.viewer2 is None:
                self.viewer2 = rendering.Viewer(self.world.width, self.world.height)
            grid_weight_map = {}
            weight_sum = 1
            for i, entity in enumerate(self.world.blues.values()):
                weight_sum += 1
                grid_weight_map[entity.grid_id] = grid_weight_map.get(entity.grid_id, 0) + 1
            for grid_id, value in grid_weight_map.items():
                x = grid_id // self.world.width
                y = (grid_id - x * self.world.width) * grid_size
                x *= grid_size

                top_left = (x, y)
                top_right = (x, y + grid_size)
                bot_left = (x + grid_size, y)
                bot_right = (x + grid_size, y + grid_size)

                v = [top_left, top_right, bot_right, bot_left]

                geom = rendering.make_polygon(v)
                geom.set_color(*[96 / 255, 198 / 255, 248 / 255], alpha=min(15 * value / weight_sum + 0.2, 0.95))
                self.viewer2.add_onetime(geom)
            dist = []
            for i in range(64):
                dist.append(grid_weight_map.get(i, 0))
            return dist
        elif style == 'ball':
            step = 0
            pos_set_arr = self.get_new_pos_arr(0.08, unit_size, grid_size)
            while True:
                # soft update the pos of blue
                update = False
                for key, value in pos_set_arr.items():
                    idx = self.blue_id_map[key]
                    if len(value) < step + 1:
                        continue
                    else:
                        update = True
                        self.blue_render_geoms_xform[idx].set_translation(*value[step])

                if not update:
                    break
                # render to display or array
                self.viewer.set_bounds(0, VIEW_BOUNDER, 0, VIEW_BOUNDER)
                self.viewer.render(return_rgb_array=(style == 'rgb_array'))
                step += 1

    def render_red(self, rendering, unit_size, grid_size, style='fill'):
        # ========== add red, one time
        self.red_render_geoms = []
        self.red_render_geoms_xform = []

        if style == 'fill':
            grid_weight_map = {}
            weight_sum = 1
            for i, entity in enumerate(self.world.reds):
                weight_sum += 1
                grid_weight_map[entity.grid_id] = grid_weight_map.get(entity.grid_id, 0) + 1

            for grid_id, value in grid_weight_map.items():
                x = grid_id // self.world.width
                y = (grid_id - x * self.world.width) * grid_size
                x *= grid_size

                top_left = (x, y)
                top_right = (x, y + grid_size)
                bot_left = (x + grid_size, y)
                bot_right = (x + grid_size, y + grid_size)

                v = [top_left, top_right, bot_right, bot_left]

                geom = rendering.make_polygon(v)
                # xform = rendering.Transform()
                geom.set_color(*[223 / 255, 60 / 255, 34 / 255], alpha=min(15 * value / weight_sum + 0.2, 0.95))
                # geom.add_attr(xform)
                self.viewer.add_onetime(geom)
            # exit(1)
            dist = []
            for i in range(64):
                dist.append(grid_weight_map.get(i, 0))
            return dist
        elif style == 'ball':
            for entity in self.world.reds:
                geom = rendering.make_circle(unit_size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                self.red_render_geoms.append(geom)
                self.red_render_geoms_xform.append(xform)

            for geom in self.red_render_geoms:
                self.viewer.add_onetime(geom)

            # update red pos
            for i, entity in enumerate(self.world.reds):
                pos = self.world.pos_of_grid(entity.grid_id, self.world.width, self.world.height, unit_size, grid_size)
                self.red_render_geoms_xform[i].set_translation(*pos)
        else:
            raise Exception('Unexcepted style: ' + str(style))

    def done(self):
        return self._get_dones()

    def close(self):
        self.viewer.close()
