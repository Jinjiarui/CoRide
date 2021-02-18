"""
Edited by Jerry Jin: run CoRide-
"""

import tensorflow as tf
import argparse
import os.path as osp

from Simulator.simulator.envs import *
from algo.cfun.worker import Worker
from algo.cfun.manager import Manager
from algo.cfun.base import *
from algo.cfun.worker_buffer import Episode_w, extract_episode_w
from algo.cfun.manager_buffer import Episode_m, extract_episode_m
from algo.base import SummaryObj
from algo.base import SimpleCommNet
from tools.create_envs import *

base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, 'data')

# def select_city(city):
#     return {
#         "xian": real_env_xian,
#         "beijing": real_env_beijing,
#         "wuhan": real_env_wuhan,
#         "chengdu": real_env_chengdu
#     }[city]


def running_example(algo='HRL', training_round=140, fleet_help=False, city="chengdu"):
    # print("\n===== You select Cheng Du =====")
    env, M, N, central_node_ids, num_vaild_grid = dispatch_env(fleet_help)

    # initialize the model
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    message_space = 32
    with tf.variable_scope('Worker'):
        worker = Worker(sess=sess, act_space=env.get_action_space(), obs_space=env.get_state_space_node(),
                        message_space=message_space, feature_space=env.get_feature_space())
    with tf.variable_scope('Manager'):
        manager = Manager(sess=sess, act_space=env.get_action_space(), obs_space=env.get_state_space_node(),
                          message_space=message_space, batch_size=len(central_node_ids))

    summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    summary.register(['KL', 'Entropy', 'Fleet-ORR', 'Fake-ORR', 'ORR', 'GMV'])


    for iteration in range(training_round):
        print('\n------- ROUND: #{} ---------'.format(iteration))
        # initialize message
        m_next_message = np.zeros([len(central_node_ids), message_space])
        w_next_message = np.zeros([7, message_space])
        m_replay_buffer = dict()
        w_replay_buffer = dict()

        order_response_rate = []
        T = 0
        max_iter =144
        states, _, _, order_idx, order_feature, _ = env.reset()
        gmv = []
        kl = []
        entropy = []
        fake_orr = []
        fleet_orr = []
        # record max reward
        m_reward_max = 0.
        w_reward_max = 0.

        while T < max_iter:
            # store transitions
            _w_message, _m_message = dict(), dict()
            _m_goal, _w_goal = dict(), dict()
            _w_weight = dict()


            m_states = generate_split_manager(M, N, env.get_state_space_node(), central_node_ids, states)
            m_goal, m_message = manager.act(m_states, m_next_message)
            for _idx, _central in enumerate(central_node_ids):
                _m_goal[_central] = m_goal[_idx]
                _m_message[_central] = m_message[_idx]
            m_next_message = m_message

            w_state, w_actions, w_action_ids, w_goal = generate_split_worker(M, N, central_node_ids, states, order_feature, order_idx, m_goal)
            action_ids = []

            for _state, _actions, _action_ids, _goal in zip(w_state, w_actions, w_action_ids, w_goal):
                assert len(_state) == len(_goal)

                w_action_ids, w_message, w_weight = worker.act(_state, w_next_message, _goal, _actions, _action_ids)

                assert len(w_message) == len(w_weight) == len(_goal) == len(w_action_ids)
                # record history
                for _g, _m, _w, _id, in zip(_goal, w_message, w_weight, w_action_ids):
                    _w_goal[_id[0][0]] = _g
                    _w_message[_id[0][0]] = _m
                    _w_weight[_id[0][0]] = _w
                w_next_message = w_message
                action_ids.extend(w_action_ids)


            orders = env.get_orders_by_id(action_ids)
            _, next_states, _, _, next_order_idx, next_order_feature, _ = env.step(orders, generate_order=1)
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            kl.append(kl_value)
            entropy.append(entr_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            if env.order_response_rate >= 0:
                order_response_rate.append(env.order_response_rate)

            # store data to buffer
            for _node_id in central_node_ids:
                d_info_manager = env.get_info_manager_node(_node_id)
                if m_replay_buffer.get(_node_id) is None:
                    m_replay_buffer[_node_id] = Episode_m()

                m_replay_buffer[_node_id].append_state(d_info_manager)
                # managers get global reward
                m_replay_buffer[_node_id].append_kl([kl_value])
                m_replay_buffer[_node_id].append_entropy([entr_value])
                m_reward = env.gmv / 1000
                if m_reward > m_reward_max:
                    m_reward_max = m_reward
                m_replay_buffer[_node_id].append_reward([m_reward])

                for key, value in _m_message.items():
                    if key == _node_id:
                        m_replay_buffer[_node_id].append_message(value)

                for key, value in _m_goal.items():
                    if key == _node_id:
                        m_replay_buffer[_node_id].append_action(value)


            for _node_id in env.get_node_ids():
                d_info_worker = env.get_info_worker_node(_node_id)
                if w_replay_buffer.get(_node_id) is None:
                    w_replay_buffer[_node_id] = Episode_w()

                w_replay_buffer[_node_id].append_state(d_info_worker)
                w_replay_buffer[_node_id].append_kl([kl_value])
                w_replay_buffer[_node_id].append_entropy([entr_value])
                w_reward = env.gmv / 1000
                if w_reward > w_reward_max:
                    w_reward_max = w_reward
                w_replay_buffer[_node_id].append_reward([w_reward])
                # workers receiver reward according global

                for key, value in _w_message.items():
                    if key == _node_id:
                        w_replay_buffer[_node_id].append_message(value)

                for key, value in _w_weight.items():
                    if key == _node_id:
                        w_replay_buffer[_node_id].append_action(value)

                for key, value in _w_goal.items():
                    if key == _node_id:
                        w_replay_buffer[_node_id].append_goal(value)


            if T % 50 == 0:
                print(
                'City_time: [{0:<5d}], Order_response_rate: [{1:<.4f}], KL: [{2:<.4f}], Entropy: [{3:<.4f}], Fake_orr: [{4:<.4f}], Fleet_arr: [{5:<.4f}], Idle_drivers: [{6}], Ori_order_num: [{7}], Fleet_drivers: [{8}]'.format(
                    env.city_time - 1, env.order_response_rate, kl_value, entr_value, env.fake_response_rate,
                    env.fleet_response_rate, env.ori_idle, env.ori_order_num, env.ori_fleet
                ))

            states = next_states
            order_feature = next_order_feature
            order_idx = next_order_idx
            T += 1
        print('>>> Mean_ORR: [{0:<.6F}] GMV: [{1}] Mean_KL: [{2}] Mean_Entropy: [{3}]'.format(
                np.mean(order_response_rate), np.sum(gmv), np.mean(kl), np.mean(entropy)))

        summary.write({
            'KL': np.mean(kl),
            'Entropy': np.mean(entropy),
            'Fake-ORR': np.mean(fake_orr),
            'Fleet-ORR': np.mean(fleet_orr),
            'ORR': np.mean(order_response_rate),
            'GMV': np.sum(gmv)
        }, iteration)

        # train model
        worker.store_transitions(extract_episode_w(w_replay_buffer))
        worker.train(w_reward_max)
        manager.store_transitions(extract_episode_m(m_replay_buffer))
        manager.train(m_reward_max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='HRL', help='Algorithm Type, choices: HRL')
    parser.add_argument('-t', '--train_round', type=int, help='Training round limit', default=1400)
    parser.add_argument('-f', '--fleet_help', type=bool, help='Trigger for fleet management', default=False)
    parser.add_argument("-c", "--city", type=str, help="City selection", default="chengdu")
    args = parser.parse_args()

    running_example(args.algo, args.train_round, city=args.city)
