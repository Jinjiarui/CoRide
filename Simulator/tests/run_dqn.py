"""
Edited by Jerry Jin & Ming Zhou: run DQN
"""

import tensorflow as tf
import argparse
import os.path as osp

from Simulator.simulator.envs import *
# from algo.kl.kl import KL
from algo.il.dqn import IL
from algo.stdqn.stdqn import DQNSpacTempoAgent
from algo.non_nueral.distance import Nearest
from algo.non_nueral.myopic import Myopic
from algo.stdqn.replay_buffer import Episode, extract_episode
from algo.base import SummaryObj

from tools.create_envs import *

base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, 'data')


# def select_city(city):
#     return {
#             "xian": real_env_xian,
#             "beijing": real_env_beijing,
#             "wuhan": real_env_wuhan,
#             "chengdu": real_env_chengdu
#             }[city]


def running_example(algo, training_round=1400, fleet_help=False, city="chengdu"):
    env, M, N, central_node_ids, num_valid_grid = dispatch_env(fleet_help)

    # initialize the model
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    summary.register(['KL', 'Entropy', 'Fleet-ORR', 'Fake-ORR', 'ORR', 'GMV'])

    if algo == 'IL':
        model = IL(sess, env.get_state_space(), env.get_action_space(), tf_device='/cpu:*')
    elif algo == 'NOD':
        model = Nearest()
    elif algo == 'LOD':
        model = Myopic()
    elif algo == 'DQN':
        model = DQNSpacTempoAgent(sess, env.get_state_space(), env.get_action_space(),
                                  144, num_valid_grid, tf_device='/cpu:*', M=M, N=N)
    else:
        raise Exception('Unaccepted algo type: {}'.format(algo))


    for iteration in range(training_round):
        print('\n---- ROUND: #{} ----'.format(iteration))
        replay_buffer = dict()

        order_response_rates = []
        T = 0
        max_iter = 144


        _, states, order_list, order_idx,_ , global_order_states = env.reset()

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []

        while T < max_iter:
            order_id_pairs = model.act(states, order_list, order_idx, global_order_states)
            orders = env.get_orders_by_id(order_id_pairs)

            serve_drivers_ids, _,  next_states, next_order_list, next_order_idx, _, next_global_order_states = env.step(orders, generate_order=1)
            # distribution should gotten after step
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
                order_response_rates.append(env.order_response_rate)
            for _driver_id in serve_drivers_ids:
                d_info = env.get_info_of_driver(_driver_id)
                if replay_buffer.get(_driver_id) is None:
                    replay_buffer[_driver_id] = Episode()

                replay_buffer[_driver_id].append(d_info.state, d_info.order_list, d_info.order, d_info.reward)
                replay_buffer[_driver_id].append_kl(kl_value)

            if T % 50 == 0:
                print('City_time: [{0:<5d}], Order_response_rate: [{1:<.4f}], KL: [{2:<.4f}], Entropy: [{3:<.4f}], Fake_orr: [{4:<.4f}], Fleet_arr: [{5:<.4f}], Idle_drivers: [{6}], Ori_order_num: [{7}], Fleet_drivers: [{8}]'.format(
                env.city_time -1, env.order_response_rate, kl_value, entr_value, env.fake_response_rate, env.fleet_response_rate, env.ori_idle, env.ori_order_num, env.ori_fleet))

            states = next_states
            order_list = next_order_list
            order_idx = next_order_idx
            global_order_states = next_global_order_states

            T += 1
        print('>>> Mean_ORR: [{0:<.6f}] GMV: [{1}] Mean_KL: [{2}] Mean_Entropy: [{3}]'.format(np.mean(order_response_rates), np.sum(gmv), np.mean(kl), np.mean(entropy)))

        summary.write({
            'KL': np.mean(kl),
            'Entropy': np.mean(entropy),
            'Fake-ORR': np.mean(fake_orr),
            'Fleet-ORR': np.mean(fleet_orr),
            'ORR': np.mean(order_response_rates),
            'GMV': np.sum(gmv)
        }, iteration)

        # model.train()
        model.store_transitions(extract_episode(replay_buffer))
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='DQN', help='Algorithm Type, choices: KL, IL, NOD, LOD, DQN')
    parser.add_argument('-t', '--train_round', type=int, help='Training round limit', default=1400)
    parser.add_argument('-f', '--fleet_help', type=bool, help='Trigger for fleet management', default=False)
    parser.add_argument("-c", "--city", type=str, default="chengdu")
    args = parser.parse_args()

    running_example(args.algo, args.train_round, city=args.city)
