import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import pandas as pd
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
import random
import math
from crowd_sim.envs.utils.human import Human

from crowd_sim.envs.policy.socialforce import SocialForce
import matplotlib.pyplot as plt
# from crowd_sim.envs.policy.ssp import SSP

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=2)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=False)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--human_policy',  type=str, default='socialforce')
    parser.add_argument('--trained_env',  type=str, default='socialforce')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # device = 'cpu'
    logging.info('Using device: %s', device)


    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim_mixed-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'


    PPL = env.human_num




    robot = Robot(env_config, 'robot')
    # print(robot.px)
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    ppl_local = []
    robot_states = []
    robot_vel = []
    policy.set_env(env)
    robot.print_info()
    non_attentive_humans = []

    for case in range(args.test_case):
        print(case)
        rewards = []

        # if args.visualize:

        ob = env.reset(test_case=case)

        # ob = env.reset(args.phase, case)
        done = False
        last_pos = np.array(robot.get_position())
        # non_attentive_humans = random.sample(env.humans, int(math.ceil(env.human_num/10)))
        non_attentive_humans = []
        non_attentive_humans = set(non_attentive_humans)

        while not done:

            # print(env.global_time)
            # count = env.global_time
            # if count != 0:
            #     #     old_non_attentive_humans = []
            #     # else:
            #     old_non_attentive_humans = non_attentive_humans
            # # only record the first time the human reaches the goal

            # if count % 4 == 0:
            #     print("true")
            #
            #     non_attentive_humans = Human.get_random_humans()
            #     old_non_attentive_humans = non_attentive_humans
            # # else:
            # non_attentive_humans = old_non_attentive_humans

            action = robot.act(ob)
            ob, _, done, info, ppl_count, robot_pose, robot_velocity, dmin = env.step(action, non_attentive_humans)
            # ob, _, done, info, ppl_count, robot_pose, robot_velocity, dmin = env.step(action)

            rewards.append(_)

            ppl_local.append(ppl_count)
            robot_states.append(robot_pose)
            robot_vel.append(robot_velocity)


            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos

        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
                                 * reward for t, reward in enumerate(rewards)])
            #
        # #
        # # # # # #
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)


        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)


        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()

            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

        # logging.info('PPl counter ', ppl_local)



        # logging.info('PPl counter ', ppl_local)
        main = "Results/"
        human_policy = args.human_policy
        trained_env = args.trained_env

        if human_policy == 'socialforce':
            maindir = 'SocialForce/'
            if not os.path.exists(main+maindir):
                os.mkdir(main+maindir)
        else:
            maindir =  'ORCA/'
            if not os.path.exists(main+maindir):
                os.mkdir(main+maindir)

        robot_policy = args.policy
        trained_env = args.trained_env

        if robot_policy == 'igp_dist':
            method_dir = 'igp_dist/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)

        if robot_policy == 'ssp':
            method_dir = 'ssp/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'model_predictive_rl'and trained_env == 'orca'):
            method_dir = 'model_predictive_rl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'model_predictive_rl' and trained_env == 'socialforce'):
            method_dir = 'model_predictive_rl_social/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)

        if (robot_policy == 'rgl'and trained_env == 'orca'):
            method_dir = 'rgl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'rgl' and trained_env == 'socialforce'):
            method_dir = 'rgl_social/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)

        if (robot_policy == 'sarl'and trained_env == 'orca'):
            method_dir = 'sarl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'sarl' and trained_env == 'socialforce'):
            method_dir = 'sarl_social/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)


        if (robot_policy == 'cadrl'and trained_env == 'orca'):
            method_dir = 'cadrl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'cadrl' and trained_env == 'socialforce'):
            method_dir = 'cadrl_social/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'cadrl'and trained_env == 'orca_new'):
            method_dir = 'cadrl_new/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'cadrl'and trained_env == 'socialforce_new'):
            method_dir = 'cadrl_social_new/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)


        if robot_policy == 'ssp2':
            method_dir = 'ssp2/'
            if not os.path.exists(main+maindir+method_dir):
                os.mkdir(main+maindir+method_dir)
        # elif robot_policy == 'cadrl':
        #     method_dir = 'cadrl/'
        #     if not os.path.exists(maindir+method_dir):
        #         os.mkdir(maindir+method_dir)
        # elif robot_policy == 'sarl':
        #     method_dir = 'sarl/'
        #     if not os.path.exists(maindir+method_dir):
        #         os.mkdir(maindir+method_dir)
        # elif robot_policy == 'lstm_rl':
        #     method_dir = 'lstm_rl/'
        #     if not os.path.exists(maindir+method_dir):
        #         os.mkdir(maindir+method_dir)
        elif robot_policy == 'orca':
            method_dir = 'orca/'
            if not os.path.exists(main+maindir+method_dir):
                os.mkdir(main+maindir+method_dir)

        robot_data = pd.DataFrame()
        robot_data['robot_x'] = np.array(robot_states)[:, 0]
        robot_data['robot_y'] = np.array(robot_states)[:, 1]
        robot_data['local_ppl_cnt'] = np.array(ppl_local)
        # robot_data['dmin'] = np.array(dmin)

        out_name = f'robot_data{case}.csv'

        if not os.path.exists(main+maindir + method_dir + f'{PPL}/'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/')
        # outdir = f'{PPL}/robot_data_{PPL}/'
        if not os.path.exists(main+maindir + method_dir + f'{PPL}/robot_data_{PPL}/'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/robot_data_{PPL}/')

        fullname = os.path.join(main+maindir + method_dir + f'{PPL}/robot_data_{PPL}/', out_name)

        robot_data.to_csv(fullname, index=True)

        if not os.path.exists(main+maindir + method_dir + f'{PPL}/time_{PPL}'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/time_{PPL}')
        Time_data = pd.DataFrame()
        Time_data['time (s)'] = [env.global_time]
        Time_data['mean_local'] = np.mean(ppl_local)
        Time_data['std_local'] = np.std(ppl_local)
        Time_data['collision_flag'] = info
        Time_data['dmin'] = dmin
        Time_data['reward'] = cumulative_reward
        Time_data.to_csv(main+maindir + method_dir + f'{PPL}/time_{PPL}/robot_time_data_seconds_{PPL}_{case}.csv')

        if not os.path.exists(main+maindir + method_dir + f'{PPL}/localdensity_{PPL}'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/localdensity_{PPL}')
        LD = pd.DataFrame()
        LD['local_ppl_cnt'] = np.array(ppl_local)
        LD['vx'] = np.array(robot_vel)[:, 0]
        LD['vy'] = np.array(robot_vel)[:, 1]
        LD.to_csv(main+maindir + method_dir + f'{PPL}/localdensity_{PPL}/localdensity_{PPL}_{case}.csv')

    #
    # else:
    #     explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
    #


if __name__ == '__main__':
    main()
