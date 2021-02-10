import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import sys
# sys.path.append("navigation-algorithms/sarl/")
# sys_path = [path for path in sys.path if 'navigation-algorithms/sarl' in path][0]
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA

from crowd_sim.envs.utils._remove_ped_stats import get_remove_ped_dict

import pickle
import pdb


# model_dir = "navigation-algorithms/sarl/crowd_nav/data/sarl_og_10"
# data_dir = "navigation-algorithms/sarl/crowd_nav/data/eth-ucy/eth_train.pkl"
data_dir = "eth_data/eth_train_with_vel.pkl" # 150 peds
plot_folder = "/home/lambda-rl/Desktop/CrowdNav/results_eth/orca/plots/orca/partial_traj/"
metrics_folder = "/home/lambda-rl/Desktop/CrowdNav/results_eth/orca/metrics/orca/partial_traj/"

start_index = None  # start robot and ped from specific time index
num_steps = None  # set goal after no. of time steps
remove_ped = None  # remove ped from data and add robot there
robot_vel = None  # set preferred robot velocity
ignore_collisions = True  # keep running robot even after collisions
time_limit = 100000

runs = get_remove_ped_dict("last_runs")

# distance_list=[]
# ignore_peds=[]
# ignore_peds = [ 10,  16,  19,  20,  21,  30,  31,  33,  35,  37,  42,  44,  45,
#                 46,  50,  53,  55,  60,  61,  62,  65,  67,  70,  75,  76,  77,
#                 78,  79,  81,  83,  86,  87,  89,  90,  96,  97,  99, 103, 112,
#                 113, 117, 121, 122, 130, 131, 134, 140, 141 ]
# new_runs = []
# new_runs_dict = {}

def set_parameters(run_number):
    global start_index, num_steps, remove_ped, robot_vel
    key_name = list(runs.keys())[run_number]
    if 'full' not in key_name:
        value = runs[key_name]
        print(value)
        # if value[0] != 88:
        #     return False
        remove_ped = value[0]
        # new_runs.append(value)
        # new_runs_dict[key_name] = value
        start_index = value[1]
        num_steps = value[2]
        robot_vel = None  # set equals to remove ped avg. vel
        return True
    else:
        logging.warn("Ignoring dictionary key %s", key_name)
        return False


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/eth_env.config')
    # parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--rect', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--traj_new', default=False, action='store_true')
    parser.add_argument('--final_file', type=str, default='final_metrics')
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--trained_env',  type=str, default='orca')
    parser.add_argument('--resumed', default=False, action='store_true')
    args = parser.parse_args()

    # if model_dir:
    #     args.model_dir = model_dir
# da
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
    # env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
    # policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
    # if args.il:
    #     model_weights = os.path.join(args.model_dir, 'il_model.pth')
    # elif args.resumed:
    #     model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
    # else:
    #     model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    # else:
    #     env_config_file = args.env_config
    #     policy_config_file = args.policy_config

    # if args.model_dir is not None:
    #     env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
    #     policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
    #     if args.il:
    #         model_weights = os.path.join(args.model_dir, 'il_model.pth')
    #     else:
    #         if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
    #             model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
    #         else:
    #             model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    # else:
    #     env_config_file = args.env_config
    #     policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
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

    step_number = start_index  # index for given data

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim_eth-v0')
    if data_dir:
        env.set_data_path(data_dir)

    # pdb.set_trace()
    env.configure(env_config)
    # env.initialize_eth_data()
    env.set_ped_traj_length(100)  # 100
    env.set_step_number(step_number)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    if args.rect:
        env.test_sim = 'rectangle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot, v_pref=robot_vel)
    robot_goal_index = start_index + num_steps
    if ignore_collisions:
        env.set_ignore_collisions()
    env.set_time_limit(time_limit)
    env.set_plot_folder(plot_folder)
    env.set_metrics_folder(metrics_folder)
    last_removed_index, removed_traj = env.set_remove_ped(remove_ped, start_index=start_index, goal_index=robot_goal_index, set_robot_vel=True)  # sets ped to (0.0, 0.0) position and sets robot start
    env.set_human_num(remove_ped)
    print(f"Number of Humans: {env.human_num}")
    # pdb.set_trace()
    # and goal positions similar to a given
    # ped
    # env.set_robot_states(start=[4.3729, 4.5884], goal=[11.9214, 5.3149])
    # env.set_robot_states(ped=10)
    explorer = Explorer(env, robot, device, gamma=0.9)

    # calc remove ped distance
    # distance_list.append(env.calc_total_distance_travelled_by_ped(remove_ped))
    # if env.calc_total_distance_travelled_by_ped(remove_ped) == 0.0:
    #     ignore_peds.append(remove_ped)
    # return

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    generated_traj = []
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        # pdb.set_trace()
        done = False
        last_pos = np.array(robot.get_position())
        generated_traj.append(last_pos)
        import time

        non_attentive_humans = []
        non_attentive_humans = set(non_attentive_humans)

        while not done:
            try:
                if env.use_eth_data:
                    if not env.set_step_number(step_number):
                        done = True
            except ValueError as e:
                logging.warn(e)
                break
            time_start = time.time()
            action = robot.act(ob)
            time_end = time.time()
            env.add_time(time_start, time_end)
            env.set_human_num(remove_ped, step_number)
            print(f"Number of Humans: {env.human_num}")
            # logging.info("time taken to select an action: {}".format(toc-tic))
            ob, _, done, info, ppl_count, robot_pose, robot_velocity, dmin = env.step(action, non_attentive_humans)
            current_pos = np.array(robot.get_position())
            logging.info('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            generated_traj.append(last_pos)
            step_number += 1
            robot_goal_index += 1
            if robot_goal_index < last_removed_index:
                robot = env.set_new_robot_goal(robot, remove_ped, robot_goal_index)
            # else:
            #     done = True
        # generated_traj = np.vstack(generated_traj)
        # with open(f'{remove_ped}_{start_index}_{num_steps}_trajectory.pkl', 'wb') as file:
        #     pickle.dump((removed_traj, generated_traj), file)
        if args.traj:
            env.render('traj', args.video_file)
        elif args.traj_new:
            env.render(mode='traj_new', plots=args.plot, output_file=args.video_file, final_file=args.final_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':

    for i in range(len(runs.keys())):
        if set_parameters(i):
            main()
            # exit()

    # print(distance_list)
    # print(ignore_peds)
    # print(len(distance_list))
    # print(new_runs_dict)
