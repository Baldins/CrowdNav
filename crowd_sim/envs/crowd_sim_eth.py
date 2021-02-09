import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import *
import pickle

import copy
import os

import pdb


class CrowdSim_eth(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        # Extra
        self.robot_start = None
        self.robot_goal = None

        self.use_eth_data = False
        self.step_number = None
        self.ped_traj_length = None
        self.limit_ped_traj_length = None
        self.x = None
        self.y = None
        self.x_vel = None
        self.y_vel = None
        self.data_dict = None
        self.data_id = None
        self.data_in_batches = None
        self.remove_ped = None
        self.data_path = 'data/eth-ucy/test_data_dict_with_vel.pkl'

        self.x_backup = None
        self.x_vel_backup = None
        self.y_backup = None
        self.y_vel_backup = None

        self.ignore_collisions = False
        self.plot_folder = None
        self.metrics_folder = None

        self.start_index = None
        self.num_steps = None

        self.ped_min_dist_list = None
        self.robot_min_dist_list = None
        self.robot_agent_path_diff_list = None
        self.time_diff_list = None

        self.goal_list = None
        self.temp_starting_step = None

    def add_time(self, time_start, time_end):
        if self.time_diff_list is None:
            self.time_diff_list = []

        self.time_diff_list.append(time_end - time_start)

    def initialize_eth_data(self):
        self.use_eth_data = True
        self.data_id = 11
        self.data_in_batches = True
        self.x = dict()
        self.y = dict()
        self.x_vel = dict()
        self.y_vel = dict()
        self.import_eth_data()

    def set_pseudo_bot(self, num):
        self.pseudo_bot = num

    def set_ignore_collisions(self):
        self.ignore_collisions = True

    def set_plot_folder(self, folder_path):
        self.plot_folder = folder_path

    def set_metrics_folder(self, metrics_folder):
        self.metrics_folder = metrics_folder

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit

    def import_eth_data(self):

        with open(self.data_path, 'rb') as f:
            self.data_dict = pickle.load(f, encoding='latin1')

        keys = [int(key) for key in self.data_dict.keys()]
        self.eth_num_peds = max(keys) + 1
        # pdb.set_trace()

        # for ped in range(self.human_num):
        for ped in range(self.eth_num_peds):

            if self.data_in_batches:  # only for test_data_dict_vel.pkl
                ped_name = str(ped)  # 'Pedestrian/' + str(ped)

                # pdb.set_trace()
                self.x[ped] = self.data_dict[ped_name][self.data_id][..., 0]
                self.x_vel[ped] = self.data_dict[ped_name][self.data_id][..., 2]
                self.y[ped] = self.data_dict[ped_name][self.data_id][..., 1]
                self.y_vel[ped] = self.data_dict[ped_name][self.data_id][..., 3]

            else:
                ped_name = 'Pedestrian/' + str(ped)
                self.x[ped] = self.data_dict[ped_name][..., 0].squeeze()
                self.x_vel[ped] = self.data_dict[ped_name][..., 2].squeeze()
                self.y[ped] = self.data_dict[ped_name][..., 1].squeeze()
                self.y_vel[ped] = self.data_dict[ped_name][..., 3].squeeze()

            ped_traj_length = self.ped_traj_length
            limit_ped_traj_length = self.limit_ped_traj_length

            if limit_ped_traj_length:
                self.x[ped] = self.x[ped][:ped_traj_length]
                self.x_vel[ped] = self.x_vel[ped][:ped_traj_length]
                self.y[ped] = self.y[ped][:ped_traj_length]
                self.y_vel[ped] = self.y_vel[ped][:ped_traj_length]

        self.x_backup = copy.deepcopy(self.x)
        self.x_vel_backup = copy.deepcopy(self.x_vel)
        self.y_backup = copy.deepcopy(self.y)
        self.y_vel_backup = copy.deepcopy(self.y_vel)

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.rectangle_length = config.getfloat('sim', 'rectangle_length')
            self.rectangle_width = config.getfloat('sim', 'rectangle_width')
            self.human_num = config.getint('sim', 'human_num')
            self.random_pos = config.getboolean('sim', 'random_pos')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        if self.test_sim == 'eth':
            # pdb.set_trace()
            self.initialize_eth_data()

    def show_config(self):
        print(f'Training and Validation Simulation: {self.train_val_sim}')
        print(f'Testing Simulation: {self.test_sim}')
        if self.train_val_sim == 'circle_crossing':
            print(f'Circle radius: {self.circle_radius}')
        elif self.train_val_sim == 'square_crossing':
            print(f'Square width: {self.square_width}')
        elif self.train_val_sim == 'rectangle_crossing':
            print(f'Rectangle width: {self.rectangle_width}')
            print(f'Rectangle length: {self.rectangle_length}')
        print(f'Number of Humans: {self.human_num}')

    def set_robot(self, robot, v_pref=None):
        self.robot = robot
        if v_pref is not None:
            self.robot.v_pref = v_pref

    def set_human_num(self, ped_num, step_number=None, fixed_num=None):
        ped_list = []
        ped_limit = []

        if fixed_num is not None:
            self.human_num = fixed_num
            count = 0
            for ped in range(self.eth_num_peds):
                if ped == ped_num:
                    continue
                if step_number is None:
                    x = self.x_backup[ped][self.start_index]
                    y = self.y_backup[ped][self.start_index]
                else:
                    x = self.x_backup[ped][step_number]
                    y = self.y_backup[ped][step_number]
                if x != 0. and y != 0.:
                    ped_list.append(ped)
                    end_index = np.nonzero(self.x_backup[ped])[0][-1]
                    ped_limit.append(end_index)
                    count = count + 1
                if count == self.human_num:
                    break
        else:
            for ped in range(self.eth_num_peds):
                if ped == ped_num:
                    continue
                if step_number is None:
                    x = self.x_backup[ped][self.start_index]
                    y = self.y_backup[ped][self.start_index]
                else:
                    x = self.x_backup[ped][step_number]
                    y = self.y_backup[ped][step_number]
                if x != 0. and y != 0.:
                    ped_list.append(ped)
                    end_index = np.nonzero(self.x_backup[ped])[0][-1]
                    ped_limit.append(end_index)

        if fixed_num is None:
            self.update_humans(ped_list, ped_limit, step_number)
        elif not hasattr(self, 'ped_list'):
            self.ped_list = ped_list
            self.ped_limit = ped_limit

        self.human_num = len(self.ped_list)
        if not hasattr(self, 'human_list'):
            self.human_list = []
        self.human_list.append(self.human_num)

    def update_humans(self, ped_list, ped_limit, step_number):
        # If list of pedestrians is different than insert and remove corresponding humans
        if hasattr(self, 'ped_list'):
            if ped_list != self.ped_list:
                offset = 0
                num = self.human_num
                if len(ped_list) < self.human_num:
                    num = len(ped_list)
                    temp = self.ped_list
                    self.ped_list = ped_list
                    ped_list = temp
                for i in range(num):
                    if ped_list[i + offset] > self.ped_list[i]:
                        self.humans.pop(i)
                        self.human_times.pop(i)
                        offset = offset - 1
                    elif ped_list[i + offset] < self.ped_list[i]:
                        human = Human(self.config, 'humans')
                        human.time_step = self.time_step
                        human.policy.time_step = self.time_step
                        ped = ped_list[i + offset]
                        px = self.x_backup[ped][step_number]
                        py = self.y_backup[ped][step_number]
                        human.set(px, py, px, py, 0., 0., 0.)
                        self.humans.insert(i, human)
                        self.human_times.insert(i, 0)
                        offset = offset + 1

                if len(ped_list) == self.human_num:
                    self.ped_limit = ped_limit
                else:
                    self.ped_list = ped_list
                    self.ped_limit = ped_limit
        else:
            self.ped_list = ped_list
            self.ped_limit = ped_limit

    def set_step_number(self, num):
        self.step_number = num
        if self.limit_ped_traj_length:
            if num >= self.ped_traj_length:
                raise ValueError("Data limit reached. Traj lengths are limited to: {}".format(self.ped_traj_length))
                return False

    def set_ped_traj_length(self, traj_length):
        self.limit_ped_traj_length = True
        self.ped_traj_length = traj_length

    def set_new_robot_goal(self, robot, remove_ped, goal_index):
        self.num_steps = goal_index - self.start_index
        if hasattr(self, 'start_pos'):
            goal_index = self.num_steps + self.start_pos
            if goal_index > self.last_index:
                return robot
        goal_x = self.x_backup[remove_ped][goal_index]
        goal_y = self.y_backup[remove_ped][goal_index]
        if (goal_x != 0.0) and (goal_y != 0.0):
            self.robot_goal = [goal_x, goal_y]
            robot.gx = goal_x
            robot.gy = goal_y
            self.goal_list.append(self.robot_goal)
        return robot

    def set_robot_states(self, start=None, goal=None, ped=None, goal_index=None, start_index=None):
        if start and goal:
            logging.info("Setting robot start position to {} and end position to {}".format(start, goal))
            self.robot_start = start
            self.robot_goal = goal
        elif ped is not None:
            ped = str(ped)
            logging.info("Setting robot start and goal positions similar to Pedestrian{}".format(ped))
            if start_index is not None:
                self.robot_start = [self.data_dict[ped][self.data_id][start_index, 0],
                                    self.data_dict[ped][self.data_id][start_index, 1]]
            else:
                self.robot_start = [self.data_dict[ped][self.data_id][0, 0], self.data_dict[ped][self.data_id][0, 1]]
            if goal_index is not None:
                self.robot_goal = [self.data_dict[ped][self.data_id][goal_index, 0],
                                   self.data_dict[ped][self.data_id][goal_index, 1]]
            else:
                self.robot_goal = [self.data_dict[ped][self.data_id][-1, 0], self.data_dict[ped][self.data_id][-1, 1]]
            logging.info("{} : {}".format(self.robot_start, self.robot_goal))

        if self.goal_list is None:
            self.goal_list = []
        self.goal_list.append(self.robot_goal)

    def set_remove_ped(self, ped_num, goal_index=None, start_index=None, set_robot_vel=False):

        self.start_index = start_index
        self.num_steps = goal_index - start_index
        self.temp_starting_step = self.num_steps

        self.last_index = np.nonzero(self.x_backup[ped_num])[0][-1]
        self.removed_x = self.x_backup[ped_num]
        self.removed_y = self.y_backup[ped_num]
        self.removed_x_vel = self.x_vel_backup[ped_num]
        self.removed_y_vel = self.y_vel_backup[ped_num]

        if self.removed_x[start_index] == 0.0 and self.removed_y[start_index] == 0.0:
            state_index = np.nonzero(self.removed_x)[0]
            start_pos = state_index[0]
            try:
                goal_pos = state_index[self.num_steps]
            except IndexError:
                pdb.set_trace()

            if start_pos != start_index:
                self.start_pos = start_pos
        else:
            start_pos = start_index
            goal_pos = goal_index

        x = self.removed_x[start_pos]
        y = self.removed_y[start_pos]
        gx = self.removed_x[goal_pos]
        gy = self.removed_y[goal_pos]
        self.set_robot_states(start=[x, y], goal=[gx, gy])

        if set_robot_vel:
            ped_x_avg = self.x_vel_backup[ped_num][np.nonzero(self.x_vel_backup[ped_num])].mean()
            ped_y_avg = self.y_vel_backup[ped_num][np.nonzero(self.y_vel_backup[ped_num])].mean()
            total_vel = (abs(ped_x_avg) + abs(ped_y_avg)) / 0.4  # 0.6667
            self.robot.v_pref = total_vel

        self.remove_ped = ped_num
        x_index = np.nonzero(self.removed_x)
        y_index = np.nonzero(self.removed_y)
        x_pos = np.array(self.removed_x[x_index]).reshape(-1, 1)
        y_pos = np.array(self.removed_y[y_index]).reshape(-1, 1)
        remove_ped_traj = np.hstack((x_pos, y_pos))
        return self.last_index, remove_ped_traj

    def set_data_path(self, path):
        self.data_path = path

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'rectangle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_rectangle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((
                                    px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        elif rule == "eth":
            """
            Humans will follow eth data
            """
            logging.info("Using eth data")
            self.humans = []
            for i in range(self.human_num):
                human = Human(self.config, 'humans')
                ped = self.ped_list[i]
                px = self.x[ped][self.start_index]
                py = self.y[ped][self.start_index]
                human.set(px, py, px, py, 0., 0., 0.)
                self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_rectangle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            if self.random_pos:
                px = (np.random.random() - 0.5) * self.rectangle_length
                py = (np.random.random() - 0.5) * self.rectangle_width
            else:
                px = np.random.random() * self.rectangle_length * 0.5 * sign
                py = (np.random.random() - 0.5) * self.rectangle_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            if self.random_pos:
                gx = (np.random.random() - 0.5) * self.rectangle_length
                gy = (np.random.random() - 0.5) * self.rectangle_width
            else:
                gx = np.random.random() * self.rectangle_length * 0.5 * -sign
                gy = (np.random.random() - 0.5) * self.rectangle_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            logging.info("setting robot start and goal state")
            rule = self.train_val_sim
            if self.random_pos:
                if rule == 'circle_crossing':
                    sign = -1 if np.random.random() > 0.5 else 1
                    init_angle = np.random.random() * np.pi * 2
                    goal_angle = np.random.random() * np.pi * 2
                    px = self.circle_radius * np.cos(init_angle)
                    py = self.circle_radius * np.sin(init_angle)
                    gx = self.circle_radius * np.cos(goal_angle)
                    gy = self.circle_radius * np.sin(goal_angle)
                    while norm((gx - px, gy - py)) < self.circle_radius / 2.:
                        goal_angle = np.random.random() * np.pi * 2
                        gx = self.circle_radius * np.cos(goal_angle)
                        gy = self.circle_radius * np.sin(goal_angle)
                    self.robot.set(px, py, gx, py, 0.0, 0.0, 0.0)
                elif rule == 'square_crossing':
                    pass
                elif rule == 'rectangle_crossing':
                    px = (np.random.random() - 0.5) * self.rectangle_length
                    py = (np.random.random() - 0.5) * self.rectangle_width
                    gx = (np.random.random() - 0.5) * self.rectangle_length
                    gy = (np.random.random() - 0.5) * self.rectangle_width
                    while norm((gx - px, gy - py)) < self.rectangle_width / 2.:
                        gx = (np.random.random() - 0.5) * self.rectangle_length
                        gy = (np.random.random() - 0.5) * self.rectangle_width
                    self.robot.set(px, py, gx, gy, 0.0, 0.0, 0.0)
            else:
                if rule == 'circle_crossing':
                    self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
                elif rule == 'square_crossing':
                    self.robot.set(0, -self.square_width, 0, self.square_width, 0, 0, 1.33)
                elif rule == 'rectange_crossing':
                    self.robot.set(self.rectangle_length * .5, -self.rectangle_width * .5, self.rectangle_length * .5,
                                   self.rectangle_width * .5, 0, 0, 1.33)
                elif rule == 'eth':
                    self.robot.set(self.robot_start[0], self.robot_start[1], self.robot_goal[0], self.robot_goal[1],
                                   0.0, 0.0, 0.0)  # px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    # human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    if self.robot.policy.multiagent_training:
                        human_num = self.human_num
                    else:
                        human_num = 1
                        self.human_num = 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action, non_attentive_humans):
        return self.step(action, non_attentive_humans, update=False)

    def step(self, action,  non_attentive_humans, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        ppl_count = 0

        if self.use_eth_data:
            # set current human positions
            for human_num in range(self.human_num):
                # try:
                ped = self.ped_list[human_num]
                if update and self.step_number <= self.ped_limit[human_num]:
                    self.humans[human_num].set(self.x_backup[ped][self.step_number],
                                               self.y_backup[ped][self.step_number],
                                               self.x_backup[ped][-1],
                                               self.y_backup[ped][-1],
                                               self.x_vel_backup[ped][self.step_number],
                                               self.y_vel_backup[ped][self.step_number],
                                               0)
                elif self.step_number <= self.ped_limit[human_num]:
                    self.humans[human_num].set(self.x[ped][self.step_number],
                                               self.y[ped][self.step_number],
                                               self.x[ped][-1],
                                               self.y[ped][-1],
                                               self.x_vel[ped][self.step_number],
                                               self.y_vel[ped][self.step_number],
                                               0)
                #     self.humans[human_num].set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        for num in range(self.human_num):
            try:
                human = self.humans[num]
            except IndexError:
                pdb.set_trace()
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]

            if self.use_eth_data:
                # try:
                ped = self.ped_list[num]
                if update and self.step_number <= self.ped_limit[num]:
                    # Adding human actions from data
                    human_actions.append(ActionXY(self.x_vel_backup[ped][self.step_number],
                                                  self.y_vel_backup[ped][self.step_number]))
                elif self.step_number <= self.ped_limit[num]:
                    # Adding human actions from data
                    human_actions.append(ActionXY(self.x_vel[ped][self.step_number],
                                                  self.y_vel[ped][self.step_number]))
                else:
                    human_actions.append(ActionXY(0.0, 0.0))

            else:
                human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False

        # for i, human in enumerate(self.humans):
        for i in range(self.human_num):
            human = self.humans[i]
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                if update:
                    if i != self.remove_ped:
                        logging.warn("Collision: distance between robot and p{} is {:.2f}".format(i, closest_dist))
                logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                if i == self.remove_ped:
                    logging.debug("Ignoring collision with p{} == remove_ped".format(self.remove_ped))
                    collision = False
                elif self.ignore_collisions:
                    logging.warn("Ignoring Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    collision = False
                else:
                    break
            elif closest_dist < dmin:
                dmin = closest_dist
            elif closest_dist < 3:
                ppl_count += 1


        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius + .45

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        elif self.step_number == 100:
            self.reached_traj_limit = True
            reward = 0
            done = True
            info = Nothing()
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info, ppl_count, self.robot.get_position(), self.robot.get_velocity(), dmin

    def calc_total_distance_travelled_by_ped(self, ped_num):
        """
        :param ped_num:
        :return: total distance travelled by a pedestrian in between start index and goal index
        """
        # get array of all positions in between start index and goal index
        travelled_distance = 0.0
        current_step = self.temp_starting_step

        x_arr = self.x_backup[ped_num][self.start_index:self.start_index + current_step]
        y_arr = self.y_backup[ped_num][self.start_index:self.start_index + current_step]

        for num in range(self.temp_starting_step - 1):
            travelled_distance += np.power(np.power(x_arr[num] - x_arr[num + 1], 2) +
                                           np.power(y_arr[num] - y_arr[num + 1], 2), 1 / 2)

        if (self.temp_starting_step + 1) <= self.num_steps:
            self.temp_starting_step += 1

        return travelled_distance

    def calc_total_distance_travelled_by_robot(self, frame_id):
        """
        :return: Distance travelled by a robot
        """
        # get robot trajectory
        travelled_distance = 0.0
        total_frames = len(self.states)
        for num in range(frame_id - 1):
            travelled_distance += np.power(np.power(self.states[num][0].px - self.states[num + 1][0].px, 2) +
                                           np.power(self.states[num][0].py - self.states[num + 1][0].py, 2), 1 / 2)

        return travelled_distance

    def calc_closest_ped_distance(self, ped_num, frame_id):
        """
        NOTE: LOOK OUT SUSPICOUS!
        :param ped_num:
        :return: closest distance of given pedestrian with others
        """

        frame_id = frame_id + self.start_index

        if self.ped_min_dist_list is None:
            self.ped_min_dist_list = []

        # closest_distance = None
        ped_x = self.x_backup[ped_num][frame_id]
        ped_y = self.y_backup[ped_num][frame_id]

        # if ped_num == 53:
        #     pdb.set_trace()

        # return old values if ped_x and ped_y are zeros
        if (ped_x != 0.0) and (ped_y != 0.0):
            dists = []
            for i in range(len(self.x_backup)):
                if i == ped_num:
                    continue
                else:
                    other_x = self.x_backup[i][frame_id]
                    other_y = self.y_backup[i][frame_id]

                    # do not calculate if other_x and other_y both are zeros
                    if (other_x != 0.0) and (other_y != 0.0):
                        dist = np.power(np.power(ped_x - other_x, 2) +
                                        np.power(ped_y - other_y, 2), 1 / 2)
                        dists.append(dist)

                        # if closest_distance is None:
                        #     closest_distance = dist
                        # elif closest_distance > dist:
                        #     closest_distance = dist
            if len(dists) > 0:
                self.ped_min_dist_list.append(np.min(dists))

        if len(self.ped_min_dist_list) > 0:
            distance_mean = np.mean(self.ped_min_dist_list)
            distance_std = np.std(self.ped_min_dist_list)
            return np.min(self.ped_min_dist_list), distance_mean, distance_std
        else:
            return 0.0, 0.0, 0.0

    def calc_closest_robot_distance(self, frame_id):
        """
        NOTE: LOOK OUT SUSPICOUS!
        :param frame_id:
        :return: closest distance w.r.t pedestrians
        """

        if self.robot_min_dist_list is None:
            self.robot_min_dist_list = []

        # closest_distance = None
        robot_x = self.states[frame_id][0].px
        robot_y = self.states[frame_id][0].py

        frame_id = frame_id + self.start_index

        dists = []
        for i in range(len(self.x_backup)):
            if i == self.remove_ped:
                continue
            else:
                other_x = self.x_backup[i][frame_id]
                other_y = self.y_backup[i][frame_id]

                if (other_x != 0.0) and (other_y != 0.0):
                    dist = np.power(np.power(robot_x - other_x, 2) +
                                    np.power(robot_y - other_y, 2), 1 / 2)

                    dists.append(dist)
                    # if closest_distance is None:
                    #     closest_distance = dist
                    # elif closest_distance > dist:
                    #     closest_distance = dist

        if len(dists) > 0:
            self.robot_min_dist_list.append(np.min(dists))

        if len(self.robot_min_dist_list) > 0:
            distance_mean = np.mean(self.robot_min_dist_list)
            distance_std = np.std(self.robot_min_dist_list)
            return np.min(self.robot_min_dist_list), distance_mean, distance_std
        else:
            return 0.0, 0.0, 0.0

    def calc_robot_ped_path_diff(self, ped_num, frame_id):
        """
        :param ped_num:
        :param frame_id:
        :return: robot and ped. path difference values
        """
        if self.robot_agent_path_diff_list is None:
            self.robot_agent_path_diff_list = []

        robot_x = self.states[frame_id][0].px
        robot_y = self.states[frame_id][0].py

        frame_id = frame_id + self.start_index

        ped_x = self.x_backup[ped_num][frame_id]
        ped_y = self.y_backup[ped_num][frame_id]

        path_diff = np.power(np.power(robot_x - ped_x, 2) +
                             np.power(robot_y - ped_y, 2), 1 / 2)

        self.robot_agent_path_diff_list.append(path_diff)

        return np.mean(self.robot_agent_path_diff_list), np.std(self.robot_agent_path_diff_list), \
               np.max(self.robot_agent_path_diff_list), path_diff

    def get_time_values(self, frame_id):
        """
        :param frame_id:
        :return: time taken to get an action output
        """
        time_now = self.time_diff_list[frame_id]
        time_mean = np.mean(self.time_diff_list[:frame_id + 1])
        time_std = np.std(self.time_diff_list[:frame_id + 1])
        if frame_id > 0:
            time_max = np.max(self.time_diff_list[:frame_id + 1])
        else:
            time_max = 0.0

        return time_now, time_mean, time_std, time_max

    def save_log(self, frame_id, logging_folder_name, file_name):
        """
        :param frame_id:
        :return:
        saves required information at given frame id
        """
        '''
        def metrics(frame, num_peds_follow, remove_ped, cmd_x, cmd_y, \
            x_follow, y_follow, x_nonzero, y_nonzero, \
            robot_history_x, robot_history_y, remove_ped_path_length, \
            chandler, p2w_x, p2w_y, p2w):
            dist_remove_ped = [0. for _ in range(num_peds_follow)]
            dist_robot = [0. for _ in range(num_peds_follow)]

              for ped in range(num_peds_follow):
                dist_robot[ped] = np.power(\
                                       np.power(p2w_x*(cmd_x-x_follow[ped][frame]), 2) + \
                                       np.power(p2w_y*(cmd_y-y_follow[ped][frame]), 2), 1/2)
                if frame < np.size(x_nonzero):
                  dist_remove_ped[ped] = np.power(\
                            np.power(p2w_x*(x_nonzero[frame]-x_follow[ped][frame]), 2) + \
                            np.power(p2w_y*(y_nonzero[frame]-y_follow[ped][frame]), 2), 1/2)
                else:
                  dist_remove_ped[ped] = 2000.

              safety_robot = np.min(dist_robot)
              safety_remove_ped = np.min(dist_remove_ped)

              if frame == 0:
                robot_path_length = 0.
                robot_agent_path_diff = 0.
              else:
                robot_path_length = 0.
                for t in range(frame):
                  robot_path_length = robot_path_length + np.power(\
                          np.power(p2w_x*(robot_history_x[t]-robot_history_x[t+1]), 2) + \
                          np.power(p2w_y*(robot_history_y[t]-robot_history_y[t+1]), 2), 1/2)
                if frame<np.size(x_nonzero):
                  robot_agent_path_diff = np.power(\
                                           np.power(p2w_x*(cmd_x-x_nonzero[frame]), 2) + \
                                           np.power(p2w_y*(cmd_y-y_nonzero[frame]), 2), 1/2)
                else:
                  robot_agent_path_diff = np.power(\
                                           np.power(p2w_x*(cmd_x-x_nonzero[-1]), 2) + \
                                           np.power(p2w_y*(cmd_y-y_nonzero[-1]), 2), 1/2)
              return safety_robot, safety_remove_ped, robot_path_length, \
                     robot_agent_path_diff, remove_ped_path_length

        '''

        ped_closest_distance, ped_dist_mean, ped_dist_std = self.calc_closest_ped_distance(self.remove_ped, frame_id)
        robot_closest_distance, robot_dist_mean, robot_dist_std = self.calc_closest_robot_distance(frame_id)
        path_diff_mean, path_diff_std, path_diff_max, path_diff_now = \
            self.calc_robot_ped_path_diff(self.remove_ped, frame_id)
        ped_distance = self.calc_total_distance_travelled_by_ped(self.remove_ped)
        robot_distance = self.calc_total_distance_travelled_by_robot(frame_id)
        time_now, time_mean, time_std, time_max = self.get_time_values(frame_id)

        print("\n")
        print("REMOVE PED: ", self.remove_ped)
        print("FRAME NUMBER: ", frame_id)
        print("TIME NOW: ", time_now)
        print("TIME MEAN: ", time_mean, "+/-", time_std)
        print("TIME MAX: ", time_max)
        print("SAFETY AGENT ", self.remove_ped, "MIN: ", ped_closest_distance)
        print("SAFETY ROBOT MIN: ", robot_closest_distance)
        print("SAFETY AGENT ", self.remove_ped, " MEAN: ", ped_dist_mean, "+/-", ped_dist_std)
        print("SAFETY ROBOT MEAN: ", robot_dist_mean, "+/-", robot_dist_std)
        print("ROBOT-AGENT PATH DIFF MEAN: ", path_diff_mean, "+/-", path_diff_std)
        print("ROBOT-AGENT PATH DIFF MAX: ", path_diff_max)
        print("ROBOT-AGENT PATH DIFF NOW: ", path_diff_now)
        print("AGENT ", self.remove_ped, " PATH LENGTH: ", ped_distance)
        print("ROBOT PATH LENGTH: ", robot_distance)

        # file_path = logging_folder_name+ '/' + file_name + '_frame_' + str(frame_id) + '.txt'
        file_path = logging_folder_name + '/' + file_name + '.txt'
        with open(file_path, 'a') as text_file:
            print(f"\n")
            print(f"REMOVE PED: {self.remove_ped}", file=text_file)
            print(f"FRAME NUMBER: {frame_id}", file=text_file)
            print(f"TIME NOW: {time_now}", file=text_file)
            print(f"TIME MEAN: {time_mean}+/-{time_std}", file=text_file)
            print(f"TIME MAX: {time_max}", file=text_file)
            print(f"SAFETY AGENT {self.remove_ped} MIN: {ped_closest_distance}", file=text_file)
            print(f"SAFETY ROBOT MIN: {robot_closest_distance}", file=text_file)
            print(f"SAFETY AGENT {self.remove_ped} MEAN: {ped_dist_mean}+/-{ped_dist_std}", file=text_file)
            print(f"SAFETY ROBOT MEAN: {robot_dist_mean}+/-{robot_dist_std}", file=text_file)
            print(f"ROBOT-AGENT PATH DIFF MEAN: {path_diff_mean}+/-{path_diff_std}", file=text_file)
            print(f"ROBOT-AGENT PATH DIFF MAX: {path_diff_max}", file=text_file)
            print(f"ROBOT-AGENT PATH DIFF NOW: {path_diff_now}", file=text_file)
            print(f"AGENT {self.remove_ped} PATH LENGTH: {ped_distance}", file=text_file)
            print(f"ROBOT PATH LENGTH: {robot_distance}", file=text_file)
            print(f" ", file=text_file)
            text_file.close()

    def render(self, mode='human', plots=False, output_file=None, final_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':

            if self.metrics_folder is not None:
                log_name = "remove_ped_" + str(self.remove_ped) + "_start_" + str(self.start_index) + "_steps_" + \
                           str(self.num_steps) + "_vel_" + str(self.robot.v_pref)
                log_directory = self.metrics_folder + log_name
                if not os.path.exists(log_directory):
                    os.makedirs(log_directory)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            # ax.set_xlim(-25, 25)
            # ax.set_ylim(-16, 16)
            ax.set_xlim(-5, 15)
            ax.set_ylim(0, 15)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)

                self.save_log(k, log_directory, log_name)
            plt.legend([robot], ['Robot'], fontsize=16)
            # plt.show()
        elif mode == 'traj_new':

            fig_name = None
            log_name = None
            fig_directory = None
            log_directory = None

            if self.plot_folder is not None and plots:
                fig_name = "remove_ped_" + str(self.remove_ped) + "_start_" + str(self.start_index) + "_steps_" + \
                           str(self.num_steps) + "_vel_" + str(self.robot.v_pref)
                fig_directory = self.plot_folder + fig_name
                if not os.path.exists(fig_directory):
                    os.makedirs(fig_directory)

            if self.metrics_folder is not None:
                log_name = "remove_ped_" + str(self.remove_ped) + "_start_" + str(self.start_index) + "_steps_" + \
                           str(self.num_steps) + "_vel_" + str(self.robot.v_pref)
                log_directory = self.metrics_folder + log_name
                if not os.path.exists(log_directory):
                    os.makedirs(log_directory)

            if plots:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.tick_params(labelsize=16)
                # ax.set_xlim(-25, 25)
                # ax.set_ylim(-16, 16)
                ax.set_xlim(-5, 15)
                ax.set_ylim(0, 15)
                ax.set_xlabel('x(m)', fontsize=16)
                ax.set_ylabel('y(m)', fontsize=16)

                robot_positions = [self.states[i][0].position for i in range(len(self.states))]
                try:
                    human_positions = [[self.states[i][1][j].position for j in range(self.human_list[i])]
                                       for i in range(len(self.states))]
                except IndexError:
                    pdb.set_trace()
                ped_names = []

                goal_iter = iter(self.goal_list)
                goal_pos = None
                goal = None

                for k in range(len(self.states)):
                    if k % 4 == 0 or k == len(self.states) - 1:
                        # ped = self.ped_list[k]
                        robot = plt.Circle(robot_positions[k], .1, fill=True, color='red')
                        # humans = [plt.Circle(human_positions[k][i], .1, fill=False, color='black')
                        # for i in range(self.human_list[k])]
                        if self.remove_ped is not None:
                            humans = [plt.Circle((self.removed_x[k], self.removed_y[k]), .1, fill=False, color='green')]

                        ax.add_artist(robot)
                        for human in humans:
                            ax.add_artist(human)

                    if k != 0:
                        nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                                   (self.states[k - 1][0].py, self.states[k][0].py),
                                                   color='red', ls='solid', linewidth=3.0)
                        # human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                        #                                (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                        #                                color='black', ls='solid', linewidth=2.0)
                        #                     for i in range(self.human_list[k])]
                        if self.remove_ped is not None:
                            human_directions = [plt.Line2D((self.removed_x[k - 1], self.removed_x[k]),
                                                           (self.removed_y[k - 1], self.removed_y[k]),
                                                           color='green', ls='solid', linewidth=3.5)]
                        ax.add_artist(nav_direction)
                        for human_direction in human_directions:
                            if (human_direction._xorig[0] == 0.0) and (human_direction._yorig[0] == 0.0):
                                continue
                            elif (human_direction._xorig[1] == 0.0) and (human_direction._yorig[1] == 0.0):
                                continue
                            ax.add_artist(human_direction)

                        # goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                        #                      color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
                    if goal:
                        goal.remove()
                    try:
                        goal_pos = next(goal_iter)
                        goal = mlines.Line2D([goal_pos[0]], [goal_pos[1]],
                                             color=goal_color, marker='*', linestyle='None', markersize=15,
                                             label='Goal')
                    except:
                        print("keeping last goal")
                        goal = mlines.Line2D([goal_pos[0]], [goal_pos[1]],
                                             color=goal_color, marker='*', linestyle='None', markersize=15,
                                             label='Goal')
                    goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                         color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
                    ax.add_artist(goal)

                    plt.legend((robot, goal), ('Robot', 'goal'), fontsize=10)
                    if (self.remove_ped is not None) and (k != 0):
                        plt.legend((nav_direction, human_directions[-1], goal),
                                   ('Robot', 'Removed Ped', 'goal'),
                                   fontsize=10)
                    # save figure
                    plt.savefig(fig_directory + '/' + fig_name + '_frame_%f.png' % k)
                    self.save_log(k, log_directory, log_name)
                    if final_file is not None and k == len(self.states) - 1:
                        self.save_log(k, self.metrics_folder, final_file)
            else:
                for k in range(len(self.states)):
                    self.save_log(k, log_directory, log_name)
                    if final_file is not None and k == len(self.states) - 1:
                        self.save_log(k, self.metrics_folder, final_file)

            # plt.show()
        elif mode == 'video':
            # fig, ax = plt.subplots(figsize=(7, 7))
            fig, ax = plt.subplots(figsize=(19.2, 10.8))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 15)
            ax.set_ylim(0, 15)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 15, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-23.5, 15 - 1 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                               agent_state.py + radius * np.sin(
                                                                                   theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
