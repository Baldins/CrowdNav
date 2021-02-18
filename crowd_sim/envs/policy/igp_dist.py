import numpy as np
from scipy.stats import multivariate_normal as mvn
import george
from copy import copy

from crowd_sim.envs.utils.igp_dist_utils import compute
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

from crowd_sim.envs.policy.igp_fun import *

class Igp_Dist(Policy):
    """igp-dist class"""

    def __init__(self):
        """
        class constructor (same as the parent class)
        :param num_agents: number of all agents (including the robot)
        :param robot_idx: index of the robot among all agents
        """
        super().__init__()
        self.name = 'IGP-DIST'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.max_speed = 1
        self.dt = 0.1
        self.pred_len = 0
        self.num_samples = 500
        self.obsv_x = []
        self.obsv_y = []
        self.obsv_len = 2
        self.num_agents = 4
        self.cov_thred_x = 1e-04
        self.cov_thred_y = 1e-04
        self.obsv_err_magnitude = 0.01
        self.a = 0.004  # a controls safety region
        self.h = 1.0  # h controls safety weight
        self.obj_thred = 0.001  # terminal condition for optimization
        self.max_iter = 150  # maximal number of iterations allowed


        # initialize gp kernels
        k1 = george.kernels.LinearKernel(np.exp(2 * 2.8770), order=1)
        k2 = 2.0 * george.kernels.Matern52Kernel(5.0)
        hyper_x = np.array([0.2925377, 9.81793274, 8.20649053])
        hyper_y = np.array([11.0121243, 10.98684898, 8.41968977])
        self.gp_x = [george.GP(k1 + k2, fit_white_noise=True) for _ in range(self.num_agents)]
        self.gp_y = [george.GP(k1 + k2, fit_white_noise=True) for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.gp_x[i].set_parameter_vector(hyper_x)
            self.gp_y[i].set_parameter_vector(hyper_y)

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        generate velocity command for robot
        :param dt: interval of simulation step
        :return: velocity command
        """
        robot_state = state.self_state
        robot_x = robot_state.px
        robot_y = robot_state.py

        self.num_agents = len(state.human_states)

        for i, human in enumerate(state.human_states):
            self.obsv_x.append(human.px)
            self.obsv_y.append(human.py)
            idx = i
        ## add observation
        self.obsv_x.append(robot_x)
        self.obsv_y.append(robot_y)
        robot_idx = idx + 1


        vel = robot_state.v_pref

        opt_robot_x, opt_robot_y = igp(state, self.obsv_x, self.obsv_y, robot_idx, self.num_samples, self.num_agents,
                                        self.a, self.h, self.obj_thred, self.max_iter, vel, self.dt,
                                        self.obsv_len, self.obsv_err_magnitude, self.cov_thred_x, self.cov_thred_y,
                                        self.gp_x, self.gp_y)

        # generate velocity command
        vel_x = (opt_robot_x - robot_x) / self.dt
        vel_y = (opt_robot_y - robot_y) / self.dt

        action = ActionXY(vel_x, vel_y)

        return action

    def get_opt_traj(self):
        """
        get joint optimal trajectories of all agents
        :return: joint optimal trajectories
        """
        traj_x = np.zeros((self.num_agents, self.pred_len))
        traj_y = np.zeros((self.num_agents, self.pred_len))
        for i in range(self.num_agents):
            opt_idx = np.argmax(self.weights[i])
            traj_x[i] = self.samples_x[i * self.num_samples + opt_idx].copy()
            traj_y[i] = self.samples_y[i * self.num_samples + opt_idx].copy()

        return traj_x, traj_y

