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
        self.obsv_xt = []
        self.obsv_yt = []
        self.obsv_len = 2

    def configure(self, config):
        return

    def set_phase(self, phase):
        return



    def weight_compute(self, a, h, obj_thred, max_iter):
        """
        core of igp computation, compute all samples' weights
        :param a: safety region parameter
        :param h: safety magnitude parameter
        :param obj_thred: terminal condition for iterations
        :param max_iter: maximal number of iterations allowed
        :return: optimized weights
        """
        samples_x = self.samples_x.copy()
        samples_y = self.samples_y.copy()
        weights = compute(samples_x, samples_y, self.human_num, self.num_samples, self.pred_len,
                          a, h, obj_thred, max_iter)
        self.weights = weights.copy()

    def predict(self, state):
        """
        generate velocity command for robot
        :param dt: interval of simulation step
        :return: velocity command
        """

        robot_state = state.self_state
        robot_x = robot_state.px
        robot_y = robot_state.py

        humans_state = state.human_states
        num_agents = len(state.human_states)

        # add the observations
        obsv_x, obsv_y = add_observation(humans_state.px, humans_state.py):


        # do gp regression here, this generate initial preference distributions of each agent
        # the returned "pred_len" is not necessary
        pred_len = gp_predict(robot_state.gx, robot_state.gy, obsv_len=self.obsv_len, obsv_err_magnitude=0.01,
                                  vel=0.028, dt=self.dt, cov_thred_x=1e-04, cov_thred_y=1e-04)

        samples_x, samples_y = gp_sampling(num_samples=self.num_samples)

        weights = weight_compute(self.a, self.h, self.obj_thred, self.max_iter)

        # select optimal sample trajectory as reference for robot navigation
        opt_idx = np.argmax(weights)
        opt_robot_x = samples_x[self.num_samples + opt_idx][0]
        opt_robot_y = samples_y[self.num_samples + opt_idx][0]

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
