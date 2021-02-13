
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
import datetime
from copy import deepcopy

####################SPECIAL LIBRARIES
import scipy as sp
import george
import autograd as ag
from autograd import value_and_grad
#reverse mode is more efficiet for scalar valued functions
import autograd.numpy as np
# import numpy as np
from autograd.numpy.linalg import solve
import mpmath as mp
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from numba import jit
from scipy import optimize

from crowd_sim.envs.objectives import igp.so_igp_diag_objectives as so_diagonal
from crowd_sim.envs.objectives import igp.so_igp_dense_objectives as so_dense
from crowd_sim.envs.objectives import igp.fo_igp_dense_objectives as fo_dense


    def __init__(self):
        """
        """
        super().__init__()
        self.name = 'so_IGP'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

        self.fo = False # FIRST OR SECOND ORDER INTERACTION?
        self.normalize = False

        self.ess_off = False
        self.ess_newton = True

        self.ess_boost = True
        self.ess_num_peds = 3

        self.ess_limit = True
        self.ess_limit_num = 7

        self.opt_iter_robot = False
        self.opt_iter_all = True
        self.ll_converge = False
        self.vg = False
        self.hess_opt = False
        self.tol = 1e-8
        self.opt_method = 'Newton-CG'
        self.dwa = False  # COUPLED IGP
        self.cov_eps = 0.000001

        self.mc = False
        self.num_mc_samples = 0

        self.linear_diag = False
        self.conditioned = True

        self.actuate_distance = True
        self.actuate_to_step = False
        self.actuate_to_index = False


################SAMPLE TESTING PARAMETERS
        self.random_sample = False
        self.num_random_samples = 0
        self.num_show = 0

        self.var_sample = True
        self.num_var_samples = 3 #8
        self.var_ratio = 1 #0.5
        self.num_var_show = 0

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        return


    def set_phase(self, phase):
        return

    def Tdex_newton(self):

        return Tdex

    def igp(self,fo, diagonal, ess_boost, ess_num_peds, ess_newton, Tdex_max, frame, \
            num_peds_follow, max_vel_robot, max_vel_ped, \
            robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, \
            vel_x, vel_y, cmd_x, cmd_y, \
            x_follow, y_follow, x_obs, y_obs, x_obs_un, y_obs_un, \
            err_magnitude_ped, err_magnitude_robot, end_point_err_ped, \
            end_point_err_robot, buffer_robot, buffer_ped, obs_duration_robot, \
            obs_duration_ped, gp_x, gp_y, num_random_samples, num_var_samples, \
            random_sample, var_sample, tol, ess_time_array, \
            ess_array, ess_off, normalize, \
            ll_converge, conditioned, var_ratio, data_set, support_boost, \
            goal_dex, x_nonzero, y_nonzero, ess_limit, ess_limit_num, \
            normal_vel, full_traj, dwa, cov_eps, linear_diag, vg, hess_opt, \
            opt_iter_robot, opt_iter_all, agent_disrupt, robot_agent_disrupt, \
            opt_method):

        Tdex = Tdex_newton(Tdex_max, num_peds, self.max_speed,  \                       robot_start_x, robot_start_y, \
                          robot_goal_x, robot_goal_y, \
                          vel_x, vel_y, cmd_x, cmd_y, \
                          data_set, support_boost, \
                          goal_dex, x_nonzero, y_nonzero, \
                          normal_vel, full_traj)

        return x_obs, y_obs, x_obs_un, y_obs_un, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, ess, \
                top_Z_indices, ess_array, ess_time, ess_time_array, ess_ave_time, \
                ess_std_time, optima, optimal_ll, optima_dex, num_optima, \
                norm_likelihood, global_optima_dex, time_gp, \
                agent_disrupt, robot_agent_disrupt


    def predict(self, state):
        """
        (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,self.attentive)
        """
        self_state = state.self_state

        T = np.size(robot_mu_x)


        robot_history_x, robot_history_y, cmd_x, cmd_y, vel_x, vel_x \
            = actuate(a, T, x_obs, y_obs, max_vel_robot, robot_history_x, \
                      robot_history_y, frame, x_follow, y_follow, num_peds_follow, \
                      p2w_x, p2w_y, p2w, actuate_distance, actuate_to_step, actuate_to_index)
        action = ActionXY(vel_x, vy)

        return action