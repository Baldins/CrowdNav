####################GENERIC LIBRARIES
from copy import deepcopy
import math
import time
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

import numpy as np
import math
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from copy import deepcopy
from numba import jit
import scipy as sp
from scipy import optimize

from crowd_sim.envs.objectives import igp.so_igp_diag_objectives as so_diagonal
from crowd_sim.envs.objectives import igp.so_igp_dense_objectives as so_dense
from crowd_sim.envs.objectives import igp.fo_igp_dense_objectives as fo_dense



####################GENERIC LIBRARIES
from copy import deepcopy
import math
import time
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

class so_IGP(Policy):
    def __init__(self):

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
        self.boost_factor = 1.1

        ################TESTING PARAMETERS

        self.fo = False  # FIRST OR SECOND ORDER INTERACTION?
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

        # opt_method = 'trust-krylov'
        self.opt_method = 'Newton-CG'
        # opt_method = 'dogleg'
        # opt_method = 'trust-ncg'

        self.dwa = False  # COUPLED IGP
        self.cov_eps = 0.000001

        self.mc = False
        self.num_mc_samples = 0

        self.linear_diag = False
        self.conditioned = True

        self.actuate_distance = True
        self.actuate_to_step = False
        self.actuate_to_index = False


    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def Tdex_newton(self):
        """
        input : Tdex_max, frame, num_peds, max_vel_robot, \
                robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, \
                vel_x, vel_y, cmd_x, cmd_y, data_set, support_boost, \
                goal_dex, x_follow, y_follow, normal_vel, follow_traj
        output: Tdex, robot_goal_x, robot_goal_y
        """

    def gp_computation_newton(self):
        """
        input :frame, num_peds, x, y, x_obs, y_obs, \
                x_obs_un, y_obs_un, err_magnitude_ped, err_magnitude_robot, \
                end_point_err_ped, end_point_err_robot, buffer_robot, buffer_ped, \
                robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, \
                cmd_x, cmd_y, obs_duration_robot, obs_duration_ped, Tdex, gp_x, gp_y, \
                num_intents, num_var_samples, var_ratio, dwa, cov_eps, linear_diag
        return : gp_x, gp_y, mu_linear_conditioned_x, mu_linear_conditioned_y, \
                mu_linear_un_x, mu_linear_un_y, \
                cov_linear_conditioned_x, cov_linear_conditioned_y, \
                cov_un_x, cov_un_y, x_obs, y_obs,\
                x_obs_un, y_obs_un, joint_sample_x, joint_sample_y, \
                var_samples_x, var_samples_y, time_gp
        """

    def fo_ess_compute_newton(self):

        """
        input: diagonal, num_peds, robot_mu_x, robot_mu_y, \
                ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
                inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, \
                inv_cov_ped_x, inv_cov_ped_y, \
                one_over_cov_sum_x, one_over_cov_sum_y, normalize
        outpu: ess, top_Z_indices

        """


    def optimize_iterate():
        """
        input: fo, tol, diagonal, frame, z0, num_peds, ess, \
      robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
      inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, \
      inv_cov_ped_x, inv_cov_ped_y, one_over_cov_sum_x, one_over_cov_sum_y,  \
      one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, ll_converge, T, \
      opt_iter_robot, opt_iter_all):
        output : z
        """


    def igp(self):



    def actuate(self):
        """
        input : f, T, x_obs, y_obs, max_vel_robot, \
				    robot_history_x, robot_history_y, frame, \
				    x_follow, y_follow, num_peds_follow, p2w_x, p2w_y, p2w, \
				    actuate_distance, actuate_to_step, actuate_to_index
        returns: robot_history_x, robot_history_y, cmd_x, cmd_y, vel_x, vel_y
        """

    def gp_computation_newton():
        """
        input = frame, num_peds, x, y, x_obs, y_obs, \
                              x_obs_un, y_obs_un, err_magnitude_ped, err_magnitude_robot, \
                              end_point_err_ped, end_point_err_robot, buffer_robot, buffer_ped, \
                              robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, \
                              cmd_x, cmd_y, obs_duration_robot, obs_duration_ped, Tdex, gp_x, gp_y, \
                              num_intents, num_var_samples, var_ratio, dwa, cov_eps, linear_diag

        """

    return robot_history_x, robot_history_y, cmd_x, cmd_y, vel_x, vel_y

    def predict(self, state):
        "(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,self.attentive)"
        self_state = state.self_state


        action = ActionXY(vx, vy)

        return action