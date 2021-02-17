
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

from crowd_sim.envs.objectives.fo_igp_ess_compute_newton import fo_ess_compute_newton
from crowd_sim.envs.functions.igp_ess_compute_Z import ess_compute_Z

from crowd_sim.envs.functions.so_igp_Tdex_newton import Tdex_newton
from crowd_sim.envs.functions.so_igp_gp_computation_newton import gp_computation_newton
from crowd_sim.envs.functions.so_igp_rename import rename
from crowd_sim.envs.functions.so_igp_optimize_newton import optimize_newton


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

            ### Tedx
            Tdex, robot_goal_x, robot_goal_y \
                = Tdex_newton(Tdex_max, frame, num_peds, max_vel_robot, \
                              robot_start_x, robot_start_y, \
                              robot_goal_x, robot_goal_y, \
                              vel_x, vel_y, cmd_x, cmd_y, \
                              data_set, support_boost, \
                              goal_dex, x_nonzero, y_nonzero, \
                              normal_vel, full_traj)


            ### GP computation newton
            gp_x, gp_y, mu_linear_conditioned_x, mu_linear_conditioned_y, \
            mu_linear_un_x, mu_linear_un_y, \
            cov_linear_conditioned_x, cov_linear_conditioned_y, cov_un_x, cov_un_y, \
            x_obs, y_obs, x_obs_un, y_obs_un, joint_sample_x, joint_sample_y, \
            var_samples_x, var_samples_y, time_gp \
                = gp_computation_newton(frame, num_peds, x, y, x_obs, y_obs, \
                                        x_obs_un, y_obs_un, err_magnitude_ped, err_magnitude_robot, \
                                        end_point_err_ped, end_point_err_robot, buffer_robot, buffer_ped, \
                                        robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, \
                                        cmd_x, cmd_y, obs_duration_robot, obs_duration_ped, \
                                        Tdex, gp_x, gp_y, num_intents, num_var_samples, var_ratio, \
                                        dwa, cov_eps, linear_diag)

            ### Rename
            robot_mu_x, robot_mu_y, robot_cov_x, robot_cov_y, \
            inv_var_robot_x, inv_var_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
            ped_mu_x, ped_mu_y, ped_cov_x, ped_cov_y, cov_sum_x, cov_sum_y, \
            inv_var_ped_x, inv_var_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
            inv_cov_sum_x, inv_cov_sum_y, one_over_robot_cov_x, one_over_robot_cov_y, \
            one_over_ped_cov_x, one_over_ped_cov_y, \
            one_over_cov_sum_x, one_over_cov_sum_y, \
            one_over_cov_sumij_x, one_over_cov_sumij_y = rename(num_peds, conditioned, \
                                                                mu_linear_conditioned_x, mu_linear_conditioned_y, \
                                                                mu_linear_un_x, mu_linear_un_y, \
                                                                cov_linear_conditioned_x, cov_linear_conditioned_y, \
                                                                cov_un_x, cov_un_y, linear_diag)



            ####################ESS COMPUTING
            if(ess_off):
            ess = num_peds
            top_Z_indices = np.asarray(range(num_peds))
            else:
            if(ess_newton):
              ess, top_Z_indices \
                 = fo_ess_compute_newton(diagonal, num_peds, robot_mu_x, robot_mu_y, \
                     ped_mu_x, ped_mu_y, robot_cov_x, robot_cov_y, inv_cov_robot_x, \
                     inv_cov_robot_y, ped_cov_x, ped_cov_y, inv_cov_ped_x, \
                     inv_cov_ped_y, one_over_cov_sum_x, one_over_cov_sum_y, \
                     normalize)
            else:
              ess, top_Z_indices \
                 = ess_compute_Z(diagonal, num_peds, robot_mu_x, robot_mu_y, ped_mu_x, \
                     ped_mu_y, robot_cov_x, robot_cov_y, inv_cov_robot_x, \
                     inv_cov_robot_y, ped_cov_x, ped_cov_y, inv_cov_ped_x, \
                     inv_cov_ped_y, one_over_cov_sum_x, one_over_cov_sum_y, \
                     normalize)
            if(ess_boost):
              if(ess < ess_num_peds):
                ess = ess_num_peds
                ess_array[frame] = ess
            if(ess_limit):
              if(ess > ess_limit_num):
                ess = ess_limit_num
                ess_array[frame] = ess

              ########################OPTIMIZATION
              T = np.size(robot_mu_x)

              optima, ll, ess_time, ess_time_array, ess_ave_time, ess_std_time, \
              agent_disrupt, robot_agent_disrupt \
                  = optimize_newton(fo, diagonal, random_sample, var_sample, \
                                    tol, num_intents, num_var_samples, T, joint_sample_x, joint_sample_y, \
                                    var_samples_x, var_samples_y, frame, num_peds, ess_time_array, ess, \
                                    top_Z_indices, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
                                    robot_cov_x, robot_cov_y, inv_cov_robot_x, inv_cov_robot_y, \
                                    ped_cov_x, ped_cov_y, inv_cov_ped_x, inv_cov_ped_y, \
                                    one_over_cov_sum_x, one_over_cov_sum_y, \
                                    one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, \
                                    ll_converge, vg, hess_opt, opt_iter_robot, opt_iter_all, \
                                    x, y, agent_disrupt, robot_agent_disrupt, opt_method)

              global_optima_dex = np.argmin(ll)

              optimal_ll, optima_dex = np.unique(ll, return_index=True)
              likelihood = np.exp(-optimal_ll)
              norm_likelihood = likelihood / np.sum(likelihood)

              num_optima = np.size(optimal_ll)
              time_gp = ess * time_gp


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

        max_vel_ped = self.max_speed

        goal_noise_multiplier_robot = 1. / 1.
        goal_noise_multiplier_ped = 10.

        normal_vel = 12.
        support_boost = math.trunc((max_vel_robot / normal_vel) * boost_factor * 1e3) / 1e3

        radius = 3.
        show_radius = False

        x, y, x_follow, y_follow, num_peds_real, num_peds_follow, num_frames, \
        p2w_x, p2w_y, p2w, scaling \
            = import_eth_data(chandler, data_set, remove_ped, home_dir, remove_ped_start)

        x_nonzero = deepcopy(x[remove_ped][np.nonzero(x[remove_ped][:])])
        y_nonzero = deepcopy(y[remove_ped][np.nonzero(y[remove_ped][:])])

        sld = np.linalg.norm([x_nonzero[0] - x_nonzero[-1], \
                              y_nonzero[0] - y_nonzero[-1]])
        sld_now = np.linalg.norm([p2w_x * (x_nonzero[0] - x_nonzero[goal_dex]), \
                                  p2w_y * (y_nonzero[0] - y_nonzero[goal_dex])])
        ####################IMPORT START AND GOAL INITIAL DATA
        robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, Tdex_max, \
        remove_ped_path_length = start_goal(x_nonzero, y_nonzero, remove_ped, \
                                            sld, chandler, p2w_x, p2w_y, p2w, \
                                            data_set, goal_dex, full_traj)
        robot_history_x = robot_start_x
        robot_history_y = robot_start_y
        ####################ARRAYS
        x_obs = {}
        y_obs = {}

        x_obs_un = {}
        y_obs_un = {}

        ess_array = [0. for _ in range(num_frames)]

        ess_time_array = [0. for _ in range(num_frames)]

        safety_robot = [0. for _ in range(num_frames)]
        safety_remove_ped = [0. for _ in range(num_frames)]
        robot_agent_path_diff = [0. for _ in range(num_frames)]

        local_density = [0. for _ in range(num_frames)]

        time = [0. for _ in range(num_frames)]

        agent_disrupt = [0. for _ in range(num_frames)]
        robot_agent_disrupt = [0. for _ in range(num_frames)]
        ####################GP PARAMETERS
        buffer_ped = 2
        buffer_robot = 1

        obs_duration_robot = 0
        obs_duration_ped = 0
        # ERR MUST MATCH THE HYPERPARAMETERS
        # magic: 2 and
        err_magnitude_ped = 2.
        err_magnitude_robot = 5.

        end_point_err_ped = goal_noise_multiplier_ped * err_magnitude_ped
        end_point_err_robot = goal_noise_multiplier_robot * err_magnitude_robot

        ####################INIT GPs
        os.chdir('../crwod_sim/utils/gp_hyperparams_pixels/k12/')
        gp_x, gp_y = gp_init(num_peds_follow, home_dir)
        os.chdir(str(home_dir))
        ####################BEGIN SIMULATION
        for frame in range(num_frames):
            print(k)
            ####################OBS_DURATION
            if obs_duration_robot < buffer_robot:
                obs_duration_robot = obs_duration_robot + 1
            else:
                obs_duration_robot = buffer_robot
            if obs_duration_ped < buffer_ped:
                obs_duration_ped = obs_duration_ped + 1
            else:
                obs_duration_ped = buffer_ped

            if frame == 0:
                vel_x = 0.
                vel_y = 0.
                cmd_x = 0.
                cmd_y = 0.
            #######################IGP
            robot_goal_x, robot_goal_y, gp_x, gp_y, x_obs, y_obs, x_obs_un, \
            y_obs_un, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, ess, \
            top_Z_indices, ess_array, ess_time, ess_time_array, ess_ave_time, \
            ess_std_time, optima, optimal_ll, optima_dex, num_optima, \
            norm_likelihood, global_optima_dex, time_gp, \
            agent_disrupt, robot_agent_disrupt \
                = igp(fo, diagonal, ess_boost, ess_num_peds, ess_newton, Tdex_max, frame, \
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
                      opt_method)

            time[frame] = ess_time_array[frame] + time_gp
            ####################ACTUATE
            T = np.size(robot_mu_x)
            if opt_iter_robot or opt_iter_all:
                a = optima[global_optima_dex]
            else:
                a = optima[global_optima_dex].x
            robot_history_x, robot_history_y, cmd_x, cmd_y, vel_x, vel_y \
                = actuate(a, T, x_obs, y_obs, max_vel_robot, robot_history_x, \
                          robot_history_y, frame, x_follow, y_follow, num_peds_follow, \
                          p2w_x, p2w_y, p2w, actuate_distance, actuate_to_step, actuate_to_index)

        action = ActionXY(vel_x, vy)

        return action