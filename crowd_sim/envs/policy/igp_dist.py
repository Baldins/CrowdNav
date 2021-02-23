import numpy as np
from scipy.stats import multivariate_normal as mvn
import george
from copy import copy
import math
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
        self.dt = 1
        self.pred_len = 0
        self.num_samples = 100
        self.obsv_xt = []
        self.obsv_yt = []
        self.obsv_x = []
        self.obsv_y = []
        self.obsv_len = 2
        self.count = 0
        self.vel = 1
        self.collision_thresh = 0.2
        self.len_scale = 5
        self.num_agents = 22
        self.cov_thred_x = 0.02
        self.cov_thred_y = 0.02
        self.obsv_err_magnitude = 0.001
        self.a = 0.03 # a controls safety region
        self.h = 1.0  # h controls safety weight
        self.obj_thred = 0.001  # terminal condition for optimization
        self.max_iter = 500  # maximal number of iterations allowed
        self.weights = np.zeros(self.num_agents)

        self.gp_pred_x = [0. for _ in range(self.num_agents)]
        self.gp_pred_x_cov = [0. for _ in range(self.num_agents)]
        self.gp_pred_y = [0. for _ in range(self.num_agents)]
        self.gp_pred_y_cov = [0. for _ in range(self.num_agents)]
        self.samples_x = [0. for _ in range(self.num_agents)]
        self.samples_y = [0. for _ in range(self.num_agents)]

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


        self.trajs_x = []
        self.trajs_y = []

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
        self.count+=1
        obsv_xt =[]
        obsv_yt =[]
        for i, human in enumerate(state.human_states):
            obsv_xt.append(human.px)
            obsv_yt.append(human.py)
            idx = i
        # ## add observation
        obsv_xt.append(robot_x)
        obsv_yt.append(robot_y)
        robot_idx = idx + 1

        self.obsv_x, self.obsv_y = add_observation(obsv_xt, obsv_yt, self.obsv_x, self.obsv_y)



        # vel = robot_state.v_pref
        vel = self.vel
        print(self.count)
        if self.count > self.obsv_len:
            print("IGP")
            opt_robot_x, opt_robot_y, traj_x, traj_y = igp(state, self.obsv_x, self.obsv_y, robot_idx, self.num_samples, self.num_agents, self.len_scale,
                                            self.a, self.h, self.obj_thred, self.max_iter, vel, self.dt,
                                            self.obsv_len, self.obsv_err_magnitude, self.cov_thred_x, self.cov_thred_y,
                                            self.gp_x, self.gp_y, self.gp_pred_x, self.gp_pred_x_cov, self.gp_pred_y, self.gp_pred_y_cov, self.samples_x, self.samples_y, self.weights)
            print("opt robot" ,opt_robot_y)

            close_obst = []
            close_obst2 = []
            self.trajs_x = traj_x
            self.trajs_y = traj_y

            # print("self_traj", self.trajs)
            #
            # for k in range(len(self.trajs[0])):
            #     print("traj = ",k, self.trajs[0][k], self.trajs[1][k])


            for k in range(self.num_agents -1):
                # print(traj_x[k][0])
                distance = math.sqrt((traj_x[k][0] - opt_robot_x) ** 2 + (traj_y[k][0] - opt_robot_y) ** 2)
                # print("distance ", distance)
                if (distance <= self.collision_thresh):
                    close_obst.append([traj_x[k], traj_y[k], distance])
                # generate velocity command

            if (len(close_obst) == 0 ):  # no obstacles

                # vel_x = (opt_robot_x - robot_x) / self.dt
                # vel_y = (opt_robot_y - robot_y) / self.dt


                theta = np.arctan2(opt_robot_y - robot_y, opt_robot_x - robot_x)
                vel_x = np.cos(theta) * robot_state.v_pref
                vel_y = np.sin(theta) * robot_state.v_pref
            else:
                # vel_x = 0.00000001 * (opt_robot_x - robot_x) / self.dt
                # vel_y = 0.00000001 * (opt_robot_y - robot_y) / self.dt
                theta = np.arctan2(opt_robot_y - robot_y, opt_robot_x - robot_x)

                vel_x = 0.000001 * np.cos(theta) * robot_state.v_pref
                vel_y = 0.000001 * np.sin(theta) * robot_state.v_pref

            action = ActionXY(vel_x, vel_y)
        else:
            print("Linear")
            theta = np.arctan2(state.self_state.gy - robot_y, state.self_state.gx - robot_x)
            vx = np.cos(theta) * state.self_state.v_pref
            vy = np.sin(theta) * state.self_state.v_pref
            action = ActionXY(vx, vy)

        self.obsv_xt = []
        self.obsv_yt = []
        return action

    def get_traj(self):
        return self.trajs_x, self.trajs_y

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

