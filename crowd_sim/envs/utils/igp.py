import numpy as np
from scipy.stats import multivariate_normal as mvn
import george
from copy import copy


class igp:
    """parent class for all igp variants"""

    def __init__(self, num_agents, robot_idx):
        """
        class constructor
        :param num_agents: number of all agents (including the robot)
        :param robot_idx: index of the robot among all agents
        """
        # initialize basic parameters
        self.num_agents = num_agents
        self.robot_idx = robot_idx
        self.pred_len = 0
        self.num_samples = 0
        self.weights = np.zeros(num_agents)

        # initialize gp related variables
        self.obsv_x = []
        self.obsv_y = []
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

    def add_observation(self, obsv_xt, obsv_yt):
        """
        add observation of all agents
        :param obsv_xt: a list/array of all agents' x positions at current time step
        :param obsv_yt: a list/array of all agents' y positions at current time step
        :return: None
        """
        self.obsv_x.append(obsv_xt)
        self.obsv_y.append(obsv_yt)

    def gp_predict(self, goals_x, goals_y, vel, dt,
                   obsv_len, obsv_err_magnitude,
                   cov_thred_x, cov_thred_y):
        """
        gp regression for all agents' trajectories
        :param goals_x: a list/array of all agents' navigation goals' x positions
        :param goals_y: a list/array of all agents' navigation goals' y positions
        :param vel: desired velocity of the robot
        :param dt: time interval of simulation
        :param obsv_len: length of the observations of all agents to be used for regression
        :param obsv_err_magnitude: noise magnitude of observations (assume x and y positions have the same magnitude)
        :param cov_thred_x: constraint on the regression uncertainty on x position
        :param cov_thred_y: constraint on the regression uncertainty on y position
        :return: length of the predicted trajectory
        """
        # extract current robot pose from collected observations
        curr_robot_x = self.obsv_x[-1][self.robot_idx]
        curr_robot_y = self.obsv_y[-1][self.robot_idx]
        # use the distance between current pose and navigation goal (of the robot) to determine length of prediction
        dist = np.sqrt((goals_x[self.robot_idx] - curr_robot_x) ** 2 + (goals_y[self.robot_idx] - curr_robot_y) ** 2)
        pred_len = int(dist / (vel * dt)) + 1

        # indices/labels for gp regression
        pred_t = np.arange(pred_len) + obsv_len
        obsv_t = np.array([i for i in range(obsv_len)] + [pred_len + obsv_len])
        obsv_err = np.ones_like(obsv_t) * obsv_err_magnitude
        for i in range(self.num_agents):
            # here we do gp regression on x coordinate
            self.gp_x[i].compute(obsv_t, obsv_err)
            obsv_x = []
            for j in range(obsv_len):
                obsv_x.append(self.obsv_x[-obsv_len + j][i])
            obsv_x.append(goals_x[i])
            pred_x, pred_x_cov = self.gp_x[i].predict(obsv_x, pred_t, return_cov=True)
            scale_x = np.diag(pred_x_cov).max() / (cov_thred_x * pred_len)
            pred_x_cov /= scale_x
            self.gp_pred_x[i] = copy(pred_x)
            self.gp_pred_x_cov[i] = copy(pred_x_cov)

            # here we do gp regression on y coordinate
            self.gp_y[i].compute(obsv_t, obsv_err)
            obsv_y = []
            for j in range(obsv_len):
                obsv_y.append(self.obsv_y[-obsv_len + j][i])
            obsv_y.append(goals_y[i])
            pred_y, pred_y_cov = self.gp_y[i].predict(obsv_y, pred_t, return_cov=True)
            scale_y = np.diag(pred_y_cov).max() / (cov_thred_y * pred_len)
            pred_y_cov /= scale_y
            self.gp_pred_y[i] = copy(pred_y)
            self.gp_pred_y_cov[i] = copy(pred_y_cov)

        self.pred_len = pred_len
        return pred_len

    def gp_sampling(self, num_samples, include_mean=True):
        """
        generate samples from gp posteriors
        :param num_samples: number of samples for each agent
        :param include_mean: whether include gp mean as a sample
        :return: generated samples
        """
        self.num_samples = num_samples
        samples_x = np.zeros((self.num_agents * num_samples, self.pred_len), dtype=np.float32)
        samples_y = np.zeros((self.num_agents * num_samples, self.pred_len), dtype=np.float32)
        for i in range(self.num_agents):
            if self.pred_len > 1:
                samples_x[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=self.gp_pred_x[i],
                                                                            cov=self.gp_pred_x_cov[i], size=num_samples)
                samples_y[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=self.gp_pred_y[i],
                                                                            cov=self.gp_pred_y_cov[i], size=num_samples)
            else:
                samples_x[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=self.gp_pred_x[i],
                                                                            cov=self.gp_pred_x_cov[i],
                                                                            size=num_samples)[:, None]
                samples_y[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=self.gp_pred_y[i],
                                                                            cov=self.gp_pred_y_cov[i],
                                                                            size=num_samples)[:, None]
        if include_mean:
            # if include gp mean as sample, replace the first sample as gp mean
            for i in range(self.num_agents):
                samples_x[i * num_samples] = self.gp_pred_x[i].copy()
                samples_y[i * num_samples] = self.gp_pred_y[i].copy()

        self.samples_x = samples_x.copy()
        self.samples_y = samples_y.copy()
        return samples_x.copy(), samples_y.copy()
