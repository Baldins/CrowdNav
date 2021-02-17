import numpy as np
from scipy.stats import multivariate_normal as mvn
import george
from copy import copy


def add_observation(obsv_xt, obsv_yt):
    """
    add observation of all agents
    :param obsv_xt: a list/array of all agents' x positions at current time step
    :param obsv_yt: a list/array of all agents' y positions at current time step
    :return: None
    """
    obsv_x.append(obsv_xt)
    obsv_y.append(obsv_yt)
    return obsv_x, obsv_y

def gp_sampling(num_samples, num_agents, pred_len, gp_pred_x, gp_pred_x_cov,gp_pred_y,gp_pred_y_cov, include_mean=True):
    """
    generate samples from gp posteriors
    :param num_samples: number of samples for each agent
    :param include_mean: whether include gp mean as a sample
    :return: generated samples
    """
    samples_x = np.zeros((num_agents * num_samples, pred_len), dtype=np.float32)
    samples_y = np.zeros((num_agents * num_samples, pred_len), dtype=np.float32)
    for i in range(num_agents):
        if pred_len > 1:
            samples_x[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=gp_pred_x[i],
                                                                        cov=gp_pred_x_cov[i], size=num_samples)
            samples_y[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=gp_pred_y[i],
                                                                        cov=gp_pred_y_cov[i], size=num_samples)
        else:
            samples_x[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=gp_pred_x[i],
                                                                        cov=gp_pred_x_cov[i],
                                                                        size=num_samples)[:, None]
            samples_y[i * num_samples: (i + 1) * num_samples] = mvn.rvs(mean=gp_pred_y[i],
                                                                        cov=gp_pred_y_cov[i],
                                                                        size=num_samples)[:, None]
    if include_mean:
        # if include gp mean as sample, replace the first sample as gp mean
        for i in range(num_agents):
            samples_x[i * num_samples] = gp_pred_x[i].copy()
            samples_y[i * num_samples] = gp_pred_y[i].copy()

    samples_x = samples_x.copy()
    samples_y = samples_y.copy()

    return samples_x.copy(), samples_y.copy()

def gp_predict(robot_state, humans_state, num_agents, vel, dt,
               obsv_len, obsv_err_magnitude,gp_x,gp_y,
               cov_thred_x, cov_thred_y, obsv_x, obsv_y):
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

    # use the distance between current pose and navigation goal (of the robot) to determine length of prediction
    dist = np.sqrt((robot_state.gx - robot_state.px) ** 2 + (robot_state.gy - robot_state.py) ** 2)
    pred_len = int(dist / (vel * dt)) + 1

    # indices/labels for gp regression
    pred_t = np.arange(pred_len) + obsv_len
    obsv_t = np.array([i for i in range(obsv_len)] + [pred_len + obsv_len])
    obsv_err = np.ones_like(obsv_t) * obsv_err_magnitude

    for i, human_state in enumerate(state.human_states):

    # for i in range(num_agents):
        # here we do gp regression on x coordinate
        gp_x[i].compute(obsv_t, obsv_err)
        obsv_x = []
        for j in range(obsv_len):
            obsv_x.append(obsv_x[-obsv_len + j][i])
        obsv_x.append(human_state.gx)
        pred_x, pred_x_cov = gp_x[i].predict(obsv_x, pred_t, return_cov=True)
        scale_x = np.diag(pred_x_cov).max() / (cov_thred_x * pred_len)
        pred_x_cov /= scale_x
        gp_pred_x[i] = copy(pred_x)
        gp_pred_x_cov[i] = copy(pred_x_cov)

        # here we do gp regression on y coordinate
        gp_y[i].compute(obsv_t, obsv_err)
        obsv_y = []
        for j in range(obsv_len):
            obsv_y.append(obsv_y[-obsv_len + j][i])
        obsv_y.append(human_state.gy[i])
        pred_y, pred_y_cov = gp_y[i].predict(obsv_y, pred_t, return_cov=True)
        scale_y = np.diag(pred_y_cov).max() / (cov_thred_y * pred_len)
        pred_y_cov /= scale_y
        gp_pred_y[i] = copy(pred_y)
        gp_pred_y_cov[i] = copy(pred_y_cov)

    return pred_len
