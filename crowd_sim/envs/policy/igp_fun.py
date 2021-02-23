import numpy as np
import time
from scipy.stats import multivariate_normal as mvn
import george
from copy import copy
from crowd_sim.envs.utils.igp_dist_utils import *


def add_observation(obsv_xt, obsv_yt, obsv_x, obsv_y):
    """
    add observation of all agents
    :param obsv_xt: a list/array of all agents' x positions at current time step
    :param obsv_yt: a list/array of all agents' y positions at current time step
    :return: None
    """
    obsv_x.append(obsv_xt)
    obsv_y.append(obsv_yt)
    return obsv_x, obsv_y


def gp_predict(state, robot_idx, num_agents, robot_state, vel, dt, len_scale,
               obsv_len, obsv_err_magnitude, gp_x, gp_y,
               cov_thred_x, cov_thred_y, obsv_x, obsv_y, gp_pred_x, gp_pred_x_cov, gp_pred_y, gp_pred_y_cov):
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
    # initialization
    gp_pred_x = [0. for _ in range(num_agents)]
    gp_pred_x_cov = [0. for _ in range(num_agents)]
    gp_pred_y = [0. for _ in range(num_agents)]
    gp_pred_y_cov = [0. for _ in range(num_agents)]

    # extract current robot pose from collected observations
    curr_robot_x = robot_state.px
    curr_robot_y = robot_state.py

    print("curr_robot: ", curr_robot_x, curr_robot_y)

    # use the distance between current pose and navigation goal (of the robot) to determine length of prediction
    dist = np.sqrt((robot_state.gx - curr_robot_x) ** 2 + (robot_state.gy - curr_robot_y) ** 2)
    pred_len = int(dist / (vel * dt)) + 1

    print("pred_len", pred_len)

    for i, human in enumerate(state.human_states):
        pred_t = np.arange(pred_len) + obsv_len
        obsv_t = np.array([i for i in range(obsv_len)] + [pred_len + obsv_len])
        obsv_err = np.ones_like(obsv_t) * obsv_err_magnitude

        # here we do gp regression on x coordinate
        gp_x[i].compute(np.asarray(obsv_t), np.asarray(obsv_err))
        obsv_xn = []
        for j in range(obsv_len):
            obsv_xn.append(obsv_x[-obsv_len + j][i])
        # predict goal's x coordinate

        curr_agent_x = human.px
        curr_vel_x = human.vx
        curr_goal_x = curr_agent_x + curr_vel_x * (pred_len - 1)
        obsv_xn.append(curr_goal_x)

        # here we do gp regression on y coordinate
        gp_y[i].compute(obsv_t, obsv_err)
        obsv_yn = []
        for j in range(obsv_len):
            obsv_yn.append(obsv_y[-obsv_len + j][i])
        # predict goal's y coordinate
        curr_agent_y = human.py
        curr_vel_y = human.vy
        curr_goal_y = curr_agent_y + curr_vel_y * (pred_len - 1)
        obsv_yn.append(curr_goal_y)

        curr_vel = np.sqrt(curr_vel_x ** 2 + curr_vel_y ** 2)
        scale = pred_len * curr_vel / 2

        pred_x, pred_x_cov = gp_x[i].predict(obsv_xn, pred_t, return_cov=True)
        scale_x = np.diag(pred_x_cov).max() / (cov_thred_x * scale)
        pred_x_cov /= scale_x
        gp_pred_x[i] = copy(pred_x)
        gp_pred_x_cov[i] = copy(pred_x_cov)

        pred_y, pred_y_cov = gp_y[i].predict(obsv_yn, pred_t, return_cov=True)
        scale_y = np.diag(pred_y_cov).max() / (cov_thred_y * scale)
        pred_y_cov /= scale_y
        gp_pred_y[i] = copy(pred_y)
        gp_pred_y_cov[i] = copy(pred_y_cov)
    #
    # for i in range(num_agents):
    #     if i == robot_idx: # for robot, we condition on real goal

       ## Robot


        # indices/labels for gp regression
        pred_t = np.arange(pred_len) + obsv_len
        obsv_t = np.array([i for i in range(obsv_len)] + [pred_len + obsv_len])
        obsv_err = np.ones_like(obsv_t) * obsv_err_magnitude

        # here we do gp regression on x coordinate
        gp_x[robot_idx].compute(np.asarray(obsv_t), np.asarray(obsv_err))

        obsv_xn = []
        for j in range(obsv_len):
            obsv_xn.append(obsv_x[-obsv_len + j][robot_idx])
        obsv_xn.append(-obsv_x[0][robot_idx])

        pred_x, pred_x_cov = gp_x[robot_idx].predict(obsv_xn, pred_t, return_cov=True)
        scale_x = np.diag(pred_x_cov).max() / (cov_thred_x * pred_len)
        pred_x_cov /= scale_x
        gp_pred_x[robot_idx] = copy(pred_x)
        gp_pred_x_cov[robot_idx] = copy(pred_x_cov)

        # here we do gp regression on y coordinate
        gp_y[robot_idx].compute(obsv_t, obsv_err)
        obsv_yn = []
        for j in range(obsv_len):
            obsv_yn.append(obsv_y[-obsv_len + j][robot_idx])
        obsv_yn.append(-obsv_y[0][robot_idx])

        pred_y, pred_y_cov = gp_y[robot_idx].predict(obsv_yn, pred_t, return_cov=True)
        scale_y = np.diag(pred_y_cov).max() / (cov_thred_y * pred_len)
        pred_y_cov /= scale_y
        gp_pred_y[robot_idx] = copy(pred_y)
        gp_pred_y_cov[robot_idx] = copy(pred_y_cov)

        # else: # for pedestrian we condition on predicted goal
        #
        #
        #     pred_t = np.arange(pred_len) + obsv_len
        #     obsv_t = np.array([i for i in range(obsv_len)] + [pred_len + obsv_len])
        #     obsv_err = np.ones_like(obsv_t) * obsv_err_magnitude
        #
        #     # here we do gp regression on x coordinate
        #     gp_x[i].compute(np.asarray(obsv_t), np.asarray(obsv_err))
        #     obsv_xn = []
        #     for j in range(obsv_len):
        #         obsv_xn.append(obsv_x[-obsv_len + j][i])
        #     # predict goal's x coordinate
        #
        #     curr_agent_x = obsv_x[-1][i]
        #     curr_vel_x = obsv_x[-1][i] - obsv_x[-2][i]
        #     curr_goal_x = curr_agent_x + curr_vel_x * (pred_len - 1)
        #     obsv_xn.append(curr_goal_x)
        #
        #     # here we do gp regression on y coordinate
        #     gp_y[i].compute(obsv_t, obsv_err)
        #     obsv_yn = []
        #     for j in range(obsv_len):
        #         obsv_yn.append(obsv_y[-obsv_len + j][i])
        #     # predict goal's y coordinate
        #     curr_agent_y = obsv_y[-1][i]
        #     curr_vel_y = obsv_y[-1][i] - obsv_y[-2][i]
        #     curr_goal_y = curr_agent_y + curr_vel_y * (pred_len - 1)
        #     obsv_yn.append(curr_goal_y)
        #
        #     curr_vel = np.sqrt(curr_vel_x ** 2 + curr_vel_y ** 2)
        #     scale = pred_len * curr_vel / 2
        #
        #     pred_x, pred_x_cov = gp_x[i].predict(obsv_xn, pred_t, return_cov=True)
        #     scale_x = np.diag(pred_x_cov).max() / (cov_thred_x * scale)
        #     pred_x_cov /= scale_x
        #     gp_pred_x[i] = copy(pred_x)
        #     gp_pred_x_cov[i] = copy(pred_x_cov)
        #
        #     pred_y, pred_y_cov = gp_y[i].predict(obsv_yn, pred_t, return_cov=True)
        #     scale_y = np.diag(pred_y_cov).max() / (cov_thred_y * scale)
        #     pred_y_cov /= scale_y
        #     gp_pred_y[i] = copy(pred_y)
        #     gp_pred_y_cov[i] = copy(pred_y_cov)

    return gp_pred_x, gp_pred_y, gp_pred_x_cov, gp_pred_y_cov, pred_len


def gp_sampling(num_samples, num_agents, pred_len, gp_pred_x, gp_pred_x_cov, gp_pred_y, gp_pred_y_cov, samples_x,
                samples_y,
                include_mean=True):
    """
    generate samples from gp posteriors
    :param num_samples: number of samples for each agent
    :param include_mean: whether include gp mean as a sample
    :return: generated samples
    """
    time_seed = int(time.time() * 1000) % 1000
    print("random seed: ", time_seed)
    np.random.seed(time_seed)

    if include_mean is True:
        print("GP mean included as sample")
    else:
        print("GP mean NOT included as sample")
    samples_x = np.zeros((num_agents * num_samples, pred_len))
    samples_y = np.zeros((num_agents * num_samples, pred_len))
    pdf_x = np.zeros((num_agents, num_samples), dtype=np.float128)
    pdf_y = np.zeros((num_agents, num_samples), dtype=np.float128)
    for i in range(num_agents):
        if pred_len > 1:
            rv_x = mvn(mean=gp_pred_x[i], cov=gp_pred_x_cov[i], allow_singular=True)
            samples_x[i * num_samples: (i + 1) * num_samples] = rv_x.rvs(size=num_samples).copy()
            rv_y = mvn(mean=gp_pred_y[i], cov=gp_pred_y_cov[i], allow_singular=True)
            samples_y[i * num_samples: (i + 1) * num_samples] = rv_y.rvs(size=num_samples).copy()
        else:
            rv_x = mvn(mean=gp_pred_x[i], cov=gp_pred_x_cov[i], allow_singular=True)
            samples_x[i * num_samples: (i + 1) * num_samples] = rv_x.rvs(size=num_samples).copy()[:, None]
            rv_y = mvn(mean=gp_pred_y[i], cov=gp_pred_y_cov[i], allow_singular=True)
            samples_y[i * num_samples: (i + 1) * num_samples] = rv_y.rvs(size=num_samples).copy()[:, None]
        if include_mean:
            # if include gp mean as sample, replace the first sample as gp mean
            samples_x[i * num_samples] = gp_pred_x[i]
            samples_y[i * num_samples] = gp_pred_y[i]
        pdf_x[i] = rv_x.pdf(samples_x[i * num_samples: (i + 1) * num_samples])
        pdf_y[i] = rv_y.pdf(samples_y[i * num_samples: (i + 1) * num_samples])
    scale = np.max(pdf_x[0])
    pdf_x /= scale
    pdf_y /= scale
    # print("sample_x", samples_x)
    # print("sample_y", samples_y)

    samples_pdf = pdf_x * pdf_y
    return samples_x, samples_y, samples_pdf


def weight_compute(a, h, obj_thred, max_iter, samples_x, samples_y, human_num, num_samples, pred_len):
    """
    core of igp computation, compute all samples' weights
    :param a: safety region parameter
    :param h: safety magnitude parameter
    :param obj_thred: terminal condition for iterations
    :param max_iter: maximal number of iterations allowed
    :return: optimized weights
    """

    weights = compute(samples_x, samples_y, human_num, num_samples, pred_len,
                      a, h, obj_thred, max_iter)
    return weights


def actuate(weights, robot_idx, num_samples, samples_x, samples_y, dt=1, samples_pdf=None):
    """
    generate velocity command for robot
    :param dt: interval of simulation step
    :return: velocity command
    """
    # select optimal sample trajectory as reference for robot navigation
    if samples_pdf is not None:
        print("pdf included for actuate")
        opt_idx = np.argmax(weights[robot_idx] * samples_pdf[robot_idx])
    else:
        print("pdf NOT included for actuate")
        opt_idx = np.argmax(weights[robot_idx])
    opt_robot_x = samples_x[robot_idx * num_samples + opt_idx][0]
    opt_robot_y = samples_y[robot_idx * num_samples + opt_idx][0]

    return opt_robot_x, opt_robot_y


def get_opt_traj(num_agents, num_samples, pred_len, samples_x, samples_y, weights, sampels_pdf):
    """
    get joint optimal trajectories of all agents
    :return: joint optimal trajectories
    """
    traj_x = np.zeros((num_agents, pred_len))
    traj_y = np.zeros((num_agents, pred_len))
    for i in range(num_agents):
        if sampels_pdf is not None:
            opt_idx = np.argmax(weights[i] * sampels_pdf[i])
        else:
            opt_idx = np.argmax(weights[i])
        traj_x[i] = samples_x[i * num_samples + opt_idx]
        traj_y[i] = samples_y[i * num_samples + opt_idx]
    return traj_x, traj_y


def igp(state, obsv_x, obsv_y, robot_idx, num_samples, num_agents, len_scale,
        a, h, obj_thred, max_iter, vel, dt, obsv_len, obsv_err_magnitude, cov_thred_x, cov_thred_y, gp_x, gp_y,
        gp_pred_x, gp_pred_x_cov, gp_pred_y, gp_pred_y_cov, samples_x, samples_y, weights, include_pdf=False, actuate_index=0):
    robot_state = state.self_state
    goals_x = []
    goals_y = []
    for i, human in enumerate(state.human_states):
        goals_x.append(-obsv_x[0][i])
        goals_y.append(-obsv_y[0][i])
    goals_x.append(state.self_state.gx)
    goals_y.append(state.self_state.gy)

    ## predict
    gp_pred_x, gp_pred_y, gp_pred_x_cov, gp_pred_y_cov, pred_len = gp_predict(state, robot_idx, num_agents, robot_state, vel, dt,
                                                                              len_scale,
                                                                              obsv_len, obsv_err_magnitude, gp_x, gp_y,
                                                                              cov_thred_x, cov_thred_y, obsv_x, obsv_y,
                                                                              gp_pred_x, gp_pred_x_cov, gp_pred_y,
                                                                              gp_pred_y_cov)

    ## sampling
    samples_x, samples_y, samples_pdf = gp_sampling(num_samples, num_agents, pred_len, gp_pred_x, gp_pred_x_cov,
                                                    gp_pred_y, gp_pred_y_cov, samples_x, samples_y,
                                                    include_mean=True)

    ## weight
    weights = weight_compute(a, h, obj_thred, max_iter, samples_x, samples_y, num_agents, num_samples, pred_len)

    ## actuate
    if include_pdf is True:
        opt_robot_x, opt_robot_y = actuate(weights, robot_idx, num_samples, samples_x, samples_y, dt,
                                           samples_pdf=samples_pdf)
        traj_x, traj_y = get_opt_traj(num_agents, num_samples, pred_len, samples_x, samples_y, weights,
                                      sampels_pdf=samples_pdf)
    else:
        opt_robot_x, opt_robot_y = actuate(weights, robot_idx, num_samples, samples_x, samples_y, dt, samples_pdf=None)
        traj_x, traj_y = get_opt_traj(num_agents, num_samples, pred_len, samples_x, samples_y, weights,
                                      sampels_pdf=None)

    return opt_robot_x, opt_robot_y, traj_x, traj_y