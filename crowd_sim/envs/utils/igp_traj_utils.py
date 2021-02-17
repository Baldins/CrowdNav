"""
utility functions for igp-traj optimization
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=False)
def interact(f1_x, f1_y, f2_x, f2_y, tlen, a, h):
    """
    interaction function between two trajectories (collision penalty)
    :param f1_x: first agent's trajectory at x coordinate
    :param f1_y: first agent's trajectory at y coordinate
    :param f2_x: second agent's trajectory at x coordinate
    :param f2_y: second agent's trajectory at y coordinate
    :param tlen: length of the trajectory
    :param a: safety region parameter
    :param h: safety weight parameter
    :return: interaction score
    """
    val = np.zeros(tlen)
    for t in range(tlen):
        val[t] = h * np.exp(-0.5 * ((f1_x[t] - f2_x[t]) ** 2 + (f1_y[t] - f2_y[t]) ** 2) / a) \
                 / np.sqrt(2 * np.pi * a)
    return val.max()


@jit(nopython=True, cache=False)
def joint_weight(traj_x, traj_y, num_agents, traj_len, a, h):
    """
    joint collision penalty
    :param traj_x: x coordinate of the joint trajectories
    :param traj_y: y coordinate of the joint trajectories
    :param num_agents: number of agents
    :param traj_len: trajectory length
    :param a: safety region parameter
    :param h: safety weight parameter
    :return: joint collision penalty
    """
    weight = 0.0
    for i in prange(num_agents):
        for j in prange(i + 1, num_agents):
            w = interact(traj_x[i], traj_y[i], traj_x[j], traj_y[j],
                         traj_len, a, h)
            weight += w
    return np.exp(-weight)


@jit(nopython=True, cache=False, parallel=True)
def compute(samples_x, samples_y, num_agents, num_samples,
            traj_len, a, h):
    """
    igp optimization (importance sampling)
    :param samples_x: all agents' trjectory samples' x coordinate
    :param samples_y: all agents' trajectory samples' y coordinate
    :param num_samples: number of samples for each agent
    :param num_agents: number of agents
    :param traj_len: trajectory length
    :param a: safety region parameter
    :param h: safety weight parameter
    :return: optimal weights for all agents' samples
    """
    weights = np.zeros(num_samples)
    for i in prange(num_samples):
        traj_x = np.zeros((num_agents, traj_len))
        traj_y = np.zeros((num_agents, traj_len))
        for j in range(num_agents):
            traj_x[j] = samples_x[j * num_samples + i].copy()
            traj_y[j] = samples_y[j * num_samples + i].copy()
        w = joint_weight(traj_x, traj_y, num_agents, traj_len, a, h)
        weights[i] = w

    return weights.copy()
