"""
utility functions for igp-dist optimization
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
def init_config(num_agents):
    """
    generate index table for computation later
    :param num_agents: number of agents
    :return: index table
    """
    sub_table = np.zeros((num_agents, num_agents - 1))
    for i in range(num_agents):
        for j in range(num_agents):
            if i < j:
                sub_table[i][j - 1] = j
            elif i > j:
                sub_table[i][j] = j
            else:
                continue
    return sub_table


@jit(nopython=True, cache=False, parallel=True)
def generate_expect_table(samples_x, samples_y, num_samples, num_agents,
                          pred_len, a, h):
    """
    generate the table containing interaction scores among all agents' all samples
    to save time for igp optimization later
    :param samples_x: all agents' trajectory samples' x coordinate
    :param samples_y: all agents' trajectory samples' y coordinate
    :param num_samples: number of samples for each agent
    :param num_agents: number of agents
    :param pred_len: trajectory length
    :param a: safety region parameter
    :param h: safety weight parameter
    :return: interaction score table
    """
    table = np.zeros((num_agents * num_samples, num_agents * num_samples), dtype=np.float32)

    for i in prange(num_agents * num_samples):
        for j in prange(num_agents * num_samples):
            f1_x = samples_x[i]
            f1_y = samples_y[i]
            f2_x = samples_x[j]
            f2_y = samples_y[j]
            table[i][j] = interact(f1_x, f1_y, f2_x, f2_y, pred_len, a, h)

    return table


@jit(nopython=True, cache=False, parallel=True)
def one_iteration(weights, table, sub_table, host_id, num_pts, num_agents):
    """
    one iteration of igp optimization
    :param weights: input weights of all agents' samples
    :param table: the interaction score table
    :param sub_table: the index table
    :param host_id: index of the agent whose sample weights would be optimized
    :param num_pts: number of samples of each agent
    :param num_agents: number of agents
    :return: updated weights of all agents' samples
    """
    new_weights = weights.copy()
    for i in prange(num_pts):
        ev = 0.
        for j in prange(num_pts):
            for k in prange(num_agents - 1):
                client_id = int(sub_table[host_id][k])
                ev += table[host_id * num_pts + i][client_id * num_pts + j] * new_weights[client_id][j]
        ev /= num_pts
        new_weights[host_id][i] *= np.exp(-ev)
    new_weights[host_id] /= np.sum(new_weights[host_id]) / num_pts
    return new_weights.copy()


@jit(nopython=True, cache=False, parallel=False)
def objective(table, weights, num_agents, num_pts):
    """
    approximation of igp objective, used for terminating optimization
    :param table: interaction score table
    :param weights: weights of all agents' samples
    :param num_agents: number of samples
    :param num_pts: number of samples of each agent
    :return: approximated objective
    """
    val = 0.
    for i in prange(num_agents):
        for j in prange(i + 1, num_agents):
            for k in prange(num_pts):
                val += table[i * num_pts + k][j * num_pts + k] * weights[i][k] * weights[j][k]
    val /= num_pts
    return val


@jit(nopython=True, cache=False)
def compute(samples_x, samples_y, num_agents, num_samples, traj_len,
            a, h, obj_thred, max_iter):
    """
    igp optimization
    :param samples_x: all agents' trjectory samples' x coordinate
    :param samples_y: all agents' trajectory samples' y coordinate
    :param num_samples: number of samples for each agent
    :param num_agents: number of agents
    :param traj_len: trajectory length
    :param a: safety region parameter
    :param h: safety weight parameter
    :param obj_thred: terminal condition for optimization
    :param max_iter: maximal number of iteration allowed
    :return: optimal weights for all agents' samples
    """
    sub_table = init_config(num_agents)
    table = generate_expect_table(samples_x, samples_y, num_samples,
                                  num_agents, traj_len, a, h)
    weights = np.ones((num_agents, num_samples), dtype=np.float32)

    it = 0
    while True:
        obj = objective(table, weights, num_agents, num_samples)
        if obj < obj_thred or it >= max_iter:
            print("terminate optimization at: iteration [", it, "], objective [", obj, "]")
            break
        for pid in range(num_agents):
            new_weights = one_iteration(weights, table, sub_table, pid, num_samples, num_agents)
            weights = new_weights.copy()
        it += 1

    return weights.copy()
