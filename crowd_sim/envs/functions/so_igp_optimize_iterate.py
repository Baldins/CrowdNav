import numpy as np
import time
from copy import deepcopy

import crowd_sim.envs.objectives.so_igp_diag_objectives as so_diagonal
import crowd_sim.envs.objectives.so_igp_dense_objectives as so_dense
import crowd_sim.envs.objectives.fo_igp_dense_objectives as fo_dense

max_iter = 50

def optimize_iterate(fo, tol, diagonal, frame, z0, num_peds, ess, \
      robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
      inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, \
      inv_cov_ped_x, inv_cov_ped_y, one_over_cov_sum_x, one_over_cov_sum_y,  \
      one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, ll_converge, T, \
      opt_iter_robot, opt_iter_all):

  T = np.size(robot_mu_x)
  z = deepcopy(z0)

  for _ in range(max_iter):
    if diagonal:
      gll = so_diagonal.d_ll(z, num_peds, ess, \
                      robot_mu_x, robot_mu_y, \
                      ped_mu_x, ped_mu_y, \
                      cov_robot_x, cov_robot_y, \
                      inv_cov_robot_x, inv_cov_robot_y, \
                      cov_ped_x, cov_ped_y, \
                      inv_cov_ped_x, inv_cov_ped_y, \
                      one_over_cov_sum_x, one_over_cov_sum_y,  \
                      one_over_cov_sumij_x, one_over_cov_sumij_y, \
                      normalize, T)
      hll = so_diagonal.dd_ll(z, num_peds, ess, \
                       robot_mu_x, robot_mu_y, \
                       ped_mu_x, ped_mu_y, \
                       cov_robot_x, cov_robot_y, \
                       inv_cov_robot_x, inv_cov_robot_y, \
                       cov_ped_x, cov_ped_y, \
                       inv_cov_ped_x, inv_cov_ped_y, \
                       one_over_cov_sum_x, one_over_cov_sum_y,  \
                       one_over_cov_sumij_x, one_over_cov_sumij_y, \
                       normalize, T)
    else:
      if fo:
        gll = fo_dense.d_ll(z, num_peds, ess, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x, ped_mu_y, \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x, cov_ped_y, \
                          inv_cov_ped_x, inv_cov_ped_y, \
                          one_over_cov_sum_x, one_over_cov_sum_y,
                          normalize)
        hll = fo_dense.dd_ll(z, num_peds, ess, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x, ped_mu_y, \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x, cov_ped_y, \
                          inv_cov_ped_x, inv_cov_ped_y, \
                          one_over_cov_sum_x, one_over_cov_sum_y,
                          normalize)
      else:
        gll = so_dense.d_ll(z, num_peds, ess, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x, ped_mu_y, \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x, cov_ped_y, \
                          inv_cov_ped_x, inv_cov_ped_y, \
                          one_over_cov_sum_x, one_over_cov_sum_y,  \
                          one_over_cov_sumij_x, one_over_cov_sumij_y, \
                          normalize, T)
        hll = so_dense.dd_ll(z, num_peds, ess, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x, ped_mu_y, \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x, cov_ped_y, \
                          inv_cov_ped_x, inv_cov_ped_y, \
                          one_over_cov_sum_x, one_over_cov_sum_y,  \
                          one_over_cov_sumij_x, one_over_cov_sumij_y, \
                          normalize, T)
    delta = np.linalg.solve(hll, -gll)

    if np.isnan(np.sum(delta)) or np.isinf(np.sum(delta)):
      print('DELTA IS NAN')
      return z
    else:
      z = z + delta
##########OPTIMIZATION
    if ll_converge:
      if so_dense.ll(z+delta, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, \
    ped_mu_y, cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
    cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
    one_over_cov_sum_x, one_over_cov_sum_y, one_over_std_sum_x, \
    one_over_std_sum_y, normalize, T) - so_dense.ll(z, num_peds, ess, robot_mu_x, \
    robot_mu_y, ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
    inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, inv_cov_ped_x, \
    inv_cov_ped_y, one_over_cov_sum_x, one_over_cov_sum_y, normalize, T) < tol:
        break
      elif _>=max_iter:
        print('ERROR: NEWTON FAILED', _)
        print('delta norm', np.linalg.norm(delta[0:2*T]))

    if opt_iter_robot:
      if np.linalg.norm(delta[:2*T]) < tol:
        break
      elif _>=max_iter:
        print('ERROR: NEWTON FAILED', _)
        print('delta norm', np.linalg.norm(delta[0:2*T]))
    if opt_iter_all:
      if np.linalg.norm(delta) < tol:
        break
      elif _>=max_iter:
        print('ERROR: NEWTON FAILED', _)
        print('delta norm', np.linalg.norm(delta))
  return z








