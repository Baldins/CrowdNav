import autograd.numpy as np
import scipy as sp
from scipy import optimize

import crowd_sim.envs.objectives.fo_igp_diag_objectives_ess as fo_diag_ess
import crowd_sim.envs.objectives.fo_igp_dense_objectives_ess as fo_dense_ess

from crowd_sim.envs.functions.so_igp_optimize_iterate import optimize_iterate

def fo_ess_compute_newton(diagonal, num_peds, robot_mu_x, robot_mu_y, \
                ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
                inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, \
                inv_cov_ped_x, inv_cov_ped_y, \
                one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  delta0 = [0. for _ in range(num_peds)]
  norm_delta0 = [0. for _ in range(num_peds)]
  norm_delta0_normalized = [0. for _ in range(num_peds)]
  T = np.size(robot_mu_x)
  for ped in range(num_peds):
    x0 = np.zeros(4*T)
    x0 = robot_mu_x
    x0 = np.concatenate((x0, robot_mu_y))
    x0 = np.concatenate((x0, ped_mu_x[ped]))
    x0 = np.concatenate((x0, ped_mu_y[ped]))
    if diagonal:
      g_ll = fo_diag_ess.d_ll(x0, T, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x[ped], ped_mu_y[ped], \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x[ped], cov_ped_y[ped], \
                          inv_cov_ped_x[ped], inv_cov_ped_y[ped], \
                          one_over_cov_sum_x[ped], one_over_cov_sum_y[ped], \
                          normalize)
      h_ll = fo_diag_ess.dd_ll(x0, T, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x[ped], ped_mu_y[ped], \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x[ped], cov_ped_y[ped], \
                          inv_cov_ped_x[ped], inv_cov_ped_y[ped], \
                          one_over_cov_sum_x[ped], one_over_cov_sum_y[ped], \
                          normalize)
    else:
      g_ll = fo_dense_ess.d_ll(x0, T, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x[ped], ped_mu_y[ped], \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x[ped], cov_ped_y[ped], \
                          inv_cov_ped_x[ped], inv_cov_ped_y[ped], \
                          one_over_cov_sum_x[ped], one_over_cov_sum_y[ped], \
                          normalize)
      h_ll = fo_dense_ess.dd_ll(x0, T, \
                          robot_mu_x, robot_mu_y, \
                          ped_mu_x[ped], ped_mu_y[ped], \
                          cov_robot_x, cov_robot_y, \
                          inv_cov_robot_x, inv_cov_robot_y, \
                          cov_ped_x[ped], cov_ped_y[ped], \
                          inv_cov_ped_x[ped], inv_cov_ped_y[ped], \
                          one_over_cov_sum_x[ped], one_over_cov_sum_y[ped], \
                          normalize)
    delta0[ped] = np.linalg.solve(h_ll, -g_ll)
    norm_delta0[ped] = np.linalg.norm(delta0[ped])
  #############################MINIMIZE ON EACH AGENT
      # x0 = np.zeros(4*T)
      # x0 = robot_mu_x
      # x0 = np.concatenate((x0, robot_mu_y))
      # x0 = np.concatenate((x0, ped_mu_x[ped]))
      # x0 = np.concatenate((x0, ped_mu_y[ped]))
      # f = sp.optimize.minimize(diag_ll_ess, x0, \
      #        args=(T, robot_mu_x, robot_mu_y, \
      #              ped_mu_x[ped], ped_mu_y[ped], \
      #              inv_cov_robot_x, inv_cov_robot_y, \
      #              inv_cov_ped_x[ped], inv_cov_ped_y[ped], \
      #              one_over_cov_sum_x[ped], one_over_cov_sum_y[ped], \
      #              one_over_std_sum_x[ped], one_over_std_sum_y[ped]), \
      #              method='trust-krylov',\
      #              jac=fo_diag_ess.d_ll, hess=so_diag_ess.dd_ll)
      # norm_delta0[ped] = np.linalg.norm(f.x[:T]-robot_mu_x) + \
                         # np.linalg.norm(f.x[T:2*T]-robot_mu_y)
  # norm_z_ess_normalized = np.divide(norm_z_ess, (np.sum(norm_z_ess)))
  # ess = 1./np.sum(np.power(norm_z_ess_normalized, 2))
  # top_Z_indices = np.argsort(norm_z_ess_normalized)[::-1]

  norm_delta0_normalized = norm_delta0/(np.sum(norm_delta0))
  ess = np.power(np.sum(np.power(norm_delta0_normalized, 2)), -1)
  if np.isnan(ess):
    ess = 1.
    print(f"ESS IS 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  else:
    ess = np.int(ess)

  top_Z_indices = np.argsort(norm_delta0_normalized)[::-1]

  return ess, top_Z_indices












