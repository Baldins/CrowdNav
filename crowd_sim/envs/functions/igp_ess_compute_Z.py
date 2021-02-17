import autograd.numpy as np
import scipy as sp
from scipy import optimize

def ess_compute_Z(diagonal, num_peds, robot_mu_x, robot_mu_y, \
                  ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
                  inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, \
                  inv_cov_ped_x, inv_cov_ped_y, \
                  one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  delta0 = [0. for _ in range(num_peds)]
  norm_delta0 = [0. for _ in range(num_peds)]
  norm_delta0_normalized = [0. for _ in range(num_peds)]
  T = np.size(robot_mu_x)

  # for var in range(np.size(var_x_ess)):
  for ped in range(num_peds):
    # if normalize == True:
    #   normalize_x = np.multiply(np.power(2*np.pi,-0.5), one_over_std_sum_x)
    #   normalize_y = np.multiply(np.power(2*np.pi,-0.5), one_over_std_sum_y)
    # else:
    normalize_x = 1.
    normalize_y = 1.

    vel_x = robot_mu_x - ped_mu_x[ped]
    vel_y = robot_mu_y - ped_mu_y[ped]
    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)

    one_over_var_sum_x = np.diag(one_over_cov_sum_x[ped])
    one_over_var_sum_y = np.diag(one_over_cov_sum_y[ped])

    quad_x = np.multiply(one_over_var_sum_x, vel_x_2)
    quad_y = np.multiply(one_over_var_sum_y, vel_y_2)

    Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_x))
    Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_y))

    Z = np.multiply(Z_x, Z_y)

    norm_delta0[ped] = np.abs(np.sum(np.log1p(-Z)))

  norm_delta0_normalized = norm_delta0/(np.sum(norm_delta0))
  ess = 1./np.sum(np.power(norm_delta0_normalized, 2))
  ess = np.int(ess)
  top_Z_indices = np.argsort(norm_delta0_normalized)[::-1]

  return ess, top_Z_indices












