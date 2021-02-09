import autograd.numpy as np

def ll(x, T, robot_mu_x, robot_mu_y, \
        ped_mu_x, ped_mu_y, \
        inv_cov_robot_x, inv_cov_robot_y, \
        inv_cov_ped_x, inv_cov_ped_y, \
        one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  quad_robot_mu_x = np.dot((x[:T]-robot_mu_x).T, np.dot(\
                                             inv_cov_robot_x, x[:T]-robot_mu_x))
  quad_robot_mu_y = np.dot((x[T:2*T] - robot_mu_y).T, np.dot( \
                                        inv_cov_robot_y, x[T:2*T] - robot_mu_y))
  llambda = -0.5*quad_robot_mu_x - 0.5*quad_robot_mu_y

  n=2
  quad_ped_mu_x = np.dot((x[n*T:(n+1)*T]-ped_mu_x).T, np.dot(inv_cov_ped_x,\
                                                       x[n*T:(n+1)*T]-ped_mu_x))
  quad_ped_mu_y = np.dot((x[(n+1)*T:(n+2)*T]-ped_mu_y).T, np.dot(inv_cov_ped_y,\
                                                   x[(n+1)*T:(n+2)*T]-ped_mu_y))
  llambda = llambda -0.5*quad_ped_mu_x -0.5*quad_ped_mu_y

  n = 2
  # if normalize == True:
  #   normalize_x = np.multiply(np.power(2*np.pi, -0.5), \
  #                                                   np.diag(one_over_std_sum_x))
  #   normalize_y = np.multiply(np.power(2*np.pi, -0.5), \
  #                                                   np.diag(one_over_std_sum_y))
  # else:
  normalize_x = 1.
  normalize_y = 1.

  vel_x = x[:T] - x[n*T:(n+1)*T]
  vel_y = x[T:2*T] - x[(n+1)*T:(n+2)*T]
  vel_x_2 = np.power(vel_x, 2)
  vel_y_2 = np.power(vel_y, 2)

  quad_robot_ped_x = np.multiply(vel_x_2, np.diag(one_over_cov_sum_x))
  quad_robot_ped_y = np.multiply(vel_y_2, np.diag(one_over_cov_sum_y))

  Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_robot_ped_x))
  Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_robot_ped_y))

  Z = np.multiply(Z_x, Z_y)

  log_znot_norm = np.sum(np.log1p(-Z))

  llambda = llambda + log_znot_norm
  return -1.*llambda

def d_ll(x, T, \
                 robot_mu_x, robot_mu_y, \
                 ped_mu_x, ped_mu_y, \
                 cov_robot_x, cov_robot_y, \
                 inv_cov_robot_x, inv_cov_robot_y, \
                 cov_ped_x, cov_ped_y, \
                 inv_cov_ped_x, inv_cov_ped_y, \
                 one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  d_alpha = [0. for _ in range(4*T)]
  d_beta = [0. for _ in range(4*T)]
  d_llambda = np.asarray([0. for _ in range(4*T)])

  n = 2
  vel_x = x[:T] - x[n*T:(n+1)*T]
  vel_y = x[T:2*T] - x[(n+1)*T:(n+2)*T]

  one_over_var_sum_x = np.diag(one_over_cov_sum_x)
  one_over_var_sum_y = np.diag(one_over_cov_sum_y)

  # if normalize == True:
  #   normalize_x = np.multiply(np.power(2*np.pi, -0.5), \
  #                                                   np.diag(one_over_std_sum_x))
  #   normalize_y = np.multiply(np.power(2*np.pi, -0.5), \
  #                                                   np.diag(one_over_std_sum_y))
  # else:
  normalize_x = 1.
  normalize_y = 1.

  quad_x = np.multiply(one_over_var_sum_x, np.power(vel_x, 2))
  quad_y = np.multiply(one_over_var_sum_y, np.power(vel_y, 2))

  Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_x))
  Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_y))

  Z = np.multiply(Z_x, Z_y)
  X = np.divide(Z, 1.-Z)

  alpha_x = np.multiply(X, np.multiply(vel_x, one_over_var_sum_x))
  alpha_y = np.multiply(X, np.multiply(vel_y, one_over_var_sum_y))
#        X and Y COMPONENT OF R DERIVATIVE
  d_alpha[:T] = np.add(d_alpha[:T], alpha_x)
  d_alpha[T:2*T] = np.add(d_alpha[T:2*T], alpha_y)

  d_alpha[n*T:(n+1)*T] = -alpha_x
  d_alpha[(n+1)*T:(n+2)*T] = -alpha_y

  d_beta[n*T:(n+1)*T] = -np.dot(x[n*T:(n+1)*T]-ped_mu_x, inv_cov_ped_x)
  d_beta[(n+1)*T:(n+2)*T] = -np.dot(x[(n+1)*T:(n+2)*T]-ped_mu_y, inv_cov_ped_y)

  d_beta[:T] = -np.dot(x[:T]-robot_mu_x, inv_cov_robot_x)
  d_beta[T:2*T] = -np.dot(x[T:2*T]-robot_mu_y, inv_cov_robot_y)

  d_llambda[0:2*T] = np.add(d_alpha[0:2*T], d_beta[0:2*T])
  d_llambda[2*T:] = np.add(d_alpha[2*T:], d_beta[2*T:])
  return -1.*d_llambda

def dd_ll(x, T, robot_mu_x, robot_mu_y, \
         ped_mu_x, ped_mu_y, cov_robot_x, cov_robot_y, \
         inv_cov_robot_x, inv_cov_robot_y, cov_ped_x, cov_ped_y, \
         inv_cov_ped_x, inv_cov_ped_y, one_over_cov_sum_x, one_over_cov_sum_y, \
         normalize):
  H = np.zeros((4*T,4*T), float)
  sum_dd_alpha = [0. for _ in range(4*T)]

# if normalize == True:
  #   normalize_x = np.multiply(np.power(2*np.pi, -0.5), \
  #                                                   np.diag(one_over_std_sum_x))
  #   normalize_y = np.multiply(np.power(2*np.pi, -0.5), \
  #                                                   np.diag(one_over_std_sum_y))
  # else:
  normalize_x = 1.
  normalize_y = 1.

  n = 2
  vel_x = x[:T] - x[n*T:(n+1)*T]
  vel_y = x[T:2*T] - x[(n+1)*T:(n+2)*T]
  vel_x_y = np.multiply(vel_x, vel_y)

  one_over_var_x = np.diag(one_over_cov_sum_x)
  one_over_var_y = np.diag(one_over_cov_sum_y)
  one_over_var_x_y = np.multiply(one_over_var_x, one_over_var_y)

  quad_x = np.multiply(one_over_var_x, np.power(vel_x, 2))
  quad_y = np.multiply(one_over_var_y, np.power(vel_y, 2))

  Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_x))
  Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_y))

  Z = np.multiply(Z_x, Z_y)
  X = np.divide(Z, 1.-Z)

  X_plus_X2 = np.add(X, np.power(X, 2))

  dd_alpha_matrix_x = np.multiply(X, one_over_var_x)
  dd_alpha_matrix_x = np.add(dd_alpha_matrix_x, -np.multiply(X_plus_X2, \
                                           np.multiply(quad_x, one_over_var_x)))

  dd_alpha_matrix_y = np.multiply(X, one_over_var_y)
  dd_alpha_matrix_y = np.add(dd_alpha_matrix_y, -np.multiply(X_plus_X2, \
                                           np.multiply(quad_y, one_over_var_y)))

  sum_dd_alpha[:T] = np.add(sum_dd_alpha[:T], dd_alpha_matrix_x)
  sum_dd_alpha[T:2*T] = np.add(sum_dd_alpha[T:2*T], dd_alpha_matrix_y)
  #        OFF DIAGONALS
  dd_off_alpha_matrix = np.multiply(X_plus_X2, np.multiply(vel_x_y, \
                                                              one_over_var_x_y))
  # ROBOT ENTRIES
  H[:T,T:2*T] = np.add(H[:T,T:2*T], -np.diag(dd_off_alpha_matrix))

  H[:T,n*T:(n+1)*T] = -np.diag(dd_alpha_matrix_x)
  H[n*T:(n+1)*T,:T] = H[:T,n*T:(n+1)*T].T

  H[T:2*T,(n+1)*T:(n+2)*T] = -np.diag(dd_alpha_matrix_y)
  H[(n+1)*T:(n+2)*T,T:2*T] = H[T:2*T,(n+1)*T:(n+2)*T].T

  H[:T,(n+1)*T:(n+2)*T] = np.diag(dd_off_alpha_matrix)
  H[(n+1)*T:(n+2)*T,:T] = H[:T,(n+1)*T:(n+2)*T].T

  H[T:2*T,n*T:(n+1)*T] = np.diag(dd_off_alpha_matrix)
  H[n*T:(n+1)*T,T:2*T] = H[T:2*T,n*T:(n+1)*T].T

  # PEDESTRIAN ENTRIES
  H[n*T:(n+1)*T,n*T:(n+1)*T] = np.diag(dd_alpha_matrix_x)-inv_cov_ped_x
  H[(n+1)*T:(n+2)*T,(n+1)*T:(n+2)*T] = np.diag(dd_alpha_matrix_y)-inv_cov_ped_y

  H[n*T:(n+1)*T,(n+1)*T:(n+2)*T] = -np.diag(dd_off_alpha_matrix)
  H[(n+1)*T:(n+2)*T,n*T:(n+1)*T] = H[n*T:(n+1)*T,(n+1)*T:(n+2)*T].T

  H[T:2*T,:T] = H[:T,T:2*T].T

  H[:T,:T] = np.diag(sum_dd_alpha[:T]) - inv_cov_robot_x
  H[T:2*T,T:2*T] = np.diag(sum_dd_alpha[T:2*T]) - inv_cov_robot_y
  return -1.*H







