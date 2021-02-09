from numba import jit
import autograd.numpy as np

def ll(x, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
       cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
       cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
       one_over_cov_sum_x, one_over_cov_sum_y, \
       one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T):

  quad_mu_x = np.dot((x[:T]-robot_mu_x).T, np.dot(inv_cov_robot_x, \
                                                              x[:T]-robot_mu_x))
  quad_mu_y = np.dot((x[T:2*T]-robot_mu_y).T, np.dot(inv_cov_robot_y, \
                                                           x[T:2*T]-robot_mu_y))
  llambda = -0.5*quad_mu_x - 0.5*quad_mu_y

  i = 2
  for ped in range(ess):
    quad_mu_x = np.dot((x[i*T:(i+1)*T]-ped_mu_x[ped]).T, np.dot(\
                            inv_cov_ped_x[ped], x[i*T:(i+1)*T]-ped_mu_x[ped]))
    quad_mu_y = np.dot((x[(i+1)*T:(i+2)*T]-ped_mu_y[ped]).T, np.dot(\
                        inv_cov_ped_y[ped], x[(i+1)*T:(i+2)*T]-ped_mu_y[ped]))
    llambda = llambda -0.5*quad_mu_x - 0.5*quad_mu_y
    i = i + 2

  i = 2#robot-agent CCA
  for ped in range(ess):
    vel_x = np.tile(x[:T],(T,1)).T - np.tile(x[i*T:(i+1)*T], (T,1))
    vel_y = np.tile(x[T:2*T],(T,1)).T - np.tile(x[(i+1)*T:(i+2)*T],(T,1))
    vel_x = np.diag(vel_x)
    vel_y = np.diag(vel_y)
    i = i + 2

    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)

    quad_x = np.multiply(vel_x_2, one_over_cov_sum_x[ped])
    quad_y = np.multiply(vel_y_2, one_over_cov_sum_y[ped])

    Z_x = np.exp(-0.5*quad_x)
    Z_y = np.exp(-0.5*quad_y)
    Z = np.multiply(Z_x, Z_y)

    log_znot_norm = np.sum(np.log1p(-Z))

    llambda = llambda + log_znot_norm
  # agent-agent CCA
  i = 2
  j = 4
  for ped_i in range(ess):
    ped_j = int(j/2) - 1
    while j<=2*ess:
      vel_x = np.tile(\
                 x[i*T:(i+1)*T],(T,1)).T - np.tile(x[(j)*T:(j+1)*T], (T,1))
      vel_y = np.tile(\
                 x[(i+1)*T:(i+2)*T],(T,1)).T - np.tile(x[(j+1)*T:(j+2)*T],(T,1))
      vel_x = np.diag(vel_x)
      vel_y = np.diag(vel_y)
      vel_x_2 = np.power(vel_x, 2)
      vel_y_2 = np.power(vel_y, 2)

      quad_x = np.multiply(vel_x_2, one_over_cov_sumij_x[ped_i][ped_j])
      quad_y = np.multiply(vel_y_2, one_over_cov_sumij_y[ped_i][ped_j])

      Z_x = np.exp(-0.5*quad_x)
      Z_y = np.exp(-0.5*quad_y)
      Z = np.multiply(Z_x, Z_y)
      log_znot_norm = np.sum(np.log1p(-Z))

      llambda = llambda + log_znot_norm
      j = j + 2
    i = i + 2
    j = i + 2
  return -1.*llambda

def d_ll(x, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
         cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
         cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
         one_over_cov_sum_x, one_over_cov_sum_y, \
         one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T):
  d_beta = [0. for _ in range(2*T*np.int(np.round(ess+1)))]
  d_llambda = np.asarray([0. for _ in range(2*T*np.int(np.round(ess+1)))])
# DERIVATIVE WRT ROBOT
  i = 2
  for ped in range(ess):
    vel_x = np.tile(x[:T],(T,1)).T - np.tile(x[i*T:(i+1)*T],(T,1))
    vel_y = np.tile(x[T:2*T],(T,1)).T - np.tile(x[(i+1)*T:(i+2)*T],(T,1))
    vel_x = np.diag(vel_x)
    vel_y = np.diag(vel_y)

    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)

    quad_x = np.multiply(one_over_cov_sum_x[ped], vel_x_2)
    quad_y = np.multiply(one_over_cov_sum_y[ped], vel_y_2)

    Z_x = np.exp(-0.5*quad_x)
    Z_y = np.exp(-0.5*quad_y)
    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)

    alpha_x = np.multiply(X, np.multiply(vel_x, one_over_cov_sum_x[ped]))
    alpha_y = np.multiply(X, np.multiply(vel_y, one_over_cov_sum_y[ped]))

    d_llambda[:T] = np.add(d_llambda[:T], np.sum(alpha_x, axis=1))
    d_llambda[T:2*T] = np.add(d_llambda[T:2*T], np.sum(alpha_y, axis=1))
    i = i + 2

  d_beta[:T] = -np.dot(x[:T]-robot_mu_x, inv_cov_robot_x)
  d_beta[T:2*T] = -np.dot(x[T:2*T]-robot_mu_y, inv_cov_robot_y)
  d_llambda[0:2*T] = np.add(d_llambda[0:2*T], d_beta[0:2*T])
# DERIVATIVE WRT PED: ROBOT-PED DERIVATIVE, FIRST TERM
  i = 2
  for ped in range(ess):
    vel_x = np.tile(x[:T],(T,1)) - np.tile(x[i*T:(i+1)*T],(T,1)).T
    vel_y = np.tile(x[T:2*T],(T,1)) - np.tile(x[(i+1)*T:(i+2)*T],(T,1)).T
    vel_x = np.diag(vel_x)
    vel_y = np.diag(vel_y)

    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)

    quad_x = np.multiply(vel_x_2, one_over_cov_sum_x[ped])
    quad_y = np.multiply(vel_y_2, one_over_cov_sum_y[ped])

    Z_x = np.exp(-0.5*quad_x)
    Z_y = np.exp(-0.5*quad_y)

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)

    alpha_x = np.multiply(X, np.multiply(vel_x, one_over_cov_sum_x[ped]))
    alpha_y = np.multiply(X, np.multiply(vel_y, one_over_cov_sum_y[ped]))

    d_llambda[i*T:(i+1)*T] = -np.sum(alpha_x, axis=1)
    d_llambda[(i+1)*T:(i+2)*T] = -np.sum(alpha_y, axis=1)
    i = i + 2
# DERIVATIVE WRT PED: PED-PED DERIVATIVE, SECOND TERM
  i = 2
  j = 2
  for ped_i in range(ess):
    for ped_j in range(ess):
      if i != j:
        vel_x = np.tile(x[i*T:(i+1)*T],(T,1)).T - np.tile(x[j*T:(j+1)*T], (T,1))
        vel_y = np.tile(\
                 x[(i+1)*T:(i+2)*T],(T,1)).T - np.tile(x[(j+1)*T:(j+2)*T],(T,1))
        vel_x = np.diag(vel_x)
        vel_y = np.diag(vel_y)
        vel_x_2 = np.power(vel_x, 2)
        vel_y_2 = np.power(vel_y, 2)

        quad_x = np.multiply(vel_x_2, one_over_cov_sumij_x[ped_i][ped_j])
        quad_y = np.multiply(vel_y_2, one_over_cov_sumij_y[ped_i][ped_j])

        Z_x = np.exp(-0.5*quad_x)
        Z_y = np.exp(-0.5*quad_y)
        Z = np.multiply(Z_x, Z_y)
        X = np.divide(Z, 1.-Z)

        alpha_x = np.multiply(X, np.multiply(vel_x, \
                                            one_over_cov_sumij_x[ped_i][ped_j]))
        alpha_y = np.multiply(X, np.multiply(vel_y, \
                                            one_over_cov_sumij_y[ped_i][ped_j]))

        d_llambda[i*T:(i+1)*T] = np.add(d_llambda[i*T:(i+1)*T], \
                                                        np.sum(alpha_x, axis=1))
        d_llambda[(i+1)*T:(i+2)*T] = np.add(d_llambda[(i+1)*T:(i+2)*T], \
                                                        np.sum(alpha_y, axis=1))
      j = j + 2
    i = i + 2
    j = 2
# DERIVATIVE WRT PED: PED-PED DERIVATIVE, THIRD TERM
  i = 2
  for ped in range(ess):
    d_beta[i*T:(i+1)*T] = -np.dot(x[i*T:(i+1)*T]-ped_mu_x[ped], \
                                                           inv_cov_ped_x[ped])
    d_beta[(i+1)*T:(i+2)*T] = -np.dot(x[(i+1)*T:(i+2)*T]-ped_mu_y[ped], \
                                                           inv_cov_ped_y[ped])
    i = i + 2
  d_llambda[2*T:] = np.add(d_llambda[2*T:], d_beta[2*T:])
  return -1.*d_llambda

def dd_ll(x, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
          cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
          cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
          one_over_cov_sum_x, one_over_cov_sum_y, \
          one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T):
  H = np.zeros((2*T*np.int(ess+1),2*T*np.int(ess+1)), float)
  sum_alpha = [0. for _ in range(2*T*np.int(np.round(ess+1)))]
# ROBOT DIAG AND OFF DIAG (ROBOT-PED) COMPUTATION
  i = 2
  for ped in range(ess):
    vel_x = np.tile(x[:T],(T,1)).T - np.tile(x[i*T:(i+1)*T],(T,1))
    vel_y = np.tile(x[T:2*T],(T,1)).T - np.tile(x[(i+1)*T:(i+2)*T],(T,1))
    vel_x = np.diag(vel_x)
    vel_y = np.diag(vel_y)
    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)
    vel_x_y = np.multiply(vel_x, vel_y)

    one_over_cov_x_y = np.multiply(one_over_cov_sum_x[ped], \
                                                        one_over_cov_sum_y[ped])
    quad_x = np.multiply(one_over_cov_sum_x[ped], vel_x_2)
    quad_y = np.multiply(one_over_cov_sum_y[ped], vel_y_2)

    Z_x = np.exp(-0.5*quad_x)
    Z_y = np.exp(-0.5*quad_y)

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)
    X_2 = np.power(X, 2)

    X_plus_X2 = np.add(X, X_2)

    alpha_x = np.multiply(X, one_over_cov_sum_x[ped])
    alpha_x = np.add(alpha_x, -np.multiply(X_plus_X2, np.power(\
                         np.multiply(vel_x, one_over_cov_sum_x[ped]), 2)))
    alpha_y = np.multiply(X, one_over_cov_sum_y[ped])
    alpha_y = np.add(alpha_y, -np.multiply(X_plus_X2, np.power(\
                         np.multiply(vel_y, one_over_cov_sum_y[ped]), 2)))

    sum_alpha[:T] = np.add(sum_alpha[:T], np.sum(alpha_x, axis=1))
    sum_alpha[T:2*T] = np.add(sum_alpha[T:2*T], np.sum(alpha_y, axis=1))

    d_off_alpha = -np.multiply(X_plus_X2, np.multiply(vel_x_y, \
                                                              one_over_cov_x_y))
#   ROBOT OFF DIAG (ROBOT-PED) ENTRY
    H[:T,T:2*T] = np.add(H[:T,T:2*T], np.diag(np.sum(d_off_alpha, axis=1)))

    H[:T,i*T:(i+1)*T] = -1.*alpha_x
    H[i*T:(i+1)*T,:T] = H[:T,i*T:(i+1)*T].T

    H[T:2*T,(i+1)*T:(i+2)*T] = -1.*alpha_y
    H[(i+1)*T:(i+2)*T,T:2*T] = H[T:2*T,(i+1)*T:(i+2)*T].T

    H[T:2*T,i*T:(i+1)*T] = np.multiply(X_plus_X2, np.multiply(vel_x_y, \
                                                              one_over_cov_x_y))
    H[i*T:(i+1)*T,T:2*T] = H[T:2*T,i*T:(i+1)*T].T

    H[:T,(i+1)*T:(i+2)*T] = np.multiply(X_plus_X2, np.multiply(vel_x_y, \
                                                              one_over_cov_x_y))
    H[(i+1)*T:(i+2)*T,:T] = H[:T,(i+1)*T:(i+2)*T].T

    i = i + 2
# ROBOT DIAG ENTRY
  H[:T,:T] = np.add(np.diag(sum_alpha[:T]), -1.*inv_cov_robot_x)
  H[T:2*T,T:2*T] = np.add(np.diag(sum_alpha[T:2*T]), -1.*inv_cov_robot_y)

  H[T:2*T,:T] = H[:T,T:2*T].T
# PED i-PED i DIAG COMPUTATION: ROBOT-PED i, FO TERMS, TAU-T INDEXING
  i = 2
  for ped in range(ess):
    vel_x = np.tile(x[:T],(T,1)) - np.tile(x[i*T:(i+1)*T],(T,1)).T
    vel_y = np.tile(x[T:2*T],(T,1)) - np.tile(x[(i+1)*T:(i+2)*T],(T,1)).T
    vel_x = np.diag(vel_x)
    vel_y = np.diag(vel_y)
    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)
    vel_x_y = np.multiply(vel_x, vel_y)

    one_over_cov_x_y = np.multiply(one_over_cov_sum_x[ped], \
                                                        one_over_cov_sum_y[ped])
    quad_x = np.multiply(one_over_cov_sum_x[ped], vel_x_2)
    quad_y = np.multiply(one_over_cov_sum_y[ped], vel_y_2)

    Z_x = np.exp(-0.5*quad_x)
    Z_y = np.exp(-0.5*quad_y)

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)
    X_2 = np.power(X, 2)

    X_plus_X2 = np.add(X, X_2)

    alpha_x = np.multiply(X, one_over_cov_sum_x[ped])
    alpha_x = np.add(alpha_x, -np.multiply(X_plus_X2, np.power(\
                           np.multiply(vel_x, one_over_cov_sum_x[ped]), 2)))
    alpha_y = np.multiply(X, one_over_cov_sum_y[ped])
    alpha_y = np.add(alpha_y, -np.multiply(X_plus_X2, np.power(\
                           np.multiply(vel_y, one_over_cov_sum_y[ped]), 2)))
# PED i-PED i DIAG ENTRY, FO TERMS
    H[i*T:(i+1)*T,i*T:(i+1)*T] = np.diag(np.sum(alpha_x, axis=1)) - \
                                                              inv_cov_ped_x[ped]
    H[(i+1)*T:(i+2)*T,(i+1)*T:(i+2)*T] = np.diag(np.sum(alpha_y, axis=1)) - \
                                                              inv_cov_ped_y[ped]
    H[i*T:(i+1)*T,(i+1)*T:(i+2)*T] = -np.diag(np.sum(np.multiply(X_plus_X2, \
                           np.multiply(vel_x_y, one_over_cov_x_y)), axis=1))

    H[(i+1)*T:(i+2)*T,i*T:(i+1)*T] = H[i*T:(i+1)*T,(i+1)*T:(i+2)*T].T

    i = i + 2
# PED i-PED i DIAG COMPUTATION: ROBOT TO PED i, SO TERMS, T-TAU INDEX
# TAU,T INDEXING = np.tile(x[:T],(T,1)) - np.tile(x[i*T:(i+1)*T],(T,1)).T
# T,TAU INDEX = np.tile(x[i*T:(i+1)*T],(T,1)).T - np.tile(x[j*T:(j+1)*T], (T,1))
  i = 2
  j = 2
  for ped_i in range(ess):
    for ped_j in range(ess):
      if i != j:
        vel_x = np.tile(x[i*T:(i+1)*T],(T,1)).T - np.tile(x[j*T:(j+1)*T], (T,1))
        vel_y = np.tile(\
                 x[(i+1)*T:(i+2)*T],(T,1)).T - np.tile(x[(j+1)*T:(j+2)*T],(T,1))
        vel_x = np.diag(vel_x)
        vel_y = np.diag(vel_y)
        vel_x_2 = np.power(vel_x, 2)
        vel_y_2 = np.power(vel_y, 2)
        vel_x_y = np.multiply(vel_x, vel_y)

        one_over_covij_x_y = np.multiply(one_over_cov_sumij_x[ped_i][ped_j], \
                                             one_over_cov_sumij_y[ped_i][ped_j])
        quad_x = np.multiply(one_over_cov_sumij_x[ped_i][ped_j], vel_x_2)
        quad_y = np.multiply(one_over_cov_sumij_y[ped_i][ped_j], vel_y_2)

        Z_x = np.exp(-0.5*quad_x)
        Z_y = np.exp(-0.5*quad_y)

        Z = np.multiply(Z_x, Z_y)
        X = np.divide(Z, 1.-Z)
        X_2 = np.power(X, 2)

        X_plus_X2 = np.add(X, X_2)

        alpha_x = np.multiply(X, one_over_cov_sumij_x[ped_i][ped_j])
        alpha_x = np.add(alpha_x, -np.multiply(X_plus_X2, np.power(\
                    np.multiply(vel_x, one_over_cov_sumij_x[ped_i][ped_j]), 2)))
        alpha_y = np.multiply(X, one_over_cov_sumij_y[ped_i][ped_j])
        alpha_y = np.add(alpha_y, -np.multiply(X_plus_X2, np.power(\
                    np.multiply(vel_y, one_over_cov_sumij_y[ped_i][ped_j]), 2)))

        H[i*T:(i+1)*T,i*T:(i+1)*T] = np.add(\
          H[i*T:(i+1)*T,i*T:(i+1)*T], np.diag(np.sum(alpha_x, axis=1)))
        H[(i+1)*T:(i+2)*T,(i+1)*T:(i+2)*T] = np.add(\
          H[(i+1)*T:(i+2)*T,(i+1)*T:(i+2)*T], np.diag(np.sum(alpha_y, axis=1)))

        d_off_alpha = np.multiply(X_plus_X2, \
                                       np.multiply(vel_x_y, one_over_covij_x_y))
        H[i*T:(i+1)*T,(i+1)*T:(i+2)*T] = np.add(\
          H[i*T:(i+1)*T,(i+1)*T:(i+2)*T], -np.diag(np.sum(d_off_alpha, axis=1)))
      j = j + 2
    H[(i+1)*T:(i+2)*T, i*T:(i+1)*T] = H[i*T:(i+1)*T,(i+1)*T:(i+2)*T]
    i = i + 2
    j = 2
# PED i-PED j OFF DIAG COMPUTATION
  i = 2
  j = 2
  for ped_i in range(ess):
    for ped_j in range(ess):
      if i != j:
        vel_x = np.tile(x[i*T:(i+1)*T],(T,1)).T - np.tile(x[j*T:(j+1)*T], (T,1))
        vel_y = np.tile(\
                 x[(i+1)*T:(i+2)*T],(T,1)).T - np.tile(x[(j+1)*T:(j+2)*T],(T,1))
        vel_x = np.diag(vel_x)
        vel_y = np.diag(vel_y)
        vel_x_2 = np.power(vel_x, 2)
        vel_y_2 = np.power(vel_y, 2)
        vel_x_y = np.multiply(vel_x, vel_y)

        one_over_covij_x_y = np.multiply(one_over_cov_sumij_x[ped_i][ped_j], \
                                             one_over_cov_sumij_y[ped_i][ped_j])
        quad_x = np.multiply(one_over_cov_sumij_x[ped_i][ped_j], vel_x_2)
        quad_y = np.multiply(one_over_cov_sumij_y[ped_i][ped_j], vel_y_2)

        Z_x = np.exp(-0.5*quad_x)
        Z_y = np.exp(-0.5*quad_y)

        Z = np.multiply(Z_x, Z_y)
        X = np.divide(Z, 1.-Z)
        X_2 = np.power(X, 2)

        X_plus_X2 = np.add(X, X_2)

        alpha_x = np.multiply(X, one_over_cov_sumij_x[ped_i][ped_j])
        alpha_x = np.add(alpha_x, -np.multiply(X_plus_X2, np.power(\
                    np.multiply(vel_x, one_over_cov_sumij_x[ped_i][ped_j]), 2)))
        alpha_y = np.multiply(X, one_over_cov_sumij_y[ped_i][ped_j])
        alpha_y = np.add(alpha_y, -np.multiply(X_plus_X2, np.power(\
                    np.multiply(vel_y, one_over_cov_sumij_y[ped_i][ped_j]), 2)))
        alpha_x_y = np.multiply(\
                            X_plus_X2, np.multiply(vel_x_y, one_over_covij_x_y))

        H[i*T:(i+1)*T,j*T:(j+1)*T] = -1.*alpha_x
        H[j*T:(j+1)*T,i*T:(i+1)*T] = H[i*T:(i+1)*T,j*T:(j+1)*T].T

        H[(i+1)*T:(i+2)*T,(j+1)*T:(j+2)*T] = -1.*alpha_y
        H[(j+1)*T:(j+2)*T,(i+1)*T:(i+2)*T] = \
                                            H[(i+1)*T:(i+2)*T,(j+1)*T:(j+2)*T].T
        H[(i+1)*T:(i+2)*T,j*T:(j+1)*T] = alpha_x_y
        H[j*T:(j+1)*T,(i+1)*T:(i+2)*T] = H[(i+1)*T:(i+2)*T,(j)*T:(j+1)*T].T

        H[i*T:(i+1)*T,(j+1)*T:(j+2)*T] = alpha_x_y
        H[(j+1)*T:(j+2)*T,i*T:(i+1)*T] = H[i*T:(i+1)*T,(j+1)*T:(j+2)*T].T

        j = j + 2
    i = i + 2
    j = 2
  return -1.*H

