import autograd.numpy as np

def ll(x, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
       cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
       cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
       one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  T = np.size(robot_mu_x)

  quad_robot_mu_x = np.dot((x[:T]-robot_mu_x).T, np.dot(inv_cov_robot_x, \
                                                              x[:T]-robot_mu_x))
  quad_robot_mu_y = np.dot((x[T:2*T]-robot_mu_y).T, np.dot(inv_cov_robot_y, \
                                                           x[T:2*T]-robot_mu_y))
  llambda = -0.5*quad_robot_mu_x - 0.5*quad_robot_mu_y

  n=2
  for ped in range(ess):
    quad_ped_mu_x = np.dot((x[n*T:(n+1)*T]-ped_mu_x[ped]).T, np.dot(\
                            inv_cov_ped_x[ped], x[n*T:(n+1)*T]-ped_mu_x[ped]))
    quad_ped_mu_y = np.dot((x[(n+1)*T:(n+2)*T]-ped_mu_y[ped]).T, np.dot(\
                        inv_cov_ped_y[ped], x[(n+1)*T:(n+2)*T]-ped_mu_y[ped]))
    llambda = llambda -0.5*quad_ped_mu_x - 0.5*quad_ped_mu_y
    n = n + 2

  n = 2
  for ped in range(ess):
    # if normalize == True:
    #   # normalize_x = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_x[ped])
    #   # normalize_y = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_y[ped])
    # else:
    normalize_x = 1.
    normalize_y = 1.

    vel_x = np.tile(x[:T],(T,1)).T - np.tile(x[n*T:(n+1)*T], (T,1))
    vel_y = np.tile(x[T:2*T],(T,1)).T - np.tile(x[(n+1)*T:(n+2)*T],(T,1))
    n = n + 2

    vel_x_2 = np.power(vel_x, 2)
    vel_y_2 = np.power(vel_y, 2)

    quad_robot_ped_x = np.multiply(vel_x_2, one_over_cov_sum_x[ped])
    quad_robot_ped_y = np.multiply(vel_y_2, one_over_cov_sum_y[ped])

    Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_robot_ped_x))
    Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_robot_ped_y))

    Z = np.multiply(Z_x, Z_y)

    log_znot_norm = np.sum(np.log1p(-Z))

    llambda = llambda + log_znot_norm
  return -1.*llambda

def d_ll(x, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
         cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
         cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
         one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  T = np.size(robot_mu_x)

  d_alpha = [0. for _ in range(2*T*np.int(np.round(ess+1)))]
  d_beta = [0. for _ in range(2*T*np.int(np.round(ess+1)))]
  d_llambda = np.asarray([0. for _ in range(2*T*np.int(np.round(ess+1)))])

  n = 2
  for ped in range(ess):
    # if normalize == True:
    #   # normalize_x = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_x[ped])
    #   # normalize_y = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_y[ped])
    # else:
    normalize_x = 1.
    normalize_y = 1.

    vel_robot_x = np.tile(x[:T],(T,1)).T - np.tile(x[n*T:(n+1)*T],(T,1))
    vel_robot_y = np.tile(x[T:2*T],(T,1)).T - np.tile(x[(n+1)*T:(n+2)*T],(T,1))
    n = n + 2

    vel_robot_x_2 = np.power(vel_robot_x, 2)
    vel_robot_y_2 = np.power(vel_robot_y, 2)

    quad_robot_x = np.multiply(one_over_cov_sum_x[ped], vel_robot_x_2)
    quad_robot_y = np.multiply(one_over_cov_sum_y[ped], vel_robot_y_2)

    Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_robot_x))
    Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_robot_y))

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)

    alpha_x = np.multiply(X, np.multiply(vel_robot_x, one_over_cov_sum_x[ped]))
    alpha_y = np.multiply(X, np.multiply(vel_robot_y, one_over_cov_sum_y[ped]))
  #        X and Y COMPONENT OF R DERIVATIVE
    d_alpha[:T] = np.add(d_alpha[:T], np.sum(alpha_x, axis=1))
    d_alpha[T:2*T] = np.add(d_alpha[T:2*T], np.sum(alpha_y, axis=1))

  d_beta[:T] = -np.dot(x[:T]-robot_mu_x, inv_cov_robot_x)
  d_beta[T:2*T] = -np.dot(x[T:2*T]-robot_mu_y, inv_cov_robot_y)

  d_llambda[0:2*T] = np.add(d_alpha[0:2*T], d_beta[0:2*T])
#        X AND Y COMPONENT OF PED DERIVATIVE
  n = 2
  for ped in range(ess):
    # if normalize == True:
    #   # normalize_x = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_x[ped])
    #   # normalize_y = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_y[ped])
    # else:
    normalize_x = 1.
    normalize_y = 1.

    vel_ped_x = np.tile(x[:T],(T,1)) - np.tile(x[n*T:(n+1)*T],(T,1)).T
    vel_ped_y = np.tile(x[T:2*T],(T,1)) - np.tile(x[(n+1)*T:(n+2)*T],(T,1)).T
    vel_ped_x_2 = np.power(vel_ped_x, 2)
    vel_ped_y_2 = np.power(vel_ped_y, 2)

    quad_ped_x = np.multiply(one_over_cov_sum_x[ped], vel_ped_x_2)
    quad_ped_y = np.multiply(one_over_cov_sum_y[ped], vel_ped_y_2)

    Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_ped_x))
    Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_ped_y))

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)

    alpha_x = np.multiply(X, np.multiply(vel_ped_x, one_over_cov_sum_x[ped]))
    alpha_y = np.multiply(X, np.multiply(vel_ped_y, one_over_cov_sum_y[ped]))

    d_alpha[n*T:(n+1)*T] = -np.sum(alpha_x, axis=1)
    d_alpha[(n+1)*T:(n+2)*T] = -np.sum(alpha_y, axis=1)

    d_beta[n*T:(n+1)*T] = -np.dot(x[n*T:(n+1)*T]-ped_mu_x[ped], \
                                                             inv_cov_ped_x[ped])
    d_beta[(n+1)*T:(n+2)*T] = -np.dot(x[(n+1)*T:(n+2)*T]-ped_mu_y[ped], \
                                                             inv_cov_ped_y[ped])
    n = n + 2

  d_llambda[2*T:] = np.add(d_alpha[2*T:], d_beta[2*T:])
  return -1.*d_llambda

def dd_ll(x, num_peds, ess, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
          cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
          cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
          one_over_cov_sum_x, one_over_cov_sum_y, normalize):
  T = np.size(robot_mu_x)

  H = np.zeros((2*T*np.int(ess+1),2*T*np.int(ess+1)), float)
  sum_d_alpha = [0. for _ in range(2*T*np.int(np.round(ess+1)))]

  n = 2
  for ped in range(ess):
    # if normalize == True:
    #   # normalize_x = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_x[ped])
    #   # normalize_y = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_y[ped])
    # else:
    normalize_x = 1.
    normalize_y = 1.

    vel_robot_x = np.tile(x[:T],(T,1)).T - np.tile(x[n*T:(n+1)*T],(T,1))
    vel_robot_y = np.tile(x[T:2*T],(T,1)).T - np.tile(x[(n+1)*T:(n+2)*T],(T,1))
    vel_robot_x_2 = np.power(vel_robot_x, 2)
    vel_robot_y_2 = np.power(vel_robot_y, 2)
    vel_robot_x_y = np.multiply(vel_robot_x, vel_robot_y)

    one_over_cov_x_y = np.multiply(one_over_cov_sum_x[ped], \
                                                        one_over_cov_sum_y[ped])
    quad_robot_x = np.multiply(one_over_cov_sum_x[ped], vel_robot_x_2)
    quad_robot_y = np.multiply(one_over_cov_sum_y[ped], vel_robot_y_2)

    Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_robot_x))
    Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_robot_y))

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)
    X_2 = np.power(X, 2)

    X_plus_X2 = np.add(X, X_2)

    d_alpha_x = np.multiply(X, one_over_cov_sum_x[ped])
    d_alpha_x = np.add(d_alpha_x, -np.multiply(X_plus_X2, np.power(\
                         np.multiply(vel_robot_x, one_over_cov_sum_x[ped]), 2)))
    d_alpha_y = np.multiply(X, one_over_cov_sum_y[ped])
    d_alpha_y = np.add(d_alpha_y, -np.multiply(X_plus_X2, np.power(\
                         np.multiply(vel_robot_y, one_over_cov_sum_y[ped]), 2)))

    sum_d_alpha[:T] = np.add(sum_d_alpha[:T], np.sum(d_alpha_x, axis=1))
    sum_d_alpha[T:2*T] = np.add(sum_d_alpha[T:2*T], np.sum(d_alpha_y, axis=1))

    d_off_alpha = -np.multiply(X_plus_X2, np.multiply(vel_robot_x_y, \
                                                              one_over_cov_x_y))
#   OFF DIAGONALS
    H[:T,T:2*T] = np.add(H[:T,T:2*T], np.diag(np.sum(d_off_alpha, axis=1)))

    H[:T,n*T:(n+1)*T] = -1.*d_alpha_x
    H[n*T:(n+1)*T,:T] = H[:T,n*T:(n+1)*T].T

    H[T:2*T,(n+1)*T:(n+2)*T] = -1.*d_alpha_y
    H[(n+1)*T:(n+2)*T,T:2*T] = H[T:2*T,(n+1)*T:(n+2)*T].T

    H[T:2*T,n*T:(n+1)*T] = np.multiply(X_plus_X2, np.multiply(vel_robot_x_y, \
                                                              one_over_cov_x_y))
    H[n*T:(n+1)*T,T:2*T] = H[T:2*T,n*T:(n+1)*T].T

    H[:T,(n+1)*T:(n+2)*T] = np.multiply(X_plus_X2, np.multiply(vel_robot_x_y, \
                                                              one_over_cov_x_y))
    H[(n+1)*T:(n+2)*T,:T] = H[:T,(n+1)*T:(n+2)*T].T

    n = n + 2

  H[:T,:T] = np.add(np.diag(sum_d_alpha[:T]), -1.*inv_cov_robot_x)
  H[T:2*T,T:2*T] = np.add(np.diag(sum_d_alpha[T:2*T]), -1.*inv_cov_robot_y)

  H[T:2*T,:T] = H[:T,T:2*T].T
#      PED DIAGONALS
  n = 2
  for ped in range(ess):
    # if normalize == True:
    #   # normalize_x = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_x[ped])
    #   # normalize_y = np.multiply(np.power(2*np.pi,-0.5), \
    # one_over_std_sum_y[ped])
    # else:
    normalize_x = 1.
    normalize_y = 1.

    vel_ped_x = np.tile(x[:T],(T,1)) - np.tile(x[n*T:(n+1)*T],(T,1)).T
    vel_ped_y = np.tile(x[T:2*T],(T,1)) - np.tile(x[(n+1)*T:(n+2)*T],(T,1)).T
    vel_ped_x_2 = np.power(vel_ped_x, 2)
    vel_ped_y_2 = np.power(vel_ped_y, 2)
    vel_ped_x_y = np.multiply(vel_ped_x, vel_ped_y)

    one_over_cov_x_y = np.multiply(one_over_cov_sum_x[ped], \
                                                        one_over_cov_sum_y[ped])
    quad_ped_x = np.multiply(one_over_cov_sum_x[ped], vel_ped_x_2)
    quad_ped_y = np.multiply(one_over_cov_sum_y[ped], vel_ped_y_2)

    Z_x = np.multiply(normalize_x, np.exp(-0.5*quad_ped_x))
    Z_y = np.multiply(normalize_y, np.exp(-0.5*quad_ped_y))

    Z = np.multiply(Z_x, Z_y)
    X = np.divide(Z, 1.-Z)
    X_2 = np.power(X, 2)

    X_plus_X2 = np.add(X, X_2)

    d_alpha_x = np.multiply(X, one_over_cov_sum_x[ped])
    d_alpha_x = np.add(d_alpha_x, -np.multiply(X_plus_X2, np.power(\
                           np.multiply(vel_ped_x, one_over_cov_sum_x[ped]), 2)))
    d_alpha_y = np.multiply(X, one_over_cov_sum_y[ped])
    d_alpha_y = np.add(d_alpha_y, -np.multiply(X_plus_X2, np.power(\
                           np.multiply(vel_ped_y, one_over_cov_sum_y[ped]), 2)))

    H[n*T:(n+1)*T,n*T:(n+1)*T] = np.diag(np.sum(d_alpha_x, axis=1)) - \
                                                              inv_cov_ped_x[ped]
    H[(n+1)*T:(n+2)*T,(n+1)*T:(n+2)*T] = np.diag(np.sum(d_alpha_y, axis=1)) - \
                                                              inv_cov_ped_y[ped]
    H[n*T:(n+1)*T,(n+1)*T:(n+2)*T] = -np.diag(np.sum(np.multiply(X_plus_X2, \
                           np.multiply(vel_ped_x_y, one_over_cov_x_y)), axis=1))

    H[(n+1)*T:(n+2)*T,n*T:(n+1)*T] = H[n*T:(n+1)*T,(n+1)*T:(n+2)*T].T

    n = n + 2
  return -1.*H

