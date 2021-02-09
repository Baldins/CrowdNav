import os
import math
import numpy as np

def data_dirs(remove_ped, optima_iterate_diag, tol, \
              ess_off_diag, ess_boost, ess_num_peds, \
              ess_newton_diag, Tdex_max_diag, goal_noise, max_vel_robot, \
              home_dir):
  p2w_x = 12./510.
  p2w_y = 5./322.
  p2w = np.sqrt(p2w_x**2 + p2w_y**2)

  if os.path.exists(str(home_dir)+'/results/igp/metrics/eth/\
agent_' + str(remove_ped) + \
'_optima_iter_'+ str(optima_iterate_diag) + \
'_tol_' + str(tol) + \
'_ess_off_diag_' + str(ess_off_diag) + \
'_ess_boost_' + str(ess_boost) + \
'_ess+_' + str(ess_num_peds) + \
'_essNewton_' + str(ess_newton_diag) + \
'_Tdex_max_' + str(Tdex_max_diag) + \
'_speed_' + str(math.trunc(max_vel_robot*p2w*1e3)/1e3) + \
'_goal_noise_' + str(goal_noise))==False:
    os.mkdir(str(home_dir)+'/results/igp/metrics/eth/\
agent_' + str(remove_ped) + \
'_optima_iter_' + str(optima_iterate_diag) + \
'_tol_' + str(tol) + \
'_ess_off_diag_' + str(ess_off_diag) + \
'_ess_boost_' + str(ess_boost) + \
'_ess+_' + str(ess_num_peds) + \
'_essNewton_' + str(ess_newton_diag) + \
'_Tdex_max_' + str(Tdex_max_diag) + \
'_speed_' + str(math.trunc(max_vel_robot*p2w*1e3)/1e3) + \
'_goal_noise_' + str(goal_noise))

  os.chdir(str(home_dir)+'/results/igp/metrics/eth/\
agent_' + str(remove_ped) + \
'_optima_iter_' + str(optima_iterate_diag) + \
'_tol_' + str(tol) + \
'_ess_off_diag_' + str(ess_off_diag) + \
'_ess_boost_' + str(ess_boost) + \
'_ess+_' + str(ess_num_peds) + \
'_essNewton_' + str(ess_newton_diag) + \
'_Tdex_max_' + str(Tdex_max_diag) + \
'_speed_' + str(math.trunc(max_vel_robot*p2w*1e3)/1e3) + \
'_goal_noise_' + str(goal_noise))
