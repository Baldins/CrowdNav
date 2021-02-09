import os
import math
import numpy as np

def dir_maker(remove_ped, tol, ess_off, ess_boost, \
              ess_num_peds, ess_newton, home_dir, chandler, conditioned, \
              ll_converge, rand, var, \
              num_random_samples, num_var_samples, var_ratio, data_set, \
              p2w_x, p2w_y, p2w, support_boost, ess_limit, ess_limit_num, \
              goal_dex, full_traj, remove_ped_start, plots_or_metrics, dwa, \
              save_dir, mc, num_mc_samples, opt_iter_robot, opt_iter_all):
  if full_traj:
    traj_dir = 'full_traj'
  else:
    traj_dir = 'partial_traj'
  if opt_iter_robot or opt_iter_all:
    optima_iterate = 'T'
  else:
    optima_iterate = 'F'
  check_dir = str(home_dir) + '/results/' + str(save_dir) + '/' + \
str(plots_or_metrics) + '/' + str(data_set) + '/' + str(traj_dir) + '/'

  save_folder = str(home_dir) + '/results/' + str(save_dir) + '/' + \
str(plots_or_metrics) + '/' + str(data_set) + '/' + str(traj_dir) + '/' + \
'agent_' + str(remove_ped) + '_num_mc_'+ str(num_mc_samples) + \
'_start_'+ str(remove_ped_start) + \
'_steps_'+ str(goal_dex) + '_cond_'+ str(conditioned) + \
'_opt_iter_'+ str(optima_iterate) + '_llconv_'+ str(ll_converge) + \
'_tol_' + str(tol) + '_ess_boost_' + str(ess_boost) + '_' + str(ess_num_peds) +\
'_ess_lim_' + str(ess_limit) + '_' + str(ess_limit_num) + \
'_essHess_' + str(ess_newton) + \
'_VAR_' + str(var) + '_' + str(num_var_samples) + '_ratio_' + str(var_ratio) + \
'_RAND_' + str(rand) + '_' +  str(num_random_samples) + \
'_supp_' + str(support_boost) + '_dwa_' + str(dwa)

  if not os.path.exists(check_dir):
    os.mkdir(check_dir)

  if not os.path.exists(save_folder):
    os.mkdir(save_folder)

  os.chdir(save_folder)



















