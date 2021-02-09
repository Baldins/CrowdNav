import matplotlib.pyplot as plt
import os
import math
import numpy as np

from _so_dir_maker import dir_maker

def save_figs(frame, remove_ped, tol, ess_off, ess_num_peds, \
              ess_boost, ess_newton, goal_noise, home_dir, chandler, \
              conditioned, ll_converge, \
              rand, var, num_random_samples, num_var_samples, \
              var_ratio, data_set, p2w_x, p2w_y, p2w, support_boost, \
              ess_limit, ess_limit_num, goal_dex, full_traj, \
              remove_ped_start, dwa, save_dir, mc, num_mc_samples, save_pdf, \
              opt_iter_robot, opt_iter_all):
  if conditioned:
    conditioned = 'T'
  else:
    conditioned = 'F'
  if opt_iter_robot or opt_iter_all:
    optima_iterate = 'T'
  else:
    optima_iterate = 'F'
  if ll_converge:
    ll_converge = 'T'
  else:
    ll_converge = 'F'
  if ess_boost:
    ess_boost = 'T'
  else:
    ess_boost = 'F'
  if ess_limit:
    ess_limit = 'T'
  else:
    ess_limit = 'F'
  if ess_newton:
    ess_newton = 'T'
  else:
    ess_newton = 'F'
  if var:
    var = 'T'
  else:
    var = 'F'
  if rand:
    rand = 'T'
  else:
    rand = 'F'
  if ess_limit:
    ess_limit = 'T'
  else:
    ess_limit = 'F'

  plots_or_metrics = 'plots'
  dir_maker(remove_ped, tol, ess_off, ess_boost, ess_num_peds, \
            ess_newton, home_dir, chandler, conditioned, ll_converge, \
            rand, var, num_random_samples, num_var_samples, var_ratio, \
            data_set, p2w_x, p2w_y, p2w, support_boost, ess_limit, \
            ess_limit_num, goal_dex, full_traj, remove_ped_start, \
            plots_or_metrics, dwa, save_dir, mc, num_mc_samples, \
            opt_iter_robot, opt_iter_all)

  if save_pdf:
    file_type = '.pdf'
  else:
    file_type = '.png'
  plt.savefig('agent_' + str(remove_ped) + \
              '_num_mc_samples_' + str(num_mc_samples) + \
              '_start_'+ str(remove_ped_start) + \
              '_steps_'+ str(goal_dex) + \
              '_cond_'+ str(conditioned) + \
              '_opt_iter_'+ str(optima_iterate) + \
              '_llconv_'+ str(ll_converge) + \
              '_tol_' + str(tol) + \
              '_ess_boost_' + str(ess_boost) + '_' + str(ess_num_peds) + \
              '_ess_lim_' + str(ess_limit) + '_' + str(ess_limit_num) + \
              '_essHess_' + str(ess_newton) + \
              '_VAR_' + str(var) + '_' + str(num_var_samples) + \
              '_ratio_' + str(var_ratio) + \
              '_RAND_' + str(rand) + '_' +  str(num_random_samples) + \
              '_supp_' + str(support_boost) + \
              '_frame_' + str(frame) + file_type)




