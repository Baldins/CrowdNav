import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import pylab
import math
import numpy as np
import os
from copy import deepcopy

def plotter_single(fig, ax, x, y, x_follow, y_follow, x_nonzero, y_nonzero, \
          frame, remove_ped, \
          num_peds_real, num_peds_follow, ped_mu_x, ped_mu_y, \
          optima_iterate_diag, optima_diag, optima_dex_diag, num_optima_diag, \
          robot_mu_x_diag, robot_mu_y_diag, \
          robot_history_x_diag, robot_history_y_diag, cmd_x_diag, cmd_y_diag, \
          ess_time_diag, ess_ave_time_diag, ess_std_time_diag, ess_diag, \
          time_diag, top_Z_indices_diag):
  p2w_x = 12./510.
  p2w_y = 5./322.
  p2w = np.sqrt(p2w_x**2 + p2w_y**2)

  cm = pylab.get_cmap('hsv')

  ax.clear()
# PEDESTRIANS
  for ped in range(num_peds_real):
    # ax.plot(ped_mu_x[ped]*p2w_x, \
    #         ped_mu_y[ped]*p2w_y, \
    #         "xb", label='robot GP mean', \
    #         markersize=2)
    x_temp = x[ped][frame:]
    y_temp = y[ped][frame:]
    x_temp = x_temp[np.nonzero(x_temp)]
    y_temp = y_temp[np.nonzero(y_temp)]
    ax.plot(x_temp*p2w_x, \
            y_temp*p2w_y, \
            "--^k", label='Not Observed', \
            markersize=1)
  ax.plot(x_nonzero*p2w_x, \
          y_nonzero*p2w_y, \
            "--^g", label='Not Observed', \
            markersize=1)
  # for ped in range(ess_diag):
  #   top = top_Z_indices_diag[ped]
  #   ax.plot(x_follow[top][frame]*p2w_x, \
  #           y_follow[top][frame]*p2w_y, \
  #           "og", label='Observed', \
  #           markersize=7)
  # for ped in range(ess_full):
  #   top = top_Z_indices_full[ped]
  #   ax.plot(x_follow[top][frame]*p2w_x, \
  #           y_follow[top][frame]*p2w_y, \
  #           "or", label='Observed', \
  #           markersize=4)
# ROBOT
  for ped in range(ess_diag):
    top = top_Z_indices_diag[ped]
    ax.annotate(ped, xy=(x_follow[top][frame]*p2w_x, \
                         y_follow[top][frame]*p2w_y), \
                         fontsize=12, fontweight='bold', \
                         color='cyan')
  # for ped in range(num_peds_real):
  #   ax.annotate(ped, xy=(x[ped][frame]*p2w_x, \
  #                        y[ped][frame]*p2w_y), \
  #                        fontsize=12, fontweight='bold', \
  #                        color='cyan')
  # ax.plot(robot_mu_x_diag*p2w_x, \
  #         robot_mu_y_diag*p2w_y, \
  #         "+r",label='robot GP mean', \
  #         markersize=1)

  color=0
  T_diag = np.size(robot_mu_x_diag)
  for intent in range(num_optima_diag):
    n=0
    for ped in range(ess_diag+1):
      if optima_iterate_diag:
        if n == 0:
          ax.plot(optima_diag[intent][n*T_diag:(n+1)*T_diag]*p2w_x, \
                optima_diag[intent][(n+1)*T_diag:(n+2)*T_diag]*p2w_y, \
                'xr', markersize=1)
        else:
          ax.plot(optima_diag[intent][n*T_diag:(n+1)*T_diag]*p2w_x, \
                optima_diag[intent][(n+1)*T_diag:(n+2)*T_diag]*p2w_y, \
                'xb', markersize=1)
                # '.', color = cm(1.*color), markersize=3)
      else:
        if n == 0:
          ax.plot(optima_diag[intent].x[n*T_diag:(n+1)*T_diag]*p2w_x, \
                optima_diag[intent].x[(n+1)*T_diag:(n+2)*T_diag]*p2w_y, \
                'xr', markersize=1)
        else:
          ax.plot(optima_diag[intent].x[n*T_diag:(n+1)*T_diag]*p2w_x, \
                optima_diag[intent].x[(n+1)*T_diag:(n+2)*T_diag]*p2w_y, \
                'xb', markersize=1)
                # '.', color = cm(1.*color), markersize=3)
      n = n + 2
    color = color + 10

  for ped in range(num_peds_real):
    if(ped == remove_ped):
      ax.plot(x[ped][frame]*p2w_x, \
              y[ped][frame]*p2w_y, \
              "og", label='Observed', \
              markersize=7)
    else:
      ax.plot(x[ped][frame]*p2w_x, \
              y[ped][frame]*p2w_y, \
              "ob", label='Observed', \
              markersize=7)

  ax.plot(cmd_x_diag*p2w_x, cmd_y_diag*p2w_y, 'py', markersize=6)

  ax.plot(robot_history_x_diag*p2w_x, \
          robot_history_y_diag*p2w_y, 'x--r', \
          markersize=2)

  plt.xlabel('X Position',fontsize=10)
  plt.ylabel('Y position',fontsize=10)

  ess_time_diag = math.trunc(ess_time_diag*1e3)/1e3
  ess_ave_time_diag = math.trunc(ess_ave_time_diag*1e3)/1e3
  ess_std_time_diag = math.trunc(ess_std_time_diag*1e3)/1e3

  if frame==0:
    ave_time_diag = time_diag[frame]
    std_time_diag = time_diag[frame]
    max_time_diag = math.trunc(time_diag[frame]*1e3)/1e3
  else:
    ave_time_diag = np.mean(time_diag[:frame])
    std_time_diag = np.std(time_diag[:frame])
    max_time_diag = math.trunc(np.max(time_diag[:frame])*1e3)/1e3

  time_now = math.trunc(time_diag[frame]*1e3)/1e3
  ave_time_diag = math.trunc(ave_time_diag*1e3)/1e3
  std_time_diag = math.trunc(std_time_diag*1e3)/1e3

  plt.title('Frame {0}, t_now={2}, t_max = {5}, t_ave={3}+/-{4}, ESS={1}'.\
format(frame, ess_diag, time_now, ave_time_diag, std_time_diag, \
       max_time_diag), \
            fontsize=7)

  # ax.set_xlim(5,30)
  # ax.set_ylim(4,12)

  ax.set_xlim(-5,35)
  ax.set_ylim(0,13)

  plt.pause(0.0005)

