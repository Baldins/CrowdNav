import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import pylab
import math
import numpy as np
import os
from copy import deepcopy

def plotter_single(fig, ax, x_follow, y_follow, x_nonzero, y_nonzero, \
      frame, remove_ped, num_peds_real, num_peds_follow, ped_mu_x, ped_mu_y, \
      optima, optima_dex, num_optima, robot_mu_x, robot_mu_y, \
      robot_history_x, robot_history_y, cmd_x, cmd_y, ess_time, ess_ave_time, \
      ess_std_time, ess, time, top_Z_indices, chandler, data_set, \
      p2w_x, p2w_y, p2w, radius, show_radius, scaling, num_var_samples, mc, \
      opt_iter_robot, opt_iter_all):
  cm = pylab.get_cmap('plasma')

  ax.clear()
# PLOT DENSITY CIRCLE
  if show_radius:
    density_circle = plt.Circle((cmd_x*p2w_x, cmd_y*p2w_y), radius, \
                                color='b', fill=False)
    ax.add_artist(density_circle)
# PLOT AGENT TRAJECTORIES[FRAME:}, REMOVE ZEROS
  lookahead = 10
  for ped in range(num_peds_follow):
    if data_set == 'eth_train':
      x_temp = x_follow[ped][frame:frame+lookahead]
      y_temp = y_follow[ped][frame:frame+lookahead]
      x_temp = x_temp[np.nonzero(x_temp)]
      y_temp = y_temp[np.nonzero(y_temp)]
    if data_set == 'eth_test':
      x_temp = x_follow[ped][frame:]
      y_temp = y_follow[ped][frame:]
      x_temp = x_temp[np.nonzero(x_temp)]
      y_temp = y_temp[np.nonzero(y_temp)]
    ax.plot(x_temp*p2w_x, y_temp*p2w_y, ".-k", markersize=2)

# PLOT ALL PED_MU
  # for ped in range(num_peds_follow):
  #   ax.plot(ped_mu_x[ped]*p2w_x, ped_mu_y[ped]*p2w_y, \
  #           "x",markersize=2)
# PLOT TOP PED_MU
  # for ped in range(ess):
  #   top = top_Z_indices[ped]
  #   ax.plot(ped_mu_x[top]*p2w_x, ped_mu_y[top]*p2w_y, "x", markersize=2)

# PLOT ROBOT_MU
  ax.plot(robot_mu_x*p2w_x, robot_mu_y*p2w_y, "+g", markersize=3)

# PLOT REMOVE PED
  ax.plot(x_nonzero*p2w_x, y_nonzero*p2w_y, "--^g", markersize=1)

# PLOT TOP ESS AGENTS WITH RANK
  # for ped in range(ess):
  #   top = top_Z_indices[ped]
  #   ax.annotate(ped, xy=(x_follow[top][frame]*p2w_x, \
  #                        y_follow[top][frame]*p2w_y), \
  #                        fontsize=6, fontweight='bold', color='cyan')

# PLOT TRUE NUMBERS OF ALL AGENTS
  # for ped in range(num_peds_follow):
  #   ax.annotate(ped, xy=(x_follow[ped][frame]*p2w_x, \
  #                        y_follow[ped][frame]*p2w_y), \
  #                        fontsize=5, fontweight='bold', color='cyan')

# PLOT OPTIMA, EITHER ITERATE OR SCIPY
  # color=0
  # T = np.size(robot_mu_x)
  # for intent in range(num_optima):
  #   n=0
  #   for ped in range(ess+1):
  #     if opt_iter_robot or opt_iter_all:
  #       if n == 0:
  #         ax.plot(optima[intent][n*T:(n+1)*T]*p2w_x, \
  #                 optima[intent][(n+1)*T:(n+2)*T]*p2w_y, \
  #                 'xr', markersize=1)
  #       else:
  #         ax.plot(optima[intent][n*T:(n+1)*T]*p2w_x, \
  #                 optima[intent][(n+1)*T:(n+2)*T]*p2w_y, \
  #                 'xb', markersize=1)
  #                 # '.', color = cm(1.*color), markersize=3)
  #     else:
  #       if n == 0:
  #         ax.plot(optima[intent].x[n*T:(n+1)*T]*p2w_x, \
  #                 optima[intent].x[(n+1)*T:(n+2)*T]*p2w_y, \
  #                 'xr', markersize=1)
  #       else:
  #         ax.plot(optima[intent].x[n*T:(n+1)*T]*p2w_x, \
  #                 optima[intent].x[(n+1)*T:(n+2)*T]*p2w_y, \
  #                 'xb', markersize=1)
  #               # '.', color = cm(1.*color), markersize=3)
  #     n = n + 2
  #   color = color + 10

# PLOT JOINT OPTIMA WITH LARGEST LL
  n = 0
  intent = 0
  T = np.size(robot_mu_x)
  for ped in range(ess+1):
    if mc:
      if n == 0:
        ax.plot(optima[n*T:(n+1)*T]*p2w_x, optima[(n+1)*T:(n+2)*T]*p2w_y, \
                  'xr', markersize=1)
      else:
        ax.plot(optima[n*T:(n+1)*T]*p2w_x, optima[(n+1)*T:(n+2)*T]*p2w_y, \
                  'xc', markersize=1)
    else:
      if opt_iter_robot or opt_iter_all:
        if n == 0:
          ax.plot(optima[optima_dex[0]][n*T:(n+1)*T]*p2w_x, \
                    optima[optima_dex[0]][(n+1)*T:(n+2)*T]*p2w_y, \
                    'xr', markersize=1)
        else:
          ax.plot(optima[optima_dex[0]][n*T:(n+1)*T]*p2w_x, \
                    optima[optima_dex[0]][(n+1)*T:(n+2)*T]*p2w_y, \
                    'xc', markersize=1)
      else:
        if n == 0:
          ax.plot(optima[optima_dex[0]].x[n*T:(n+1)*T]*p2w_x, \
                    optima[optima_dex[0]].x[(n+1)*T:(n+2)*T]*p2w_y, \
                    'xr', markersize=1)
        else:
          ax.plot(optima[optima_dex[0]].x[n*T:(n+1)*T]*p2w_x, \
                    optima[optima_dex[0]].x[(n+1)*T:(n+2)*T]*p2w_y, \
                    'xc', markersize=1)
    n = n + 2

# PLOT CURRENT POSE OF ALL AGENTS EXCEPT REMOVE_PED
  for ped in range(num_peds_follow):
    ax.plot(x_follow[ped][frame]*p2w_x, y_follow[ped][frame]*p2w_y, \
            "ob", markersize=3)

# PLOT CURRENT POSE OF REMOVE_PED
  if frame<np.size(x_nonzero):
    ax.plot(x_nonzero[frame]*p2w_x, y_nonzero[frame]*p2w_y, "og", markersize=4)

# PLOT CURRENT POSE OF ROBOT
  ax.plot(cmd_x*p2w_x, cmd_y*p2w_y, 'py', markersize=4)

# PLOT HISTORY OF ROBOT
  ax.plot(robot_history_x*p2w_x, robot_history_y*p2w_y, 'x--r', markersize=2)

  plt.xlabel('X Position',fontsize=10)
  plt.ylabel('Y position',fontsize=10)

  ess_time = math.trunc(ess_time*1e3)/1e3
  ess_ave_time = math.trunc(ess_ave_time*1e3)/1e3
  ess_std_time = math.trunc(ess_std_time*1e3)/1e3

  if frame==0:
    ave_time = time[frame]
    std_time = time[frame]
    max_time = math.trunc((1/(num_var_samples+1))*time[frame]*1e3)/1e3
  else:
    ave_time = np.mean(time[:frame])
    std_time = np.std(time[:frame])
    max_time = math.trunc((1/(num_var_samples+1))*np.max(time[:frame])*1e3)/1e3

  time_now = math.trunc((1/(num_var_samples+1))*time[frame]*1e3)/1e3
  ave_time = math.trunc((1/(num_var_samples+1))*ave_time*1e3)/1e3
  std_time = math.trunc((1/(num_var_samples+1))*std_time*1e3)/1e3

  plt.title('Frame {0}, t_now={2}, t_max = {5}, t_ave={3}+/-{4}, ESS={1}'.\
format(frame, ess, time_now, ave_time, std_time, max_time), fontsize=7)

  if data_set == 'eth_test':
    ax.set_xlim(-scaling*9,scaling*15)
    ax.set_ylim(scaling*2,scaling*9)
  if data_set == 'eth_train':
    if cmd_x*p2w_x > scaling*3. and cmd_x*p2w_x < scaling*10.:
      if cmd_y*p2w_y > scaling*7. and cmd_y*p2w_y < scaling*19.:
        ax.set_xlim(scaling*1,scaling*11)
        ax.set_ylim(scaling*5,scaling*15)
    if cmd_x*p2w_x >= scaling*0. and cmd_x*p2w_x < scaling*8.:
      if cmd_y*p2w_y >= scaling*0. and cmd_y*p2w_y < scaling*7:
        ax.set_xlim(-1.,scaling*9)
        ax.set_ylim(-1.,scaling*9)
    if cmd_x*p2w_x > scaling*8. and cmd_x*p2w_x < scaling*19.:
      if cmd_y*p2w_y > scaling*0. and cmd_y*p2w_y < scaling*10:
        ax.set_xlim(scaling*3,scaling*16)
        ax.set_ylim(scaling*0,scaling*13)

  plt.pause(0.001)

