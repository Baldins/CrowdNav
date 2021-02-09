import numpy as np
import math

p2w_x = 12./510.
p2w_y = 5./322.
p2w = np.sqrt(p2w_x**2 + p2w_y**2)

def metrics(frame, num_peds_follow, remove_ped, cmd_x, cmd_y, x, y, \
            x_follow, y_follow, x_nonzero, y_nonzero, \
            robot_history_x, robot_history_y):
  dist_remove_ped = [0. for _ in range(num_peds_follow)]
  dist_robot = [0. for _ in range(num_peds_follow)]

  for ped in range(num_peds_follow):
    dist_robot[ped] = np.power(\
                           np.power(p2w_x*(cmd_x-x_follow[ped][frame]), 2) + \
                           np.power(p2w_y*(cmd_y-y_follow[ped][frame]), 2), 1/2)
    if frame < np.size(x_nonzero):
      dist_remove_ped[ped] = np.power(\
            np.power(p2w_x*(x_nonzero[frame]-x_follow[ped][frame]), 2) + \
            np.power(p2w_y*(y_nonzero[frame]-y_follow[ped][frame]), 2), 1/2)
    else:
      dist_remove_ped[ped] = 2000.

  safety_robot = np.min(dist_robot)
  safety_remove_ped = np.min(dist_remove_ped)

  if frame == 0:
    remove_ped_path_length = 0.
    robot_path_length = 0.
    robot_agent_path_diff = 0.
  else:
    remove_ped_path_length = 0.
    robot_path_length = 0.
    if frame<np.size(x_nonzero):
      for t in range(frame):
        remove_ped_path_length = remove_ped_path_length + \
                np.power(np.power(p2w_x*(x_nonzero[t]-x_nonzero[t+1]), 2) + \
                         np.power(p2w_y*(y_nonzero[t]-y_nonzero[t+1]), 2), 1/2)
    else:
      for t in range(np.size(x_nonzero)-1):
        remove_ped_path_length = remove_ped_path_length + \
                np.power(np.power(p2w_x*(x_nonzero[t]-x_nonzero[t+1]), 2) + \
                         np.power(p2w_y*(y_nonzero[t]-y_nonzero[t+1]), 2), 1/2)
    for t in range(frame):
      robot_path_length = robot_path_length + np.power(\
              np.power(p2w_x*(robot_history_x[t]-robot_history_x[t+1]), 2) + \
              np.power(p2w_y*(robot_history_y[t]-robot_history_y[t+1]), 2), 1/2)
    if frame<np.size(x_nonzero):
      robot_agent_path_diff = np.power(\
                               np.power(p2w_x*(cmd_x-x_nonzero[frame]), 2) + \
                               np.power(p2w_y*(cmd_y-y_nonzero[frame]), 2), 1/2)
    else:
      robot_agent_path_diff = np.power(\
                               np.power(p2w_x*(cmd_x-x_nonzero[-1]), 2) + \
                               np.power(p2w_y*(cmd_y-y_nonzero[-1]), 2), 1/2)
  return dist_robot, dist_remove_ped, safety_robot, safety_remove_ped, \
         remove_ped_path_length, robot_path_length, robot_agent_path_diff
