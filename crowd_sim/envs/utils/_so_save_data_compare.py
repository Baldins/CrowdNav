import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import math

from _so_dir_maker_compare import dir_maker_compare

def save_data_compare(date_time, frame, opt_iter_robot, opt_iter_all, remove_ped, \
          remove_ped_path_length, chandler, p2w_x, p2w_y, p2w, \
          remove_ped_start, goal_dex, home_dir, save_dir, data_set, full_traj, \
          random_sample, var_sample, num_random_samples, num_var_samples, \
          ess_1, top_Z_indices_1, num_optima_1, \
          optimal_ll_1, optima_dex_1, norm_likelihood_1, optima_1,  \
          robot_path_length_1, safety_remove_ped_1, \
          safety_robot_1, robot_agent_path_diff_1, time_1, \
          local_density_1, robot_history_x_1, robot_history_y_1, \
          ess_2, top_Z_indices_2, num_optima_2, \
          optimal_ll_2, optima_dex_2, norm_likelihood_2, optima_2,  \
          robot_path_length_2, safety_remove_ped_2, \
          safety_robot_2, robot_agent_path_diff_2, time_2, \
          local_density_2, robot_history_x_2, robot_history_y_2, \
          agent_disrupt_fo, robot_agent_disrupt_fo, \
          agent_disrupt_so, robot_agent_disrupt_so, \
          label_1, label_2):
  def truncate(value):
    return math.trunc(value*1e4)/1e4

  if frame==0:
    ave_time = time_1[frame]
    std_time = time_1[frame]
    max_time = math.trunc(time_1[frame]*1e4)/1e4

    ave_agent_disrupt_fo = agent_disrupt_fo[frame]
    ave_agent_disrupt_so = agent_disrupt_so[frame]
    std_agent_disrupt_fo = agent_disrupt_fo[frame]
    std_agent_disrupt_so = agent_disrupt_so[frame]

    ave_robot_agent_disrupt_fo = robot_agent_disrupt_fo[frame]
    ave_robot_agent_disrupt_so = robot_agent_disrupt_so[frame]
    std_robot_agent_disrupt_fo = robot_agent_disrupt_fo[frame]
    std_robot_agent_disrupt_so = robot_agent_disrupt_so[frame]

    ave_density = local_density_1[frame]
    std_density = local_density_1[frame]
    max_density = math.trunc(local_density_1[frame]*1e4)/1e4
  else:
    ave_time = np.mean(time_1[:frame])
    std_time = np.std(time_1[:frame])
    max_time = math.trunc(np.max(time_1[:frame])*1e4)/1e4

    ave_agent_disrupt_fo = np.mean(agent_disrupt_fo[:frame])
    ave_agent_disrupt_so = np.mean(agent_disrupt_so[:frame])
    std_agent_disrupt_fo = np.std(agent_disrupt_fo[:frame])
    std_agent_disrupt_so = np.std(agent_disrupt_so[:frame])

    ave_robot_agent_disrupt_fo = np.mean(robot_agent_disrupt_fo[:frame])
    ave_robot_agent_disrupt_so = np.mean(robot_agent_disrupt_so[:frame])
    std_robot_agent_disrupt_fo = np.std(robot_agent_disrupt_fo[:frame])
    std_robot_agent_disrupt_so = np.std(robot_agent_disrupt_so[:frame])

    ave_density = np.mean(local_density_1[:frame])
    std_density = np.std(local_density_1[:frame])
    max_density = math.trunc(np.max(local_density_1[:frame])*1e4)/1e4

  time_now = math.trunc(time_1[frame]*1e4)/1e4
  ave_time = math.trunc(ave_time*1e4)/1e4
  std_time = math.trunc(std_time*1e4)/1e4

  density_now = math.trunc(local_density_1[frame]*1e4)/1e4
  ave_density = math.trunc(ave_density*1e4)/1e4
  std_density = math.trunc(std_density*1e4)/1e4

  robot_diff = [p2w_x*(robot_history_x_1-robot_history_x_2), \
                                    p2w_y*(robot_history_y_1-robot_history_y_2)]

  plots_or_metrics = 'metrics'
  dir_maker_compare(remove_ped, remove_ped_start, goal_dex, home_dir, \
            save_dir, data_set, full_traj, plots_or_metrics)

  filename = 'agent_' + str(remove_ped) + \
'_start_'+ str(remove_ped_start) + \
'_steps_'+ str(goal_dex) + '_compare_' + str(date_time) + '.txt'
  with open(filename, 'a') as text_file:
    print(f"REMOVE PED: {remove_ped}", file=text_file)
    print(f"FRAME NUMBER: {frame}", file=text_file)
    print(f"ESS {label_1}: {ess_1}", file=text_file)
    print(f"TOP Z INDICES {label_1}: {top_Z_indices_1[:ess_1]}", file=text_file)
    print(f"ESS {label_2}: {ess_2}", file=text_file)
    print(f"TOP Z INDICES {label_2}: {top_Z_indices_2[:ess_2]}", file=text_file)
    print(f"NUM OPTIMA {label_1}: {num_optima_1}", file=text_file)
    print(f"NUM OPTIMA {label_2}: {num_optima_2}", file=text_file)
    # for i in range(num_optima_1):
    #   print(f"LL {label_1} VALUES: {math.trunc(optimal_ll_1[i]*1e4)/1e4}", file=text_file)
    # for i in range(num_optima_2):
    #   print(f"LL {label_2} VALUES: {math.trunc(optimal_ll_2[i]*1e4)/1e4}", file=text_file)
    # for i in range(num_optima_1):
    #   if opt_iter_robot or opt_iter_all:
    #     zz = math.trunc(\
    #             np.linalg.norm((optima_1[0][0]-optima_1[i][0])*p2w)*1e4)/1e4
    #   else:
    #     zz = math.trunc(\
    #               np.linalg.norm((optima_1[0].x-optima_1[i].x)*p2w)*1e4)/1e4
    #   print(f"NORM DIFF BETWEEN {label_1} OPTIMA: {zz}", file=text_file)
    print(f"TIME NOW: {time_now}", file=text_file)
    print(f"TIME MEAN: {ave_time}+/-{std_time}", file=text_file)
    print(f"TIME MAX: {max_time}", file=text_file)
    print(f"SAFETY AGENT MIN: \
{math.trunc(np.min(safety_remove_ped_1)*1e4)/1e4}", file=text_file)
    print(f"SAFETY ROBOT {label_1} MIN: {math.trunc(np.min(safety_robot_1)*1e4)/1e4}", file=text_file)
    print(f"SAFETY ROBOT {label_2} MIN: {math.trunc(np.min(safety_robot_2)*1e4)/1e4}", file=text_file)
    print(f"SAFETY AGENT MEAN: \
{math.trunc(np.mean(safety_remove_ped_1)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_remove_ped_1)*1e4)/1e4}", file=text_file)
    print(f"SAFETY ROBOT {label_1} MEAN: \
{math.trunc(np.mean(safety_robot_1)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_robot_1)*1e4)/1e4}", file=text_file)
    print(f"SAFETY ROBOT {label_2} MEAN: \
{math.trunc(np.mean(safety_robot_2)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_robot_2)*1e4)/1e4}", file=text_file)
    if frame>0:
      print(f"{label_1}-{label_1} PATH DIFF NORM \
{truncate(np.linalg.norm(robot_diff))}", file=text_file)
      print(f"{label_1}-{label_1} PATH DIFF MEAN \
{truncate(np.mean(robot_diff))}+/-{truncate(np.std(robot_diff))}", file=text_file)
      print(f"ROBOT {label_1}-AGENT PATH DIFF MEAN \
{math.trunc(1e4*np.mean(robot_agent_path_diff_1[:frame]))/1e4}+/-\
{math.trunc(1e4*np.std(robot_agent_path_diff_1[:frame]))/1e4}", file=text_file)
      print(f"ROBOT {label_1}-AGENT PATH DIFF MAX \
{math.trunc(np.max(robot_agent_path_diff_1)*1e4)/1e4}", file=text_file)
      print(f"ROBOT {label_2}-AGENT PATH DIFF MEAN \
{math.trunc(1e4*np.mean(robot_agent_path_diff_2[:frame]))/1e4}+/-\
{math.trunc(1e4*np.std(robot_agent_path_diff_2[:frame]))/1e4}", file=text_file)
      print(f"ROBOT {label_2}-AGENT PATH DIFF MAX \
{math.trunc(np.max(robot_agent_path_diff_2)*1e4)/1e4}", file=text_file)
    print(f"AGENT PATH LENGTH: \
{math.trunc(remove_ped_path_length*1e4)/1e4}")
    print(f"ROBOT {label_1} PATH LENGTH: {math.trunc(robot_path_length_1*1e4)/1e4}", file=text_file)
    print(f"ROBOT {label_2} PATH LENGTH: {math.trunc(robot_path_length_2*1e4)/1e4}", file=text_file)
    print(f"AGENT DISRUPT NOW {label_1}: {truncate(agent_disrupt_fo[frame])}", file=text_file)
    print(f"AGENT DISRUPT NOW {label_2}: {truncate(agent_disrupt_so[frame])}", file=text_file)
    print(f"ROBOT-AGENT DISRUPT NOW {label_1}: {truncate(robot_agent_disrupt_fo[frame])}", file=text_file)
    print(f"ROBOT-AGENT DISRUPT NOW {label_2}: {truncate(robot_agent_disrupt_so[frame])}", file=text_file)
    print(f"AGENT DISRUPT MEAN {label_1}: \
{truncate(ave_agent_disrupt_fo)}+/-\
{truncate(std_agent_disrupt_fo)}", file=text_file)
    print(f"AGENT DISRUPT MEAN {label_2}: \
{truncate(ave_agent_disrupt_so)}+/-\
{truncate(std_agent_disrupt_so)}", file=text_file)
    print(f"ROBOT-AGENT DISRUPT MEAN {label_1}: \
{truncate(ave_robot_agent_disrupt_fo)}+/-\
{truncate(std_robot_agent_disrupt_fo)}", file=text_file)
    print(f"ROBOT-AGENT DISRUPT MEAN {label_2}: \
{truncate(ave_robot_agent_disrupt_so)}+/-\
{truncate(std_robot_agent_disrupt_so)}", file=text_file)
    print(f"DENSITY NOW: {density_now}", file=text_file)
    print(f"DENSITY MEAN: {ave_density}+/-{std_density}", file=text_file)
    print(f"DENSITY MAX: {max_density}", file=text_file)
    print('', file=text_file)











