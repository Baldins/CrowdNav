import numpy as np
import os
import math

def stopping_condition(frame, remove_ped, tol, ess_off, \
                       ess_boost, ess_num_peds, ess_newton, home_dir, \
                       chandler, conditioned, ll_converge, rand, var, \
                       num_random_samples, num_var_samples, var_ratio, \
                       data_set, p2w_x, p2w_y, p2w, support_boost, ess_limit, \
                       ess_limit_num, goal_dex, full_traj, remove_ped_start, \
                       dwa, robot_goal_x, robot_goal_y, cmd_x, cmd_y, \
                       saving_final, ess, top_Z_indices, num_optima, \
                       optima_dex, optimal_ll, optima, ess_time, \
                       ess_ave_time, ess_std_time, time, local_density, \
                       safety_remove_ped, safety_robot,\
                       robot_agent_path_diff, remove_ped_path_length, \
                       robot_path_length, sld, save_dir, mc, num_mc_samples, \
                       opt_iter_robot, opt_iter_all):
  closeness_robot = np.power(np.power(p2w_x*(cmd_x-robot_goal_x), 2) + \
                             np.power(p2w_y*(cmd_y-robot_goal_y), 2), 1/2)
  stop_thresh = 1.
  if closeness_robot < stop_thresh:
    print('')
    print('AT GOAL')
    robot_path_length = robot_path_length + closeness_robot
    stop = True
    if frame==0:
      ave_time = time[frame]
      std_time = time[frame]
      max_time = math.trunc((1/(2*(num_var_samples+1)))*time[frame]*1e4)/1e4

      ave_density = local_density[frame]
      std_density = local_density[frame]
      max_density = math.trunc(local_density[frame]*1e4)/1e4
    else:
      ave_time = np.mean(time[:frame])
      std_time = np.std(time[:frame])
      max_time = math.trunc((1/(2*(num_var_samples+1)))*np.max(time[:frame])*1e4)/1e4

      ave_density = np.mean(local_density[:frame])
      std_density = np.std(local_density[:frame])
      max_density = math.trunc(np.max(local_density[:frame])*1e4)/1e4

    time_now = math.trunc((1/(2*(num_var_samples+1)))*time[frame]*1e4)/1e4
    ave_time = math.trunc((1/(2*(num_var_samples+1)))*ave_time*1e4)/1e4
    std_time = math.trunc((1/(2*(num_var_samples+1)))*std_time*1e4)/1e4

    density_now = math.trunc(local_density[frame]*1e4)/1e4
    ave_density = math.trunc(ave_density*1e4)/1e4
    std_density = math.trunc(std_density*1e4)/1e4

    if full_traj:
      traj_dir = 'full_traj'
    else:
      traj_dir = 'partial_traj'

    curr_dir = '/data_processing/' + '/' + str(save_dir) + '/' + \
str(traj_dir) + '/'
    if mc:
      filename = 'raw_data_ends_' + str(save_dir) + '_' \
+ str(num_mc_samples) + '_' + str(traj_dir) + '.txt'
    else:
      filename = 'raw_data_ends_' + str(save_dir) + '_' + str(traj_dir) + '.txt'

    if saving_final:
      os.chdir(str(home_dir) + str(curr_dir))
      with open(filename, 'a') as text_file:
        print(f"REMOVE PED: {remove_ped}", file=text_file)
        if mc:
          print(f"NUM MC SAMPLES: {num_mc_samples}", file=text_file)
        print(f"FRAME NUMBER: {frame}", file=text_file)
        print(f"ESS: {ess}", file=text_file)
        print(f"TOP Z INDICES: {top_Z_indices[:ess]}", file=text_file)
        if rand:
          print(f"NUM RANDOM SAMPLES: {num_random_samples+1}", file=text_file)
        if var:
          print(f"NUM VAR SAMPLES: {2*num_var_samples}", file=text_file)
        print(f"NUM OPTIMA: {num_optima}", file=text_file)
        print(f"OPTIMA INDICES RANKED: {optima_dex}", file=text_file)
        for i in range(num_optima):
          print(f"LL VALUES: {optimal_ll[i]}", \
                file=text_file)
        if not mc:
          for i in range(num_optima):
            if opt_iter_robot or opt_iter_all:
              zz = math.trunc(\
                  np.linalg.norm((optima[0][0]-optima[i][0])*p2w)*1e4)/1e4
            else:
              zz = math.trunc(\
                    np.linalg.norm((optima[0].x-optima[i].x)*p2w)*1e4)/1e4
            print(f"NORM DIFFERENCE: {zz}", file=text_file)
        print(f"OPTIMIZATION TIME NOW: {ess_time}", file=text_file)
        print(f"OPTIMIZATION TIME MEAN: {ess_ave_time}+/-{ess_std_time}", \
                                                                 file=text_file)
        print(f"TIME NOW: {time_now}", file=text_file)
        print(f"TIME MEAN: {ave_time}+/-{std_time}", file=text_file)
        print(f"TIME MAX: {max_time}", file=text_file)
        print(f"SAFETY AGENT MIN: \
{math.trunc(np.min(safety_remove_ped)*1e4)/1e4}", file=text_file)
        print(f"SAFETY ROBOT MIN: {math.trunc(np.min(safety_robot)*1e4)/1e4}", \
                                                                 file=text_file)
        print(f"SAFETY AGENT MEAN: \
{math.trunc(np.mean(safety_remove_ped)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_remove_ped)*1e4)/1e4}", file=text_file)
        print(f"SAFETY ROBOT MEAN: \
{math.trunc(np.mean(safety_robot)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_robot)*1e4)/1e4}", file=text_file)
        if frame>0:
          print(f"ROBOT-AGENT PATH DIFF MEAN \
{math.trunc(1e4*np.mean(robot_agent_path_diff[:frame]))/1e4}+/-\
{math.trunc(1e4*np.std(robot_agent_path_diff[:frame]))/1e4}", file=text_file)
          print(f"ROBOT-AGENT PATH DIFF MAX \
{math.trunc(np.max(robot_agent_path_diff)*1e4)/1e4}", file=text_file)
          print(f"ROBOT-AGENT PATH DIFF NOW: \
{math.trunc(robot_agent_path_diff[frame]*1e4)/1e4}", file=text_file)

        print(f"AGENT PATH LENGTH: \
{math.trunc(remove_ped_path_length*1e4)/1e4}", file=text_file)
        print(f"ROBOT PATH LENGTH: {math.trunc(robot_path_length*1e4)/1e4}", \
                                                                 file=text_file)
        print(f"DENSITY NOW: {density_now}", file=text_file)
        print(f"DENSITY MEAN: {ave_density}+/-{std_density}", file=text_file)
        print(f"DENSITY MAX: {max_density}", file=text_file)
        print(f"STRAIGHT LINE DISTANCE: {math.trunc(np.max(sld)*1e4)/1e4}", \
                                                                 file=text_file)
        print(f" ", file=text_file)

  else:
    stop = False
  return stop




