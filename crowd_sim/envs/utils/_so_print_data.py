import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import math

def print_data(diag_or_full, ess, top_Z_indices, frame, \
							 num_optima, optimal_ll, optima_dex, norm_likelihood, \
							 optima, ess_time, ess_ave_time, \
							 ess_std_time, remove_ped, \
							 robot_path_length, safety_remove_ped, safety_robot, \
							 robot_agent_path_diff, time, remove_ped_path_length, \
							 chandler, robot_history_x, robot_history_y, p2w_x, p2w_y, p2w, \
							 rand, var, num_random_samples, num_var_samples, local_density, \
							 mc, num_mc_samples, opt_iter_robot, opt_iter_all):
	ess_time = math.trunc(ess_time*1e4)/1e4
	ess_ave_time = math.trunc(ess_ave_time*1e4)/1e4
	ess_std_time = math.trunc(ess_std_time*1e4)/1e4

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
	if var:
		time_now = math.trunc((1/(2*(num_var_samples+1)))*time[frame]*1e4)/1e4
		ave_time = math.trunc((1/(2*(num_var_samples+1)))*ave_time*1e4)/1e4
		std_time = math.trunc((1/(2*(num_var_samples+1)))*std_time*1e4)/1e4
	else:
		time_now = math.trunc(time[frame]*1e4)/1e4
		ave_time = math.trunc(ave_time*1e4)/1e4
		std_time = math.trunc(std_time*1e4)/1e4

	density_now = math.trunc(local_density[frame]*1e4)/1e4
	ave_density = math.trunc(ave_density*1e4)/1e4
	std_density = math.trunc(std_density*1e4)/1e4

	if mc:
		print(f"NUM MC SAMPLES: {num_mc_samples}")
	print(f"DIAG IS : {diag_or_full}")
	print(f"REMOVE PED: {remove_ped}")
	print(f"FRAME NUMBER: {frame}")
	print(f"ESS: {ess}")
	print(f"TOP Z INDICES: {top_Z_indices[:ess]}")
	if rand:
		print(f"NUM RANDOM SAMPLES: {num_random_samples+1}")
	if var:
		print(f"NUM VAR SAMPLES: {2*num_var_samples}")
	print(f"NUM OPTIMA: {num_optima}")
	print(f"OPTIMA INDICES RANKED: {optima_dex}")
	for i in range(num_optima):
		print(f"LL VALUES: {optimal_ll[i]}")
	# for i in range(num_optima):
	# 	print(f"NORM LIKELIHOOD: {math.trunc(norm_likelihood[i]*1e4)/1e4}")
	if not mc:
		for i in range(num_optima):
			if opt_iter_robot or opt_iter_all:
				zz = math.trunc(\
								  np.linalg.norm((optima[0][0]-optima[i][0])*p2w)*1e4)/1e4
			else:
				zz = math.trunc(\
										np.linalg.norm((optima[0].x-optima[i].x)*p2w)*1e4)/1e4
			print(f"NORM DIFFERENCE: {zz}")

	print(f"OPTIMIZATION TIME NOW: {ess_time}")
	print(f"OPTIMIZATION TIME MEAN: {ess_ave_time}+/-{ess_std_time}")
	print(f"TIME NOW: {time_now}")
	print(f"TIME MEAN: {ave_time}+/-{std_time}")
	print(f"TIME MAX: {max_time}")
	print(f"SAFETY AGENT MIN: \
{math.trunc(np.min(safety_remove_ped)*1e4)/1e4}")
	print(f"SAFETY ROBOT MIN: {math.trunc(np.min(safety_robot)*1e4)/1e4}")
	print(f"SAFETY AGENT MEAN: \
{math.trunc(np.mean(safety_remove_ped)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_remove_ped)*1e4)/1e4}")
	print(f"SAFETY ROBOT MEAN: \
{math.trunc(np.mean(safety_robot)*1e4)/1e4}+/-\
{math.trunc(np.std(safety_robot)*1e4)/1e4}")
	if frame>0:
		print(f"ROBOT-AGENT PATH DIFF MEAN \
{math.trunc(1e4*np.mean(robot_agent_path_diff[:frame]))/1e4}+/-\
{math.trunc(1e4*np.std(robot_agent_path_diff[:frame]))/1e4}")
		print(f"ROBOT-AGENT PATH DIFF MAX \
{math.trunc(np.max(robot_agent_path_diff)*1e4)/1e4}")
		print(f"ROBOT-AGENT PATH DIFF NOW: \
{math.trunc(robot_agent_path_diff[frame]*1e4)/1e4}")
	print(f"AGENT PATH LENGTH: \
{math.trunc(remove_ped_path_length*1e4)/1e4}")
	print(f"ROBOT PATH LENGTH: {math.trunc(robot_path_length*1e4)/1e4}")
	print(f"DENSITY NOW: {density_now}")
	print(f"DENSITY MEAN: {ave_density}+/-{std_density}")
	print(f"DENSITY MAX: {max_density}")
	print('')














