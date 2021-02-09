import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import math

from _so_dir_maker import dir_maker

def save_data(date_time, diag_or_full, ess, top_Z_indices, frame, \
						  num_optima, optimal_ll, optima_dex, norm_likelihood, optima, \
						  ess_time, ess_ave_time, ess_std_time, \
						  robot_path_length, safety_remove_ped, \
						  safety_robot, robot_agent_path_diff, remove_ped, \
						  tol, ess_off, ess_num_peds, ess_boost,\
						  ess_newton, time, home_dir, remove_ped_path_length, \
						  chandler, conditioned, ll_converge, \
						  robot_history_x, robot_history_y, \
						  rand, var, \
						  num_random_samples, num_var_samples, var_ratio, data_set, \
						  p2w_x, p2w_y, p2w, support_boost, ess_limit, ess_limit_num, \
						  goal_dex, full_traj, remove_ped_start, local_density, dwa, \
						  save_dir, mc, num_mc_samples, opt_iter_robot, opt_iter_all):
	ess_time = math.trunc(ess_time*1e3)/1e3
	ess_ave_time = math.trunc(ess_ave_time*1e3)/1e3
	ess_std_time = math.trunc(ess_std_time*1e3)/1e3

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

	plots_or_metrics = 'metrics'
	dir_maker(remove_ped, tol, ess_off, ess_boost, ess_num_peds, \
			 			ess_newton, home_dir, chandler, conditioned, ll_converge, \
			 			rand, var, num_random_samples, num_var_samples, var_ratio, \
			 			data_set, p2w_x, p2w_y, p2w, support_boost, ess_limit, \
			 			ess_limit_num, goal_dex, full_traj, remove_ped_start, \
			 			plots_or_metrics, dwa, save_dir, mc, num_mc_samples, \
			 			opt_iter_robot, opt_iter_all)

	filename = 'agent_' + str(remove_ped) + \
'_start_'+ str(remove_ped_start) + \
'_steps_'+ str(goal_dex) + \
'_cond_'+ str(conditioned) + \
'_opt_iter_'+ str(optima_iterate) + \
'_llconv_'+ str(ll_converge) + \
'_tol_' + str(tol) + \
'_ess_boost_' + str(ess_boost) + '_' + str(ess_num_peds) + \
'_ess_lim_' + str(ess_limit) + '_' + str(ess_limit_num) + \
'_essHess_' + str(ess_newton) + \
'_VAR_' + str(var) + '_' + str(num_var_samples) + '_ratio_' + str(var_ratio) + \
'_RAND_' + str(rand) + '_' +  str(num_random_samples) + \
'_supp_' + str(support_boost) + '_' + str(date_time)+'.txt'
	with open(filename, 'a') as text_file:
		if mc:
			print(f"NUM MC SAMPLES: {num_mc_samples}", file=text_file)
		print(f"DIAG IS: {diag_or_full}", file=text_file)
		print(f"REMOVE PED: {remove_ped}", file=text_file)
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
			print(f"LL VALUES: {optimal_ll[i]}", file=text_file)
		# for i in range(num_optima):
			# print(f"NORM LIKELIHOOD: {math.trunc(norm_likelihood[i]*1e4)/1e4}", \
																																 # file=text_file)
		# if not mc:
		# 	for i in range(num_optima):
		# 		if optima_iterate:
		# 			zz = math.trunc(\
		# 						  np.linalg.norm((optima[0][0]-optima[i][0])*p2w)*1e4)/1e4
		# 		else:
		# 			zz = math.trunc(\
		# 								np.linalg.norm((optima[0].x-optima[i].x)*p2w)*1e4)/1e4
		# 		print(f"NORM DIFFERENCE: {zz}", file=text_file)

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
		print(f" ", file=text_file)










