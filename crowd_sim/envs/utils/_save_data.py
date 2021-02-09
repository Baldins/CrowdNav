import numpy as np
import math
import os

from ._data_dirs import data_dirs

def save_data(date_time, diag_or_full, ess, top_Z_indices, frame, num_intents, \
							 num_optima, optimal_ll, optima_dex, norm_likelihood, \
							 optima_ess, optima_iterate, ess_time, ess_ave_time, \
							 ess_std_time, remove_ped_path_length, \
							 robot_path_length, safety_remove_ped, safety_robot, \
							 robot_agent_path_diff, remove_ped, optima_iterate_diag, \
							 tol, ess_off_diag, ess_num_peds, ess_boost,\
							 ess_newton_diag, time_diag, Tdex_max_diag, goal_noise, \
							 max_vel_robot, home_dir):
	p2w_x = 12./510.
	p2w_y = 5./322.
	p2w = np.sqrt(p2w_x**2 + p2w_y**2)

	ess_time = math.trunc(ess_time*1e3)/1e3
	ess_ave_time = math.trunc(ess_ave_time*1e3)/1e3
	ess_std_time = math.trunc(ess_std_time*1e3)/1e3

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

	data_dirs(remove_ped, optima_iterate_diag, \
			 tol, ess_off_diag, ess_boost, ess_num_peds, \
			 ess_newton_diag, Tdex_max_diag, goal_noise, max_vel_robot, home_dir)
	filename = 'agent_' + str(remove_ped) + \
						 '_optima_iter_'+ str(optima_iterate_diag) + \
						 '_tol_' + str(tol) + \
						 '_ess_off_diag_' + str(ess_off_diag) + \
						 '_ess_boost_' + str(ess_boost) + \
						 '_ess+_' + str(ess_num_peds) + \
						 '_essNewton_' + str(ess_newton_diag) + \
						 '_Tdex_max_' + str(Tdex_max_diag) + \
						 '_speed_' + str(math.trunc(max_vel_robot*p2w*1e3)/1e3) + \
						 '_goal_noise_' + str(goal_noise) + '_' + str(date_time)+'.txt'
	with open(filename, 'a') as text_file:
		print(f"DIAG OR FULL: {diag_or_full}", file=text_file)
		print(f"FRAME NUMBER: {frame}", file=text_file)
		print(f"GOAL NOISE: {goal_noise}", file=text_file)
		print(f"ESS: {ess}", file=text_file)
		print(f"TOP Z INDICES: {top_Z_indices[:ess]}", file=text_file)
		print(f"NUM INTENTS: {num_intents+1}", file=text_file)
		print(f"NUM OPTIMA: {num_optima}", file=text_file)
		print(f"LL VALUES: {optimal_ll}", file=text_file)
		print(f"OPTIMA INDICES RANKED: {optima_dex}", file=text_file)
		print(f"NORM LIKELIHOOD: {norm_likelihood}", file=text_file)

		for i in range(num_optima):
			if optima_iterate:
				zz = np.linalg.norm((optima_ess[0][0]-optima_ess[i][0])*p2w)
			else:
				zz = np.linalg.norm((optima_ess[0].x-optima_ess[i].x)*p2w)
			print(f"NORM DIFFERENCE: {zz}", file=text_file)

		print(f"TIME NOW: {time_now}", file=text_file)
		print(f"TIME STATS: {ave_time_diag}+/-{std_time_diag}", file=text_file)
		print(f"TIME MAX: {max_time_diag}", file=text_file)
		print(f"SAFETY AGENT {remove_ped}: {safety_remove_ped}", file=text_file)
		print(f"SAFETY ROBOT: {safety_robot}", file=text_file)
		print(f"PATH LENGTH AGENT {remove_ped}: {remove_ped_path_length}", \
					file=text_file)
		print(f"PATH LENGTH ROBOT: {robot_path_length}", file=text_file)
		if frame>0:
			print(f"ROBOT AGENT MEAN \
{math.trunc(1e3*np.mean(robot_agent_path_diff[:frame]))/1e3} +/- \
{math.trunc(1e3*np.std(robot_agent_path_diff[:frame]))/1e3}", \
						file=text_file)
			print(f"ROBOT AGENT MAX {np.max(robot_agent_path_diff)}", file=text_file)
			print(f"ROBOT AGENT PATH DIFF: {robot_agent_path_diff[frame]}", \
						file=text_file)
		print(f'', file=text_file)