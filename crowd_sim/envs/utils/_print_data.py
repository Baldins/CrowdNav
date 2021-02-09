import numpy as np
import math

def print_data(diag_or_full, ess, top_Z_indices, frame, num_intents, \
							 num_optima, optimal_ll, optima_dex, norm_likelihood, \
							 optima_ess, optima_iterate, ess_time, ess_ave_time, \
							 ess_std_time, remove_ped, remove_ped_path_length, \
							 robot_path_length, safety_remove_ped, safety_robot, \
							 robot_agent_path_diff, time_diag):

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

	print(f"DIAG OR FULL: {diag_or_full}")
	print(f"REMOVE PED: {remove_ped}")
	print(f"FRAME NUMBER: {frame}")
	print(f"ESS: {ess}")
	print(f"TOP Z INDICES: {top_Z_indices[:ess]}")
	print(f"NUM INTENTS: {num_intents+1}")
	print(f"NUM OPTIMA: {num_optima}")
	print(f"LL VALUES: {optimal_ll}")
	print(f"OPTIMA INDICES RANKED: {optima_dex}")
	print(f"NORM LIKELIHOOD: {norm_likelihood}")

	for i in range(num_optima):
		if optima_iterate:
			zz = np.linalg.norm((optima_ess[0][0]-optima_ess[i][0])*p2w)
		else:
			zz = np.linalg.norm((optima_ess[0].x-optima_ess[i].x)*p2w)
		print(f"NORM DIFFERENCE: {zz}")

	print(f"OPTIMIZATION TIME NOW: {ess_time}")
	print(f"OPTIMIZATION TIME STATS: {ess_ave_time}+/-{ess_std_time}")
	print(f"TIME NOW: {time_now}")
	print(f"TIME STATS: {ave_time_diag}+/-{std_time_diag}")
	print(f"TIME MAX: {max_time_diag}")
	print(f"SAFETY AGENT {remove_ped}: {safety_remove_ped}")
	print(f"SAFETY ROBOT: {safety_robot}")
	print(f"PATH LENGTH AGENT {remove_ped}: {remove_ped_path_length}")
	print(f"PATH LENGTH ROBOT: {robot_path_length}")
	if frame>0:
		print(f"ROBOT AGENT MEAN \
{math.trunc(1e3*np.mean(robot_agent_path_diff[:frame]))/1e3} +/- \
{math.trunc(1e3*np.std(robot_agent_path_diff[:frame]))/1e3}")
		print(f"ROBOT AGENT MAX {np.max(robot_agent_path_diff)}")
		print(f"ROBOT AGENT PATH DIFF: {robot_agent_path_diff[frame]}")
	print('')