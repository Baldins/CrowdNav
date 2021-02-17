import scipy as sp
from scipy import optimize
import numpy as np
import math
import time
from autograd import value_and_grad
from scipy.optimize import minimize

import crowd_sim.envs.objectives.so_igp_diag_objectives as so_diagonal
import crowd_sim.envs.objectives.so_igp_dense_objectives as so_dense
import crowd_sim.envs.objectives.fo_igp_dense_objectives as fo_dense

from crowd_sim.envs.functions.so_igp_optimize_iterate import optimize_iterate

def optimize_newton(fo, diagonal, random_sample, var_sample, tol, \
							num_intents, num_var_samples, T, joint_sample_x, joint_sample_y, \
     					var_samples_x, var_samples_y, frame, num_peds, time_array, \
							ess, top_Z_indices, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
              cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
              cov_ped_x, cov_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
              one_over_cov_sum_x, one_over_cov_sum_y, \
              one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, \
              ll_converge, vg, hess_opt, opt_iter_robot, opt_iter_all, \
              x_ped, y_ped, agent_disrupt, robot_agent_disrupt, opt_method):
	if random_sample:
		f = [0. for _ in range(num_intents+1)]
		ll = [0. for _ in range(num_intents+1)]
	if var_sample:
		f = [0. for _ in range(2*num_var_samples+1)]
		ll = [0. for _ in range(2*num_var_samples+1)]

	ped_mu_x_ess = [0. for _ in range(ess)]
	ped_mu_y_ess = [0. for _ in range(ess)]

	inv_cov_ped_x_ess = [0. for _ in range(ess)]
	inv_cov_ped_y_ess = [0. for _ in range(ess)]

	one_over_cov_sum_x_ess = [0. for _ in range(ess)]
	one_over_cov_sum_y_ess = [0. for _ in range(ess)]

	one_over_std_sum_x_ess = [0. for _ in range(ess)]
	one_over_std_sum_y_ess = [0. for _ in range(ess)]

	for ped in range(ess):
		top = top_Z_indices[ped]
		ped_mu_x_ess[ped] = ped_mu_x[top]
		ped_mu_y_ess[ped] = ped_mu_y[top]

		inv_cov_ped_x_ess[ped] = inv_cov_ped_x[top]
		inv_cov_ped_y_ess[ped] = inv_cov_ped_y[top]

		one_over_cov_sum_x_ess[ped] = one_over_cov_sum_x[top]
		one_over_cov_sum_y_ess[ped] = one_over_cov_sum_y[top]

		# one_over_std_sum_x_ess[ped] = one_over_std_sum_x[top]
		# one_over_std_sum_y_ess[ped] = one_over_std_sum_y[top]
	t0 = time.time()

	if random_sample:
		for intent in range(num_intents+1):
			if intent == 0:
				x0 = robot_mu_x
				x0 = np.concatenate((x0, robot_mu_y))
				for ped in range(ess):
				    top = top_Z_indices[ped]
				    x0 = np.concatenate((x0, ped_mu_x[top]))
				    x0 = np.concatenate((x0, ped_mu_y[top]))
			else:
				x0 = joint_sample_x[num_peds, intent-1,:]
				x0 = np.concatenate((x0, joint_sample_y[num_peds, intent-1,:]))
				for ped in range(ess):
				    top = top_Z_indices[ped]
				    x0 = np.concatenate((x0, joint_sample_x[top, intent-1,:]))
				    x0 = np.concatenate((x0, joint_sample_y[top, intent-1,:]))
			if opt_iter_robot or opt_iter_all:
				f[intent] = optimize_iterate(fo, tol, diagonal, frame, x0, num_peds, ess,\
                   robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
                   cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
                   cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
             		   one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
                   normalize, ll_converge, T, opt_iter_robot, opt_iter_all)
				if diagonal:
					ll[intent] = so_diagonal.ll(f[intent], num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
		               normalize, T)
				else:
					if fo:
						ll[intent] = fo_dense.ll(f[intent], num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,
		               normalize)
					else:
						ll[intent] = so_dense.ll(f[intent], num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
		               normalize, T)
			else:
				if diagonal:
					f[intent] = sp.optimize.minimize(so_diagonal.ll, x0, \
									 args=(num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		         	   	 cov_robot_x, cov_robot_y, \
		               inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
		               one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T), \
		               method=opt_method, jac=so_diagonal.d_ll, \
		               hess=so_diagonal.dd_ll, \
								   options={'xtol': tol})
								   #trust-ncg--VERY SLOW
								   #trust-krylov---VERY SLOW
								   #Newton-CG---.56 SECONDS, GOOD RESULT
								   #trust-exact---.52 seconds.
					ll[intent] = so_diagonal.ll(f[intent].x, num_peds, ess, \
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
		               normalize, T)
				else:
					if fo:
						f[intent] = sp.optimize.minimize(fo_dense.ll, x0, args=(num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		         	   	 cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,
		               normalize), \
		               method=opt_method, jac=so_dense.d_ll, hess=fo_dense.dd_ll, \
							   	 options={'xtol': tol})
						ll[intent] = fo_dense.ll(f[intent].x, num_peds, ess, \
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,
		               normalize)
					else:
						f[intent] = sp.optimize.minimize(so_dense.ll, x0, args=(num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		         	   	 cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
		               normalize, T), \
		               method=opt_method, jac=so_dense.d_ll, hess=so_dense.dd_ll, \
							   	 options={'xtol': tol})
						ll[intent] = so_dense.ll(f[intent].x, num_peds, ess, \
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
		               normalize, T)
			ll[intent] = math.trunc(ll[intent]*1e3)/1e3

			print('intent =', intent, end =" ", flush=True)
	if var_sample:
		# high_value_var_sampler(var_samples_x, var_samples_y)
		for var in range(2*num_var_samples+1):
			if var == 0:
				x0 = robot_mu_x
				x0 = np.concatenate((x0, robot_mu_y))
				for ped in range(ess):
				    top = top_Z_indices[ped]
				    x0 = np.concatenate((x0, ped_mu_x[top]))
				    x0 = np.concatenate((x0, ped_mu_y[top]))
			else:
				x0 = var_samples_x[num_peds, var-1,:]
				x0 = np.concatenate((x0, var_samples_y[num_peds, var-1,:]))
				for ped in range(ess):
				    top = top_Z_indices[ped]
				    x0 = np.concatenate((x0, var_samples_x[top, var-1,:]))
				    x0 = np.concatenate((x0, var_samples_y[top, var-1,:]))
			if opt_iter_robot or opt_iter_all:
				print('OPT ITER FO', fo)
				f[var] = optimize_iterate(fo, tol, diagonal, frame, x0, num_peds, ess,\
                   robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
                   cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
                   cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
             		   one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, \
                   normalize, ll_converge, T, opt_iter_robot, opt_iter_all)
				if diagonal:
					ll[var] = so_diagonal.ll(f[var], num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T)
				else:
					if fo:
						ll[var] = fo_dense.ll(f[var], num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 normalize)
					else:
						ll[var] = so_dense.ll(f[var], num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T)
			else:
				if diagonal:
					f[var] = sp.optimize.minimize(so_diagonal.ll, x0, args=(num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		         	   	 cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T), \
		               method=opt_method, jac=so_diagonal.d_ll, \
		               hess=so_diagonal.dd_ll, \
				   				 options={'xtol': tol})
					ll[var] = so_diagonal.ll(f[var].x, num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T)
				elif vg:
					print('VALUE AND GRAD')
					print('')
					f[var] = sp.optimize.minimize(value_and_grad(so_dense.ll), x0, \
												args=(num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		         	   	 cov_robot_x, cov_robot_y, \
		               inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
		               one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T), \
									jac=True, method='BFGS', options={'xtol': 1e-8, 'disp': True})
					# f[var] = sp.optimize.minimize(so_dense.ll, x0, \
					# 							args=(num_peds, ess,\
		   #             robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		   #       	   	 cov_robot_x, cov_robot_y, \
		   #             inv_cov_robot_x, inv_cov_robot_y, \
		   #             cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		   #             one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
		   #             one_over_cov_sumij_x, one_over_cov_sumij_y, normalize), \
					# 				jac=so_dense.d_ll, method='BFGS', options={'xtol': 1e-8, 'disp': True})
					ll[var] = so_dense.ll(f[var].x, num_peds, ess,\
		               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
		               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
		               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
		               one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T)
				elif hess_opt:
					if fo:
						print('HAND DERIVED SCIPY HESS OPT FO')
						print('')
						f[var] = sp.optimize.minimize(fo_dense.ll, x0, args=(num_peds, ess,\
	               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
	         	     cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
	               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
	               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     						 normalize), \
	               method=opt_method, jac=fo_dense.d_ll, hess=fo_dense.dd_ll, \
							   options={'xtol': tol})
						ll[var] = fo_dense.ll(f[var].x, num_peds, ess,\
	               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
	               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
	               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
	               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     						 normalize)
					else:
						print('HAND DERIVED SCIPY HESS OPT SO')
						print('')
						f[var] = sp.optimize.minimize(so_dense.ll, x0, args=(num_peds, ess,\
	               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
	         	     cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
	               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
	               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T), \
	               method=opt_method, jac=so_dense.d_ll, hess=so_dense.dd_ll, \
							   options={'xtol': tol})
						ll[var] = so_dense.ll(f[var].x, num_peds, ess,\
	               robot_mu_x, robot_mu_y, ped_mu_x_ess, ped_mu_y_ess, \
	               cov_robot_x, cov_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
	               cov_ped_x, cov_ped_y, inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
	               one_over_cov_sum_x_ess, one_over_cov_sum_y_ess,  \
     							 one_over_cov_sumij_x, one_over_cov_sumij_y, normalize, T)
			if not math.isinf(ll[var]) and not math.isnan(ll[var]):
				ll[var] = math.trunc(ll[var]*1e3)/1e3

			print('variance sample = ', var, end =" ", flush=True)
		#######################HAND ROLLED GRAD+HESS
		# f = optimize_iterate(frame, x0, num_peds, ess,\
		# 	                 robot_mu_x, robot_mu_y, \
		# 	                 ped_mu_x_ess, ped_mu_y_ess, \
		# 	                 inv_cov_robot_x, inv_cov_robot_y, \
		# 	                 inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		# 	                 one_over_cov_sum_x_ess, \
							 # one_over_cov_sum_y_ess, \
		# 	                 one_over_std_sum_x_ess, \
							 # one_over_std_sum_y_ess)
		#######################SCIPY LL+GRAD+HESS

		# f[intent] = sp.optimize.minimize(so_diagonal.ll, x0, \
		#            		 args=(num_peds, ess,\
		#                  robot_mu_x, robot_mu_y, \
		#                  ped_mu_x_ess, ped_mu_y_ess, \
	 #                 	 cov_robot_x, cov_robot_y, \
		#                  inv_cov_robot_x, inv_cov_robot_y, \
		#                  cov_ped_x, cov_ped_y, \
		#                  inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		#                  one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
		#                  one_over_std_sum_x_ess, one_over_std_sum_y_ess), \
		#                  method=opt_method, jac=so_diagonal.d_ll, \
											# hess=so_diagonal.dd_ll)
		# 				 # options={'xtol': tol})
		# ll[intent] = so_diagonal.ll(f[intent].x, num_peds, ess,\
		# 	                 robot_mu_x, robot_mu_y, \
		# 	                 ped_mu_x_ess, ped_mu_y_ess, \
		# 	                 cov_robot_x, cov_robot_y, \
		# 	                 inv_cov_robot_x, inv_cov_robot_y, \
		# 	                 cov_ped_x, cov_ped_y, \
		# 	                 inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		# 	                 one_over_cov_sum_x_ess, one_over_cov_sum_y_ess, \
		# 	                 one_over_std_sum_x_ess, one_over_std_sum_y_ess)
		# ll[intent] = math.trunc(ll[intent]*1e3)/1e3

		# print(intent, end =" ", flush=True)
	    # newton iterate, tol=1e-8, num_peds: .138+/-.077s
		######################### TIMING ON DIFFERENT APPROACHES
		# num_peds, Tdex_max=25,no collisions:
		# trust-krlov w/ gtol=1e-8 .708+/-.223s
		#      trust-krylov: no tol: .523+/-.192s
		# Newton-CG: xtol=1e-8: 0.701+/-.229s
		#   Newton-CG, no tol: .568+/-.243s
		# newton-cg/ess=True, xtol=1e-8, .227+/-.099
		# 	NCG/ess=True, no xtol .2+/-.098
		# Newton-CG, gtol=1e-8: 0.943+/-0.419s
		#     Newton-CG, no tol: .791+/-.288s
		# 	  Newton-CG,ess=True, .116+/-.059
		# trust-ncg, gtol=1e-8: 1.622+/-0.962s
		#  trust-ncg, no tol: .539+/-.123s
		#######################SCIPY LL+GRAD
		# f = sp.optimize.minimize(so_diagonal.ll, x0, \
		# 				args=(num_peds, ess,\
		#                  robot_mu_x, robot_mu_y, \
		#                  ped_mu_x_ess, ped_mu_y_ess, \
		#                  inv_cov_robot_x, inv_cov_robot_y, \
		#                  inv_cov_ped_x_ess, inv_cov_ped_y_ess, \
		#                  one_over_cov_sum_x_ess, \
						   # one_over_cov_sum_y_ess, \
		#                  one_over_std_sum_x_ess, \
						   # one_over_std_sum_y_ess), \
		#                     method='BFGS', jac=so_diagonal.d_ll, \
		#                     options={'disp': True})
		#######################SCIPY LL
		# f = sp.optimize.minimize(\
		#                 value_and_grad(ll_diag_slice_grad), x0, \
		#                 jac=True, method='BFGS',\
		                  # options={'xtol': 1e-8, 'disp': True})

	def coupling(f, x0, one_over_cov_sum_x, one_over_cov_sum_y):
		n = 2
		uncoupling = 0.
		for ped in range(ess):
			vel_x = f[:T] - x0[n*T:(n+1)*T]
			vel_y = f[T:2*T] - x0[(n+1)*T:(n+2)*T]
			n = n + 2

			vel_x_2 = np.power(vel_x, 2)
			vel_y_2 = np.power(vel_y, 2)

			quad_x = np.multiply(vel_x_2, np.diag(one_over_cov_sum_x[ped]))
			quad_y = np.multiply(vel_y_2, np.diag(one_over_cov_sum_y[ped]))

			Z_x = np.exp(-0.5*quad_x)
			Z_y = np.exp(-0.5*quad_y)

			Z = np.multiply(Z_x, Z_y)

			log_znot = np.sum(np.log1p(-Z))
			uncoupling = uncoupling + log_znot
		return -1*uncoupling #WE WANT TO MAKE UNCOUPLING LARGE; SO -UNCOUPLING SHOULD
		# BE SMALL.  LARGE VALUE OF -UNCOUPLING MEANS LOTS OF COUPLING

	global_optima_dex = np.argmin(ll)
	if opt_iter_robot or opt_iter_all:
		agent_disrupt[frame] = np.linalg.norm(f[global_optima_dex][2*T]-x0[2*T:])
		robot_agent_disrupt[frame] = coupling(f[global_optima_dex], x0, \
																				 one_over_cov_sum_x, one_over_cov_sum_y)
	else:
		agent_disrupt[frame] = np.linalg.norm(f[global_optima_dex].x[2*T]-x0[2*T:])
		robot_agent_disrupt[frame] = coupling(f[global_optima_dex].x, x0, \
																				 one_over_cov_sum_x, one_over_cov_sum_y)

	opt_time = time.time()-t0
	time_array[frame] = opt_time
	ave_time = math.trunc(1e3*np.mean(time_array[:frame+1]))/1e3
	std_time = math.trunc(1e3*np.std(time_array[:frame+1]))/1e3

	return f, ll, opt_time, time_array, ave_time, std_time, \
				 agent_disrupt, robot_agent_disrupt













