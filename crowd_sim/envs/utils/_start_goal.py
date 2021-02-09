import numpy as np

p2w_x = 12./510.
p2w_y = 5./322.
p2w = np.sqrt(p2w_x**2 + p2w_y**2)

def start_goal(x, y, remove_ped, Tdex_10, dist_agent_10):
	#with max_vel = .08, err_ped=2,err_robot=7, thinks 5 is cooperative,
	# recovers
	#bad prediction on 0, narrow recovery
	# robot_start_x = 1./p2w_x
	# robot_goal_x_full = 4/p2w_x
	# robot_goal_x_ess = 4/p2w_x
	# robot_start_y = 4./p2w_y
	# robot_goal_y_full = 4./p2w_y
	# robot_goal_y_ess = 4./p2w_y

# ETH POSITIONS

	# robot_start_x = 10./p2w_x
	# robot_goal_x_full = 20./p2w_x
	# robot_goal_x_diag = 20./p2w_x
	# robot_start_y = 8/p2w_y
	# robot_goal_y_full = 8./p2w_y
	# robot_goal_y_diag = 8./p2w_y
	# AGENT 10 IN ETH

	robot_start_x = x[0]
	robot_goal_x_diag = x[-1]
	robot_start_y = y[0]
	robot_goal_y_diag = y[-1]

	dist_agent = np.power(np.power(robot_start_x-robot_goal_x_diag, 2) + \
												np.power(robot_start_y-robot_goal_y_diag, 2), 1/2)
	Tdex_max = np.int((Tdex_10/dist_agent_10)*dist_agent)

		# robot_start_x = 20./p2w_x
		# robot_goal_x_full = 20./p2w_x
		# robot_goal_x_diag = 20./p2w_x
		# robot_start_y = 4./p2w_y
		# robot_goal_y_full = 6./p2w_y
		# robot_goal_y_diag = 6./p2w_y

	# robot_start_x = 1./p2w_x
	# robot_goal_x_full = 3/p2w_x
	# robot_goal_x_ess = 3/p2w_x
	# robot_start_y = 4./p2w_y
	# robot_goal_y_full = 5./p2w_y
	# robot_goal_y_ess = 5./p2w_y

	# robot_start_x = 1./p2w_x
	# robot_goal_x_full = 2.5/p2w_x
	# robot_goal_x_ess = 2.5/p2w_x
	# robot_start_y = 2./p2w_y
	# robot_goal_y_full = 3.5/p2w_y
	# robot_goal_y_ess = 3.5	/p2w_y

	#easy
	# robot_start_x = 3./p2w_x
	# robot_goal_x_full = 7./p2w_x
	# robot_goal_x_ess = 7./p2w_x
	# robot_start_y = 4./p2w_y
	# robot_goal_y_full = 4./p2w_y
	# robot_goal_y_ess = 4./p2w_y

	# LONG
	################TEST
	# robot_start_x = 4./p2w_x
	# robot_goal_x_full = 4./p2w_x
	# robot_goal_x_ess = 4./p2w_x
	# robot_start_y = 2.5/p2w_y
	# robot_goal_y_full = 4./p2w_y
	# robot_goal_y_ess = 4./p2w_y

	# robot_start_x = 5./p2w_x
	# robot_goal_x_full = 5./p2w_x
	# robot_goal_x_ess = 5./p2w_x
	# robot_start_y = 2.5/p2w_y
	# robot_goal_y_full = 4./p2w_y
	# robot_goal_y_ess = 4./p2w_y

	# robot_start_x = 6./p2w_x
	# robot_goal_x_full = 6./p2w_x
	# robot_goal_x_ess = 6./p2w_x
	# robot_start_y = 2.5/p2w_y
	# robot_goal_y_full = 4./p2w_y
	# robot_goal_y_ess = 4./p2w_y
	################TEST
	# LONG
	# robot_start_x = 3./p2w_x
	# robot_goal_x_full = 5./p2w_x
	# robot_goal_x_ess = 5./p2w_x
	# robot_start_y = 2.5/p2w_y
	# robot_goal_y_full = 7./p2w_y
	# robot_goal_y_ess = 7./p2w_y

	#nice interaction; has to stop and turn back when 8 doesnt cooperate
	#robot_start_x = 4./p2w_x
	#robot_goal_x_full = 6./p2w_x
	#robot_goal_x_ess = 6./p2w_x
	#robot_start_y = 3./p2w_y
	#robot_goal_y_full = 4.5/p2w_y
	#robot_goal_y_ess = 4.5/p2w_y

	#excellent with kimbo with max_vel = .05, err_ped=2,err_robot=7
	#robot_start_x = 3.17/p2w_x
	#robot_goal_x_full = 6./p2w_x
	#robot_goal_x_ess = 6./p2w_x
	#robot_start_y = 4./p2w_y
	#robot_goal_y_full = 4.42/p2w_y
	#robot_goal_y_ess = 4.42/p2w_y

	#excellent with kimbo with max_vel = .08, err_ped=2,err_robot=7
	#robot_start_x = 4.5/p2w_x
	#robot_goal_x = 4.5/p2w_x
	#robot_start_y = 2./p2w_y
	#robot_goal_y = 5./p2w_y

	#robot_start_x = 1./p2w_x
	#robot_goal_x = 5.53/p2w_x
	#robot_start_y = 4.3/p2w_y
	#robot_goal_y = 4.42/p2w_y

	#######EVIDENCE
	#kimbo stuck between 1 and 8; collides with 8! collision with 8 for diag
	# norm.
	#diag no normalize collides with 1
	#robot_start_x = 3./p2w_x
	#robot_goal_x = 5.5/p2w_x
	#robot_start_y = 6./p2w_y
	#robot_goal_y = 4/p2w_y

	#robot_start_x = 3./p2w_x
	#robot_goal_x = 6./p2w_x
	#robot_start_y = 6./p2w_y
	#robot_goal_y = 3.8/p2w_y

	#KIMBO: weird collision with 0. gets past 9 and 11
	#DIAG, NO NORM: extremely large predictions from mean for agents.
	#--> very evasive when fine grained GPs.  hits 2.  hits 1.  disaster.
	#robot_start_x = 50.
	#robot_goal_x = 320
	#robot_start_y = 290
	#robot_goal_y = 350

	#this length (about 4 meters) keeps Tdex at about 25.  robot makes it
	# through
	#scene in meaningful way.
	#at agent 11, robot is making space in advance when agent is stationary.
	# at frame 31, agent "jumps" towards robot.  robot goes toward agent 11
	#beacause that deconflicts the whole trajectory?  Going down would conflict
	#the trajectories more?  also, the agents are perpendicular to each other.
	#robot_start_x = 50.
	#robot_goal_x = 240
	#robot_start_y = 290
	#robot_goal_y = 350

	#KIMBO: fun route.  sometimes see newton fail.
	#
	#robot_start_x = 170.
	#robot_goal_x = 180
	#robot_start_y = 190
	#robot_goal_y = 450
	#######EVIDENCE


	#h_add = 50 works beautiful
	#h_mult = 30 works great
	#kimbo collision with 1
	#robot_start_x = 170.
	#robot_goal_x = 220
	#robot_start_y = 190
	#robot_goal_y = 450


	#if start is too close to 2 (start_y>160), robot gets confused by early
	#prediction, causes a  collision.  This is sensible.  at start_y<150,
	#robot is able to recover.  very
	#robot_start_x = 170.
	#robot_goal_x = 150
	#robot_start_y = 140
	#robot_goal_y = 450

	#gorgeous
	#robot_start_x = 3./p2w_x
	#robot_goal_x = 7.5/p2w_x
	#robot_start_y = 6./p2w_y
	#robot_goal_y = 5.5/p2w_y

	#gorgeous
	#robot_start_x = 3./p2w_x
	#robot_goal_x = 7.5/p2w_x
	#robot_start_y = 6./p2w_y
	#robot_goal_y = 4/p2w_y

	#collision with 1
	#robot_start_x = 3./p2w_x
	#robot_goal_x = 5.5/p2w_x
	#robot_start_y = 6./p2w_y
	#robot_goal_y = 3./p2w_y

	#h_mult = 50 goes backwards and collides with 0
	#h_add=50 COLLIDES WITH 11; 11 IS OSCILLATING; ROBOT THINKS ITS GOING DOWN
	# AND
	#AGENT 11 DOESNT.  AGENT 11 IS STANDING STILL
	#robot_start_x = 30.
	#robot_goal_x = 320
	#robot_start_y = 280
	#robot_goal_y = 250

	#agent 11 collision.  prediction went bad.  figure this out
	#robot_start_x = 30.
	#robot_goal_x = 320
	#robot_start_y = 280
	#robot_goal_y = 350

	#WORKS BEAUTIFUL WITH H_ADD=50
	#robot_start_x = 30.
	#robot_goal_x = 320
	#robot_start_y = 280
	#robot_goal_y = 250

	#robot_start_x = 1./p2w_x
	#robot_goal_x = 7.5/p2w_x
	#robot_start_y = 3.8/p2w_y
	#robot_goal_y = 5.5/p2w_y

	#robot_start_x = 4./p2w_x
	#robot_goal_x = 4./p2w_x
	#robot_start_y = 2.4/p2w_y
	#robot_goal_y = 7.5/p2w_y

	#robot_start_x = 2./p2w_x
	#robot_goal_x = 7.5/p2w_x
	#robot_start_y = 2./p2w_y
	#robot_goal_y = 6.5/p2w_y

	#robot_start_x = 4./p2w_x
	#robot_goal_x = 4./p2w_x
	#robot_start_y = 2.5/p2w_y
	#robot_goal_y = 6.5/p2w_y
	return robot_start_x, robot_start_y, \
				 robot_goal_x_diag, robot_goal_y_diag, \
	   		 Tdex_max











