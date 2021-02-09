import matplotlib.pyplot as plt
import os
import math
import numpy as np

from _so_dir_maker_compare import dir_maker_compare

def save_figs_compare(frame, remove_ped, remove_ped_start, goal_dex, home_dir, \
                      save_dir, data_set, full_traj):

  plots_or_metrics = 'plots'
  dir_maker_compare(remove_ped, remove_ped_start, goal_dex, home_dir, \
                    save_dir, data_set, full_traj, plots_or_metrics)

  plt.savefig('agent_' + str(remove_ped) + \
              '_start_'+ str(remove_ped_start) + \
              '_steps_'+ str(goal_dex) + \
              '_frame_' + str(frame) + '.png')




