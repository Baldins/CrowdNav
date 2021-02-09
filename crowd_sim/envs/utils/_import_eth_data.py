import os
import scipy.io as sio

import pickle

# try igp university test

def import_eth_data(num_frames, num_peds, remove_ped, file_dir):
  x = {}
  y = {}

  x_follow = {}
  y_follow = {}

  with open(str(file_dir) + '/eth_data/test_data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f, encoding='latin1')

  data_id = 11 # Total - train_data_dict:189 & test_data_dict:48

  for ped in range(num_peds):
    x[ped] = data_dict[str(ped)][data_id][:,0]
    # converting into cm
    x[ped] = x[ped]*100
    x[ped] = x[ped][:num_frames]

    y[ped] = data_dict[str(ped)][data_id][:,1]
    # converting into cm
    y[ped] = y[ped]*100
    y[ped] = y[ped][:num_frames]

  n = 0
  for ped in range(num_peds):
    if(ped < remove_ped):
      x_follow[n] = data_dict[str(n)][data_id][:,0]
      # converting into cm
      x_follow[n] = x_follow[n]*100
      x_follow[n] = x_follow[n][:num_frames]

      y_follow[n] = data_dict[str(n)][data_id][:,1]
      # converting into cm
      y_follow[n] = y_follow[n]*100
      y_follow[n] = y_follow[n][:num_frames]
      n = n + 1
    elif(ped > remove_ped):
      x_follow[n] = data_dict[str(ped)][data_id][:,0]
      # converting into cm
      x_follow[n] = x_follow[n]*100
      x_follow[n] = x_follow[n][:num_frames]

      y_follow[n] = data_dict[str(ped)][data_id][:,1]
      # converting into cm
      y_follow[n] = y_follow[n]*100
      y_follow[n] = y_follow[n][:num_frames]
      n = n + 1

  return x, y, x_follow, y_follow