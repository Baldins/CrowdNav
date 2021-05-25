# script to convert given data into igp required format

# Read data

# dump data 
# required format: x -> dict (ped_number:[positions])
import os
import scipy.io as sio

import pickle

# import stg_node
# import copy

x = {}
y = {}
x_vel = {}
y_vel = {}

file_dir = os.path.dirname(os.path.abspath(__file__))
# with open(str(file_dir) + '/test_data_dict_with_vel.pkl', 'rb') as f:
with open('eth_train.pkl', 'rb') as f:
    data_dict = pickle.load(f, encoding='latin1')

# print(data_dict['input_dict'].keys())
# print(data_dict)
# exit()
# print(data_dict['dt'])


# format need to be saved in
# output_dict = {}
# for key in data_dict['input_dict']:
#	if key != 'traj_lengths' and key != 'extras':
#		# print(key)
#		num = str(key).split('/')[1]
#		output_dict[num] = data_dict['input_dict'][key][...,(0,1,2,3)]

# print(output_dict)
# exit()
# data_id = 11 # Total - train_data_dict:189 & test_data_dict:48


########################### SAVE DATA ##################################
# with open('test_data_dict_with_vel.pkl', 'wb') as handle:
# 	pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
########################################################################

# data_dict = copy.deepcopy(output_dict)
# data_id = 0


def import_eth_data(num_peds):
    print(os.getcwd())
    # os.chdir("utils/ped_truth/")

    for ped in range(num_peds):
        x[ped] = data_dict['Pedestrian/' + str(ped)][..., 0]
        x_vel[ped] = data_dict['Pedestrian/' + str(ped)][..., 2]
        # converting into cm
        #        x[ped] = x[ped] * 100
        # x[ped] = x[ped][:101]
        # x_vel[ped] = x_vel[ped][:101]

    for ped in range(num_peds):
        y[ped] = data_dict['Pedestrian/' + str(ped)][..., 1]
        y_vel[ped] = data_dict['Pedestrian/' + str(ped)][..., 3]
        # converting into cm
        #       y[ped] = y[ped] * 100
        # y[ped] = y[ped][:101]
        # y_vel[ped] = y_vel[ped][:101]

    print(data_dict['Pedestrian/' + str(0)])
    print(x[0])
    print(x_vel[0])

    return x, y


import_eth_data(10)
