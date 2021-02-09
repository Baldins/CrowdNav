
import os
import pdb
import pickle
import argparse
import numpy as np
from scipy.interpolate import SmoothBivariateSpline, Rbf

# from visual_utils import GrandCentralAnimator

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--avi_file', type=str, default='grand_central_data/grandcentral.avi')
parser.add_argument('-d', '--datafile', type=str, default='grand_central_data/grand-central-trajectories.pkl')
parser.add_argument('-b', '--background', type=str, default='grand_central_data/background.jpg')
parser.add_argument('-s', '--start_time', type=float, default=0., help='start time in seconds')
parser.add_argument('-e', '--end_time', type=float, default=None, help='end time in seconds')
parser.add_argument('-r', '--removed_ped', type=int, default=None, help='agent trajectory to run over')
parser.add_argument('-i', '--interpolate', action='store_true', default=False)
parser.add_argument('--fill', action='store_true', default=False)
parser.add_argument('-f', '--frames', type=str, default='grand_central_data/frames')

fps = 25. # Tracking fps ?
video_fps = 23.
time_limit = 2000.4 # seconds

def load_data(filename):
	with open(filename, 'rb') as file:
		dataset = pickle.load(file)

	return dataset

def seconds_to_frame(start_time, end_time, fps, time_limit):
	''' Returns starting frame and end frame based on time'''
	assert end_time <= time_limit, 'Specified time duration exceeds time limit of {0} seconds'.format(str(time_limit))
	start_frame = np.floor(start_time * fps).astype(int) if start_time != 0. else 1
	end_frame = np.ceil(end_time * fps).astype(int)
	return start_frame, end_frame

def fill_frames_with_poses(ped_frames, ped_x, ped_y):
	'''Fills missing frames with last pose till new keypoint'''
	start_frame = ped_frames[0]
	end_frame = ped_frames[-1]
	frame_pos = 0
	ped_track = None
	while(frame_pos < ped_frames.shape[0]):
		if start_frame == ped_frames[frame_pos]:
			if ped_track is None:
				x = ped_x[frame_pos]
				y = ped_y[frame_pos]
				ped_track = np.asarray([x, y])[np.newaxis, ...]
			else:
				x = ped_x[frame_pos]
				y = ped_y[frame_pos]
				pose = np.asarray([x, y])[np.newaxis, ...]
				ped_track = np.concatenate((ped_track, pose), axis=0)
			frame_pos = frame_pos + 1
		else:
			frame_diff = ped_frames[frame_pos] - start_frame
			if frame_diff == 1:
				start_frame = ped_frames[frame_pos]
			else:
				pose_x = np.repeat(ped_x[frame_pos - 1], frame_diff - 1)[..., np.newaxis]
				pose_y = np.repeat(ped_y[frame_pos - 1], frame_diff - 1)[..., np.newaxis]
				pose = np.concatenate((pose_x, pose_y), axis=-1)
				ped_track = np.concatenate((ped_track, pose), axis=0)
				start_frame = ped_frames[frame_pos]

	return ped_track, np.arange(ped_frames[0], end_frame + 1)

def spline_approximation(ped_indices, ped_x, ped_y):
	'''
	Approximates missing frame poses through 2D spline approximation
	1st function assumption is y = f(x, t) where f is a radial basis function
	2nd function assumption is x = g(y, t) where g is a radial basis function
	Goal: Find best v, w s.t. v approx = g(f(x, t), t) and w approx = f(g(y, t), t)
	'''
	y_spline_fnc = Rbf(ped_x, ped_indices, ped_y) # y = f(x, t)
	x_spline_fnc = Rbf(ped_y, ped_indices, ped_x) # x = f(y, t)
	max_radius = 2 # search range for missing data best # are 2 or 3
	tol = 1
	# From start_index to end_index determine the value of x, y for the missing_index
	start_index = ped_indices[0]
	end_index = ped_indices[-1]
	ped_track = None
	frame_pos = 0
	for i in range(start_index, end_index + 1):
		if i in ped_indices:
			x = ped_x[frame_pos]
			y = ped_y[frame_pos]
			if ped_track is None:
				ped_track = np.asarray([x, y])[np.newaxis, ...]
			else:
				pose = np.asarray([x, y])[np.newaxis, ...]
				ped_track = np.concatenate((ped_track, pose), axis=0)
			frame_pos = frame_pos + 1
		else:
			# Determining missing observation x, y values given a range from max_radius
			x_prev = ped_track[-1, 0]
			y_prev = ped_track[-1, 1]
			x_space = np.arange(x_prev - max_radius, x_prev + max_radius)
			y_space = np.arange(y_prev - max_radius, y_prev + max_radius)
			i_repeat = np.repeat(i, x_space.shape[0])
			y_approx = y_spline_fnc(x_space, i_repeat)
			x_approx = x_spline_fnc(y_space, i_repeat)
			y_guess = y_spline_fnc(x_approx, i_repeat)
			x_guess = x_spline_fnc(y_approx, i_repeat)
			x_diff = np.power(x_space - x_guess, 2)
			y_diff = np.power(y_space - y_guess, 2)
			x_index = np.where(x_diff == np.min(x_diff))[0]
			y_index = np.where(y_diff == np.min(y_diff))[0]
			x = x_space[x_index[0]]
			y = y_space[x_index[0]]
			pose = np.asarray([x, y])[np.newaxis, ...]
			ped_track = np.concatenate((ped_track, pose), axis=0)

	return ped_track, np.arange(start_index, end_index + 1)

def import_grand_central_data(data_file, start_time, end_time, removed_ped=None, interpolate=False, fill=False):
	'''
	inputs:
		dataset_filename
		start_time (seconds)
		end_time (seconds)
		removed_peds (int)
	returns:
		all_peds_tracks dict()
		removed_peds_track (n_frames x 2) np.array
		n_frames
		n_peds
	'''
	if interpolate and fill:
		print('May only interpolate missing data or fill missing data with current observation')
		return
	file_path = os.path.join(os.getcwd(), data_file)
	dataset = load_data(file_path)
	tracklets = dataset['trks']
	start_frame, end_frame = seconds_to_frame(start_time, end_time, fps, time_limit)
	n_frames = end_frame - start_frame + 1
	all_ped_tracks = dict()
	removed_ped_track = None
	for n, track in enumerate(tracklets[0, :]):
		ped_start_indices = np.where(track[3] >= start_frame)[0]
		ped_start_frames = track[3][ped_start_indices, 0]
		ped_end_indices = np.where(ped_start_frames <= end_frame)[0]
		if ped_end_indices.size == 0:
			continue
		ped_frame_indices = ped_start_frames[ped_end_indices] - 1
		if n == tracklets.shape[-1] - 1:
			pdb.set_trace()
		ped_x = track[1][ped_end_indices, 0]
		ped_y = track[2][ped_end_indices, 0]
		ped_track = np.zeros((n_frames, 2))
		if interpolate and ped_x.shape[0] > 1:
			adjusted_ped_track, ped_frame_indices = spline_approximation(ped_frame_indices, ped_x, ped_y)
		elif fill:
			adjusted_ped_track, ped_frame_indices = fill_frames_with_poses(ped_frame_indices, ped_x, ped_y)
		else:
			adjusted_ped_track = np.concatenate((ped_x[..., np.newaxis], ped_y[..., np.newaxis]), axis=-1)
		# pdb.set_trace()
		ped_track[ped_frame_indices, 0] = adjusted_ped_track[:, 0]
		ped_track[ped_frame_indices, 1] = adjusted_ped_track[:, 1]
		if removed_ped is not None and n == removed_ped:
			removed_ped_track = ped_track
		all_ped_tracks[n] = ped_track

	n_peds = len(list(all_ped_tracks))
	return removed_ped_track, all_ped_tracks, n_peds, n_frames

def load_avi_frames(avi_file):
	import cv2
	cap = cv2.VideoCapture(avi_file, cv2.CAP_FFMPEG)
	success, img = cap.read()
	count = 0
	while success:
		cv2.imwrite("./grand_central_data/frames/grandcentral_frame_%d.jpg" % count, img)
		success, img = cap.read()
		print('Read a new frame: ', success)
		count = count + 1

if __name__ == '__main__':
	args = parser.parse_args()
	# pdb.set_trace()

	# Reading video file and saving frames
	# Testing avi file reading
	# Must use python 2
	# load_avi_frames(args.avi_file)
	# pdb.set_trace()

	#
	if args.end_time is None: args.end_time = time_limit
	track_info = import_grand_central_data(args.datafile, args.start_time, args.end_time, args.removed_ped, args.interpolate, args.fill)
	removed_ped_track, all_ped_tracks, n_peds, n_frames = track_info
	cwd = os.getcwd()
	# filename = os.path.join(cwd, args.datafile)
	imgname = os.path.join(cwd, args.background)
	frame_period = seconds_to_frame(args.start_time, args.end_time, fps, time_limit)

	# pdb.set_trace()
	from visual_utils import GrandCentralAnimator
	animator = GrandCentralAnimator(all_ped_tracks, frame_period, n_peds, imgname, frames=args.frames)
	animator.run()
	pdb.set_trace()
