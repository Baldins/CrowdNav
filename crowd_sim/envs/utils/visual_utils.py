
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
# from avifilelib import AviFile

class CrowdAnimator:

	def __init__(self):
		pass

	def init(self):
		pass

	def update(self, i):
		pass

	def run(self):
		pass

class GrandCentralAnimator(CrowdAnimator):

	def __init__(self, tracklets, frame_period, n_peds, background=None, frames=None):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot()
		self.tracklets = tracklets
		# pdb.set_trace()
		self.n_peds = n_peds
		self.min_frame = frame_period[0]
		self.max_frame = frame_period[1]
		if background is not None:
			if frames is None:
				img = plt.imread(background)
				self.back = self.ax.imshow(img)
			else:
				print('Provided static background and frames. Using frames.')
				self.frames_folder = frames
		elif frames is not None:
			self.frames_folder = frames
		# pdb.set_trace()
		self.text_template = 'Frame %d'
		self.timeframe_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
		self.peds = dict()
		self.peds_hist = dict()
		# self.peds = [self.ax.plot([], [], 'b.', lw=.5, markersize=1)[0] for _ in range(self.n_peds)]
		# self.peds_hist = [self.ax.plot([], [], 'r.', lw=.5, markersize=1)[0] for _ in range(self.n_peds)]

	def init(self):
		self.timeframe_text.set_text(' ')
		if hasattr(self, 'frames_folder'):
			self.frame_filename = 'grandcentral_frame'
			img = plt.imread(os.path.join(self.frames_folder, self.frame_filename + '0.jpg'))
			self.back = self.ax.imshow(img)

	def update(self, t):
		frame = t + self.min_frame
		if hasattr(self, 'frames_folder'):
			img = plt.imread(os.path.join(self.frames_folder, self.frame_filename + str(t) + '.jpg'))
			self.ax.imshow(img)
		for i in self.tracklets.keys():
			# pdb.set_trace()
			x = self.tracklets[i][t, 0]
			y = self.tracklets[i][t, 1]
			if x == 0 and y == 0:
				continue
			if i in self.peds.keys():
				x_hist, y_hist = self.peds_hist[i].get_data()
				x_prev, y_prev = self.peds[i].get_data()
				if x_hist.size != 0:
					try:
						x_hist = np.concatenate((np.asarray(x)[..., np.newaxis], x_hist))
						y_hist = np.concatenate((np.asarray(y)[..., np.newaxis], y_hist))
					except ValueError:
						pdb.set_trace()
				else:
					x_hist = np.asarray(x_prev)[..., np.newaxis]
					y_hist = np.asarray(y_prev)[..., np.newaxis]
				self.peds_hist[i].set_data((x_hist, y_hist))
				self.peds[i].set_data((x, y))
			else:
				self.peds_hist[i] = self.ax.plot([], [], 'r+', lw=.5, markersize=1)[0]
				self.peds[i] = self.ax.plot([], [], 'b.', lw=1, markersize=3)[0]
				self.peds[i].set_data((x, y))

		self.timeframe_text.set_text(self.text_template % (frame))

	def run(self):
		print('Starting Animation')
		print()
		self.ani = FuncAnimation(self.fig, self.update, frames=range(self.max_frame), interval=1, init_func=self.init, repeat=False)
		plt.show()


class TrainAnimator(CrowdAnimator):

	def __init__(self, time_sorted_data, map_file, max_ped):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot()
		self.dataset = time_sorted_data
		self.time = time_sorted_data[0][0]
		self.index = 0
		img = plt.imread(map_file)
		self.ax.imshow(img)
		self.n_peds = max_ped
		self.peds = [self.ax.plot([],[],'b.', lw=.5, markersize=1)[0] for _ in range(max_ped)]
		self.peds_hist = [self.ax.plot([],[],'r', lw=.5, markersize=1)[0] for _ in range(max_ped)]
		self.curr_annotations = []

	def init(self):
		for i in range(self.n_peds):
			self.peds[i].set_data([],[])
			self.peds_hist[i].set_data([],[])

	def update(self, i):
		if i > 0: self.time = self.update_time(self.time)
		peds = self.get_pedestrians(self.time)

	def run(self):
		print('Starting Animation')
		print()
		n_frames = self.time_to_frames(self.dataset[0][0], self.dataset[-1][0])
		self.ani = FuncAnimation(self.fig, self.update, frames=n_frames, interval=100, init_func=self.init, repeat=False)
		plt.show()

	def get_pedestrians(self, time):
		h, m, s, ms = time
		for	j in range(self.index, len(self.dataset)):
			ped = self.dataset[j]
			p_h, p_m, p_s, p_ms = ped[0]
			if m == p_m:
				if s == p_s:
					if int(ms/100) == int(p_ms/100):
						self.peds[ped[-1] - 1].set_data([ped[2]/67], [ped[3]/67])
						self.index = j
						print(f'Dataset Index: {j}')
					elif int(ms/100) > int(p_ms/100):
						break
					else:
						print(f'Current Time - Ped Time: {ms} - {p_ms}')

	@staticmethod
	def update_time(time):
		'''
		Updates the time by 100 ms
		'''
		h, m, s, ms = time
		if ms + 100 >= 1000:
			ms = ms - 900
			s = s + 1
			if s >= 60:
				s = 0
				m = m + 1
		else:
			ms = ms + 100

		return h, m, s, ms

	@staticmethod
	def time_to_frames(start_time, end_time):
		'''
		Converts time to number of frames where the frame rate is 100 ms
		'''
		start_hour, start_min, start_sec, start_msec = start_time
		end_hour, end_min, end_sec, end_msec = end_time
		if end_msec < start_msec:
			end_sec = end_sec - 1
			msec_diff = 1000 + end_msec - start_msec
		else:
			msec_diff = end_msec - start_msec
		if end_sec < start_sec:
			end_min = end_min - 1
			sec_diff = 60 + end_sec - start_sec
		else:
			sec_diff = end_sec - start_sec
		min_diff = end_min - start_min
		n_frames = int(np.ceil((min_diff * 60000 + sec_diff * 1000 + msec_diff) / 100))
		return range(n_frames)

class ETHAnimator(CrowdAnimator):

	def __init__(self, frames, pred_states, obstacles=None, **kwargs):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-25,25), ylim=(-25,25))
		self.ax.grid()
		self.frames = frames
		self.pred_states = pred_states
		self.view_obs = kwargs.get('view_obs') if 'view_obs' in kwargs else None
		self.obstacles = obstacles
		self.n_peds = kwargs.get('max_ped')
		self.peds = [self.ax.plot([],[],'bo', lw=.5)[0] for i in range(self.n_peds)]
		self.peds_hist = [self.ax.plot([], [],'k.', lw=.5, markersize=2)[0] for i in range(self.n_peds)]
		self.peds_pred = [self.ax.plot([], [], 'r.', lw=.5, markersize=2)[0] for i in range(self.n_peds)]
		self.timeframe_template = 'frame = %d, time = %.1fs'
		self.timeframe_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
		self.dt = .4
		self.curr_annotations = []

		# Generate the plots for the pedestrian (past, present, future)

	def init(self):
		for i in range(self.n_peds):
			self.peds[i].set_data([],[])
			self.peds_pred[i].set_data([], [])
			self.peds_hist[i].set_data([],[])

		if hasattr(self, 'view_obs') and self.view_obs: self.ax.scatter(self.obstacles[0,:], self.obstacles[1,:], s=1, facecolor='m', edgecolors='face')
		self.timeframe_text.set_text(' ')
		return (tuple(self.peds), tuple(self.peds_pred), tuple(self.peds_hist), self.timeframe_text)

	def run(self, frames=None):
		print('Starting Animation')
		print()
		if frames is not None:
			if len(frames) == 2:
				n_frames = range(frames[0], frames[1])
			else:
				n_frames = range(frames[0], len(self.frames) - 1)
		else:
			n_frames = range(len(self.frames) - 1)
		self.ani = FuncAnimation(self.fig, self.update, frames=n_frames, interval=100, blit=False, init_func=self.init, repeat=False)
		plt.show()

	def update(self, i):
		for ann in self.curr_annotations:
			ann.remove()

		self.curr_annotations.clear()
		frame = self.frames[i]
		pred_states = self.pred_states[i]
		indices = []
		for ped in frame:
			j = int(ped[1]) - 1
			indices.append(j)
			pred_state = pred_states[str(j)]
			# pdb.set_trace()
			ped_histx, ped_histy = self.peds_hist[j].get_data()
			ped_x, ped_y = self.peds[j].get_data()
			ped_histx = ped_histx + ped_x
			ped_histy = ped_histy + ped_y
			self.peds_hist[j].set_data((ped_histx, ped_histy))
			self.peds[j].set_data([ped[2]], [ped[4]])
			self.peds_pred[j].set_data((pred_state[:, 0], pred_state[:, 1]))
			self.curr_annotations.append(self.ax.annotate(j, (ped[2], ped[4])))

		[self.peds[j].set_data([], []) for j in range(self.n_peds) if j not in indices]
		[self.peds_hist[j].set_data([], []) for j in range(self.n_peds) if j not in indices]
		[self.peds_pred[j].set_data([], []) for j in range(self.n_peds) if j not in indices]
		self.timeframe_text.set_text(self.timeframe_template % (i, i * self.dt))
		return self.peds, self.peds_pred, self.peds_hist, self.timeframe_text

	def visualize_data(self, data):
		ins, outs = data
		_, n = np.shape(ins)
		for i in range(n):			
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			ax.scatter(ins[:,0], ins[:,1], outs[:,i])
			ax.set_xlabel('Position in X (m)')
			ax.set_ylabel('Position in Y (m)')
			ax.set_zlabel('Velocity (m/s)')
			ax.set_title('Velocity over state-space')

		plt.show()

def plot2d(x, y, std):
	pass

def plot3d(x, y, z, std):
	pass
