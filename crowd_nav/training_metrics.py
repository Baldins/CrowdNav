
import pdb
import numpy as np
import matplotlib.pyplot as plt

with open('./data/rect_3k_10k_30_15_rand/output.log') as file:
	data = file.read().split('\n')

data = [entry for entry in data if 'TRAIN' in entry]
# pdb.set_trace()
data = data[::-1][0:10000][::-1]
succ_rate = [float(entry.split('success rate: ')[-1].split(',')[0]) for entry in data]
coll_rate = [float(entry.split('collision rate: ')[-1].split(',')[0]) for entry in data]
nav_time = [float(entry.split('nav time: ')[-1].split(',')[0]) for entry in data]
reward = [float(entry.split('reward: ')[-1]) for entry in data]

avg_succ = []
avg_coll = []
avg_nav = []
avg_reward = []
step = 50
for i in range(0, 10000, step):
	avg_succ.append(np.mean(succ_rate[i:i + step]))
	avg_coll.append(np.mean(coll_rate[i:i + step]))
	avg_nav.append(np.mean(nav_time[i:i + step]))
	avg_reward.append(np.mean(reward[i:i + step]))

fig = plt.figure()
plt.plot(avg_succ)
plt.xlabel(f'Episode x{step}')
plt.ylabel('Avg. Success Rate')
plt.title('Reactangle Random Init')

fig = plt.figure()
plt.plot(avg_coll)
plt.xlabel(f'Episode x{step}')
plt.ylabel('Avg. Collision Rate')
plt.title('Reactangle Random Init')

fig = plt.figure()
plt.plot(avg_nav)
plt.xlabel(f'Episode x{step}')
plt.ylabel('Avg. Navigation Time')
plt.title('Reactangle Random Init')

fig = plt.figure()
plt.plot(avg_reward)
plt.xlabel(f'Episode x{step}')
plt.ylabel('Avg Reward')
plt.title('Reactangle Random Init')

plt.show()