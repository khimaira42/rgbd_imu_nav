# save the position_draw
import numpy as np
import os
import pandas as pd
path = '/Users/david/Documents/thesis/Thesis_code/data_fusion_plot/23-32_fusion1'
position_draw = np.load(os.path.join(path,'position_draw.npy'))

gt_draw = np.load(os.path.join(path,'gt_draw.npy'))

position_measurement_list = np.load(os.path.join(path,'position_measurement_list.npy'))

position_imu_list = np.load(os.path.join(path,'position_imu_list.npy'))


# save the position_draw
t_start = 1701886838.241981
t_end = 1701886849.541981
t_duration = t_end - t_start
time_gt = np.linspace(t_start, t_end, num=len(gt_draw))
time_fusion = np.linspace(t_start, t_end, num=len(position_draw))
# Interpolating the ground truth data to match the time points of the fusion data
from scipy.interpolate import interp1d
print('time_gt', time_gt)
interpolator = interp1d(time_fusion, position_draw, axis=0, kind='linear', fill_value='extrapolate')
position_draw_interpolated = interpolator(time_gt)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

combined_mark = pd.read_csv('combined_mark.csv')
imu_mark = pd.read_csv('imu_mark.csv')
visual_mark = pd.read_csv('visual_mark.csv')
combined_mark = combined_mark[combined_mark['timestamp'] > t_start]
combined_mark = combined_mark[combined_mark['timestamp'] < t_end]
imu_mark = imu_mark[imu_mark['timestamp'] > t_start]
imu_mark = imu_mark[imu_mark['timestamp'] < t_end]
visual_mark = visual_mark[visual_mark['timestamp'] > t_start]
visual_mark = visual_mark[visual_mark['timestamp'] < t_end]

timestamp_imu = imu_mark['timestamp'].values
timestamp_visual = visual_mark['timestamp'].values
print(timestamp_imu)
print(timestamp_visual)
time_fusion = np.linspace(t_start, t_end, num=len(position_draw))


# plt.figure(figsize=(5, 5))
# # plt.subplot(1, 2, 1)

# plt.plot(position_draw[:,0], position_draw[:,1], label='position fusion')
# plt.plot(gt_draw[:,0], gt_draw[:,1], label='ground truth')
# # add grid
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Trajectory in x-y plane')
# plt.subplot(1, 2, 2)
# plt.plot(time_gt, gt_draw[:,2], label='ground truth')
# plt.plot(time_fusion, position_draw[:,2], label='position fusion')
# plt.grid()
# plt.xlabel('timestamp')
# plt.ylabel('z')
# plt.title('Comparison in z axis')
# # show the trajectory and ground truth

#plt.figure(figsize=(10, 4))
# plt.subplot(3, 1, 1)
# plt.plot(gt_draw[:,0], gt_draw[:,0], label='ground truth')
# plt.plot(position_draw[:,0], position_draw[:,1], label='position x fusion')
# # add grid
# plt.grid()
# plt.xlabel('timestamp')
# plt.ylabel('x')
# plt.title('Comparison in x axis')
# plt.subplot(3, 1, 2)
# plt.plot(time_gt, gt_draw[:,1], label='ground truth')
# plt.plot(time_fusion, position_draw[:,2], label='position y fusion')
# plt.grid()
# plt.xlabel('timestamp')
# plt.ylabel('y')
# plt.title('Comparison in y axis')
# plt.subplot(3, 1, 3)
# plt.plot(time_fusion, position_draw[:,2], label='position z fusion')
# plt.plot(time_gt, gt_draw[:,2], label='ground truth')
# plt.grid()
# plt.xlabel('timestamp')
# plt.ylabel('z')
# plt.title('Comparison in z axis')


# plt.figure(figsize=(10, 6))
# plt.plot(time_fusion, position_draw[:,2], label='position z fusion')
# plt.plot(time_gt, gt_draw[:,2], label='ground truth')
# # Drawing vertical lines for each timestamp in IMU data
# # for ts in timestamp_imu:
# #     plt.axvline(x=ts, color='green', linestyle='-',linewidth=0.2, ymin=0, ymax=1, label='IMU' if ts == timestamp_imu[0] else "")

# # Drawing vertical lines for each timestamp in visual data
# for ts in timestamp_visual:
#     plt.axvline(x=ts, color='red', linestyle='-', linewidth=0.2, ymin=0,  ymax=1,label='Visual' if ts == timestamp_visual[0] else "")

# # Adding labels and title
# plt.xlim(t_start, t_end) 
# plt.xlabel('Timestamp')
# plt.ylabel('z')
# plt.title('Visual Marks')
# plt.grid(True)

plt.figure(figsize=(10, 4))
for ts in timestamp_visual:
    plt.axvline(x=ts, color='red', linestyle='-', linewidth=0.2, ymin=0,  ymax=1,label='Visual Mark' if ts == timestamp_visual[0] else "")
#plt.plot(time_fusion, position_draw[:,0], label='Position x')
#plt.plot(time_fusion, position_draw[:,1], label='Position y')
plt.plot(time_fusion, position_draw[:,2], label='Position z')
# plot a base line in y = 0
plt.plot(time_gt, gt_draw[:,2], label='ground truth')
#plt.plot(time_gt, gt_draw[:,1], label='ground truth')
# Adding labels and title
# show label
plt.legend()
plt.xlim(t_start, t_end) 
plt.xlabel('Timestamp (s)')
plt.ylabel('Position (m))')
plt.title('Visual Marks')
plt.grid(True)

plt.figure(figsize=(10, 4))
for ts in timestamp_visual:
    plt.axvline(x=ts, color='red', linestyle='-', linewidth=0.2, ymin=0,  ymax=1,label='Visual Mark' if ts == timestamp_visual[0] else "")
plt.plot(time_gt, gt_draw[:,0] - position_draw_interpolated[:,0], label='Error in x axis', color='blue')
plt.plot(time_gt, gt_draw[:,1] - position_draw_interpolated[:,1], label='Error in y axis', color='green')
plt.plot(time_gt, gt_draw[:,2] - position_draw_interpolated[:,2], label='Error in z axis', color='red')
# plot a base line in y = 0
plt.plot(time_gt, np.zeros(len(time_gt)), label='base line', color='black')
#plt.plot(time_gt, gt_draw[:,1], label='ground truth')
# Adding labels and title
# show label
plt.legend()
plt.xlim(t_start, t_end) 
plt.xlabel('Timestamp (s)')
plt.ylabel('Error (m))')
plt.title('Visual Marks')
plt.grid(True)

plt.figure(figsize=(10, 4))
for ts in timestamp_visual:
    plt.axvline(x=ts, color='red', linestyle='-', linewidth=0.2, ymin=0,  ymax=1,label='Visual Mark' if ts == timestamp_visual[0] else "")
# plt.plot(time_gt, gt_draw[:,0] - position_draw_interpolated[:,0], label='Error in x axis', color='blue')
# plt.plot(time_gt, gt_draw[:,1] - position_draw_interpolated[:,1], label='Error in y axis', color='green')
# plt.plot(time_gt, gt_draw[:,2] - position_draw_interpolated[:,2], label='Error in z axis', color='red')
# all error
plt.plot(time_gt, np.sqrt((gt_draw[:,0] - position_draw_interpolated[:,0])**2 + (gt_draw[:,1] - position_draw_interpolated[:,1])**2 + (gt_draw[:,2] - position_draw_interpolated[:,2])**2), label='Error in Position', color='orange')
# plot a base line in y = 0
plt.plot(time_gt, np.zeros(len(time_gt)), label='base line', color='black')
#plt.plot(time_gt, gt_draw[:,1], label='ground truth')
# Adding labels and title
# show label
plt.legend()
plt.xlim(t_start, t_end) 
plt.xlabel('Timestamp (s)')
plt.ylabel('Error (m))')
plt.title('Visual Marks')
plt.grid(True)

# plt.figure(figsize=(10, 4))
# for ts in timestamp_visual:
#     plt.axvline(x=ts, color='red', linestyle='-', linewidth=0.2, ymin=0,  ymax=1,label='Visual' if ts == timestamp_visual[0] else "")
# plt.plot(time_fusion, position_draw[:,0], label='position z fusion')
# plt.plot(time_gt, gt_draw[:,0], label='ground truth')
# # Adding labels and title
# plt.xlim(t_start, t_end) 
# plt.xlabel('Timestamp')
# plt.ylabel('x')
# plt.title('Visual Marks')
# plt.grid(True)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # Recommended way to set 3D projection
# ax.plot(position_draw[:,0], position_draw[:,1], position_draw[:,2], label='position fusion')
# ax.plot(gt_draw[:,0], gt_draw[:,1], gt_draw[:,2], label='ground truth')
# #ax.plot(position_measurement_list[:,0], position_measurement_list[:,1], position_measurement_list[:,2], label='position measurement')
# #ax.plot(position_imu_list[:,0], position_imu_list[:,1], position_imu_list[:,2], label='position imu')
# ax.legend()

plt.show()
