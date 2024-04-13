import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

path_train_position = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/fortrain/vrpn_client_node-HaoranDrone-pose.csv'
path_train_global = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/forGlobalmap2/vrpn_client_node-HaoranDrone-pose.csv'
path_test_position = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/fortesttest2/vrpn_client_node-HaoranDrone-pose.csv'
df_train = pd.read_csv(path_train_position)
df_global = pd.read_csv(path_train_global)
df_test = pd.read_csv(path_test_position)
#draw the 3d trajectory of the drone
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(df_train['pose.position.x'],df_train['pose.position.y'])
# add grid
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('position for training')
plt.subplot(1, 3, 2)
plt.plot(df_global['pose.position.x'],df_global['pose.position.y'])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('position for global map')
plt.subplot(1, 3, 3)
plt.plot(df_test['pose.position.x'],df_test['pose.position.y'])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('position for testing')
#plt.plot(df_test['pose.position.x'],df_test['pose.position.y'])

# # load the position_draw
# position_draw = np.load('position_draw.npy')
# print(position_draw.shape)
# # load ground truth
# gt_draw = np.load('gt_draw.npy')

# # load the position_measurement_list
# position_measurement_list = np.load('position_measurement_list.npy')

# # load the position_imu_list
# position_imu_list = np.load('position_imu_list.npy')
# #
# show the trajectory and ground truth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# end = -5
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # Recommended way to set 3D projection
# ax.plot(position_draw[:end,0], position_draw[:end,1], position_draw[:end,2], label='position fusion')
# ax.plot(gt_draw[:end,0], gt_draw[:end,1], gt_draw[:end,2], label='ground truth')
# #ax.plot(position_measurement_list[:,0], position_measurement_list[:,1], position_measurement_list[:,2], label='position measurement')
# #ax.plot(position_imu_list[:,0], position_imu_list[:,1], position_imu_list[:,2], label='position imu')
# ax.legend()
# plt.show()

fig = plt.figure()
# plot in 3 plots and in 3d

ax = fig.add_subplot(111, projection='3d')  # Recommended way to set 3D projection
ax.plot(df_train['pose.position.x'], df_train['pose.position.y'], df_train['pose.position.z'], label='position for training')
#ax.plot(df_global['pose.position.x'], df_global['pose.position.y'], df_global['pose.position.z'], label='position for global map')
#ax.plot(df_test['pose.position.x'], df_test['pose.position.y'], df_test['pose.position.z'], label='position for test')   
# x y z label
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 
ax.legend()
plt.show()
