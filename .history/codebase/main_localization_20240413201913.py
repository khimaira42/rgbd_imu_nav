import os
import numpy as np
import json
import tools.label_color_class
import pandas as pd
from tqdm import tqdm
from tools.ekf import ExtendedKalmanFilter
from seg_localization import get_local_segments, matching_global_descriptors, get_global_info
import math

'''
Introduction:
This script is used to localize the drone using the Extended Kalman Filter (EKF) with the IMU and Visual data.
'''

def calculate_trajectory(time_array, gyro_array, accel_array, initial_position, initial_orientation):
    """
    Calculate the trajectory (position and orientation) based on arrays of IMU data.

    :param time_array: Array of timestamps in microseconds
    :param gyro_array: Array of gyroscope readings [x, y, z] in rad/s
    :param accel_array: Array of accelerometer readings [x, y, z] in m/s^2
    :param initial_position: Initial position as a numpy array [x, y, z] in meters
    :param initial_orientation: Initial orientation as a numpy array [roll, pitch, yaw] in radians

    :return: Array of positions and orientations
    """
    print(initial_orientation)
    # add bias to accel 0.28478655780245 -0.024718655019298263 -9.828971270850001
    gyro_bias = np.array([0.0020178240054855253, 0.0017592206368581425, 0.0017427009552286577])
    acc_bias = np.array([0.28478655780245, -0.024718655019298263, -9.828971270850001])
    # set bias to 0
    # gyro_bias = np.array([0, 0, 0])
    # acc_bias = np.array([0, 0, 0])
    num_data_points = len(time_array)
    positions = np.zeros((num_data_points, 3))
    orientations = np.zeros((num_data_points, 3))

    positions[0] = initial_position
    orientations[0] = initial_orientation

    for i in range(1, num_data_points):
        dt = (time_array[i] - time_array[i-1])  # Delta time in seconds

        # Update Orientation using Gyroscope Data
        gyro = gyro_array[i] - gyro_bias
        orientations[i] = orientations[i-1] + gyro * dt

        # Update Position using Accelerometer Data
        # Assuming initial velocity is zero for simplicity
        accel = accel_array[i] - acc_bias
        velocity = accel * dt
        positions[i] = positions[i-1] + velocity * dt

    return positions, orientations


def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
    
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z

test_gt_path = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/fortesttest2/vrpn_client_node-HaoranDrone-pose.csv'
imu_path = '/Users/david/Documents/thesis/Thesis_code/data_collection/data_imu/imu_compensated.csv'
segments_path = '/Users/david/Documents/thesis/Thesis_code/segments/test_dataset/segments_database.csv'
seg_time_path = '/Users/david/Documents/thesis/Thesis_code/segments/test_dataset/timestamps_database.csv'
seg_label_path = '/Users/david/Documents/thesis/Thesis_code/segments/test_dataset/labels_database.csv'
seg_centers_path = '/Users/david/Documents/thesis/Thesis_code/segments/test_dataset/centers_database.csv'
combined_mark = pd.read_csv('combined_mark.csv')
# start from 4752
combined_mark = combined_mark[combined_mark['timestamp'] > 1701886824.2419815]
df_gt= pd.read_csv(test_gt_path)
imu_data = pd.read_csv(imu_path)

segments = pd.read_csv(segments_path, sep=' ',names=['image_id', 'segment_recount', 'camera_position[0]', 'camera_position[1]', 'camera_position[2]'])
seg_time = pd.read_csv(seg_time_path, sep=' ',names=['image_id', 'segment_recount', 'timestamp'])
seg_label = pd.read_csv(seg_label_path, sep=' ',names=['segments_id', 'label'])
seg_centers = pd.read_csv(seg_centers_path, sep=' ',names=['image_id','segment_recount', 'center_x', 'center_y'])


ekf = ExtendedKalmanFilter()  # Initialize your EKF here

initialization_set = True
last_time = None
position_draw = []
position_imu_list = []
position_measurement_list = []
gt_draw = []
import time
#from seg_localization import segment_based_localization
t_start = 1701886823.241981
t_end = 1701886831.541981
# Generate global descriptors for all segments
global_descriptors, centers_in_3d_global = get_global_info()

for index, row in combined_mark.iterrows():
    
    current_time = row['timestamp']
    if current_time < t_start: #1701886861.241981: #1701886838.241981: #1701886861.412507: #23-31 fine
        continue
    # stop after 1000 rows
    if current_time > t_end: #1701886849.541981: #1701886849.891981:#1701886865.5472007:   #max 30  good 6824.24-6825 fine 1701886838.2419815-49.89
        break
    if initialization_set:
        # Calculate the absolute difference
        df_gt['TimeDifference'] = (df_gt['Time'] - current_time).abs()
        # Find the index of the smallest difference
        closest_index = df_gt['TimeDifference'].idxmin()
        # Get the row with the closest time
        closest_time_row = df_gt.loc[closest_index]
        # If you want to drop the 'TimeDifference' column afterwards
        df_gt = df_gt.drop(columns=['TimeDifference'])
        # Initialize the EKF
        initial_position = [closest_time_row['pose.position.x'], closest_time_row['pose.position.y'], closest_time_row['pose.position.z']]
        #[5.698331356048584,-3.0916879177093506,1.1035140752792358,]  # Replace with your initial position data
        initial_velocity = [0, 0, 0]     # Assuming no initial knowledge of velocity
        # quaternion to euler
        orientation_x, orientation_y, orientation_z = euler_from_quaternion(closest_time_row['pose.orientation.w'], closest_time_row['pose.orientation.x'], closest_time_row['pose.orientation.y'], closest_time_row['pose.orientation.z'])
        #(-0.4000707268714905,0.047468122094869614,0.012585824355483055,-0.9151676893234253)
        initial_orientation = [orientation_x, orientation_y, orientation_z] 
        initial_state = initial_position + initial_orientation + initial_velocity
        ekf.state = np.array(initial_state)
        print('initial state', ekf.state)
        # wait for 3 second
        time.sleep(3)
        initialization_set = False

    # Simulate real-time delay
    # if last_time is not None:
    #     time.sleep(current_time - last_time)

    if row['data_type'] == 'IMU':
        # find the matched imu data
        imu_row = imu_data[imu_data['timestamp'] == current_time]
        #print(imu_row)
        # Process IMU data
        acc = np.array([imu_row['accelerometer_m_s2[0]'], imu_row['accelerometer_m_s2[1]'], imu_row['accelerometer_m_s2[2]']]).reshape(-1)
        gyro = np.array([imu_row['gyro_rad[0]'], imu_row['gyro_rad[1]'], imu_row['gyro_rad[2]']]).reshape(-1)
        
        if last_time is not None:
            dt = (current_time - last_time)
            #print(dt)
            ekf.predict(acc=acc, gyro=gyro, dt=dt)
            print('prediction in time:', current_time)
            position_imu_list.append(ekf.state[0:3])
    elif row['data_type'] == 'Visual':  
        # Process Visual data
        # Get the current state for visual processing
        current_state = ekf.state
        
        timestamp = row['timestamp']
        # find the nearest time row in ground truth
        #print('timestamp', timestamp)
        # find nearest timestamp for df_imu['timestamp'][0] in df_gt['Time']
        # Calculate the absolute difference
        df_gt['TimeDifference'] = (df_gt['Time'] - timestamp).abs()
        # Find the index of the smallest difference
        closest_index = df_gt['TimeDifference'].idxmin()
        # Delete the TimeDifference column
        df_gt = df_gt.drop(columns=['TimeDifference'])
        # Get the row with the closest time
        closest_time_row = df_gt.loc[closest_index]
        roll, pitch, yaw = euler_from_quaternion(closest_time_row['pose.orientation.w'], closest_time_row['pose.orientation.x'], closest_time_row['pose.orientation.y'], closest_time_row['pose.orientation.z'])
        orientation = np.array([roll, pitch, yaw])
        segments_local, centers_list_2d = get_local_segments(segments, seg_time, row['timestamp'], seg_centers, current_state, orientation ,closest_time_row['pose.position.z'])
        translation_vector, rotation_vector_pnp, success= matching_global_descriptors(segments_local, centers_list_2d, global_descriptors, centers_in_3d_global)
        
        rotation_vector = np.array([roll, pitch, yaw])
        position_measurement = current_state[0:3] + translation_vector
        # visual_measurement = np.array([translation_vector[0], translation_vector[1], closest_time_row['pose.position.z'] - current_state[2],
        #                                 rotation_vector[0], rotation_vector[1], rotation_vector[2]])
        if success == False:
            # number of segment is less than 4 so that use another method
            visual_measurement = np.array([position_measurement[0], position_measurement[1], position_measurement[2],
                                            current_state[3], current_state[4], current_state[5]])
                                        #roll, pitch, yaw])
        #[row['translation_x'], row['translation_y'], row['translation_z'], 
                                       #row['rotation_x'], row['rotation_y'], row['rotation_z']])
        else :
            print("PnP found!")
            time.sleep(5)
            orientation_measurement = current_state[3:6] + rotation_vector_pnp
            visual_measurement = np.array([position_measurement[0], position_measurement[1], position_measurement[2],
                                            orientation_measurement[0], orientation_measurement[1], orientation_measurement[2]])
            
        gt_draw.append([closest_time_row['pose.position.x'], closest_time_row['pose.position.y'], closest_time_row['pose.position.z']])
        #if np.sum(translation_vector) == 0: # or abs(translation_vector[2]) > 0.3:
            #continue
        
        ekf.update(measurement=visual_measurement)
        print('update in time:', current_time)
        print('translation_vector of update', translation_vector)
        position_measurement_list.append(position_measurement)
        if translation_vector.sum() > 10:
            print('Interupted by large translation vector')
            break
    position_draw.append(ekf.state[0:3])  
    last_time = current_time
    

# save the position_draw
position_draw = np.array(position_draw)
# save with current time
#np.save('position_draw_' + str(current_time) + '.npy', position_draw)
np.save('position_draw.npy', position_draw)
# load the position_draw
position_draw = np.load('position_draw.npy')

# save ground truth
gt_draw = np.array(gt_draw)
np.save('gt_draw.npy', gt_draw)
# load ground truth
gt_draw = np.load('gt_draw.npy')

# save the position_measurement_list
position_measurement_list = np.array(position_measurement_list)
np.save('position_measurement_list.npy', position_measurement_list)
# load the position_measurement_list
position_measurement_list = np.load('position_measurement_list.npy')

# save the position_imu_list
position_imu_list = np.array(position_imu_list)
np.save('position_imu_list.npy', position_imu_list)
# load the position_imu_list
position_imu_list = np.load('position_imu_list.npy')

# save the position_draw
#position_draw = np.array(position_draw)
# save with current time
#np.save('position_draw_' + str(current_time) + '.npy', position_draw)
# # get the current time
# time_name = time.time()
# np.save('position_draw_' + str(time_name) + '.npy', position_draw)
# # load the position_draw
# position_draw = np.load('position_draw_' + str(time_name) + '.npy')

# # save ground truth
# gt_draw = np.array(gt_draw)
# np.save('gt_draw_' + str(time_name) + '.npy', gt_draw)
# # load ground truth
# gt_draw = np.load('gt_draw_' + str(time_name) + '.npy')

# # save the position_measurement_list
# position_measurement_list = np.array(position_measurement_list)
# np.save('position_measurement_list_' + str(time_name) + '.npy', position_measurement_list)
# # load the position_measurement_list
# position_measurement_list = np.load('position_measurement_list_' + str(time_name) + '.npy')

# # save the position_imu_list
# position_imu_list = np.array(position_imu_list)
# np.save('position_imu_list.npy', position_imu_list)
# # load the position_imu_list
# position_imu_list = np.load('position_imu_list_' + str(time_name) + '.npy')


t_duration = t_end - t_start
time_gt = np.linspace(t_start, t_end, num=len(gt_draw))
time_fusion = np.linspace(t_start, t_end, num=len(position_draw))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.plot(position_draw[:,0], position_draw[:,1], label='position fusion')
plt.plot(gt_draw[:,0], gt_draw[:,1], label='ground truth')
# add grid
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory in x-y plane')
plt.subplot(1, 2, 2)

plt.plot(time_fusion, position_draw[:,2], label='position fusion')
plt.plot(time_gt, gt_draw[:,2], label='ground truth')
plt.grid()
plt.xlabel('timestamp')
plt.ylabel('z')
plt.title('Comparison in z axis')
# show the trajectory and ground truth


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Recommended way to set 3D projection
ax.plot(position_draw[:,0], position_draw[:,1], position_draw[:,2], label='position fusion')
ax.plot(gt_draw[:,0], gt_draw[:,1], gt_draw[:,2], label='ground truth')
#ax.plot(position_measurement_list[:,0], position_measurement_list[:,1], position_measurement_list[:,2], label='position measurement')
#ax.plot(position_imu_list[:,0], position_imu_list[:,1], position_imu_list[:,2], label='position imu')
ax.legend()
plt.show()
