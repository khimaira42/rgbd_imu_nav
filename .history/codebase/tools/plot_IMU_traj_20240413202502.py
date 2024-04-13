from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from ekf import ExtendedKalmanFilter

'''
Introduction:
This script is used to plot the trajectory of the drone using the IMU data and the ground truth data.
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
    # add bias to accel 0.28478655780245 -0.024718655019298263 -9.828971270850001
    gyro_bias = np.array([0.0020178240054855253, 0.0017592206368581425, 0.0017427009552286577])
    acc_bias = np.array([0.28478655780245, -0.024718655019298263, -9.828971270850001])
    # set bias to 0
    # gyro_bias = np.array([0, 0, 0])
    # acc_bias = np.array([0, 0, 0])
    num_data_points = len(time_array)
    positions = np.zeros((num_data_points, 3))
    orientations = np.zeros((num_data_points, 3))
    velocity = np.zeros((num_data_points, 3))

    positions[0] = initial_position
    
    orientations[0] = initial_orientation
    velocity[0] = np.array([0, 0, 0])

    for i in range(1, num_data_points):
        dt = (time_array[i] - time_array[i-1])  # Delta time in seconds
        # Define the rotation matrix for a minus 90-degree rotation around the Z-axis
        R_z = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
        
        #rot_to_drone = rotation_matrix_from_degrees(180, 0, -90)
        # rot_to_drone = rotation_matrix_from_degrees(250, 0, -45)
        # -15 -65
        rot_to_drone = rotation_matrix_from_degrees(-35, 0, -55)
        # Update Orientation using Gyroscope Data
        gyro = gyro_array[i] - gyro_bias
        rot_0 = rotation_matrix_from_rad(orientations[i-1][0], orientations[i-1][1], orientations[i-1][2])
        gyro = rot_to_drone.dot(gyro)
        gyro = rot_0.dot(gyro)
        orientation = gyro * dt
        #orientation = rot_to_drone.dot(orientation)
        #orientation = rot_0.dot(orientation)
        orientations[i] = orientations[i-1] + orientation
        
        # Update Position using Accelerometer Data
        # Assuming initial velocity is zero for simplicity
        rot = rotation_matrix_from_rad(orientations[i][0], orientations[i][1], orientations[i][2])

        accel = accel_array[i] - acc_bias
        # accel = R_z.dot(accel)
        # accel = rot.dot(accel)
        accel = rot_to_drone.dot(accel)
        accel = rot.dot(accel)
        velocity[i] = velocity[i-1] + accel * dt
        translations = velocity[i-1] * dt + 0.5 * accel * dt**2
        # Add bias to translation
        #bias_vector = np.array([0, 0, -0.03])
        #bias_vector = rot.dot(bias_vector)
        translations = translations #+ bias_vector
        #translations = R_z.dot(translations)
        #translations = rot_to_drone.dot(translations)
        #translations = rot.dot(translations)
        #positions[i] = positions[i-1] + velocity[i] * dt
        positions[i] = positions[i-1] + translations

    return positions, orientations
#import quaternion
import math
def rotation_matrix_from_degrees(roll, pitch, yaw):
    """
    Create a rotation matrix from roll, pitch, and yaw in degrees
    """
    # roll is rotation around x in degrees (counterclockwise)
    # pitch is rotation around y in degrees (counterclockwise)
    # yaw is rotation around z in degrees (counterclockwise)
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(math.radians(roll)), -math.sin(math.radians(roll))],
                    [0, math.sin(math.radians(roll)), math.cos(math.radians(roll))]])
    R_y = np.array([[math.cos(math.radians(pitch)), 0, math.sin(math.radians(pitch))],
                    [0, 1, 0],
                    [-math.sin(math.radians(pitch)), 0, math.cos(math.radians(pitch))]])
    R_z = np.array([[math.cos(math.radians(yaw)), -math.sin(math.radians(yaw)), 0],
                    [math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0],
                    [0, 0, 1]])
    rot = R_z.dot(R_y.dot(R_x))
    return rot

def rotation_matrix_from_rad(roll, pitch, yaw):
    """
    Create a rotation matrix from roll, pitch, and yaw in radians
    """
    # roll is rotation around x in radians (counterclockwise)
    # pitch is rotation around y in radians (counterclockwise)
    # yaw is rotation around z in radians (counterclockwise)
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    rot = R_z.dot(R_y.dot(R_x))
    return rot

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


path_test_position = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/fortesttest2/vrpn_client_node-HaoranDrone-pose.csv'
path = '/Users/david/Documents/thesis/Thesis_code/data_collection/data_imu/imu_compensated.csv'
df_gt= pd.read_csv(path_test_position)
df_imu2 = pd.read_csv(path)
# imu start from row 68 and reset index
num = 300
start = 1500
df = df_imu2.iloc[start:start+num,:].reset_index(drop=True)
accel_array = np.array([df['accelerometer_m_s2[0]'], df['accelerometer_m_s2[1]'], df['accelerometer_m_s2[2]']]).reshape(3, -1).T
gyro_array = np.array([df['gyro_rad[0]'], df['gyro_rad[1]'], df['gyro_rad[2]']]).reshape(3, -1).T
print(gyro_array.shape)
# find nearest timestamp for df_imu['timestamp'][0] in df_gt['Time']

# Calculate the absolute difference
df_gt['TimeDifference'] = (df_gt['Time'] - df['timestamp'][0]).abs()
# Find the index of the smallest difference
closest_index = df_gt['TimeDifference'].idxmin()
# Delete the TimeDifference column
df_gt = df_gt.drop(columns=['TimeDifference'])
# Get the row with the closest time
closest_time_row = df_gt.loc[closest_index]
# get the initial position and orientation
initial_position = np.array([closest_time_row['pose.position.x'], closest_time_row['pose.position.y'], closest_time_row['pose.position.z']])
#initial_orientation = np.quaternion(closest_time_row['pose.orientation.w'], closest_time_row['pose.orientation.x'], closest_time_row['pose.orientation.y'], closest_time_row['pose.orientation.z'])
# get the initial orientation in euler angle
orientation_x, orientation_y, orientation_z = euler_from_quaternion(closest_time_row['pose.orientation.w'], closest_time_row['pose.orientation.x'], closest_time_row['pose.orientation.y'], closest_time_row['pose.orientation.z'])
initial_orientation = np.array([orientation_x, orientation_y, orientation_z])

# delete column before ckosest_time_row
df_gt = df_gt.iloc[closest_index:,:].reset_index(drop=True)
print(df_gt['Time'][0])
print(df['timestamp'][0])
# Calculate the absolute difference of last timestamp
df_gt['TimeDifference'] = (df_gt['Time'] - df['timestamp'][len(df)-1]).abs()
# Find the index of the smallest difference
closest_index = df_gt['TimeDifference'].idxmin()
# Delete the TimeDifference column
df_gt = df_gt.drop(columns=['TimeDifference'])
# delete column after ckosest_time_row
df_gt = df_gt.iloc[:closest_index,:].reset_index(drop=True)
ekf = ExtendedKalmanFilter()
visual_measurement = np.array([initial_position[0], initial_position[1], initial_position[2],
                                initial_orientation[0], initial_orientation[1], initial_orientation[2]])
initial_state = np.array([initial_position[0], initial_position[1], initial_position[2],
                                initial_orientation[0], initial_orientation[1], initial_orientation[2],
                                0, 0, 0])
#ekf.update(measurement=visual_measurement)
ekf.state = initial_state
print(ekf.state)
print(initial_state)
states = []
for acc, gyro in zip(accel_array, gyro_array):
    
        # find the matched imu data
    dt = 0.005
    ekf.predict(acc=acc, gyro=gyro, dt=dt)
    states.append(ekf.state.copy())

# calculate the trajectory
#print(df_imu['timestamp'])
positions, orientations = calculate_trajectory(df['timestamp'], gyro_array, accel_array, initial_position, initial_orientation)
# convert quaternion to euler angle
euler_angles = []
# plt.figure(figsize=(20,10))
# plt.subplot(2,1,1)
# plt.plot(positions[:,1], positions[:,2])
# # plot 2 in the same row
# plt.subplot(2,1,2)
# plt.plot(df_gt['pose.position.y'],df_gt['pose.position.z'])
#plt.plot(positions[:,0], positions[:,1])
# plot the ground truth
#plt.plot(df_gt['pose.position.x'],df_gt['pose.position.y'])
print(initial_position)
states = np.array(states).reshape(-1,9)
#plot 3d trajectory
from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(figsize=(20,10))
fig = plt.figure()
# plot in 3 plots and in 3d
#draw the 3d trajectory of the drone
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(states[:,0],states[:,1])
# add grid
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pose estimation of IMU')
plt.subplot(1, 2, 2)
plt.plot(df_gt['pose.position.x'],df_gt['pose.position.y'])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ground truth')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(states[:,1],states[:,2])
# add grid
plt.grid()
plt.xlabel('y')
plt.ylabel('z')
plt.title('Pose estimation of IMU')
plt.subplot(1, 2, 2)
plt.plot(df_gt['pose.position.y'],df_gt['pose.position.z'])
plt.grid()
plt.xlabel('y')
plt.ylabel('z')
plt.title('Ground truth')

ax = fig.add_subplot(111, projection='3d')  # Recommended way to set 3D projection
#ax = fig.gca(projection='3d')
print('length of positions', len(positions))
#ax.plot(positions[:,0], positions[:,1], positions[:,2], label='drone trajectory')
ax.plot(states[:,0], states[:,1], states[:,2], label='drone trajectory KF')
ax.plot(df_gt['pose.position.x'],df_gt['pose.position.y'],df_gt['pose.position.z'], label='ground truth')
ax.legend()
plt.show()
