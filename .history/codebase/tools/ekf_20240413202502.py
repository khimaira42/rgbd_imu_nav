import numpy as np
import math

class ExtendedKalmanFilter:
    def __init__(self):
        # State vector: [x, y, z, rx, ry, rz, vx, vy, vz] 
        # rx, ry, rz could be orientation in Euler angles or a rotation vector format
        # Initialize Q with estimated variances
        # These are example values; need to adjust them based on drone's characteristics
        var_position = 0.1  # Variance for position
        var_orientation = 0.01  # Variance for orientation (roll, pitch, yaw)
        var_velocity = 0.2  # Variance for velocity

        Q_init = np.diag([var_position, var_position, var_position, 
                    var_orientation, var_orientation, var_orientation, 
                    var_velocity, var_velocity, var_velocity])
        # Initial variances for each measurement
        var_x, var_y, var_z = 10, 10, 50  # Variances for position measurements
        var_rx, var_ry, var_rz = 10, 10, 10  # Variances for orientation measurements

        R_init = np.diag([var_x, var_y, var_z, var_rx, var_ry, var_rz])
        self.state = np.zeros(9)
        self.prev_state = np.zeros(9)  # Initialize the previous state as well
        self.P = np.eye(9) * 0.1  # Initial state covariance
        self.Q = Q_init #np.eye(9) * 0.02  # Process noise covariance
        self.R = R_init #np.eye(6) * 30.0  # Measurement noise covariance for position and orientation
        self.H = np.eye(6, 9)  # Measurement matrix, assuming first 6 states are directly measured
        self.rot_drone_to_world = np.eye(3)  # Rotation matrix from drone frame to world frame
    def predict(self, acc, gyro, dt):
         # Update state [x, y, z,  roll, pitch, yaw, vx, vy, vz] using IMU data
        # imu to drone, need to adjust them based on drone's characteristics
        # rot_imu_to_drone = np.array([[ 0,  0, -1],
        #                             [0,  1,  0],
        #                             [1,  0,  0]])#[0,-90,0]
        # rot_imu_to_drone = np.array([[ -1,  0, 0],
        #                              [0,  1,  0],
        #                              [0,  0,  -1]])#[180,0,90]???

        # rot_imu_to_drone = np.array([[ 0,  1, 0],
        #                              [1,  0,  0],
        #                              [0,  0,  -1]])#[180,0,90]

        # rot_imu_to_drone2 = np.array([[ -1, 0, 0],
        #                              [0,  -1,  0],
        #                              [0,  0,  1]])
        # rot_imu_to_drone = rot_imu_to_drone1.dot(rot_imu_to_drone2)
       # print('rot_imu_to_drone', rot_imu_to_drone)
        rot_imu_to_drone = np.array([[0,  -1, 0],
                                    [-1,  0,  0],
                                    [0,  0,  -1]]) # (180, 0, -90)
        # rot_imu_to_drone = np.array([[ 0.57357644,  0.67101007,  0.46984631],
        #                             [-0.81915204,  0.46984631,  0.32898993],
        #                             [ 0.        , -0.57357644,  0.81915204]])

        # add bias
        # 1 pitch 2 roll 3 yaw
        gyro_bias = [0.001851052, 0.0017697088, 0.0019698385]  # Replace with actual values
        acc_bias = [0.015662003, 0.056381565, -0.02278524]  
        acc = acc - acc_bias
        gyro = gyro - gyro_bias

        g = 9.81  # Acceleration due to gravity (m/s^2)

    
 
        self.rotation_matrix_from_rad(self.state[3], self.state[4], self.state[5])
        gyro = rot_imu_to_drone.dot(gyro)
        gyro = self.rot_drone_to_world.dot(gyro)
        # Intupegrate gyroscope data to update orientation
        orientation = gyro * dt
        #orientation = rot_imu_to_drone.dot(orientation)
        # get rotation matrix from self.state[6:9]
        #self.rotation_matrix_from_rad(self.state[6], self.state[7], self.state[8])
        #orientation = self.rot_to_world.dot(orientation)
        self.prev_state = self.state.copy()
        self.state[3:6] += orientation
        # Calculate the gravity vector
        roll, pitch, yaw = self.state[3:6]
        gravity = np.array([
            g * np.sin(pitch),
            -g * np.sin(roll) * np.cos(pitch),
            -g * np.cos(roll) * np.cos(pitch)
        ])

        # Remove the gravity component
        acc = acc - gravity
        # Integrate acceleration to update velocity
        self.rotation_matrix_from_rad(self.state[3], self.state[4], self.state[5])
        acc = rot_imu_to_drone.dot(acc)
        acc = self.rot_drone_to_world.dot(acc)
        self.state[6:9] += acc * dt
        
        translation_vector = self.prev_state[6:9] * dt + 0.5 * acc * dt**2
        # rotation matrix from imu to world
        #translation_vector = rot_imu_to_drone.dot(translation_vector)
        #translation_vector = self.rot_to_world.dot(translation_vector)
        # Integrate velocity to update position
        self.state[0:3] += translation_vector
        # add bias to position
        self.state[0:3] += self.rot_drone_to_world.dot(rot_imu_to_drone.dot(np.array([-0.002, 0.003, 0.0002]).reshape(-1)))

        # Update state covariance matrix P
        # Update state covariance matrix P
        # F = self.calculate_F_jacobian(dt, acc, gyro)
        # self.P = F @ self.P @ F.T + self.Q

        F = np.eye(9)  # State transition matrix, modify if necessary
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        # Update the state with visual measurement
        # Assuming measurement = [translation_vector, rotation_vector]
        y = measurement - self.H @ self.state  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        #print('state before update', self.state)
        self.state = self.state + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        #print('state after update', self.state)
    
    def rotation_matrix_from_rad(self, roll, pitch, yaw):
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
        self.rot_drone_to_world = R_z.dot(R_y.dot(R_x))

    def adjust_Q(self, factor_position, factor_orientation, factor_velocity):
        """
        Adjust the process noise covariance matrix Q.
        :param Q: The original Q matrix.
        :param factor_position: Multiplicative factor to adjust position variance.
        :param factor_orientation: Multiplicative factor to adjust orientation variance.
        :param factor_velocity: Multiplicative factor to adjust velocity variance.
        :return: Adjusted Q matrix.
        """
        self.Q[0:3, 0:3] *= factor_position
        self.Q[3:6, 3:6] *= factor_orientation
        self.Q[6:9, 6:9] *= factor_velocity
        
    def adjust_R(R, factors):
        """
        Adjust the measurement noise covariance matrix R.
        :param R: The original R matrix.
        :param factors: A list of six factors to adjust the variances for x, y, z, rx, ry, rz.
        :return: Adjusted R matrix.
        """
        for i in range(6):
            R[i, i] *= factors[i]
        return R
