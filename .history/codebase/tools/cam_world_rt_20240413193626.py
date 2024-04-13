import numpy as np

def cam_to_world_rt(camera_position, camera_orientation, point_cloud):
    """
    Transforms a point cloud from the camera frame to the world frame
    :param camera_position: position of the camera in the world frame
    :param camera_orientation: orientation of the camera in the world frame
    :param point_cloud: point cloud in the camera frame
    :return: point cloud in the world frame
    """
    import numpy as np

    def rotation_matrix(roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        rot = np.dot(R_z, np.dot(R_y, R_x))
        return rot

    # adjust
    camera_translation = np.array([-0.3, 0, 0])
    point_cloud = point_cloud + camera_translation
    # Rotation of -90 degrees around the x-axis in radians
    angle_rad = np.deg2rad(90)

    # Rotation matrix
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    point_cloud = np.array([np.dot(R, point) for point in point_cloud])
    
    # Convert orientation(raw, pitch, yaw) to rotation matrix

    rotation_matrix = rotation_matrix(camera_orientation[0], camera_orientation[1], camera_orientation[2])
    
    # Function to transform a point from the camera frame to the world frame
    def transform_point(point, rotation_matrix, translation_vector):
        rotated_point = np.dot(rotation_matrix, point)
        transformed_point = rotated_point + translation_vector
        return transformed_point

    # Transforming each point in the point cloud
    transformed_point_cloud = np.array([transform_point(point, rotation_matrix, camera_position) for point in point_cloud])

    return transformed_point_cloud