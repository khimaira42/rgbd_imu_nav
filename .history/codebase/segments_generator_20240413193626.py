import open3d as o3d
import numpy as np
import json
import os
import pandas as pd
import csv

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    

    
def get_pcd_from_rgbd(rgb_image, depth_image, depth_intrin, depth_scale, filter_black_points=True):


    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image), 
        o3d.geometry.Image(depth_image), 
        depth_scale=1.0 / depth_scale,
        depth_trunc= 4.0,

        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            depth_intrin.width,
            depth_intrin.height,
            depth_intrin.fx,
            depth_intrin.fy,
            depth_intrin.ppx,
            depth_intrin.ppy))
    # Filter out black points
    if filter_black_points:
        colors = np.asarray(pcd.colors)
        non_black_indices = np.any(colors != [0, 0, 0], axis=1)
        pcd = pcd.select_by_index(np.where(non_black_indices)[0])

    # Flip pcd to align with the correct coordinate frame
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0,-1, 0], [0, 0, 0, 1]])
    # downsample
    #pcd = pcd.voxel_down_sample(voxel_size=0.08)

    return pcd

def save_segments(point_clouds, timestamp, labels, depth_intrin, depth_scale, scores):
    data_list = []

    for segment, label, score in zip(point_clouds, labels, scores):

        timestamp = timestamp
        additional_info = {
            'depth_intrin.width': depth_intrin.width,
            'depth_intrin.height': depth_intrin.height,
            'depth_intrin.fx': depth_intrin.fx,
            'depth_intrin.fy': depth_intrin.fy,
            'depth_intrin.ppx': depth_intrin.ppx,
            'depth_intrin.ppy': depth_intrin.ppy,
            'depth_scale': depth_scale
        }

        points = np.asarray(segment.points)
        colors = np.asarray(segment.colors)

        data_entry = {
            'timestamp': timestamp,
            'label': label,
            'score': score,
            'additional_info': additional_info,
            'points': points.tolist(),  # Convert NumPy array to Python list
            'colors': colors.tolist()  # Convert NumPy array to Python list
        }
        # Use the custom NumpyEncoder to handle int64 and float conversion
        data_entry_json = json.dumps(data_entry, cls=NumpyEncoder)

        data_list.append(json.loads(data_entry_json))

    output_folder = 'segments'
    output_file = 'segments_' + str(timestamp) + '.json'
    output_path = os.path.join(output_folder, output_file)

    with open(output_path, 'w') as file:
        json.dump(data_list, file)

def read_segments(file_path):
    with open(file_path, 'r') as file:
        data_list = json.load(file)

    point_clouds = []
    labels = []
    score = []
    other_info = []
    
    for data_entry in data_list:
        timestamp = data_entry['timestamp'] 
        label = data_entry['label']
        
        other_info = data_entry['additional_info']
        # Convert Python list back to NumPy array
        points = np.asarray(data_entry['points'])
        colors = np.asarray(data_entry['colors'])
        score = data_entry['score']
        # Create an Open3D point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)
        point_clouds.append(pc)
        labels.append(label)

    return point_clouds, timestamp, labels, score, other_info

def save_segments_as_csv(point_clouds, timestamp):
    segments = []
    ids = []
    id_segment = 0
    times = []
    id_image = str(timestamp).split('.')[0][-5:]
    for segment in point_clouds:

        timestamp = timestamp
        points = np.asarray(segment.points)
        segments.append(points)
        id_segment += 1
        ids.append(id_segment)
        # keep the last 5 digits before the decimal point
        times.append(id_image)

    # Create a DataFrame first colume is the id of the image, second column is the id of the segment, third to fifth column is the points
    segments = np.array(segments).reshape(-1, 3)

    data = {'id_images': times*len(segments), 'id_segments': ids*len(segments), 'points': segments}
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    output_folder = 'segments/train'
    # if not exists, create the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv = id_image + '.csv'
    output_path = os.path.join(output_folder, output_csv)
    df.to_csv(output_path, index=False)
    
    print(f'Data saved to {output_path}')




def find_nearest_time_row(df, time_test):
    # Calculate the absolute difference
    df['TimeDifference'] = (df['Time'] - time_test).abs()
    # Find the index of the smallest difference
    closest_index = df['TimeDifference'].idxmin()
    # Get the row with the closest time
    closest_time_row = df.loc[closest_index]
    # If you want to drop the 'TimeDifference' column afterwards
    df = df.drop(columns=['TimeDifference'])
    return closest_time_row

import quaternion  # numpy-quaternion package
import open3d as o3d
def cam_to_world_o3d(camera_position, camera_orientation, point_cloud):
    """
    Transforms a point cloud from the camera frame to the world frame
    :param camera_position: position of the camera in the world frame
    :param camera_orientation: orientation of the camera in the world frame
    :param point_cloud: point cloud in the camera frame
    :return: point cloud in the world frame
    """
        
    angle_rad = np.deg2rad(90)
    # Rotation matrix
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    point_cloud.rotate(rotation, center=(0, 0, 0))
    camera_translation = np.array([-0.3, 0, 0])
    point_cloud.translate(camera_translation)
    point_cloud.rotate(quaternion.as_rotation_matrix(camera_orientation), center=(0, 0, 0))
    point_cloud.translate(camera_position)

    return point_cloud

def cam_to_world(camera_position, camera_orientation, point_cloud):
    """
    Transforms a point cloud from the camera frame to the world frame
    :param camera_position: position of the camera in the world frame
    :param camera_orientation: orientation of the camera in the world frame
    :param point_cloud: point cloud in the camera frame
    :return: point cloud in the world frame
    """
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
    
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion.as_rotation_matrix(camera_orientation)

    # Function to transform a point from the camera frame to the world frame
    def transform_point(point, rotation_matrix, translation_vector):
        rotated_point = np.dot(rotation_matrix, point)
        transformed_point = rotated_point + translation_vector
        return transformed_point

    # Transforming each point in the point cloud
    transformed_point_cloud = np.array([transform_point(point, rotation_matrix, camera_position) for point in point_cloud])

    return transformed_point_cloud

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


def save_segments_as_csv2(point_clouds, timestamp, output_folder):
    segments = []
    id_segment = 0

    id_image = timestamp
    # Save DataFrame to CSV
    # if not exists, create the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv = id_image + '.csv'
    output_path = os.path.join(output_folder, output_csv)

    # Specify the CSV file name
    with open(output_path, mode='w') as file:
        pass  
    # Writing data to CSV
    with open(output_path, mode='w', newline='') as file:
        for segment in point_clouds:

            points = segment
            segments.append(points)

            writer = csv.writer(file)
            for x, y, z in points:
                writer.writerow([id_image, id_segment, x, y, z])
            id_segment += 1
def save_all_info(pcd_in_world, timestamp, label, segments_id, segments_recount, image_id, camera_position, 
                  output_labels, output_segments, output_timestamps, output_positions):


    def write_to_segments(pcd_in_world, image_id, segments_recount):
        # Writing data to CSV
        with open(output_segments, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            for x, y, z in pcd_in_world:
                writer.writerow([image_id, segments_recount, x, y, z])
            
    def write_to_timestamps(image_id, segments_recount, timestamp):
        # Writing data to CSV
        with open(output_timestamps, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerow([image_id, segments_recount, timestamp])

    def write_to_positions(image_id, segments_recount, camera_position):
        # Writing data to CSV
        with open(output_positions, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerow([image_id, segments_recount, camera_position[0], camera_position[1], camera_position[2]])

    def write_to_labels(segments_id, label):
        # Writing data to CSV
        with open(output_labels, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerow([segments_id, label])

    write_to_segments(pcd_in_world, image_id, segments_recount)
    write_to_timestamps(image_id, segments_recount, timestamp)
    write_to_positions(image_id, segments_recount, camera_position)
    write_to_labels(segments_id, label)

    print('Data saved to csv files')