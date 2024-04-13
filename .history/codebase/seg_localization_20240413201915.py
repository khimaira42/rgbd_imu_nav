
import cv2
import numpy as np
import tensorflow as tf
import numpy as np
from segmappy.segmappy.core.config import Config
from segmappy.segmappy.core.dataset import Dataset
from segmappy.segmappy.core.generator import Generator
from segmappy.segmappy.tools.classifiertools import get_default_preprocessor, get_default_dataset
from cam_world_rt import cam_to_world_rt
from sklearn.neighbors import KDTree
import os
from tqdm import tqdm

'''
Introduction:
This script contains functions to get descriptors and match global descriptors with local descriptors.
'''

def get_global_info():
    # read config file
    configfile = "haoranDronev1.ini"
    config = Config(configfile)
    trained_model_folder = "/Users/david/Documents/Thesis/Thesis_codes/model_cnn"

    tf.compat.v1.disable_eager_execution()
    # load dataset and preprocessor
    dataset = get_default_dataset(config, config.cnn_test_folder)
    segments, positions, classes, n_classes = dataset.load_for_pred()
    
    # get centers of segments
    centers_in_3d = []
    for segment in segments:
        centers_in_3d.append(np.mean(segment,axis=0))
    preprocessor = get_default_preprocessor(config)
    # Generate descriptors, segments and segments_processed
    preprocessor.init_segments(segments, [0] * len(segments), positions=positions)
    segments_processed = preprocessor.process(segments, train=False, normalize=False)

    gen_global = Generator(
    preprocessor,
    range(len(segments)),
    1,
    train=False,
    batch_size=1,
    shuffle=False,
    )
    tf.compat.v1.reset_default_graph()

    # restore variable names from previous session
    saver = tf.compat.v1.train.import_meta_graph(
        os.path.join(config.cnn_model_folder, "model.ckpt.meta")
    )

    graph = tf.compat.v1.get_default_graph()
    cnn_input = graph.get_tensor_by_name("InputScope/input:0")
    descriptor = graph.get_tensor_by_name("OutputScope/descriptor_read:0")
    scales = graph.get_tensor_by_name("scales:0")

    # generate descriptors
    descriptors_global = []
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(config.cnn_model_folder))
        
        # generate descriptors for global
        print('Generating global descriptors')
        for step in tqdm(range(0, gen_global.n_batches)):
            
            batch_segments, batch_classes = gen_global.next()

            # calculate descriptors
            batch_descriptor = batch_descriptors = sess.run(
                descriptor,
                feed_dict={cnn_input: batch_segments, scales: preprocessor.last_scales},
            )

            descriptors_global.append(batch_descriptor)

        descriptors_global = np.concatenate(descriptors_global, axis=0)

    return descriptors_global, centers_in_3d

def matching_global_descriptors(local_segments, centers_in_2d, descriptors_global, centers_in_3d_global):
    # read config file
    configfile = "haoranDronev1.ini"
    config = Config(configfile)
    trained_model_folder = "/Users/david/Documents/Thesis/Thesis_codes/model_cnn"

    tf.compat.v1.disable_eager_execution()

    preprocessor_local = get_default_preprocessor(config)

    preprocessor_local.init_segments(local_segments, [0] * len(local_segments))
    centers_in_3d_local = []
    for segment in local_segments:
        centers_in_3d_local.append(np.mean(segment,axis=0))

    gen_local = Generator(
        preprocessor_local,
        range(len(local_segments)),
        1,
        train=False,
        batch_size=1,
        shuffle=False,
    )

    tf.compat.v1.reset_default_graph()

    # restore variable names from previous session
    saver = tf.compat.v1.train.import_meta_graph(
        os.path.join(config.cnn_model_folder, "model.ckpt.meta")
    )

    graph = tf.compat.v1.get_default_graph()
    cnn_input = graph.get_tensor_by_name("InputScope/input:0")
    descriptor = graph.get_tensor_by_name("OutputScope/descriptor_read:0")
    scales = graph.get_tensor_by_name("scales:0")

    # generate descriptors
    #descriptors_global = []
    descriptors_local = []

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(config.cnn_model_folder))
        
        for step in range(0, gen_local.n_batches):
            #print("step: ", step)
            batch_segments, batch_classes = gen_local.next()

            # calculate descriptors
            batch_descriptor = batch_descriptors = sess.run(
                descriptor,
                feed_dict={cnn_input: batch_segments, scales: preprocessor_local.last_scales},
            )

            descriptors_local.append(batch_descriptor)
    # print len of local segments
    # print('len of local segments:', len(local_segments))
    # print('Matching global descriptors with local descriptors')
    match_index = []
    for descriptor_pick, center_in_3d_local in zip(descriptors_local, centers_in_3d_local):

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Build a kd-tree from the global descriptors
        tree = KDTree(descriptors_global)

        # Find the k-nearest neighbors of each local descriptor
        knn_dists, indices = tree.query(descriptor_pick, k=5)

        # Compute the centroids of the local and global point cloud
        #print('descriptor_pick', descriptor_pick)
        #x, y, z = np.mean(descriptor_pick, axis=0)
        #centroid_local = np.array([x, y, z])

        # Initialize variables to keep track of the best match
        best_distance = float('inf')
        best_match_index = -1
        combined_dist_list = []
        euclidean_dist_list = []
        normalized_dist_list = []
        # Compute the centroids of the global point clouds with indices from the previous step
        for i, knn_dist in zip(indices[0], knn_dists[0]):
            centroid_global = centers_in_3d_global[i]

            # Calculate the Euclidean distance between the centroids
            euclidean_dist = np.linalg.norm(center_in_3d_local - centroid_global)  
            # normalize the euclidean distance and knn distance to same scale
            #knn_dist2 = knn_dist / np.max(knn_dists[0])
            #print('knn_dist2', knn_dist2)
            #normalized_dist_list.append(knn_dist2)
            combined_dist = 0.3 * knn_dist + 0.7 * euclidean_dist  
            euclidean_dist_list.append(euclidean_dist)
            combined_dist_list.append(combined_dist)

            # Update the best match if the current one is better
            if combined_dist < best_distance:
                best_distance = combined_dist
                best_match_index = i

        match_index.append(best_match_index)

        print('KNN distances', knn_dists)
       # print('normalized distances', normalized_dist_list)
        print('Euclidean distances ', euclidean_dist_list)
        print('best_match_index', best_match_index,'from list', indices[0],'\nwith combined distances', combined_dist_list)

    # Use OpenCV's solvePnP function to estimate the rotation and translation vectors
    point_3d_world = []
    point_2d_camera = []

    # load camera parameters
    from depth_intrinsics import DepthIntrinsics
    import json
    path_test='/Users/david/Documents/thesis/Thesis_code/data_collection/data_test'

    cam_pam_path = os.path.join(path_test, 'cam')
    with open(os.path.join(cam_pam_path,'updated_camera_intrinsics.json') , 'r') as file:
        cam_pam = json.load(file)

    depth_intrin = DepthIntrinsics()
    depth_intrin.width = int(cam_pam['depth_intrin.width'])
    depth_intrin.height = int(cam_pam['depth_intrin.height'])
    depth_intrin.fx = float(cam_pam['depth_intrin.fx'])
    depth_intrin.fy = float(cam_pam['depth_intrin.fy'])
    depth_intrin.ppx = float(cam_pam['depth_intrin.ppx'])
    depth_intrin.ppy = float(cam_pam['depth_intrin.ppy'])
    depth_scale = float(cam_pam['depth_scale'])
    camera_matrix = np.array([[depth_intrin.fx, 0, depth_intrin.ppx], [0, depth_intrin.fy, depth_intrin.ppy], [0, 0, 1]])

    for i,j in zip(match_index, centers_in_2d):
    # skip when i is -1
        if i == -1:
            continue
        point_3d_world.append(centers_in_3d_global[i])
        point_2d_camera.append(j)
    # print('point_3d_world', point_3d_world)
    # print('point_3d_world_local', center_in_3d_local)
    # list to array
    point_3d_world = np.array(point_3d_world).reshape(-1,3)
    point_2d_camera = np.array(point_2d_camera).reshape(-1,2)
    center_in_3d_local = np.array(center_in_3d_local).reshape(-1,3)

    translation_vector = point_3d_world - center_in_3d_local
    print('translation_vector', translation_vector)
    # filter out the outliers
    dist = np.linalg.norm(translation_vector, axis=1)
    # filter out dist > 1
    filtered_translation_vector = translation_vector[dist < 1.5]
    print('dist', dist)    
    print('filtered_translation_vector', filtered_translation_vector)
    success = False
    if len(filtered_translation_vector) == 0:
        return np.array([0, 0, 0]), np.eye(3), success
    # mean in column
    translation_vector = np.mean(filtered_translation_vector, axis=0)
    print('translation_vector mean', translation_vector)
    #print('translation_vector', translation_vector)
    #print('end of translation vector')
    rotation_matrix = np.eye(3)
    
    if len(filtered_translation_vector) >= 4:
        success, rotation_vector, translation_vector = cv2.solvePnP(point_3d_world, point_2d_camera, camera_matrix, dist_coeffs)
        print('PnP success?', success)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return translation_vector, rotation_matrix
    return translation_vector, rotation_matrix, success

def get_local_segments(segments, seg_time, timestamp, centers, current_state, orientation, z):

    seg_id = seg_time[seg_time['timestamp'] == timestamp]
    segments_list = []
    centers_list_2d = []
    for i in range(len(seg_id)):
        segments_cache1 = segments[segments['image_id'] == seg_id['image_id'].values[0]]
        segments_cache2 = segments_cache1[segments_cache1['segment_recount'] == i]

        centers_cache1 = centers[centers['image_id'] == seg_id['image_id'].values[0]]
        centers_cache2 = centers_cache1[centers_cache1['segment_recount'] == i]
        # keep the last 2 columns
        center = centers_cache2.iloc[:,-2:].values
        centers_list_2d.append(np.array(center))
        # keep the last 3 columns
        segment = segments_cache2.iloc[:,-3:].values
        segment_world_est = cam_to_world_rt(np.array([current_state[0], current_state[1], current_state[2]]), orientation, segment)
        segments_list.append(segment_world_est)
        #print('center of segment', i, 'is', center)

    return segments_list, centers_list_2d
