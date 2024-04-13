from segments_generator import  get_pcd_from_rgbd, save_segments_as_csv2, find_nearest_time_row, cam_to_world, save_all_info, cam_to_world_o3d
from depth_intrinsics import DepthIntrinsics
import os
import cv2
import numpy as np
import json
import pycocotools.mask as mask_util
import numpy as np
import label_color_class
import pandas as pd
import quaternion 
import open3d as o3d
from tqdm import tqdm
from filtering_pc  import cluster_and_filter_points

'''
Introduction:
This script is used to generate the global map from the segmentation data and the global position data.
images are used to generate the point cloud, and the global position data is used to adjust the position of the point cloud.
mask from the segmentation data is used to generate the point cloud of each segment.

:Check segments_generator.py for the crucial functions used in this script
:Similar to the codebase/tools/save_segments.py and codebase/tools/save_segments_train.py
'''

path_train_position = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/fortrain/vrpn_client_node-HaoranDrone-pose.csv'
path_global_position = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/forGlobalmap2/vrpn_client_node-HaoranDrone-pose.csv'
path_test_position = '/Users/david/Documents/thesis/Thesis_code/data_collection/bag_all/fortesttest2/vrpn_client_node-HaoranDrone-pose.csv'
df_train = pd.read_csv(path_train_position)
df_global = pd.read_csv(path_global_position)
df_test = pd.read_csv(path_test_position)

label_to_color = label_color_class.label_to_color
# load all segmentation files
path_global='/Users/david/Documents/thesis/Thesis_code/data_collection/data_global'

path_global_rgb = os.path.join(path_global, 'rgb')
path_global_depth = os.path.join(path_global, 'depth')
cam_pam_path = os.path.join(path_global, 'cam')
path_global_annotation = os.path.join(path_global, 'preds')
path_vis = os.path.join(path_global, 'vis')
path_vis_pcd = os.path.join(path_global, 'pcd')
files = os.listdir(path_global_annotation)
# file end with .json
files = [file for file in files if file.endswith('.json')]
files.sort()
# ignore the first 10 files and the last 10 files
# files = files[20:]
# files = files[:-20]
count = 0
image_id = 0
segments_id = 0
viewer = o3d.visualization.Visualizer()
viewer.create_window()
#opt = viewer.get_render_option()
#opt.show_coordinate_frame = False
#opt.background_color = np.asarray([0, 0, 0])

# output_labels =  'segments/global_dataset/labels_database.csv'
# output_segments = 'segments/global_dataset/segments_database.csv'
# output_timestamps = 'segments/global_dataset/timestamps_database.csv'
# output_positions = 'segments/global_dataset/positions_database.csv'
# # if not exists, create the folder
# if not os.path.exists('segments/global_dataset'):
#     os.makedirs('segments/global_dataset')
# # Specify the CSV file name
# with open(output_labels, mode='w') as file:
#     pass
# with open(output_segments, mode='w') as file:
#     pass
# with open(output_timestamps, mode='w') as file:
#     pass
# with open(output_positions, mode='w') as file:
#     pass

for file_anno in tqdm(files):
    count += 1
    # continue each 8 files
    #if count%3 == 0:
        #continue
    # stop after 10 files
    # load annotation file[0] to test the code
    with open(os.path.join(path_global_annotation,file_anno), 'r') as file:
        annotation = json.load(file)

    # find rgb and depth image with the same file name with annotation file without extension and after second '.'
    timestamp = file_anno.split('.')[0].split('_')[1] + '.' + file_anno.split('.')[1].split('.')[0]
    if float(timestamp) != 1701885960.9507787:
        continue
    print('find the timestamp')
    file_name = file_anno.split('.')[0] + '.' + file_anno.split('.')[1].split('.')[0]
    file_name_rgb = file_name + '.png'
    file_name_depth = 'depth_' + file_name.split('_')[1] + '.png'
    # load rgb and depth image
    bgr = cv2.imread(os.path.join(path_global_rgb, file_name_rgb)) # bgr

    #cv2.waitKey(100)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(path_global_depth, file_name_depth), -1)
    cv2.imshow('depth', depth)
    # load camera parameters
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

    mask_rgb = np.zeros_like(rgb)
    segments_rgb_list = []
    segments_label_list = []
    segments_score_list = []
    segments_pcd_list = []

    masks = annotation['masks']
    bboxes = annotation['bboxes']
    labels = annotation['labels']
    scores = annotation['scores']

    df_row = find_nearest_time_row(df_global, float(timestamp))
    camera_position = np.array([df_row['pose.position.x'], df_row['pose.position.y'], df_row['pose.position.z']]) 
    camera_orientation = np.quaternion(df_row['pose.orientation.w'], df_row['pose.orientation.x'], df_row['pose.orientation.y'], df_row['pose.orientation.z']) 
    pcd_real = get_pcd_from_rgbd(rgb, depth, depth_intrin, depth_scale, filter_black_points=True)
    pcd_real = pcd_real.voxel_down_sample(voxel_size=0.08)
    # translation and rotation
    # adjust the camera position and orientation
    # pcd_real = cam_to_world_o3d(camera_position, camera_orientation, pcd_real)
    # save pcd with the same name with annotation file
    #o3d.io.write_point_cloud(os.path.join(path_vis_pcd, file_name + '.pcd'), pcd_real)
    #viewer.add_geometry(pcd_real)
    #opt = viewer.get_render_option()
    #opt.background_color = np.asarray([0, 0, 0])
    #viewer.run()
    segments_recount = 0
    for mask, label, score in zip(masks, labels, scores):

        if score < 0.5: # threshold to filter out low-confidence instances
            continue
        color = label_to_color.get(label)
        mask_decode = mask_util.decode(mask)
        mask_rgb[mask_decode > 0] = color
        segment_rgb = np.zeros_like(rgb)  
        #segment_rgb[mask_decode > 0] = rgb[mask_decode > 0]
        segment_rgb[mask_decode > 0] = color
        segment_pcd = get_pcd_from_rgbd(segment_rgb, depth, depth_intrin, depth_scale, filter_black_points=True)
        

        # continue when the pcd is not empty
        if segment_pcd.is_empty():
            continue
        #segment_pcd = segment_pcd.voxel_down_sample(voxel_size=0.02) 
        #pcd_in_world = cam_to_world(camera_position, camera_orientation, np.asarray(segment_pcd.points))
        pcd_in_world2 = cam_to_world_o3d(camera_position, camera_orientation, segment_pcd)
        
        # if np.asarray(pcd_in_world2.points)[:,2].mean() < 0.1:
        #     continue
        #pcd_in_world = cluster_and_filter_points(pcd_in_world, size_threshold=1500)

        # color normarlization
        #color = np.asarray(color) / 255.0
        #pcd.paint_uniform_color(color)
        # if len(np.asarray(pcd_in_world2.points)) < 2500:
        #     continue
        pcd_in_world2 = pcd_in_world2.voxel_down_sample(voxel_size=0.06) 
        # add pcd each 20 frames
            # down sample pcd
        #o3d.io.write_point_cloud(os.path.join('data_collection/global_data/pcd_semantic_map', file_name + '_' + str(segments_recount) + '_'+ str(label) + '.pcd'), pcd_in_world2)
        
        downpcd = pcd_in_world2
        #viewer.add_geometry(downpcd)
        #print('pcd size', len(downpcd.points))

        #viewer.add_geometry(downpcd)
        # opt = viewer.get_render_option()

        # opt.background_color = np.asarray([0, 0, 0])
        # viewer.run()
        segments_pcd_list.append(np.asarray(downpcd.points))
        #print('shape', np.asarray(downpcd.points).shape)

        segments_recount += 1
        segments_id += 1
        segments_rgb_list.append(segment_rgb)
        segments_label_list.append(label)
        segments_score_list.append(score)
    segment_pcd = get_pcd_from_rgbd(mask_rgb, depth, depth_intrin, depth_scale, filter_black_points=True)
    segment_pcd = segment_pcd.voxel_down_sample(voxel_size=0.06) 
    viewer.add_geometry(segment_pcd)
   # opt = viewer.get_render_option()
    cache = segment_pcd
    pcd_in_world2 = cam_to_world_o3d(camera_position, camera_orientation, cache)
    #pcd_in_world2 = pcd_in_world2.voxel_down_sample(voxel_size=0.06) 
    viewer.add_geometry(pcd_in_world2)
    o3d.io.write_point_cloud(os.path.join('data_collection/data_global/pcd_semantic_map', 'picked_frame_real.pcd'), segment_pcd)
    # window for cv2
    # cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    # cv2.imshow('vis', rgb)
    image_id += 1


opt = viewer.get_render_option()
#opt.show_coordinate_frame = True
opt.background_color = np.asarray([0, 0, 0])
viewer.run()
viewer.destroy_window()