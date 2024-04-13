from segments_generator import  get_pcd_from_rgbd, save_segments_as_csv2, find_nearest_time_row, cam_to_world, save_all_info
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
:Check global_map_real.py for more information
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

output_labels =  'segments/global_mapping2/labels_database.csv'
output_segments = 'segments/global_mapping2/segments_database.csv'
output_timestamps = 'segments/global_mapping2/timestamps_database.csv'
output_positions = 'segments/global_mapping2/positions_database.csv'
# if not exists, create the folder
if not os.path.exists('segments/global_mapping2'):
    os.makedirs('segments/global_mapping2')
# Specify the CSV file name
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
    file_name = file_anno.split('.')[0] + '.' + file_anno.split('.')[1].split('.')[0]
    file_name_rgb = file_name + '.png'
    file_name_depth = 'depth_' + file_name.split('_')[1] + '.png'
    # load rgb and depth image
    bgr = cv2.imread(os.path.join(path_global_rgb, file_name_rgb)) # bgr

    #cv2.waitKey(100)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(path_global_depth, file_name_depth), -1)

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
    # print time and position and orientation
    # print('time: ', timestamp)
    # print('position: ', camera_position)
    # print('orientation: ', quaternion.as_rotation_matrix(camera_orientation))
    segments_recount = 0
    for mask, label, score in zip(masks, labels, scores):

        if score < 0.5: # threshold to filter out low-confidence instances
            continue
        color = label_to_color.get(label)
        mask_decode = mask_util.decode(mask)
        mask_rgb[mask_decode > 0] = color
        segment_rgb = np.zeros_like(rgb)  
        segment_rgb[mask_decode > 0] = color
        segment_pcd = get_pcd_from_rgbd(segment_rgb, depth, depth_intrin, depth_scale, filter_black_points=True)
        # continue when the pcd is not empty
        if segment_pcd.is_empty():
            continue
        #segment_pcd = segment_pcd.voxel_down_sample(voxel_size=0.02) 
        pcd_in_world = cam_to_world(camera_position, camera_orientation, np.asarray(segment_pcd.points))
        if pcd_in_world[:,2].mean() < 0.1:
            continue
        #pcd_in_world = cluster_and_filter_points(pcd_in_world, size_threshold=1500)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_in_world)
        # color normarlization
        color = np.asarray(color) / 255.0
        pcd.paint_uniform_color(color)
        #if len(pcd_in_world) < 2500:
            #continue
        
        # add pcd each 20 frames
            # down sample pcd
        downpcd = pcd.voxel_down_sample(voxel_size=0.06) 
        print('pcd size', len(downpcd.points))
        #downpcd.estimate_normals(
    #search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        viewer.add_geometry(downpcd)
        #update
        #viewer.run()
        ##vis = cv2.imread(os.path.join(path_vis, file_name_rgb)) # bgr
        #cv2.imshow('vis', vis)
        #cv2.waitKey(200)
        segments_pcd_list.append(np.asarray(downpcd.points))
        print('shape', np.asarray(downpcd.points).shape)
        # save_all_info(np.asarray(downpcd.points), timestamp, label, segments_id, segments_recount, image_id, camera_position,
        #               output_labels, output_segments, output_timestamps, output_positions)
        segments_recount += 1
        segments_id += 1
        segments_rgb_list.append(segment_rgb)
        segments_label_list.append(label)
        segments_score_list.append(score)
    #print('segments num', segments_pcd_list.__len__())
    # save segments as csv file
    # save when timestamp = 1701885390.3409352
    
    #save_segments_as_csv2(segments_pcd_list, timestamp, output_folder = 'segments/train')
    #save_timestamps(image_id,segments_num,timestamp)
    #save_positions(image_id,segments_id,positiion)
    #save_labels(segment_count,label_number)
    image_id += 1

opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0, 0, 0])
viewer.run()
viewer.destroy_window()