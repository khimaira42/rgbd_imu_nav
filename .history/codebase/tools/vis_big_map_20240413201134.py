import open3d as o3d
import numpy as np
import os

'''
:Visualize the big map
'''

path_global='/Users/david/Documents/thesis/Thesis_code/data_collection/global_data'
path_global_pcd = os.path.join(path_global, 'pcd')
#path pcd_segments
path_pcd_segments = os.path.join(path_global, 'pcd_segments')
#path pcd_semantic_map
path_pcd_semantic_map = os.path.join(path_global, 'pcd_semantic_map')
#new path to restore the pcd
path_pcd = os.path.join(path_global, 'pcd_big_map')
#os make dir
if not os.path.exists(path_pcd):
    os.makedirs(path_pcd)
#load all pcd files
path_load = path_pcd_segments
files = os.listdir(path_load)
# file end with .pcd
files = [file for file in files if file.endswith('.pcd')]
files.sort()
pcd_big_map = o3d.geometry.PointCloud()
#load with open3d
pcd_list = []
for file in files:
    print(file)
    pcd = o3d.io.read_point_cloud(os.path.join(path_load,file))
    pcd_list.append(pcd)
#concatenate the pcd_list
    pcd_big_map += pcd
#save the pcd_big_map
o3d.io.write_point_cloud(os.path.join(path_pcd,'pcd_segments.pcd'),pcd_big_map)
#visualize the pcd_big_map
o3d.visualization.draw_geometries(pcd_list)