import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("data\dataset5_r180\\raftstereo\ptcloud_sum_predictions\\raft_ptcloud_predictions.ply")
print("Downsample the point cloud with a voxel of 0.01")
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
o3d.visualization.draw_geometries([downpcd],zoom=0.1,
                                  front=[200, 200, 300],
                                  lookat=[412,612,100],
                                  up=[-0.0694, -0.9768, 0.2024])