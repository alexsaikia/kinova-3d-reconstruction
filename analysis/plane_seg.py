import open3d as o3d
import numpy as np

print("Load a ply point cloud")
pcd = o3d.io.read_point_cloud("data\dataset5_r180\\raftstereo\ptcloud_sum_predictions\\cropped_raft_ptcloud_predictions.ply")

plane_model, inliers = pcd.segment_plane(distance_threshold=1.0,
                                         ransac_n=5,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
