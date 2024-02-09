import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(
    "data\dataset8\\r300\\raftstereo\ptcloud_sum_predictions\\rs_ptcloud_predictions.ply")
pcd = pcd.voxel_down_sample(voxel_size=0.5)
print("Statistical oulier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.1)
pcd = pcd.select_by_index(ind)
# cl,ind = pcd.remove_radius_outlier(nb_points=100,radius=3)
# pcd = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd])

