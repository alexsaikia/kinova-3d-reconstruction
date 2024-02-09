import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("data\dataset5_r180\\raftstereo\ptcloud_sum_predictions\\raft_ptcloud_predictions.ply")
alpha = 0.03
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)