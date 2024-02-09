import open3d as o3d
import numpy as np

print("Load a ply point cloud")
pcd = o3d.io.read_point_cloud("data\dataset12\\r300\\raftstereo\\raft_sfw_pcd_pred\\raft_ptcloud_predictions_tsdf.ply")
corners = np.asarray([[ 0.0, 350.0,  70.0],
 [ 150.0, 850.0,  70.0],
 [ 650.0, 350.0,  70.0],
 [ 650.0, 850.0,  70.0],
 [ 150.0, 350.0,  400.0],
 [ 150.0, 850.0,  400.0],
 [ 650.0, 350.0,  400.0],
 [ 650.0, 850.0,  400.0]])

# Convert the corners array to have type float64
bounding_polygon = corners

# Create a SelectionPolygonVolume
vol = o3d.visualization.SelectionPolygonVolume()

# You need to specify what axis to orient the polygon to.
# I choose the "Y" axis. I made the max value the maximum Y of
# the polygon vertices and the min value the minimum Y of the
# polygon vertices.
vol.orthogonal_axis = "Y"
vol.axis_max = np.max(bounding_polygon[:, 1])
vol.axis_min = np.min(bounding_polygon[:, 1])

# Set all the Y values to 0 (they aren't needed since we specified what they
# should be using just vol.axis_max and vol.axis_min).
bounding_polygon[:, 1] = 0

# Convert the np.array to a Vector3dVector
vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

# Crop the point cloud using the Vector3dVector
cropped_pcd = vol.crop_point_cloud(pcd)

# Get a nice looking bounding box to display around the newly cropped point cloud
# (This part is optional and just for display purposes)
bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (1, 0, 0)

# Draw the newly cropped PCD and bounding box
o3d.visualization.draw_geometries([cropped_pcd])
# o3d.io.write_point_cloud("data\dataset5_r180\\raftstereo\ptcloud_sum_predictions\\cropped_raft_ptcloud_predictions.ply",cropped_pcd)
print("hi")