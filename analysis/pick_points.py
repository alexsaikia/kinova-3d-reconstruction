import open3d as o3d
import numpy as np

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("data\LaserScans\KukaTable.ply")
pcd = pcd.voxel_down_sample(voxel_size=0.01)
print(pcd)
print(np.asarray(pcd.points))
g = pick_points(pcd)
print(g)