

import numpy as np
import copy
import open3d as o3d




def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


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


def demo_manual_registration():
    print("Demo for manual ICP")
    source = o3d.io.read_point_cloud("data\LaserScans\dataset12.ply").voxel_down_sample(voxel_size=0.5)
    target = o3d.io.read_point_cloud("data\dataset12\\r200\\raftstereo\\raft_sfw_pcd_pred\\raft_ptcloud_predictions.ply").voxel_down_sample(voxel_size=0.5)
    cl, ind = target.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.1)
    target = target.select_by_index(ind)
    cl,ind = target.remove_radius_outlier(nb_points=100,radius=3)
    target = target.select_by_index(ind)
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))
    print(trans_init)
    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 1  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")
    


if __name__ == "__main__":
    demo_manual_registration()