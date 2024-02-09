import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import argparse

def pairwise_registration(source, target, max_correspondence_distance_coarse,
                          max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in tqdm(range(n_pcds)):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id],
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def main(args):
    voxel_size = 0.5
    
    # Set Bounding corners
    bounding_polygon = np.asarray([[200, 300.0,  100.0],
                                    [200.0, 800.0,  100.0],
                                    [600.0, 300.0,  100.0],
                                    [600.0, 800.0,  100.0],
                                    [600.0, 300.0,  200.0],
                                    [600.0, 800.0,  200.0],
                                    [200.0, 300.0,  200.0],
                                    [200.0, 800.0,  200.0]
                                    ]).astype("float64")
    # Create a SelectionPolygonVolume
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Y"
    vol.axis_max = np.max(bounding_polygon[:, 1])
    vol.axis_min = np.min(bounding_polygon[:, 1])

    # Set all the Y values to 0 (they aren't needed since we specified what they
    # should be using just vol.axis_max and vol.axis_min).
    bounding_polygon[:, 1] = 0

    # Convert the np.array to a Vector3dVector
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

    # Set the pointcloud paths
    ply_dir = Path(args.ply_dir)
    ply_paths = sorted(list(ply_dir.iterdir()))
    (ply_dir.parent/'ptcloud_sum_predictions').mkdir(parents=True, exist_ok=True)
    (ply_dir.parent/'camera_poses').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(str(ply_dir.parent.parent/'poses' /
                            'infra1_optical_frame_poses.csv'))
    idx = 0
    pcd_sum = o3d.geometry.PointCloud()
    pcds = []
    for input_file in tqdm(ply_paths):
        pcd = (o3d.io.read_point_cloud(str(input_file),
                print_progress=True)).remove_non_finite_points()
        r = R.from_quat(np.array(df.loc[idx, 'Q_x':'Q_w']))
        r = (np.reshape(np.append(np.append(np.array(r.as_matrix()), np.array([0, 0, 0])[
            :, np.newaxis], axis=1), [[0, 0, 0, 1]]), (4, 4)).astype(np.float32))
        trans = 1000 * np.array(df.loc[idx, 'X':'Z'])
        t = np.array([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [
                        0, 0, 1, trans[2]], [0, 0, 0, 1]]).astype(np.float32)
        pcd.transform(r)
        pcd.transform(t)
        #Crop to bounding box
        pcd = vol.crop_point_cloud(pcd)
                
        #Remove statistical outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.1)
        pcd = pcd.select_by_index(ind).voxel_down_sample(voxel_size)
        pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
        pcd_sum += pcd
        pcds.append(pcd)
        # o3d.io.write_point_cloud(str(
        #     ply_dir.parent/'ptcloud_sum_predictions'/input_file.name), pcd, print_progress=True)
        # np.savetxt(str(ply_dir.parent/'camera_poses'/input_file.name).replace('.ply','.txt'), np.matmul(t,r), delimiter=' ')
        idx = idx + 1
    # o3d.io.write_point_cloud(str(ply_dir.parent/'ptcloud_sum_predictions' /
    #                             ply_dir.name)+'.ply', pcd_sum, print_progress=True)
    
     
    
     # Set Bounding corners
    bounding_polygon = np.asarray([[200, 300.0,  100.0],
                                    [200.0, 800.0,  100.0],
                                    [600.0, 300.0,  100.0],
                                    [600.0, 800.0,  100.0],
                                    [600.0, 300.0,  200.0],
                                    [600.0, 800.0,  200.0],
                                    [200.0, 300.0,  200.0],
                                    [200.0, 800.0,  200.0]
                                    ]).astype("float64")
    # Create a SelectionPolygonVolume
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Y"
    vol.axis_max = np.max(bounding_polygon[:, 1])
    vol.axis_min = np.min(bounding_polygon[:, 1])

    # Set all the Y values to 0 (they aren't needed since we specified what they
    # should be using just vol.axis_max and vol.axis_min).
    bounding_polygon[:, 1] = 0
    
    # Convert the np.array to a Vector3dVector
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
    
    # Set Bounding corners
    # source_bounding_polygon = np.asarray([[370.0, 570.0,  100.0],
    #                                     [370.0, 660.0,  100.0],
    #                                     [450.0, 570.0,  100.0],
    #                                     [450.0, 660.0,  100.0],
    #                                     [450.0, 570.0,  200.0],
    #                                     [450.0, 660.0,  200.0],
    #                                     [370.0, 570.0,  200.0],
    #                                     [370.0, 660.0,  200.0]
    #                                     ]).astype("float64")
    
    # Set Bounding corners
    source_bounding_polygon = np.asarray([[370.0, 570.0,  100.0],
                                        [370.0, 660.0,  100.0],
                                        [450.0, 570.0,  100.0],
                                        [450.0, 660.0,  100.0],
                                        [450.0, 570.0,  200.0],
                                        [450.0, 660.0,  200.0],
                                        [370.0, 570.0,  200.0],
                                        [370.0, 660.0,  200.0]
                                        ]).astype("float64")
    
    
    # Create a SelectionPolygonVolume
    source_vol = o3d.visualization.SelectionPolygonVolume()
    source_vol.orthogonal_axis = "Y"
    source_vol.axis_max = np.max(source_bounding_polygon[:, 1])
    source_vol.axis_min = np.min(source_bounding_polygon[:, 1])

    # Set all the Y values to 0 (they aren't needed since we specified what they
    # should be using just vol.axis_max and vol.axis_min).
    source_bounding_polygon[:, 1] = 0
    
    # Convert the np.array to a Vector3dVector
    source_vol.bounding_polygon = o3d.utility.Vector3dVector(source_bounding_polygon)
    
    # trans_init = np.array([[ 9.38561123e-01, -3.45032982e-01, -7.43367834e-03, -3.39084303e+02],
    #                                      [ 3.45045542e-01,  9.38585841e-01,  4.38515902e-04,  3.21847313e+02],
    #                                      [ 6.82584279e-03, -2.97653155e-03,  9.99972274e-01,  7.88710666e+01],
    #                                      [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # print("Demo for manual ICP")
    # target = o3d.io.read_point_cloud("data\LaserScans\SpinePhantom.ply").voxel_down_sample(voxel_size=0.5)
    # target = target.transform(trans_init)
    # target = vol.crop_point_cloud(target)
    # target.estimate_normals(
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    # source = o3d.io.read_point_cloud("data\dataset6\\r200\\raftstereo\ptcloud_sum_predictions\\raft_ptcloud_predictions.ply").voxel_down_sample(voxel_size=0.5)
    # cl, ind = source.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.1)
    # source = source.select_by_index(ind)
    # source = source_vol.crop_point_cloud(source)
    # o3d.visualization.draw(pcds)
    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 5
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as mm:
        pose_graph = full_registration(pcds,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as mm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and display")
    for point_id in range(len(pcds)):
        print(pose_graph.nodes[point_id].pose)
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    # o3d.visualization.draw(pcds)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_dir',
                        nargs='?',
                        const='/home/alex/data/dataset5_r180/raftstereo/raft_ptcloud_predictions',
                        type=str,
                        default='/home/alex/data/dataset5_r180/raftstereo/raft_ptcloud_predictions'
                        )
    SystemExit(main(parser.parse_args()))