

import numpy as np
import copy
import open3d as o3d
from matplotlib import pyplot as plt
treg = o3d.t.pipelines.registration
from pathlib import Path
import csv
import time


def follow(thefile):
    thefile.seek(0,2) # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1) # Sleep briefly
            continue
        yield line

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
def t_draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()])

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
    # Set Bounding corners
    bounding_polygon = np.asarray([[300, 500.0,  80.0],
                                    [300.0, 700.0,  80.0],
                                    [600.0, 500.0,  80.0],
                                    [600.0, 700.0,  80.0],
                                    [600.0, 500.0,  200.0],
                                    [600.0, 700.0,  200.0],
                                    [300.0, 500.0,  200.0],
                                    [300.0, 700.0,  200.0]
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
    
    source_bounding_polygon = np.asarray([[350, 525.0,  105.0],
                                    [350.0, 675.0,  105.0],
                                    [500.0, 525.0,  105.0],
                                    [500.0, 675.0,  105.0],
                                    [500.0, 525.0,  200.0],
                                    [500.0, 675.0,  200.0],
                                    [350.0, 525.0,  200.0],
                                    [350.0, 675.0,  200.0]
                                    ]).astype("float64")
    # Create a SelectionPolygonVolume
    
    # # Set Bounding corners
    # source_bounding_polygon = np.asarray([[370.0, 570.0,  100.0],
    #                                     [370.0, 660.0,  100.0],
    #                                     [450.0, 570.0,  100.0],
    #                                     [450.0, 660.0,  100.0],
    #                                     [450.0, 570.0,  200.0],
    #                                     [450.0, 660.0,  200.0],
    #                                     [370.0, 570.0,  200.0],
    #                                     [370.0, 660.0,  200.0]
    #                                     ]).astype("float64")
    
    
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
    trans_init = np.array([[ 8.73352961e-01,  4.87037648e-01, -6.99526737e-03, -4.82389745e+02],
                            [-4.87004788e-01,  8.72848658e-01, -3.10089798e-02,  6.63276730e+02],
                            [-8.99673086e-03,  3.04885130e-02,  9.99494627e-01,  7.17985830e+01],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    print("Demo for manual ICP")
    
    
    
  
    content_list = "data\dataset12\\r200\\raftstereo\\rs_pcd_pred\\rs_ptcloud_predictions_tsdf.ply"
    path = Path(content_list)
    source = o3d.io.read_point_cloud(content_list)
    if path.parts[1] == 'dataset6':
        dataset = 'Phantom'
        target = o3d.io.read_point_cloud("data\LaserScans\SpinePhantom.ply").voxel_down_sample(voxel_size=0.05)
    elif path.parts[1] == 'dataset7':
        dataset = 'Cube'
        target = o3d.io.read_point_cloud("data\LaserScans\Rubik.ply").voxel_down_sample(voxel_size=0.25)
    elif path.parts[1] == 'dataset8':
        dataset = 'Tissue'
        target = o3d.io.read_point_cloud("data\LaserScans\Chops.ply").voxel_down_sample(voxel_size=0.25)
    elif path.parts[1] == 'dataset12':
        dataset = 'Liver'
        target = o3d.io.read_point_cloud("data\LaserScans\dataset12.ply").voxel_down_sample(voxel_size=0.05)
    elif path.parts[1] == 'dataset14':
        dataset = 'Porcine1'
        target = o3d.io.read_point_cloud("data\LaserScans\dataset14.ply").voxel_down_sample(voxel_size=0.05)
    elif path.parts[1] == 'dataset15':
        dataset = 'Porcine2'
        target = o3d.io.read_point_cloud("data\LaserScans\dataset15after.ply").voxel_down_sample(voxel_size=0.05)
    elif path.parts[1] == 'dataset16':
        dataset = 'Bovine'
        target = o3d.io.read_point_cloud("data\LaserScans\dataset16after.ply").voxel_down_sample(voxel_size=0.05)
        
    target = target.transform(trans_init)
    target = vol.crop_point_cloud(target)
    target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    
    target_temp = copy.deepcopy(target)
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.io.write_point_cloud("data\\temp\\temp_target.ply",target_temp)
    target = o3d.t.io.read_point_cloud("data\\temp\\temp_target.ply")
        
    rad = path.parts[2].replace('r','')
    if 'raft_sfw' in path.parts[4]:
        recon_type = 'RAFT-Stereo'
    elif 'rs_' in path.parts[4]:
        recon_type = 'RealSense'
        
    if '_tsdf' in content_list:
        source=source.scale(1000,[0,0,0]).voxel_down_sample(voxel_size=0.5)
        sum_type = 'TSDF'
    else:
        source=source.voxel_down_sample(voxel_size=0.5)
        sum_type = 'Summation'
        
    source = source_vol.crop_point_cloud(source)
    cl, ind = source.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.1)
    source = source.select_by_index(ind)
    source_temp = copy.deepcopy(source)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    o3d.io.write_point_cloud("data\\temp\\temp_source.ply",source_temp)
    source1 = source
    source = o3d.t.io.read_point_cloud("data\\temp\\temp_source.ply")

    print("Visualization of two point clouds before automatic alignment")
    # draw_registration_result(source, target, np.identity(4))
    print(trans_init)
    # point-to-plane ICP for refinement
    print("Perform point-to-plane ICP refinement")
    voxel_sizes = o3d.utility.DoubleVector([2.0,1.0, 0.5,0.25])
    threshold = o3d.utility.DoubleVector([6.0, 3.0, 1.5,0.75])
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=2000),
        treg.ICPConvergenceCriteria(relative_fitness=0.00001, relative_rmse=0.00001, max_iteration=1500),
        treg.ICPConvergenceCriteria(relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=1000),
        treg.ICPConvergenceCriteria(relative_fitness=0.0000001, relative_rmse=0.0000001, max_iteration=1000)
    ]
    estimation = treg.TransformationEstimationPointToPlane(treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.CauchyLoss,1.5))
    save_loss_log=True
            
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as mm:
        reg_p2p = treg.multi_scale_icp(source, 
                            target, 
                            voxel_sizes,
                            criteria_list,
                            threshold, 
                            o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32),
                            estimation,
                            save_loss_log
                            )
    
    t_draw_registration_result(source, target, reg_p2p.transformation)
  
    

if __name__ == "__main__":
    demo_manual_registration()