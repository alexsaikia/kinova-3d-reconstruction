import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import argparse

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj



def main(args):
    # Set Bounding corners
    bounding_polygon = np.asarray([[200, 400.0,  00.0],
                          [200.0, 800.0,  00.0],
                          [600.0, 400.0,  00.0],
                          [600.0, 800.0,  00.0],
                          [200.0, 400.0,  400.0],
                          [200.0, 800.0,  400.0],
                          [600.0, 400.0,  400.0],
                          [600.0, 800.0,  400.0]]).astype("float64")
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
    
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=848,
                                                          height=480,
                                                          fx=422.119,
                                                          fy=422.119,
                                                          cx=416.549,
                                                          cy=240.874)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.0005,
                                                          sdf_trunc=0.001,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                                                          )
    
    # Set the pointcloud paths
    ply_dir = Path(args.ply_dir)
    ply_paths = sorted(list(ply_dir.iterdir()))
    out_dir = Path(args.output_directory)
    depth_dir = Path(args.depth_dir)
    
    (ply_dir.parent/'camera_poses').mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(str(ply_dir.parent.parent/'poses' /
                         'HE_poses.csv'))
    pcd_sum = o3d.geometry.PointCloud()
    #Delete camera pose file if it exists
    (ply_dir.parent/'camera_poses/camera_poses.txt').unlink(missing_ok=True)
    
    idx = 0
    for input_file in tqdm(ply_paths):
        color = o3d.io.read_image(str(sorted(list((ply_dir.parent.parent/'rgb').iterdir()))[idx]))
        depth = o3d.io.read_image(str(sorted(list((depth_dir).iterdir()))[idx]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=0.4, convert_rgb_to_intensity=False)
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
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10.0)
        pcd = pcd.select_by_index(ind)
        pcd_sum += pcd
        o3d.io.write_point_cloud(str(out_dir/input_file.name), pcd)
        
        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(np.matmul(np.array([[1, 0, 0, trans[0]/1000], 
                                              [0, 1, 0, trans[1]/1000], 
                                              [0, 0, 1, trans[2]/1000], 
                                              [0, 0, 0, 1]]).astype(np.float32),r))
            )
        
        with open(str(ply_dir.parent/'camera_poses/camera_poses.txt'), "a") as f:
            f.write(f'0 0 {idx}\n')
            np.savetxt(f, np.matmul(np.array([[1, 0, 0, trans[0]/1000], [0, 1, 0, trans[1]/1000], [
                     0, 0, 1, trans[2]/1000], [0, 0, 0, 1]]).astype(np.float32),r), delimiter=' ')
        f.close()
        # o3d.io.write_point_cloud(str(out_dir / ply_dir.name)+f'{idx}.ply', pcd_sum)
        idx = idx + 1
    o3d.io.write_point_cloud(str(out_dir / ply_dir.name)+'.ply', pcd_sum)
    o3d.io.write_point_cloud(str(out_dir / ply_dir.name)+'_tsdf.ply',volume.extract_point_cloud())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_dir')
    parser.add_argument('--output_directory')
    parser.add_argument('--depth_dir')
    #,nargs='?',const='/home/alex/data/dataset5_r180/raftstereo/raft_ptcloud_predictions',type=str,default='/home/alex/data/dataset5_r180/raftstereo/raft_ptcloud_predictions')
    SystemExit(main(parser.parse_args()))
