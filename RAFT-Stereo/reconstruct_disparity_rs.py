import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from pathlib import Path
from typing import Union
import argparse


def read_disp_subpix(filename):
    return (cv2.imread(str(filename), -1).astype(np.float32)/256)


def disparity_to_img3d(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Convert disparity to 3D image

    Convert disparity to 3D image format, similar to what is used to store
    ground truth information in scared. The resulting 3D image is expressed in
    the same frame of reference with the disparity, thus it cannot directly used
    to create 3D images suitable for evaluation on the provided sequence. The 
    unprojection is done using the Q matrix computed during the stereo 
    calibration and rectification phase.
    Args:
        disparity (np.ndarray): HxW disparity map float array
        Q (np.ndarray): Q matrix computed during stereo calibration and 
        rectification phase.
    Returns:
        np.ndarray: HxWx3 img3d output array. Each pixel location (u,v) encodes
        the 3D coordinates of the 3D point that projects to u,v.
    """
    assert disparity.dtype == np.float32
    disparity = np.nan_to_num(disparity)
    valid = disparity >= 0
    img3d = cv2.reprojectImageTo3D(disparity, Q)
    img3d[~valid] = np.nan
    return img3d


def save_ptcloud_as_ply(
    path: Union[Path, str], ptcloud: np.array, color=None
) -> Path:
    """save an Nx3 array as a .ply file

    This functions accept a N element pointcloud stored as an Nx3 np.array
    and stores is at as .ply. The function can store pointcloud both in 
    ascii and binary mode, according to the save_binary flag. Additionally
    if the destination folder does not exist, the function creates it and stores
    the resulting pointcloud. Currently the function does not support storing 
    of RGB information.
    Args:
        path (Union[Path, str]): path to store ptcloud as a .ply file
        ptcloud (np.array): N element pointcloud stored as a Nx3 np.array
        save_binary (bool, optional): Save .ply in binary mode. Defaults to True.

    Returns:
        Path: the input path argument as a pathlib.Path 
    """
    assert ptcloud.dtype == np.float32

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = ptcloud.shape[0]

    if color is not None:
        vertices = np.empty(n, dtype=[(
            'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        vertices = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertices['x'] = ptcloud[:, 0].astype('f4')
    vertices['y'] = ptcloud[:, 1].astype('f4')
    vertices['z'] = ptcloud[:, 2].astype('f4')
    if color is not None:
        vertices['red'] = color[:, -1].astype('u1')
        vertices['green'] = color[:, -2].astype('u1')
        vertices['blue'] = color[:, -3].astype('u1')

    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)

    ply.write(str(path))
    print(path)

    return path


def main(args):

    # left = cv2.imread('./LEFT-KID1.png'); right= cv2.imread('./RIGHT-KID1.png')
    disparity_dir = Path(args.disparity_dir)
    disparity_paths = sorted(list(disparity_dir.iterdir()))
    if args.save_depth:
        (disparity_dir.parent/'depth_predictions').mkdir(parents=True, exist_ok=True)
        (disparity_dir.parent/'visible_depth_predictions').mkdir(parents=True, exist_ok=True)
    if args.save_ptcloud:
        (disparity_dir.parent/'raft_ptcloud_predictions').mkdir(parents=True, exist_ok=True)
        (disparity_dir.parent/'rs_ptcloud_predictions').mkdir(parents=True, exist_ok=True)

    # define the Q matrix to reconstruct disparity. you can construct the Q matrix
    # based on the P(projection) matrix of the CameraInfo.csv inside the right_rect_raw directory.
    # the T_x value bellow follows the OpenCV convention and corresponds to the baseline
    # of the stereo camera expressed in metric units. The metric camera baseline in ros
    # is denoted as B. Based on this http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
    # we can compute the metric baselines for B from the CameraInfo.csv of the left camera
    # as P[0,3]/-P[0,0]

    # Also for OpenCV and a conventional stereo setup this values should be negative
    # multiply T_x(mm) by scale_factor with the baseline to change scale. if scale_factor=1
    # the depthmap is going to stored in meters. If scale_factor=1000, the depthmap is going to be expressed
    # in mm
    scale_factor = 1000
    T_x = -(-7.60071/-422.119)*scale_factor
    c_x = 416.549
    c_y = 240.874
    f = 422.119
    c_x_right = 416.549

    Q = np.array([[1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -c_x],
                 [0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -c_y],
                 [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  f],
                 [0.00000000e+00,  0.00000000e+00,  -1/T_x, (c_x-c_x_right)/T_x]])

    
    for disparity_img in disparity_paths:
        disp = read_disp_subpix(disparity_img).astype(np.float32)
        pointmap = cv2.reprojectImageTo3D(disp, Q)
        rs_depth = cv2.imread(str(disparity_img.parent.parent.parent /
                              'aligned_depth'/disparity_img.name), -1).astype(np.float32)
        rs_disp = -(f*T_x)/rs_depth
        rs_pointmap = cv2.reprojectImageTo3D(rs_disp, Q)
        if args.save_depth:
            img_clipped = np.clip(
                (pointmap[..., 2]), 0, 2**16-1).astype(np.uint16)
            vis_img_clipped = np.clip(
                (pointmap[..., 2]*256), 0, 2**16-1).astype(np.uint16)
            cv2.imwrite(str(disparity_dir.parent /
                        'depth_predictions'/disparity_img.name), img_clipped)
            cv2.imwrite(str(disparity_dir.parent/'visible_depth_predictions' /
                        disparity_img.name), vis_img_clipped)
        if args.save_ptcloud:
            save_ptcloud_as_ply(disparity_dir.parent/'raft_ptcloud_predictions'/(disparity_img.stem+'.ply'),
                                pointmap.reshape(-1, 3), color=cv2.imread(str(disparity_img.parent.parent.parent/'rgb'/disparity_img.name), -1).astype(np.uint8).reshape(-1, 3))
            save_ptcloud_as_ply(disparity_dir.parent/'rs_ptcloud_predictions'/(disparity_img.stem+'.ply'),
                                rs_pointmap.reshape(-1, 3), color=cv2.imread(str(disparity_img.parent.parent.parent/'rgb'/disparity_img.name), -1).astype(np.uint8).reshape(-1, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('disparity_dir')
    parser.add_argument('--save_depth', action='store_true', default=True)
    parser.add_argument('--save_ptcloud', action='store_true', default=True)

    SystemExit(main(parser.parse_args()))
