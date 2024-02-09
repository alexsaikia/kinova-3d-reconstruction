# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import os
import sys
import argparse
from pathlib import Path


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
    dir = Path(args.ply_dir)
    camera_poses = read_trajectory(str(dir/'camera_poses/camera_poses.txt'))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=848,
                                                          height=480,
                                                          fx=422.119,
                                                          fy=422.119,
                                                          cx=416.549,
                                                          cy=240.874)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.00025,
        sdf_trunc=0.001,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(str(sorted(list((dir.parent/'rgb').iterdir()))[i]))
        depth = o3d.io.read_image(str(sorted(list((dir/'depth_predictions').iterdir()))[i]))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=0.3, convert_rgb_to_intensity=False)
        
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,camera_intrinsics)
        # o3d.visualization.draw_geometries([pcd])
        
        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(camera_poses[i].pose),
        )
# np.linalg.inv(camera_poses[i].pose)
    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    o3d.visualization.draw_geometries([voxel_pcd])

    # print("Extract voxel-aligned debugging voxel grid")
    # voxel_grid = volume.extract_voxel_grid()
    # o3d.visualization.draw_geometries([voxel_grid])

    print("Extract point cloud")
    pcd = volume.extract_point_cloud()
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_dir',
                        nargs='?',
                        const='data\dataset6\\r200\\raftstereo',
                        type=str,
                        default='data\dataset6\\r200\\raftstereo'
                        )
    SystemExit(main(parser.parse_args()))
