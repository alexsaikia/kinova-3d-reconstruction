#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate testing

mkdir /home/alex/data/dataset12/r200/raftstereo/
CUDA_VISIBLE_DEVICES=0 python /home/alex/RAFT-Stereo/demo.py \
                        --restore_ckpt /home/alex/RAFT-Stereo/models/raftstereo-sceneflow.pth \
                        --save_numpy \
                        -l '/home/alex/data/dataset12/r200/left_rect_raw/*.png' \
                        -r '/home/alex/data/dataset12/r200/right_rect_raw/*.png' \
                        --output_directory /home/alex/data/dataset12/r200/raftstereo/

python /home/alex/RAFT-Stereo/reconstruct_disparity_rs.py /home/alex/data/dataset12/r200/raftstereo/disparity --save_depth --save_ptcloud 
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r200/raftstereo/rs_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r200/raftstereo/rs_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r200/aligned_depth
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r200/raftstereo/raft_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r200/raftstereo/raft_sfw_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r200/raftstereo/depth_predictions


mkdir /home/alex/data/dataset12/r225/raftstereo/
CUDA_VISIBLE_DEVICES=0 python /home/alex/RAFT-Stereo/demo.py \
                        --restore_ckpt /home/alex/RAFT-Stereo/models/raftstereo-sceneflow.pth \
                        --save_numpy \
                        -l '/home/alex/data/dataset12/r225/left_rect_raw/*.png' \
                        -r '/home/alex/data/dataset12/r225/right_rect_raw/*.png' \
                        --output_directory /home/alex/data/dataset12/r225/raftstereo/

python /home/alex/RAFT-Stereo/reconstruct_disparity_rs.py /home/alex/data/dataset12/r225/raftstereo/disparity --save_depth --save_ptcloud
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r225/raftstereo/rs_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r225/raftstereo/rs_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r225/aligned_depth
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r225/raftstereo/raft_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r225/raftstereo/raft_sfw_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r225/raftstereo/depth_predictions


mkdir /home/alex/data/dataset12/r250/raftstereo/
CUDA_VISIBLE_DEVICES=0 python /home/alex/RAFT-Stereo/demo.py \
                        --restore_ckpt /home/alex/RAFT-Stereo/models/raftstereo-sceneflow.pth \
                        --save_numpy \
                        -l '/home/alex/data/dataset12/r250/left_rect_raw/*.png' \
                        -r '/home/alex/data/dataset12/r250/right_rect_raw/*.png' \
                        --output_directory /home/alex/data/dataset12/r250/raftstereo/

python /home/alex/RAFT-Stereo/reconstruct_disparity_rs.py /home/alex/data/dataset12/r250/raftstereo/disparity --save_depth --save_ptcloud
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r250/raftstereo/rs_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r250/raftstereo/rs_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r250/aligned_depth
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r250/raftstereo/raft_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r250/raftstereo/raft_sfw_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r250/raftstereo/depth_predictions


mkdir /home/alex/data/dataset12/r275/raftstereo/
CUDA_VISIBLE_DEVICES=0 python /home/alex/RAFT-Stereo/demo.py \
                        --restore_ckpt /home/alex/RAFT-Stereo/models/raftstereo-sceneflow.pth \
                        --save_numpy \
                        -l '/home/alex/data/dataset12/r275/left_rect_raw/*.png' \
                        -r '/home/alex/data/dataset12/r275/right_rect_raw/*.png' \
                        --output_directory /home/alex/data/dataset12/r275/raftstereo/

python /home/alex/RAFT-Stereo/reconstruct_disparity_rs.py /home/alex/data/dataset12/r275/raftstereo/disparity --save_depth --save_ptcloud
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r275/raftstereo/rs_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r275/raftstereo/rs_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r275/aligned_depth
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r275/raftstereo/raft_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r275/raftstereo/raft_sfw_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r275/raftstereo/depth_predictions


mkdir /home/alex/data/dataset12/r300/raftstereo/
CUDA_VISIBLE_DEVICES=0 python /home/alex/RAFT-Stereo/demo.py \
                        --restore_ckpt /home/alex/RAFT-Stereo/models/raftstereo-sceneflow.pth \
                        --save_numpy \
                        -l '/home/alex/data/dataset12/r300/left_rect_raw/*.png' \
                        -r '/home/alex/data/dataset12/r300/right_rect_raw/*.png' \
                        --output_directory /home/alex/data/dataset12/r300/raftstereo/

python /home/alex/RAFT-Stereo/reconstruct_disparity_rs.py /home/alex/data/dataset12/r300/raftstereo/disparity --save_depth --save_ptcloud
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r300/raftstereo/rs_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r300/raftstereo/rs_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r300/aligned_depth
python /home/alex/RAFT-Stereo/test_colourply.py \
                        --ply_dir /home/alex/data/dataset12/r300/raftstereo/raft_ptcloud_predictions \
                        --output_directory /home/alex/data/dataset12/r300/raftstereo/raft_sfw_pcd_pred \
                        --depth_dir /home/alex/data/dataset12/r300/raftstereo/depth_predictions