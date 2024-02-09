import open3d as o3d
import numpy as np
import copy

print("Load a ply point cloud, print it, and render it")
# pcd_sum = o3d.geometry.PointCloud()
# for i in range(63):

# Set Bounding corners
bounding_polygon = np.asarray([[200, 400.0,  0.0],
                                [200.0, 800.0,  0.0],
                                [600.0, 400.0,  0.0],
                                [600.0, 800.0,  0.0],
                                [600.0, 400.0,  200.0],
                                [600.0, 800.0,  200.0],
                                [200.0, 400.0,  200.0],
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

pcd = o3d.io.read_point_cloud("data\dataset16\\r300\\raftstereo\\rs_pcd_pred\\rs_ptcloud_predictions_tsdf.ply")
pcd=pcd.scale(1000,[0,0,0])
pcd = vol.crop_point_cloud(pcd)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.1)
pcd = pcd.select_by_index(ind)


pcd1 = o3d.io.read_point_cloud("data\dataset12\\r200\\raftstereo\\raft_sfw_pcd_pred\\raft_ptcloud_predictions_tsdf.ply")
# pcd = pcd.voxel_down_sample(voxel_size=1.0)
print(pcd)
print(np.asarray(pcd.points))
    # pcd_sum +=pcd
o3d.visualization.draw_geometries([pcd])



# ,
#                                   front = [1, 0.0414, 0 ],
#                                   lookat = [ 412.5, 620, 100 ],
#                                   up = [ 0, 0, 1 ],
#                                   zoom = 0.3

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 539.59262339430802, 799.99501279730612, 119.21364385293309 ],
# 			"boundingbox_min" : [ 261.10660847708607, 451.77388489048508, 59.586545767357364 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.89970716683280705, 0.041028146217382266, 0.43456150907282243 ],
# 			"lookat" : [ 406.19359610003977, 622.62875294164121, 101.80672975512287 ],
# 			"up" : [ 0.43593958247350678, 0.034297717823011463, 0.8993221597319605 ],
# 			"zoom" : 0.2999999999999996
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 544.22018119812105, 799.98876576498878, 119.90445081831091 ],
# 			"boundingbox_min" : [ 258.06669232998638, 463.04666312346308, 53.855198907307397 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.78065245672715911, -0.20495713097193946, 0.59040182610625658 ],
# 			"lookat" : [ 412.5, 612.5, 100.0 ],
# 			"up" : [ 0.60218365308948618, 0.0060600662771313485, 0.79833459373154958 ],
# 			"zoom" : 0.40000000000000002
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }


# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 526.2615867434356, 715.15872397376916, 152.93170358102958 ],
# 			"boundingbox_min" : [ 289.48634002990684, 522.52110965677593, 100.0000410800549 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.0, 0.0, 1.0 ],
# 			"lookat" : [ 407.87396338667122, 618.83991681527255, 126.46587233054224 ],
# 			"up" : [ 0.0, 1.0, 0.0 ],
# 			"zoom" : 0.39999999999999969
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }