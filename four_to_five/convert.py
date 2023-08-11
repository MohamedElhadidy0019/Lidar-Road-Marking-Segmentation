import numpy as np
import open3d as o3d

points_file='/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/four_to_five/conti.bin'
points = np.fromfile(points_file, dtype=np.float32).reshape((-1, 4))
# add column of zeros
points = np.hstack((points, np.zeros((points.shape[0], 1))))
points.astype(np.float32).tofile('/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/four_to_five/conti_5.bin')

points_file_again='/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/four_to_five/conti_5.bin'
points_again = np.fromfile(points_file_again, dtype=np.float32).reshape((-1, 5))
points=points_again

# points_c=np.c_[points, np.zeros((points.shape[0], 1))]
# zeros_column=np.zeros((points.shape[0], 1))
# points_c=np.append(points,zeros_column,axis=1)
# points_c=np.array(points_c)
# print(points_c.shape)
# points_c.astype(np.float32).tofile('/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/four_to_five/waymo_points_converted.bin')
# print(points.shape)
# print('SHAPE =',points_c.shape)

# ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
o3d.visualization.draw_geometries([pcd])
# downpcd = pcd.voxel_down_sample(voxel_size=1)

# downpcd to numpy array
# points_downpcd = np.asarray(downpcd.points)
# zeros_column=np.zeros((points_downpcd.shape[0], 1))
# points_downpcd=np.append(points_downpcd,zeros_column,axis=1)
# points_downpcd=np.append(points_downpcd,zeros_column,axis=1)
# print(points_downpcd.shape)
# points_downpcd.astype(np.float32).tofile('/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/four_to_five/down_sampled.bin')
# o3d.visualization.draw_geometries([downpcd])
