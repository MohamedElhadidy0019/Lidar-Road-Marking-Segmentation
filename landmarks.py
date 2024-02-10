from cv2 import threshold
import numpy as np
import yaml
import open3d
from pathlib import Path
import os
import cv2
import nthresh
import open3d as o3d
import matplotlib.pyplot as plt


from config.config import load_config_data


def get_rgb_list(_label):

    # c = color_dict[_label]
    if(_label):
        c = [255,0,0]
    else:
        c = [0,0,0]

    return np.array((c[0], c[1], c[2]))


def draw_pc(pc_xyzrgb):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)

    def custom_draw_geometry_with_key_callback(pcd):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 1
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        open3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

    custom_draw_geometry_with_key_callback(pc)


def concate_color(_points, _label):
    color = np.zeros((_points.shape[0], 3))
    label_id = np.unique(_label)
    label_filter = [40, 48, 70, 72]    # object's label which you wan't to show
    # print(label_id)
    # print('---------------------------------------------')
    pass
    for cls in label_id:
        if label_filter.__len__() == 0:
            color[_label == cls] = get_rgb_list(cls)
        elif label_filter.count(cls) == 0:
            color[_label == cls] = get_rgb_list(cls) # kol el 7agat ely leha el label bta3 l unique label cls , ta5d el loon bta3ha
    _points = np.concatenate([_points, color], axis=1)
    return _points

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False


    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image

    opt = o3d.vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


import nthresh
import open3d as o3d
def ring_local_thresholding(points_to_threshold,vis_bool=False, vis_save=False,bin_name=None,points_full=None):
    '''
    points_to_threshold: numpy array of shape (n,5) where n is the number of points and 5 is the x,y,z,intensity,ring
                         these points should be the drivable surface points only
    vis_bool: boolean to visualize the results
    vis_save: boolean to save the results
    bin_name: name of the bin file to save the results
    points_full: numpy array of shape (n,5) where n is the number of points and 5 is the x,y,z,intensity,ring
                    these points should be the full point cloud
    
    returns :  numpy array of the road marks points           
    '''
    list_land_marks_points = []
    np_list_land_marks_points = []
    list_return = []
    n_rings=32

    # get the threshold for each ring
    for i in range(n_rings):
        ring_points=points_to_threshold[points_to_threshold[:,4]==i]
        intensity=ring_points[:,3]
        if(intensity.shape[0]<2):
            continue
        try:
            # get the threshold for each ring using Otsu's method
            threshold = nthresh.nthresh(intensity, n_classes=2, bins=255, n_jobs=1)
        except:
            continue
        # get points of intenisty higher than theh threshold, these points are the road marking points
        binary_intensity=intensity>(threshold[0]+10)
        land_marking_points=ring_points[binary_intensity]
        if(land_marking_points.shape[0]<2):
            continue

        list_return.append(land_marking_points)
        pc_land_marks = o3d.geometry.PointCloud()
        pc_land_marks.points = o3d.utility.Vector3dVector(land_marking_points[:,:3])
        pc_land_marks.paint_uniform_color([1,0, 0])
        list_land_marks_points.append(pc_land_marks)
        np_list_land_marks_points.append(land_marking_points[:,:3])


    # the following code is to visualize the results using open3d library
    if(vis_bool or vis_save):
        visgeom = o3d.visualization.Visualizer()
        visgeom.create_window()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # create coordinate frame
        visgeom.add_geometry(mesh_frame)
        # for box in bounding_boxes:
        #     visgeom.add_geometry(box)
        points_full=points_full[:,:3]+np.array([0,0,-0.5])
        pc_full=o3d.geometry.PointCloud()
        pc_full.points = o3d.utility.Vector3dVector(points_full[:,:3])
        pc_full.paint_uniform_color([0.8,0.8, 0.8])
        visgeom.add_geometry(pc_full)
        for pc in list_land_marks_points:
            visgeom.add_geometry(pc)


        #camera
        ctr = visgeom.get_view_control()
        ctr.set_front([0, -3, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.5)

        # if (vis_bool):
        #     visgeom.run()
        #     visgeom.destroy_window()

        # if (vis_save):
        save3DPath='./output_vis_folder/'
        visgeom.capture_screen_image( save3DPath + "/" + str(bin_name)+ ".jpg", do_render=True)

    return np_list_land_marks_points



def global_thresholding(points_to_threshold,vis_bool=False,bounding_box=False,bin_name=None,points_full=None,labels_full=None):

    list_land_marks_points = []
    np_list_land_marks_points = []
    list_return = []
    bounding_boxes=[]
    n_rings=32

    intensity=points_to_threshold[:,3]
    intensity=(intensity/100.0)*255.0
    intensity=np.array(intensity,dtype=np.float32)
    threshold = nthresh.nthresh(intensity, n_classes=2, bins=255, n_jobs=1)
    binary_intensity=intensity>(threshold[0])
    land_marking_points=points_to_threshold[binary_intensity]



    if(vis_bool):
        visgeom = o3d.visualization.Visualizer()
        visgeom.create_window()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # create coordinate frame
        visgeom.add_geometry(mesh_frame)
        if(bounding_box):
            for box in bounding_boxes:
                visgeom.add_geometry(box)
        points_full=points_full[:,:3]+np.array([0,0,-0.5])
        pc_full=o3d.geometry.PointCloud()
        pc_full.points = o3d.utility.Vector3dVector(points_full[:,:3])
        pc_full.paint_uniform_color([0.8,0.8, 0.8])
        visgeom.add_geometry(pc_full)

        land_marking_pc=o3d.geometry.PointCloud()
        land_markings_xyz=land_marking_points[:,:3]+np.array([0,0,-0.5])
        land_marking_pc.points = o3d.utility.Vector3dVector(land_markings_xyz)
        land_marking_pc.paint_uniform_color([1,0, 0])

        visgeom.add_geometry(land_marking_pc)



        #camera
        ctr = visgeom.get_view_control()
        ctr.set_front([0, -3, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.04)

        visgeom.run()
        visgeom.destroy_window()


        # save3DPath='/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/o3d_output_conti_old/'
        # visgeom.capture_screen_image( save3DPath + "/" + str(bin_name)+ ".jpg", do_render=True)




def draw_grey_scale(ground_points):
    '''
    ground_points: (5*n) numpy array, the ground points only

    description: draws the
    '''

    vis_points=ground_points[:,:3]

    intensity=ground_points[:,3]
    # intensity=intensity+0.001
    color_of_points=(np.stack([intensity, intensity, intensity], axis=1)/np.max(intensity).astype(np.float32))

    background_color=np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_points)
    pcd.colors = o3d.utility.Vector3dVector(color_of_points)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()





def draw_grey_scale_ground_w_whole_scene(ground_points,whole_pc):
    '''
    ground_points: (5*n) numpy array, the ground points only

    whole_pc: (5*n) numpy array, the whole point cloud

    description:
        Draws the full point cloud with the background black and the points white accroding to its intensity [0-1]
    '''

    # temp=ground_points[ ground_points[:,3]>=0.1  ]

    # ground_points=ground_points[ ground_points[:,3]<0.1  ]
    vis_points=ground_points[:,:3]
    vis_points+=np.array([0,0,0.5])

    intensity=ground_points[:,3]

    print('max=',np.max(intensity))


    # color_of_points=(np.stack([intensity, intensity, intensity], axis=1)/np.max(intensity).astype(np.float32))
    color_of_points=(np.stack([intensity, intensity, intensity], axis=1))
    # color_of_points=(  (np.stack([intensity, intensity, intensity], axis=1)/(0.09)) .astype(np.float32))

    background_color=np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_points)
    pcd.colors = o3d.utility.Vector3dVector(color_of_points)

    pcd_whole_pc = o3d.geometry.PointCloud()
    pcd_whole_pc.points = o3d.utility.Vector3dVector(whole_pc[:,:3])

    intensity_whole_pc=whole_pc[:,3]
    intensity_whole_pc= np.stack([intensity_whole_pc, intensity_whole_pc, intensity_whole_pc], axis=1)/np.max(intensity_whole_pc)
    intensity_whole_pc=intensity_whole_pc.astype(np.float32)
    pcd_whole_pc.colors = o3d.utility.Vector3dVector(intensity_whole_pc)
    # pcd_whole_pc.paint_uniform_color([0.2,0, 0])
    # pcd_whole_pc.paint_uniform_color(np.stack([intensity_whole_pc, intensity_whole_pc, intensity_whole_pc], axis=1))



    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0.2, 0.2, 0.2])
    # vis.add_geometry(pcd)
    vis.add_geometry(pcd_whole_pc)
    vis.run()
    vis.destroy_window()

def extract_landmarks( pointlcoud:np.array, labels:np.array, name:str):
    '''
    pointlcoud: (5*n) numpy array, the whole point cloud

    labels: (n) numpy array, the labels of the whole point cloud

    description:
        Extracts the landmarks from the point cloud and returns them as a numpy array

    returns:
        points_np : same as pointcloud but with different order
        new_labels_np : label of point 1 -> landmark
                                       0 -> not landmark
    '''


    drivable_surface_points = pointlcoud[labels==11]


    new_labels_list = []
    points_list = []

    drivable_surface_points = pointlcoud[labels==11]

    n_rings = 32

    for i in range(n_rings):
        
        ring_points = drivable_surface_points[drivable_surface_points[:,4]==i]
        intensity=ring_points[:,3]
        if(intensity.shape[0]<2):
            continue
        try:
            # get the threshold for each ring using Otsu's method
            threshold = nthresh.nthresh(intensity, n_classes=2, bins=255, n_jobs=1)
        except:
            continue

        ring_labels = intensity > (threshold[0]+10)
        # map true and false to 1 and 0
        ring_labels = ring_labels.astype(np.uint32)
        new_labels_list.append(ring_labels)
        points_list.append(ring_points)
    
    non_drivable_surface_points = pointlcoud[labels!=11]
    non_drivable_surface_labels = np.zeros(non_drivable_surface_points.shape[0]).astype(np.uint32)

    new_labels_list.append(non_drivable_surface_labels)
    points_list.append(non_drivable_surface_points)

    new_labels_np = np.concatenate(new_labels_list).astype(np.uint32)
    
    points_np = np.concatenate(points_list)


    visualiser = o3d.visualization.Visualizer()
    visualiser.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # create coordinate frame
    visualiser.add_geometry(mesh_frame)

    points_l = points_np[new_labels_np==1]
    points_o = points_np[new_labels_np==0]

    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(points_l[:,:3])
    pcd_l.paint_uniform_color([1,0, 0])

    pcd_o = o3d.geometry.PointCloud()
    pcd_o.points = o3d.utility.Vector3dVector(points_o[:,:3])
    pcd_o.paint_uniform_color([0.8,0.8, 0.8])

    visualiser.add_geometry(pcd_l)
    visualiser.add_geometry(pcd_o)

    # save
    save3DPath='./output_vis_folder/'
    visualiser.capture_screen_image( save3DPath + "/" + name+ ".jpg", do_render=True)


    return points_np, new_labels_np
    

def main():

    points_dir = Path('./lidar_data/') # path to .bin data
    label_dir = Path('./lidar_data_labels_all/') # path to .label data
    save_dir = Path('./lidar_data_labels_road_marking/') # path to save the results

    with open('./config/label_mapping/nuscenes.yaml', 'r') as stream: # label_mapping configuration file
        label_mapping = yaml.safe_load(stream)
        color_dict = label_mapping['color_map']

    # rename the label names to be the same as the poincloud name
    for it_scan, it_label in zip(sorted(points_dir.iterdir()), sorted(label_dir.iterdir())):
        bin_name=str(it_scan).split('/')[-1]
        label_name=str(it_label).split('/')[-1]
        old_name=str(it_label)
        new_name=str(label_dir)+'/'+bin_name[:-4]+'.label'
        os.rename(old_name,new_name)


    config_path='config/nuScenes.yaml'
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    with open(dataset_config["label_mapping"], 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)
    labels_16 = nuscenesyaml['labels_16']


    dataset_config = configs['dataset_params']
    for it in sorted(label_dir.iterdir()):

        label_file = it
        points_file = points_dir / (str(it.stem) + '.bin')
        labels = np.fromfile(label_file, dtype=np.uint32)

        # x, y, z, inetnsity, ring_index
        points_full = np.fromfile(points_file, dtype=np.float32).reshape((-1, 5))

        new_points, new_labels_np = extract_landmarks(points_full, labels, str(it.stem))
        # new_labels_np as int
        new_labels_np = new_labels_np.astype(np.uint32)
        # new_labels_np as float
        # new_labels_np = new_labels_np.astype(np.float32)
        # save new_points
        new_points.tofile(save_dir / (str(it.stem) + '.bin'))
        # save new_labels_np
        new_labels_np.tofile(save_dir / (str(it.stem) + '.label'))


# main
if __name__ == '__main__':
    main()
