from hamcrest import none
import numpy as np
import yaml
import open3d
import open3d as o3d
from pathlib import Path
import os
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering


from config.config import load_config_data


def get_rgb_list(_label):

    c = color_dict[_label]


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




################################______MAIN______################################


points_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2/demo_lidar_input/') # path to .bin data
label_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2/demosave/') # path to .label data
# points_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2/inference_cluster/lidar_pc_val/') # path to .label data
# label_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2/inference_cluster/lidar_labels/') # path to .bin data

label_filter = [40, 48, 70, 72]    # object's label which you wan't to show
with open('/home/mnabail/repos/Cylinder3D_spconv_v2/config/label_mapping/nuscenes.yaml', 'r') as stream: # label_mapping configuration file
    label_mapping = yaml.safe_load(stream)
    color_dict = label_mapping['color_map']

#print('----------------------------------------------------------------------------------')
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
    print("BIN NAME=",str(it.stem))
    points_file = points_dir / (str(it.stem) + '.bin')
    labels = np.fromfile(label_file, dtype=np.uint32)
    points = np.fromfile(points_file, dtype=np.float32).reshape((-1, 5))[:, 0:3]


    for i in labels_16:
        if(i!=4):
            continue
        points_to_draw=points[labels==i]
        # labels_to_draw=labels[labels==i]

        # db=DBSCAN(eps=1,min_samples=10).fit(points_to_draw)
        # db=OPTICS(min_samples=10, max_eps=10,eps=2,metric='euclidean').fit(points_to_draw)
        #db=Birch(n_clusters=8).fit(points_to_draw)
        #db= AgglomerativeClustering(linkage='single', distance_threshold=3,n_clusters=None).fit(points_to_draw)
        db= AgglomerativeClustering(linkage='single', distance_threshold=3,n_clusters=None).fit(points_to_draw)


        labels_db=np.array(db.labels_)
        n_clusters=np.unique(labels_db).shape[0]
        bounding_boxes=[]
        for i in np.unique(labels_db):
            if(i==-1):
                continue
            cluter_points=points_to_draw[labels_db==i]
            cluster_pc = o3d.geometry.PointCloud()
            cluster_pc.points = o3d.utility.Vector3dVector(cluter_points)
            #get bounding box
            bounding_box=cluster_pc.get_axis_aligned_bounding_box()
            bounding_box.color = (1, 0, 0)
            bounding_boxes.append(bounding_box)


        visgeom = o3d.visualization.Visualizer()
        visgeom.create_window()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # create coordinate frame
        visgeom.add_geometry(mesh_frame)
        pc=o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_to_draw)
        pc.paint_uniform_color([0.1, 0.1, 0.9])
        visgeom.add_geometry(pc)
        for box in bounding_boxes:
            visgeom.add_geometry(box)
        visgeom.run()
        visgeom.destroy_window()
