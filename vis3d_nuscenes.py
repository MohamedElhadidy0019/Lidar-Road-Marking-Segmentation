import numpy as np
import yaml
import open3d
from pathlib import Path
import os

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


points_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/demo_lidar_input/') # path to .bin data
label_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2_LANDMARKINGS/demosave/') # path to .label data
# points_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2/inference_cluster/lidar_pc_val/') # path to .label data
# label_dir = Path('/home/mnabail/repos/Cylinder3D_spconv_v2/inference_cluster/lidar_labels/') # path to .bin data

label_filter = [40, 48, 70, 72]    # object's label which you wan't to show
with open('./config/label_mapping/nuscenes.yaml', 'r') as stream: # label_mapping configuration file
    label_mapping = yaml.safe_load(stream)
    color_dict = label_mapping['color_map']

#print('----------------------------------------------------------------------------------')
for it_scan, it_label in zip(sorted(points_dir.iterdir()), sorted(label_dir.iterdir())):
    bin_name=str(it_scan).split('/')[-1]
    label_name=str(it_label).split('/')[-1]
    old_name=str(it_label)
    new_name=str(label_dir)+'/'+bin_name[:-4]+'.label'
    os.rename(old_name,new_name)
    # print("bin name=",bin_name, "  label name=",label_name)
    # print('old name: ',old_name)
    # print("new name=",new_name)
    # print('--------------------------------------------')

# input("break now")

#list of 50 different rgb colors

for it in sorted(label_dir.iterdir()):
    label_file = it
    print("BIN NAME=",str(it.stem))
    points_file = points_dir / (str(it.stem) + '.bin')
    label = np.fromfile(label_file, dtype=np.uint32)
    points = np.fromfile(points_file, dtype=np.float32).reshape((-1, 5))[:, 0:3]
    print("points shape=",points.shape)
    print("label shape=",label.shape)



    #print(points.shape)
    #print(label.shape, points.shape)
    colorful_points = concate_color(points, label)
    draw_pc(colorful_points)
    #print("=============================================================================")
    #break