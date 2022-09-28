import os
from pathlib import Path
from matplotlib.pyplot import annotate
import numpy as np
from torch.utils import data
import yaml
import pickle

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import numba as nb
from torch.utils import data
from dataloader.dataset_semantickitti import register_dataset


import warnings

warnings.filterwarnings("ignore")



def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)



def preprocess_pointcloud(data_tuple):
    grid_size=np.array([480,360,2])
    ignore_label=0
    data_pt_cloud,sig=data_tuple
    sig=sig.reshape(sig.shape[0],)
    #sig=data_pt_cloud[:, :3]
    xyz=data_pt_cloud[:, :3]
    xyz_pol = cart2polar(xyz)
    max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
    min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
    max_bound = np.max(xyz_pol[:, 1:], axis=0)
    min_bound = np.min(xyz_pol[:, 1:], axis=0)
    max_bound = np.concatenate(([max_bound_r], max_bound))
    min_bound = np.concatenate(([min_bound_r], min_bound))

    max_bound=np.array([50. , 3.1415926, 3.])
    min_bound=np.array([0. , -3.1415926, -5.])

    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = grid_size
    intervals = crop_range / (cur_grid_size - 1)


    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

    voxel_position = np.zeros(grid_size, dtype=np.float32)
    dim_array = np.ones(len(grid_size) + 1, int)
    dim_array[0] = -1
    voxel_position = np.indices(grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
    voxel_position = polar2cat(voxel_position)


        # process labels
    # processed_label = np.ones(grid_size, dtype=np.uint8) * ignore_label
    # label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
    # label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
    # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
    data_tuple = (voxel_position, )

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
    return_xyz = xyz_pol - voxel_centers
    return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

    return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

    return return_fea,grid_ind






    pass










@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz




################################### MAIN CODE #######################################

def main(args):
    POINTS_DIR=Path('demo_lidar_input/')
    OUTPUT_PATH=Path('demosave/')
    model_load_path = Path['model_load_path']

    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    with open(dataset_config["label_mapping"], 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)
    learning_map = nuscenesyaml['learning_map']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)


    counter=0
    for it in sorted(POINTS_DIR.iterdir()):
        points_file = POINTS_DIR / (str(it.stem) + '.bin')
        print(str(points_file))
        raw_data = np.fromfile(points_file, dtype=np.float32).reshape((-1, 5))
        points_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        data_tuple = (raw_data[:, :3], points_label.astype(np.uint8),raw_data[:, 3:5])
        print(raw_data.shape)
        val_pt_fea,val_grid=preprocess_pointcloud(data_tuple)
        #print("SHAPES=", np.array(val_pt_fea).shape," , ",np.array(val_grid).shape)
        val_pt_fea=np.reshape(val_pt_fea,(1,val_pt_fea.shape[0],val_pt_fea.shape[1]))
        val_grid=np.reshape(val_grid,(1,val_grid.shape[0],val_grid.shape[1]))

        #print("SHAPES= ",val_pt_fea.shape, " , ", val_grid.shape)

        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]


        predict_labels = my_model(val_pt_fea_ten, val_grid_ten, 1)
        predict_labels = torch.argmax(predict_labels, dim=1)
        predict_labels = predict_labels.cpu().detach().numpy()
        for count, i_val_grid in enumerate(val_grid):
            inv_labels = np.vectorize(learning_map.__getitem__)(predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]])
            inv_labels = inv_labels.astype('uint32')
            output_path='demosave/'
            outputPath = output_path + str(counter).zfill(6) + '.label'
            inv_labels.tofile(outputPath)
            print("save " + outputPath)
            #print("\n\n BIN_NAME: ", bin_name, "\n\n")
        counter+=1





if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)


