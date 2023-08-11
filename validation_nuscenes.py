# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml


from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import warnings

warnings.filterwarnings("ignore")

OUTPUT_PATH = '/scratch/perstk/repos/cylinderical3d_testing/CylinderEVALUATION/output_nuscenes/'
def main(args):
    #pytorch_device = torch.device('cuda:0')
    pytorch_device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = 1
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    with open(dataset_config["label_mapping"], 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)
    learning_map = nuscenesyaml['learning_map']

    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    # loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
    #                                                num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    if True:
        my_model.eval()
        hist_list = []
        val_loss_list = []
        counter=0
        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                    val_dataset_loader):
                if(counter>=10):
                    break
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                    val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]


                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)# ground truth

                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)  # model output
                # loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                #                         ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()

                val_label_tensor = val_label_tensor.cpu().detach().numpy()
                for count, i_val_grid in enumerate(val_grid):
                    # hist_list.append(fast_hist_crop(predict_labels[
                    #                                     count, val_grid[count][:, 0], val_grid[count][:, 1],
                    #                                     val_grid[count][:, 2]], val_pt_labs[count],
                    #                                 unique_label))


                    labels = np.vectorize(learning_map.__getitem__)(predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]])
                    labels = labels.astype('uint32')


                    ground_truth = np.vectorize(learning_map.__getitem__)(val_label_tensor[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]])
                    ground_truth = ground_truth.astype('uint32')


                    save_dir=OUTPUT_PATH
                    output_path_label = save_dir +'label_'+ str(counter).zfill(6) + '.label'
                    output_path_truth = save_dir + 'truth_'+str(counter).zfill(6) + '.label'
                    labels.tofile(output_path_label)
                    ground_truth.tofile(output_path_truth)

                    # print("save " + output_path_label)
                    #print("\n\n BIN_NAME: ", bin_name, "\n\n")
                counter+=1

        print("COUNTER = ",counter)








if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
