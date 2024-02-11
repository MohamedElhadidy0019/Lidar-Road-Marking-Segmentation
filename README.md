## Lidar Road Marking Segmentation
#### This repo is based on cylinderical 3d paper and code, this repo is actually a fork of cylinderical 3d repo

#### This code is tested on Ubuntu 20.04 and CUDA version 11.4
#### To run the code on newer CUDA version, you may need to check the new updates in the original repo of cylinderical 3d model(the one we forked from)

#### we trained the cylindircal 3d model on nuscenes, intially it was trained only on KITTI.
#### Our approach is tested on nuscenes dataset only as it is the one that have the drivable surface labels

#### Model weights link: https://drive.google.com/file/d/1mCXL2INlabwm7ExOvhgponQUBQNM6HKf/view?usp=sharing

## Data in folders
- `lidar_data/` put in it the raw lidar bins of nuscenes having structure of (x,y,z,intensity,ring), so each bin have size of N x 5, where N is the number of points in the lidar scan.
- `model_load_dir_nuscenes/` put in the weights of the trained model, name must be `model_weight.pt` ,  you can change the path though from `config/nuScenes.yaml` 
- After inference, `lidar_data_labels_all/` folder will have a label for each lidar bin, the label mapping can be found in `config/label_mapping/nuscenes.yaml`,  the labels used in this yaml file are `labels_16`
- After running the `lane_marking_segmentation.py` script, the `lidar_data_labels_road_marking/` folder will have label files also, but with label `1` for road marking and `0` for non-road marking.

## How to run the code
- install requirements `conda create --name <env> --file environment.yml`
- `conda activate <env>`
- run  `python inference_nuscenes.py` to invoke cylinderical 3d model, labels will be saved in `lidar_data_labels_all/` folder
- run `python landmarks.py`, pointcloud and its labels will be saved in `lidar_data_labels_road_marking/` and visualisation images will be saved in `output_vis_folder/`



<!-- add gif -->

### End result (red is road marking)
![Alt Text](all_output/output.gif)




