## How to extract road marking guide will be available soon.

## Data in folders
- `lidar_data/` put in it the raw lidar bins of nuscenes having structure of (x,y,z,intensity,ring), so each bin have size of N x 5, where N is the number of points in the lidar scan.
- `model_load_dir_nuscenes/` put in the weights of the trained model, name must be `model_weight.pt` ,  you can change the path though from `config/nuScenes.yaml` 
- After inference, `lidar_data_labels_all/` folder will have a label for each lidar bin, the label mapping can be found in `config/label_mapping/nuscenes.yaml`,  the labels used in this yaml file are `labels_16`
- After running the `lane_marking_segmentation.py` script, the `lidar_data_labels_road_marking/` folder will have label files also, but with label `1` for road marking and `0` for non-road marking.

## How to run the code
- install requirements `pip install -r requirements.txt`
- run  `python inference_nuscenes.py` to invoke cylinderical 3d model, labels will be saved in `lidar_data_labels_all/` folder
- 

