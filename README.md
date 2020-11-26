# Point Cloud Backdoor Attacks

## Requirement 
This is tested with environment `quangbd/dc-miniconda:3-cuda10.0-cudnn7-ubuntu16.04`

Other required packages include numpy, joblib, sklearn, open3d etc.

## Install dependencies
```
sh install.sh 
```
## Prepare dataset and pretrained model
ModelNet40 : [link](https://drive.google.com/file/d/1KJBDpP7H_o2hjN1DTGKaf9djBBxT_dry/view?usp=sharing)

ScanObjectNN : [link](https://drive.google.com/file/d/1fRrTadhl98IUxZlRyiQCY9d2zGuiIHS5/view?usp=sharing)

You will need download data , unzip files and put all directoy into `data/`

Your data folder structure: 
```
└── data/
    └── h5_files/
    └── modelnet40_ply_hdf5_2048/
```

The pretrained model can be downloaded [here]()
## Usage 

There are two python scripts for train clean model: 
- train/train_cls.py 
- train/train_cls_scan.py

There are three python scripts for different attacks: 
- backdoor_attack/perturbation.py -- Point Perturbations 
- backdoor_attack/duplicate.py -- Duplicate Points
- backdoor_attack/local.py -- Local Points 

The code logics of these scripts are similar. 

The basic usage: 
```
python backdoor_attack/perturbation.py --random
```

Other parameters can be founded in the script, or run `python backdoor_attack/perturbation.py -h`. 
## Other files 
- critical_point/ -- contains files to represent critical point in model. 
- visualization/ -- used to visualize point cloud data by mitsuba2, matplotlib, open3d
- point_cloud_analysis/ -- used saliency map, grad_cam to analysis property of point
- defense/ -- some defense method for backdoor attack in point cloud

## Evaluation
To evaluate backdoor model, you need download pretrained model by link above and run scripts. 
```
sh scripts/eval_perturbation.sh
sh scripts/eval_duplicate.sh
sh scripts/eval_local.sh
```
## Visualization 
Run `python critical_point/critical_pointnet.py`
 
You will get output like:
 
![](images/example.gif)



## Detail dataset 
- OBJ_BG / OBJ_ONLY: *objectdataset.h5
- PB_T25: *objectdataset_augmented25_norot.h5
- PB_T25_R: *objectdataset_augmented25rot.h5
- PB_T50_R: *objectdataset_augmentedrot.h5
- PB_T50_RS: *objectdataset_augmentedrot_scale75.h5
# Run