#!/bin/bash
python ba_local.py --model pointnet_cls --radius 0.05 --random
python ba_local.py --model pointnet_cls --radius 0.05 --random --dataset scanobjectnn_pb_t50_rs --epochs 500