#!/bin/bash
python ba_local.py --model pointnet_cls --radius 0.01 --random
python ba_local.py --model pointnet_cls --radius 0.01 --random --dataset scanobjectnn_pb_t50_rs --epochs 500