#!/bin/bash
python ba_attack.py --random --mode_attack corner_point --model pointnet_cls
python ba_attack.py --random --mode_attack corner_point --model pointnet_cls --dataset scanobjectnn_pb_t50_rs