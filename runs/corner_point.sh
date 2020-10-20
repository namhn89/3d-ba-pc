#!/bin/bash
python ba_attack.py --random --attack_method corner_point --model pointnet_cls
python ba_attack.py --random --attack_method corner_point --model pointnet_cls --dataset scanobjectnn_pb_t50_rs --epochs 500