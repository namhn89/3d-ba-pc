import numpy as np
import argparse
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='independent', help='dump folder path [independent]')

parser.add_argument('--add_num', type=int, default=512, help='number of added points [default: 512]')
parser.add_argument('--target', type=int, default=5, help='target class index')
parser.add_argument('--constraint', default='c', help='type of constraint. h for Hausdoff; c for Chamfer')
parser.add_argument('--lr_attack', type=float, default=0.01, help='learning rate for optimization based attack')

parser.add_argument('--initial_weight', type=float, default=5000, help='initial value for the parameter lambda')#200 for Hausdorff
parser.add_argument('--upper_bound_weight', type=float, default=40000, help='upper_bound value for the parameter lambda')#900 for Hausdorff
parser.add_argument('--step', type=int, default=10, help='binary search step')
parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')