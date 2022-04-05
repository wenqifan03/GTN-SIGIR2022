'''
## GTN-pytorch
"Graph Trend Filtering Networks for Recommendations", Accepted by SIGIR'2022.
Pytorch Implementation of GTN in Graph Trend Filtering Networks for Recommendations
The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch

@inproceedings{fan2022graph,
  title={Graph Trend Filtering Networks for Recommendations},
  author={Fan, Wenqi and Liu, Xiaorui and Jin, Wei and Zhao, Xiangyu and Tang, Jiliang and Li, Qing},
  booktitle={International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2022}
}
'''



import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys

sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['gowalla', 'yelp2018', 'amazon-book', 'last-fm']
all_models = ['mf', 'gtn', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['K'] = args.K
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['args'] = args
config['dataset'] = args.dataset
config['epochs'] = args.epochs
config['lambda2'] = args.lambda2

GPU = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
