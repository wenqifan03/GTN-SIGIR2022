'''
## GTN-pytorch
"Graph Trend Filtering Networks for Recommendations", Accepted by SIGIR'2022.
Pytorch Implementation of GTN in Graph Trend Networks for Recommendations
The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch

@inproceedings{fan2022graph,
  title={Graph Trend Filtering Networks for Recommendations},
  author={Fan, Wenqi and Liu, Xiaorui and Jin, Wei and Zhao, Xiangyu and Tang, Jiliang and Li, Qing},
  booktitle={International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2022}
}

'''



import argparse
import torch

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Go GTN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")  # 512 1024 2048 4096
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")

    parser.add_argument('--epochs', type=int, default=1000)  # 1000, ...

    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing, 100")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="gtn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    parser.add_argument('--prop_dropout', type=float, default=0.1)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--ogb', type=bool, default=True)
    parser.add_argument('--incnorm_para', type=bool, default=True)

    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--alpha1', type=float, default=0.25)
    parser.add_argument('--alpha2', type=float, default=0.25)

    parser.add_argument('--lambda2', type=float, default=4.0) #2, 3, 4,...

    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate:0.001")  # 0.001
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [gowalla,  last-fm, yelp2018, amazon-book]")
    parser.add_argument('--model', type=str, default='gtn', help='rec-model, support [gnt, lgn]')
    parser.add_argument('--avg', type=int, default=0)
    parser.add_argument('--recdim', type=int, default=256,
                        help="the embedding size of GTN: 128, 256")
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--gcn_model', type=str,
                        default='GTN', help='GTN')
    return parser.parse_args()
