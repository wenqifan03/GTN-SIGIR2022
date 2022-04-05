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
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from sklearn.model_selection import train_test_split
import random

seed = 2020
import random
import numpy as np

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset,['gowalla', 'yelp2018', 'amazon-book']:
    """

    def __init__(self, config=world.config, path="../data/gowalla", flag_test=0):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        # self.pre_norm = config['pre_norm']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.args = config['args']

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        with open(test_file) as f:
            for l in f.readlines():

                if len(l) > 0:
                    l = l.strip('\n').split(' ')

                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    # items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1

        # original setting
        testUser = testUser
        testItem = testItem
        valUser = testUser
        valItem = testItem

        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        # R 
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.__valDict = self.__build_val()
        print(f"{world.dataset} is ready to go")

    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            # t = tuple(random.sample(range(0, adj.shape[0]), 2))
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def valDict(self):
        return self.__valDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            # if self.flag_test == 0:
            R = self.UserItemNet.tolil()

            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()

            norm_adj = adj_mat.tocsr()
            end = time()
            print(f"costing {end - s}s, saved mat...")

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            self.norm_adj = norm_adj
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_val(self):
        """ validation
        return:
            dict: {user: [items]}
        """
        val_data = {}
        for i, item in enumerate(self.valItem):
            user = self.valUser[i]
            if val_data.get(user):
                val_data[user].append(item)
            else:
                val_data[user] = [item]
        return val_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
