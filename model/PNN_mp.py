# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:37:32 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
import sys


class InnerProductNetwork(nn.Module):

    def forward(self, x):
        """
        x : (num_fields, batch_size, embed_dim)
        """
        num_fields = x.shape[1]
        row, col = [],[]
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:,row] * x[:,col], dim=2)


class OuterProductNetwork(nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        x : (batch_size, num_fields, embed_dim)
        """
        num_fields = x.shape[1]
        row, col = [],[]
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)
    
    
class PNN(nn.Module):
    """
        Product-based Neural Network
    """
    
    def create_emb(self, ln, dim, sparse=True, ndevices=4):
        self.embedding = nn.ModuleList()
        for i in range(0, len(ln)):
            n = ln[i]
            EE = nn.EmbeddingBag(n, dim, mode="sum" , sparse=sparse)
            torch.nn.init.xavier_uniform_(EE.weight.data)
            d = torch.device("cuda:" + str(i % ndevices))
            self.embedding.append(EE.to(d))
    
    def create_mlp(self, input_dim, mlp_dims, dropout):
        layers = nn.ModuleList()
        self.mlp_dims = mlp_dims
        for mlp_dim in mlp_dims:
            LL = nn.Linear(input_dim, mlp_dim)
            nn.init.xavier_uniform_(LL.weight.data)
            nn.init.zeros_(LL.bias)
            layers.append(LL)
            layers.append(nn.BatchNorm1d(mlp_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        bot_LL = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(bot_LL.weight.data)
        nn.init.zeros_(bot_LL.bias)
        layers.append(bot_LL)
        self.mlp = nn.Sequential(*layers)
        
    
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout, method = 'inner', ndevices=4):
        super(PNN, self).__init__()
        self.ndevices = ndevices
        self.feature_fields = feature_fields
        if method not in ['inner', 'outer']:
            raise ValueError ('unknown product type : %s' % method)
        else:
            self.method = method
        
        # Embedding layer
        self.create_emb(feature_fields, embed_dim, ndevices=ndevices)
        # self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim, sparse=True)
        
        self.embedding_out_dim = len(feature_fields) * embed_dim
        
        # mlp layer
        num_fields = len(feature_fields)
        input_dim = self.embedding_out_dim + (num_fields * (num_fields - 1))// 2 
        self.create_mlp(input_dim, mlp_dims, dropout)
        device_ids = range(ndevices)
        self.mlp = self.mlp.to("cuda:0")
        
        #Product layer
        if self.method == 'inner':
            self.pn = InnerProductNetwork()
        else:
            self.pn = OuterProductNetwork(num_fields, embed_dim)
        
        self.pn_replicas = replicate(self.pn, device_ids)
    
    
    def apply_emb(self, lS_i):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = torch.zeros_like(sparse_index_group_batch)
            E = self.embedding[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)
            ly.append(V)
        
        return ly
    
    def forward(self, x):
        # tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        x_list = []
        for k, _ in enumerate(self.embedding):
            d = torch.device("cuda:" + str(k % self.ndevices))
            x_list.append(x[:,k].to(d))


        
        # apply emb
        ly = self.apply_emb(x_list)
        # print(ly)
        
        if len(self.embedding) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        device_ids = range(self.ndevices)
        t_list = []
        for k, _ in enumerate(self.embedding):
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        
        emb_x_list = []
        for k in range(self.ndevices):
            emb_x = torch.stack(ly[k], dim=1)
            emb_x_list.append(emb_x)
        
        p = parallel_apply(self.pn_replicas, emb_x_list, None, device_ids)
        p0 = gather(p, 0, dim=0)
        
        emb_x_0 = gather(emb_x_list, 0, dim=0)
        mlp_in = torch.cat( [emb_x_0.view(-1, self.embedding_out_dim), p0], dim=1)
        
        mlp_out = self.mlp(mlp_in)
        
        
        x = torch.sigmoid(mlp_out.squeeze(1))
        return x
    
    