import argparse

from model import DeepFM_mp
import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
import os
import json


from lsecp import LSECP
from ckpt_with_pickle import CkptWithPickle
from ckpt_with_rocksdb import CkptWithRocksdb
from tracker import *


def run():
    
    parser = argparse.ArgumentParser(
        description="Train DeepFM Model"
    )
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--dataset-path", type=str, default="/mnt/ssd/dataset/kaggle/train_sample.txt")
    
    #ckpt methods: diff, incre, rocksdb, lsecp
    parser.add_argument("--ckpt-method", type=str, default="lsecp")
    #ckpt freq: checkpoint for every {ckpt_freq} iterations
    parser.add_argument("--ckpt-freq", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="/mnt/ssd/deepfm")
    parser.add_argument("--lsecp-eperc", type=float, default=0.01)
    parser.add_argument("--lsecp-clen", type=int, default=10)
    
    # the file name to store perf data
    parser.add_argument("--perf-out-path", type=str, default="")

    args = parser.parse_args()
    
    
    #load dataset 
    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_feature + sparse_feature
    # data_kaggle = pd.read_csv('/mnt/ssd/dataset/kaggle/train_sample.txt', names=col_names, sep='\t')
    data_kaggle = pd.read_csv(args.dataset_path, names=col_names, sep='\t')
    # data_kaggle = data_kaggle.iloc[:100000,:]
    data_X_ka = data_kaggle.iloc[:,1:]
    data_y_ka = data_kaggle.label.values
    data_X_ka = data_X_ka.apply(LabelEncoder().fit_transform)
    fields_ka = data_X_ka.max().values + 1

    tmp_X_ka, test_X_ka, tmp_y_ka, test_y_ka = train_test_split(data_X_ka, data_y_ka, test_size = 0.2, random_state=42, stratify=data_y_ka)
    train_X_ka, val_X_ka, train_y_ka, val_y_ka = train_test_split(tmp_X_ka, tmp_y_ka, test_size = 0.25, random_state=42, stratify=tmp_y_ka)

    train_X_ka = torch.from_numpy(train_X_ka.values).long()
    val_X_ka = torch.from_numpy(val_X_ka.values).long()
    test_X_ka = torch.from_numpy(test_X_ka.values).long()

    train_y_ka = torch.from_numpy(train_y_ka).long()
    val_y_ka = torch.from_numpy(val_y_ka).long()
    test_y_ka = torch.from_numpy(test_y_ka).long()

    train_set_ka = Data.TensorDataset(train_X_ka, train_y_ka)
    val_set_ka = Data.TensorDataset(val_X_ka, val_y_ka)
    train_loader_ka = Data.DataLoader(dataset=train_set_ka,
                                batch_size=128,
                                shuffle=True)
    # val_loader_ka = Data.DataLoader(dataset=val_set_ka,
    #                             batch_size=32,
    #                             shuffle=False)
    
    
    epoches = 1
    model = DeepFM_mp.DeepFM(feature_fields=fields_ka, embed_dim=1024, mlp_dims=(32, 16), dropout=0.0)
    
    emb_names = []
    for name in model.state_dict().keys():
        if "emb" in name:
            emb_names.append(name)
    
    track = Tracker()
    ckpt_sys = None
    ckpt_path = args.ckpt_dir
    if args.ckpt_method == "lsecp":
        ckpt_path = args.ckpt_dir + "/lsecp"
        ckpt_sys = LSECP(ckpt_path, emb_names, args.lsecp_eperc, args.lsecp_clen)
    elif args.ckpt_method == "diff":
        ckpt_path = args.ckpt_dir + "/diff"
        ckpt_sys = CkptWithPickle(ckpt_path, emb_names)
    elif args.ckpt_method == "incre":
        ckpt_path = args.ckpt_dir + "/incre"
        ckpt_sys = CkptWithPickle(ckpt_path, emb_names)
    elif args.ckpt_method == "rocksdb":
        ckpt_path = args.ckpt_dir + "/rocksdb"
        ckpt_sys = CkptWithRocksdb(ckpt_path, emb_names)
        
    perf_count = {}
    do_perf = False
    if args.perf_out_path != "":
        perf_count = {
            "emb_time": [],
            "mlp_time": [],
            "storage_consumption": []
        }
        do_perf = True
        
    
    for epoch in range(epoches):
        train_loss = []
        criterion = nn.BCELoss(reduction='mean')
        # optimizer = optim.Adam(model.parameters(), lr = 0.001)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
        model.train()
        
        for iter, (x, y) in enumerate(train_loader_ka):
            if iter >= args.num_batches:
                break
            
            y = y.to("cuda:0")
            pred = model(x)
            indexes = x.transpose(0,1)
            loss = criterion(pred, y.float().detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # x = torch.tensor(x)
            
            if iter == 0 :
                # checkpoint base model
                torch.save(model.state_dict(), ckpt_path + "/base.pt")
            elif iter % args.ckpt_freq == 0:
                # checkpoint diff view
                track.add(indexes)
                diff_view = track.cur_view()
                
                emb_start = time.time()
                ckpt_sys.ckpt_emb(diff_view, model, iter)
                emb_time = time.time() - emb_start
                
                # save non-emb params
                # user-defined space
                mlp_start = time.time()
                ckpt_non_emb(model, ckpt_path, iter)
                mlp_time = time.time() - mlp_start
                
                if do_perf:
                    perf_count["emb_time"].append(emb_time)
                    perf_count["mlp_time"].append(mlp_time)
                    
                if args.ckpt_method != "diff":
                    track.reset()
            else :
                # do track only
                track.add(indexes)
    
    if args.ckpt_method == "lsecp" or args.ckpt_method == "rocksdb":
        ckpt_sys.finish()

    if do_perf:
        storage_con = get_folder_size(ckpt_path)
        perf_count["storage_consumption"].append(storage_con / (1024*1024))
        
        with open(args.perf_out_path, "w") as json_file:
            json.dump(perf_count, json_file, indent=4)
    
    

if __name__ == "__main__":
    run()