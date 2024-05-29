import argparse

from model import PNN_mp
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

from incrcp import IncrCP
from naive_ckpt import NaiveCkpt
from tracker import *


def run():
    
    parser = argparse.ArgumentParser(
        description="Train PNN Model"
    )
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--dataset-path", type=str, default="/mnt/ssd/dataset/kaggle/train_sample.txt")
    
    #ckpt methods: naive_diff, naive_incre, incrcp
    parser.add_argument("--ckpt-method", type=str, default="incrcp")
    #ckpt freq: checkpoint for every {ckpt_freq} iterations
    parser.add_argument("--ckpt-freq", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="/mnt/3dx/checkpoint")
    parser.add_argument("--eperc", type=float, default=0.01)
    parser.add_argument("--concat", type=int, default=0)
    parser.add_argument("--incrcp-reset-thres", type=int, default=100)
    
    # the file name to store perf data
    parser.add_argument("--perf-out-path", type=str, default="./perf_res/incrcp.json")

    args = parser.parse_args()
    
    
    #load dataset 
    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_feature + sparse_feature
    data_kaggle = pd.read_csv(args.dataset_path, names=col_names, sep='\t')
    # data_kaggle = pd.read_csv('../data/data.csv', names=col_names, sep='\t')
    # data_kaggle = data_kaggle.iloc[:100000,:]
    data_X_ka = data_kaggle.iloc[:,1:]
    data_y_ka = data_kaggle.label.values
    data_X_ka = data_X_ka.apply(LabelEncoder().fit_transform)
    fields_ka = data_X_ka.max().values + 1 # 模型输入的feature_fields

    #train, validation, test 集合
    tmp_X_ka, test_X_ka, tmp_y_ka, test_y_ka = train_test_split(data_X_ka, data_y_ka, test_size = 0.2, random_state=42, stratify=data_y_ka)
    train_X_ka, val_X_ka, train_y_ka, val_y_ka = train_test_split(tmp_X_ka, tmp_y_ka, test_size = 0.25, random_state=42, stratify=tmp_y_ka)


    # 数据量小, 可以直接读
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
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PNN_mp.PNN(feature_fields=fields_ka, embed_dim=1024, mlp_dims=(32, 16), dropout=0.0)
    
    emb_names = []
    for name in model.state_dict().keys():
        if "emb" in name:
            emb_names.append(name)
    
    tracker = Tracker()
    ckpt_sys = None
    ckpt_base = True
    ckpt_path = args.ckpt_dir + "/" + args.ckpt_method
    if args.ckpt_method == "incrcp":
        ckpt_sys = IncrCP(ckpt_path, emb_names, args.eperc, args.concat, args.incrcp_reset_thres, 0)
    elif args.ckpt_method == "diff":
        ckpt_sys = NaiveCkpt(ckpt_path, emb_names)
    elif args.ckpt_method == "naive_incre":
        ckpt_sys = NaiveCkpt(ckpt_path, emb_names)

    perf_count = {}
    do_perf = False
    if args.perf_out_path != "":
        perf_count = {
            "save_time": [],
            "list_time": [],
            "ckpt_time": [],
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
            pred = model(x) # x shape: [32, 39]
            indexes = x.transpose(0,1) # [39,32]
            loss = criterion(pred, y.float().detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # x = torch.tensor(x)
            
            if iter % args.ckpt_freq == 0:
                if ckpt_base:
                    ckpt_base = False
                    base_name = ckpt_path + "/base." + str(iter) + ".pt"
                    # checkpoint base model
                    if args.ckpt_method == "diff":
                        ckpt_sys.diff_hist = [1]
                        tracker.reset()
                    ckpt_time = time.time()
                    torch.save(model.state_dict(), base_name)
                    ckpt_time = time.time() - ckpt_time
                    save_time, list_time = 0, 0
                else :
                    # checkpoint diff view
                    tracker.add(indexes)
                    diff_view = tracker.cur_view()
                    
                    ckpt_time = time.time()
                    fraction, save_time, list_time = ckpt_sys.ckpt_emb(diff_view, model, iter)
                    ckpt_time = time.time() - ckpt_time
                    
                    if args.ckpt_method != "diff":
                        tracker.reset()
                    if args.ckpt_method != "naive_incre":
                        # may reset baseline ckpt
                        ckpt_base = ckpt_sys.may_reset_base(fraction)
                
                if do_perf:
                    perf_count["save_time"].append(save_time)
                    perf_count["list_time"].append(list_time)
                    perf_count["ckpt_time"].append(ckpt_time)
                    storage_con = get_folder_size(ckpt_path)
                    perf_count["storage_consumption"].append(storage_con / (1024*1024))
            else :
                # do track only
                tracker.add(indexes)
    
    if args.ckpt_method == "incrcp":
        ckpt_sys.finish()
    if do_perf:
        with open(args.perf_out_path, "w") as json_file:
            json.dump(perf_count, json_file, indent=4)
    
    

if __name__ == "__main__":
    run()