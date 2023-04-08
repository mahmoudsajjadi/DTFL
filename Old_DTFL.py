#============================================================================
# DTFL
# This program is Version1: Single program simulation 
# This program is for ResNet-56 , wandb.ai is used to log data
# ============================================================================

# tier naming is different from the paper. and it is in reverse order

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random
import numpy as np
import os
from collections import Counter

import time
import sys
import wandb
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from model.resnet56_7t import resnet56_SFL_local_tier_7
from model.resnet56_7t import resnet56_SFL_fedavg_base
from utils.loss import dis_corr
from utils.fedavg import aggregated_fedavg

from utils.dynamic_tier import dynamic_tier8
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10


import copy

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "DTFL_v1"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def printRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def printGreen(skk): print("\033[92m {}\033[00m" .format(skk))    


def add_args(parser):
    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--running_name', default="DTFL", type=str)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--lr_factor', default=0.95, type=float)
    parser.add_argument('--lr_patience', default=20, type=float)
    parser.add_argument('--rounds', default=2000, type=int)
    parser.add_argument('--whether_local_loss', default=True, type=bool)
    parser.add_argument('--whether_dcor', default=False, type=bool)
    parser.add_argument('--dcor_coefficient', default=0.5, type=float)  
    parser.add_argument('--tier', default=7, type=int)
    parser.add_argument('--client_epoch', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
    
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')  
    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)') # this part is from fedGKT code
    
    parser.add_argument('--model', type=str, default='resnet56_7', metavar='N',
                        help='neural network used in training')
    

    
    args = parser.parse_args()
    return args

lr_threshold = 0.0001
frac = 1        # participation of clients; if 1 then 100% clients participate

parser = argparse.ArgumentParser()
args = add_args(parser)
logging.info(args)
    
wandb.init(mode="disabled")
    
wandb.init(
    project="DTFL",
    name="DTFL",# + str(args.tier),
    config=args,
    tags="Tier1_7",
)


SFL_local_tier = resnet56_SFL_local_tier_7


if args.model == 'resnet56_7' and args.whether_local_loss:
    SFL_local_tier = resnet56_SFL_local_tier_7
    num_tiers = 7


whether_local_loss = args.whether_local_loss
whether_dcor = args.whether_dcor
dcor_coefficient = args.dcor_coefficient
tier = args.tier
client_epoch = args.client_epoch
client_epoch = np.ones(args.client_number,dtype=int) * client_epoch


client_type_percent = [0.0, 0.0, 0.0, 0.0, 1.0]

if num_tiers == 7:
    client_type_percent = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    tier = 1

client_number_tier = (np.dot(args.client_number , client_type_percent))

delay_coefficient = [16,20,32,72,256] * 100 # this used to simulate CPU profile when the program run on a single machine

delay_coefficient = list(np.array(delay_coefficient)/10)

delay_coefficient_list = [16,20,32,72,256]
delay_coefficient_list = list(np.array(delay_coefficient_list)/10)


total_time = 0 
avg_tier_time_list = []
max_time_list = pd.DataFrame({'time' : []})
    
client_delay_computing = 0.1
client_delay_net = 0.1


num_users = args.client_number
epochs = args.rounds
lr = args.lr



# =====
#   load dataset
# ====

class_num = 7

def load_data(args, dataset_name):
    if dataset_name == "HAM10000":
        return
    elif dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10
        args.data_dir = './data/cinic10/'
    else:
        data_loader = load_partition_data_cifar10

    if dataset_name == "cinic10":
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
        
    else:
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size)
        
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset
if args.dataset != "HAM10000" and args.dataset != "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
if args.dataset == "cinic10":
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts] = dataset

    dataset_test = test_data_local_dict
    dataset_train = train_data_local_dict
    
    sataset_size = {}
    for i in range(0,len(traindata_cls_counts)):
        sataset_size[i] = sum(traindata_cls_counts[i].values())
    avg_dataset = sum(sataset_size.values()) / len(sataset_size)

sataset_size = {}
if args.dataset != "HAM10000" and args.dataset != "cinic10":
    for i in range(0,args.client_number):
        sataset_size[i] = sum(dict(Counter(dataset_train[i].dataset.target)).keys())
    avg_dataset = sum(sataset_size.values()) / len(sataset_size)
    


## global model
init_glob_model = resnet56_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)
    
net_glob_client_tier = {}
net_glob_client_tier[1],_ = SFL_local_tier(classes=class_num,tier=5)
net_glob_client,_ = SFL_local_tier(classes=class_num,tier=tier)
for i in range(1,num_tiers+1):
    net_glob_client_tier[i],_ = SFL_local_tier(classes=class_num,tier=i)

    

if torch.cuda.device_count() > 1:
    
    net_glob_client = nn.DataParallel(net_glob_client, device_ids=list(range(torch.cuda.device_count())))  
    for i in range(1, num_tiers+1):
        net_glob_client_tier[i] = nn.DataParallel(net_glob_client_tier[i], device_ids=list(range(torch.cuda.device_count())))

for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)

net_glob_client.to(device)

net_glob_server_tier = {}
_, net_glob_server = SFL_local_tier(classes=class_num,tier=tier) # local loss SplitFed
for i in range(1,num_tiers+1):
    _, net_glob_server_tier[i] = SFL_local_tier(classes=class_num,tier=i)
    
    
if torch.cuda.device_count() > 1:
    
    net_glob_server = nn.DataParallel(net_glob_server, device_ids=list(range(torch.cuda.device_count())))   # to use the multiple GPUs 
    for i in range(1, num_tiers+1):
        net_glob_server_tier[i] = nn.DataParallel(net_glob_server_tier[i], device_ids=list(range(torch.cuda.device_count())))
        
        
for i in range(1, num_tiers+1):
    net_glob_server_tier[i].to(device)

net_glob_server.to(device)

#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []


criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

time_train_server_train = 0
time_train_server_train_all = 0

  

#====================================================================================================
#                                  Server Side Training
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    len_min = float('inf')
    index_len_min = 0
    for j in range(0, len(w)):
        if len(w[j]) < len_min:
            len_min = len(w[j])
            index_len_min = j
    w[0],w[index_len_min] = w[index_len_min],w[0]
            
            
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        c = 1
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            c += 1
        w_avg[k] = torch.div(w_avg[k], c)
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
best_acc = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []
w_glob_server = net_glob_server.state_dict()
w_glob_server_tier ={}
net_glob_server_tier[tier].load_state_dict(w_glob_server)
for i in range(1, num_tiers+1):
   w_glob_server_tier[i] = net_glob_server_tier[i].state_dict()
w_locals_server = []
w_locals_server_tier = {}
for i in range(1,num_tiers+1):
    w_locals_server_tier[i]=[]
idx_collect = []
l_epoch_check = False
fed_check = False
net_model_server_tier = {}
net_model_client_tier = {}
client_tier = {}
for i in range (0, num_users):
    client_tier[i] = num_tiers
k = 0
net_model_server = [net_glob_server for i in range(num_users)]

for i in range(len(client_number_tier)):
    for j in range(int(client_number_tier[i])):
        net_model_server_tier[k] = net_glob_server_tier[i+1]
        client_tier[k] = i+1
        k +=1
net_server = copy.deepcopy(net_model_server[0]).to(device)
net_server = copy.deepcopy(net_model_server_tier[0]).to(device)

        
optimizer_server_glob =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) 
scheduler_server = ReduceLROnPlateau(optimizer_server_glob, 'max', factor=0.8, patience=0, threshold=0.0000001)
patience = args.lr_patience
factor= args.lr_factor
wait=0
new_lr = lr

times_in_server = []
        
        
# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, extracted_features):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server, time_train_server_train, time_train_server_train_all, w_glob_server_tier, w_locals_server_tier, w_locals_tier
    global loss_train_collect_user, acc_train_collect_user, lr, total_time, times_in_server, new_lr
    batch_logits = extracted_features
    time_train_server_s = time.time()
    

    net_server = copy.deepcopy(net_model_server_tier[idx]).to(device)

        
    net_server.train()
    # optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
    lr = new_lr
    if args.optimizer == "Adam":# and False:
        optimizer_server =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) 
    elif args.optimizer == "SGD":
        optimizer_server =  torch.optim.SGD(net_server.parameters(), lr=lr, momentum=0.9,
                                              nesterov=True,
                                              weight_decay=args.wd)
    
    time_train_server_s = time.time()
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    if args.dataset != 'HAM10000':
        y = y.to(torch.long)
    loss = criterion(fx_server, y) # to solve change dataset
    
                    
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()  #original
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch

    net_model_server[idx] = copy.deepcopy(net_server)
    net_model_server_tier[idx] = copy.deepcopy(net_server)

    time_train_server_train += time.time() - time_train_server_s
    # count1: to track the completion of the local batch associated with one client
    # like count1 , aggregate time_train_server_train # this part comes from SplitFed paper
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        times_in_server.append(time_train_server_train)
        time_train_server_train_all += time_train_server_train
        total_time += time_train_server_train
        time_train_server_train = 0
        
        printRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            w_locals_server_tier[client_tier[idx]].append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
        
        # This is for federation process--------------------
        if len(idx_collect) == m: 
            fed_check = True 

                             
            w_locals_tier = w_locals_server
            w_locals_server = []
            w_locals_server_tier = {}
            for i in range(1,num_tiers+1):
                w_locals_server_tier[i]=[]
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
            new_lr = optimizer_server.param_groups[0]['lr']
            
    
    return dfx_client  # output of server 

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server, net_glob_server_tier 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check, w_glob_server_tier
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, acc_avg_all_user, loss_avg_all_user_train, best_acc
    global wait, new_lr
    
    net = copy.deepcopy(net_model_server_tier[idx]).to(device)


    net.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        if args.dataset != 'HAM10000':
            y = y.to(torch.long)
        loss = criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
    
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            printGreen('Server Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(acc_avg_test, loss_avg_test))


            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                
                
                if (acc_avg_all_user/100) > best_acc  * ( 1 + lr_threshold ):
                    best_acc = (acc_avg_all_user/100)
                    wait = 0
                else:
                     wait += 1 
                if wait > patience:   #https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
                    new_lr = max(float(optimizer_server.param_groups[0]['lr']) * factor, 0)
                    wait = 0
                              
                print("====================== DTFL V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
                
                wandb.log({"Server_Training_Accuracy": acc_avg_all_user_train, "epoch": ell}, commit=False)
                wandb.log({"Server_Test_Accuracy": acc_avg_all_user, "epoch": ell}, commit=False)
         
    return 

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = client_epoch[idx]
        #self.selected_clients = []
        batch_size = args.batch_size
        if args.dataset == "HAM10000":
            self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = batch_size, shuffle = True, drop_last=True)
            self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = batch_size, shuffle = True, drop_last=True)
        else:
            self.ldr_train = dataset_train[idx]
            self.ldr_test = dataset_test[idx]
            

    def train(self, net):
        net.train()
        self.lr , lr = new_lr, new_lr

        if args.optimizer == "Adam":
            optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) 
        elif args.optimizer == "SGD":
            optimizer_client =  torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                                      nesterov=True,
                                                      weight_decay=args.wd)
        
        time_client=0

        CEloss_client_train = []
        Dcorloss_client_train = []
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                if args.optimizer == "Adam":
                    optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) 
                elif args.optimizer == "SGD":
                    optimizer_client =  torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                                              nesterov=True,
                                                              weight_decay=args.wd)
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                
                if True:
                    
                    #---------forward prop-------------
                    if whether_local_loss:
                        extracted_features, fx = net(images)
                    else:
                        fx = net(images)
                    client_fx = fx.clone().detach().requires_grad_(True)
                    time_client += time.time() - time_s
                    dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, _)
                    
                    
                    #--------backward prop -------------
                    time_s = time.time()
                    if whether_local_loss:
                        
                        if args.dataset != 'HAM10000':
                            labels = labels.to(torch.long)
                        loss = criterion(extracted_features, labels) # to solve change dataset)
                        CEloss_client_train.append(((1 - dcor_coefficient)*loss.item()))                    
                        
                            
                        if whether_dcor:
                            loss += dcor_coefficient * dis_corr(images,fx)
                            Dcorloss_client_train.append(((dcor_coefficient)*dis_corr(images,fx)))                    
                        loss.backward()
    
                    else:
                        fx.backward(dfx) # backpropagation
    

                    optimizer_client.step()
                    time_client += time.time() - time_s
                    


        
        return net.state_dict(), time_client
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------

                if whether_local_loss :
                    extracted_features, fx = net(images)
                # Sending activations to server 
                    evaluate_server(fx, labels, self.idx, len_batch, ell)

                else:
                    fx = net(images)
                # Sending activations to server 
                    evaluate_server(fx, labels, self.idx, len_batch, ell)
            
        return 

    def evaluate_glob(self, net, ell):
        net.eval()
        epoch_acc = []
        epoch_loss = []
           
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                if args.dataset != 'HAM10000':
                    labels = labels.to(torch.long)
                loss = criterion(fx, labels)
                acc = calculate_accuracy(fx, labels)
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            printGreen('Client{} Test =>                     \tAcc: {:.3f} \tLoss: {:.4f}'
                    .format(self.idx, epoch_acc[-1], epoch_loss[-1])) 
                
            return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
            
                
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users): # this is only for HAM10000
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          
#=============================================================================
#                         Data loading 
#============================================================================= 
if args.dataset == "HAM10000":
        
    #os.chdir('../')
    df = pd.read_csv('data/HAM10000_metadata.csv') # need to load data for HAM10000
    print(df.head())
    
    
    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    
    # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join("data", '*', '*.jpg'))}
    

    df['path'] = df['image_id'].map(imageid_path.get)
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes


#==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y
#=============================================================================
# Train-test split          
if args.dataset == "HAM10000":
    
    train, test = train_test_split(df, test_size = 0.2)
    
    train = train.reset_index()
    test = test.reset_index()


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                            transforms.RandomVerticalFlip(),
                            transforms.Pad(3),
                            transforms.RandomRotation(10),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])
        
    test_transforms = transforms.Compose([
                            transforms.Pad(3),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])    
    
    
    # With augmentation
    dataset_train = SkinData(train, transform = train_transforms)
    dataset_test = SkinData(test, transform = test_transforms)

#----------------------------------------------------------------
    dict_users = dataset_iid(dataset_train, num_users)
    dict_users_test = dataset_iid(dataset_test, num_users)

# Data transmission

client_tier_all = []
client_tier_all.append(copy.deepcopy(client_tier))
total_training_time = 0
time_train_server_train_all_list = []

client_sample = np.ones(num_users)
#------------ Training And Testing  -----------------
net_glob_client.train()
w_glob_client_tier ={}

#copy weights
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()

w_glob_client_tier[tier] = net_glob_client_tier[tier].state_dict()

client_sample = np.ones(num_tiers)
# to start with same weigths 
for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)
    

w_glob = copy.deepcopy(init_glob_model.state_dict())

for t in range(1, num_tiers+1):

    for k in w_glob_client_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            #k1 = 'module'+k
            k1 = k1[7:]
        
        #if (k == 'fc.bias' or k == 'fc.weight'):
        if (k == 'module.fc.bias' or k == 'module.fc.weight'):
            continue 
        
        w_glob_client_tier[t][k] = w_glob[k1]

    for k in w_glob_server_tier[t].keys():
        k1 = k
        if k.startswith('module'):
            #k1 = 'module'+k
            k1 = k1[7:]
        w_glob_server_tier[t][k] = w_glob[k1]
        
    net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
    net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
    
w_locals_tier, w_locals_client, w_locals_server = [], [], []


net_model_client_tier = {}
for i in range(1, num_tiers+1):
    net_model_client_tier[i] = net_glob_client_tier[i]
    net_model_client_tier[i].train()
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()


# optimizer for every elient
optimizer_client_tier = {}
for i in range(0, num_users): # one optimizer for every tier/ client
    if args.optimizer == "Adam":
        optimizer_client_tier[i] =  torch.optim.Adam(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) 
    elif args.optimizer == "SGD":
        optimizer_client_tier[i] =  torch.optim.SGD(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, momentum=0.9,
                                                          nesterov=True,
                                                          weight_decay=args.wd)
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

df_delay = pd.DataFrame()
start_time = time.time() 

client_times = pd.DataFrame()
torch.manual_seed(SEED)
delay_actual= np.zeros(num_users)

for i in range(0, num_users):
    data_server_to_client = 0

    delay_actual[i] = data_server_to_client / 1

for iter in range(epochs):
    if iter % 1000 == 500:
        for c in range(0, num_users):
            if c % 5 == 0 :
                delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(60, 25, 10, 5, 0))[0]
            elif c % 5 == 1:
                delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(15, 60, 15, 10, 0))[0]
            elif c % 5 == 2:
                delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(5, 15, 60, 15, 5))[0]
            elif c % 5 == 3:
                delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(0, 10, 15, 60, 15))[0]
            elif c % 5 == 4:
                delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(0, 5, 10, 25, 60))[0]
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []
    w_locals_client_tier = {}
    for i in range(1,num_tiers+1):
        w_locals_client_tier[i]=[]
    client_time = np.zeros(num_users)
    for i in range(0, num_users):
        wandb.log({"Client{}_Tier".format(i): client_tier[i], "epoch": -1}, commit=False)
    if args.dataset != "HAM10000": 
        client_sample = []
    for idx in idxs_users:
        time_train_test_s = time.time()
        net_glob_client = net_model_client_tier[client_tier[idx]]
        w_glob_client_tier[client_tier[idx]] = net_glob_client_tier[client_tier[idx]].state_dict() # may be I can eliminate this line
        if args.dataset == "HAM10000":
            local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        else:
            local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = [], idxs_test = [])
        if idx == idxs_users[0]:
            local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        # Training ------------------
        [w_client, duration] = local.train(net = copy.deepcopy(net_glob_client).to(device))
        if args.dataset != "HAM10000": 
            client_sample.append(train_data_local_num_dict[idx] / sum(train_data_local_num_dict.values()) * num_users)
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_client_tier[client_tier[idx]].append(copy.deepcopy(w_client))
        # Testing -------------------  
        client_time[idx] = duration
        delay_actual[idx] += (delay_coefficient[idx] * duration * (1 + np.random.rand()* client_delay_computing) ) # this is per epoch
        total_time += delay_actual[idx]
        wandb.log({"Client{}_Actual_Delay".format(idx): delay_actual[idx], "epoch": iter}, commit=False)
    server_wait_time = (max(delay_actual * client_epoch) - min(delay_actual * client_epoch))
    training_time = (max(delay_actual) + max(times_in_server))
    total_training_time += training_time
    if iter == 0:
        first_training_time = training_time
    wandb.log({"Training_time": total_training_time, "epoch": iter}, commit=False)
    times_in_server = []
    time_train_server_train_all_list.append(time_train_server_train_all)
    time_train_server_train_all = 0
     
    delay_actual[delay_actual==0] = np.nan  # convert zeros to nan, for when some clients not involved in the epoch
    df_delay = df_delay.append(pd.DataFrame(delay_actual).T, ignore_index = True)  
    client_times = client_times.append(pd.DataFrame(client_time).T, ignore_index = True) # this is only time for training
    client_epoch_last = client_epoch.copy()

    
    [client_tier, client_epoch, avg_tier_time_list, max_time_list, client_times] = dynamic_tier8(client_tier_all[:], df_delay, 
                                                num_tiers, server_wait_time, client_epoch,
                                                time_train_server_train_all_list, num_users, iter,
                                                sataset_size = sataset_size, avg_tier_time_list = avg_tier_time_list,
                                                max_time_list = max_time_list, idxs_users = idxs_users) # assign next tier and model
    client_tier_all.append(copy.deepcopy(client_tier))
    
    delay_actual= np.zeros(num_users)
    for i in range(0, num_users):
        data_server_to_client = 0
        delay_actual[i] = data_server_to_client / 1
    

    for i in client_tier.keys():  
        net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]
            
    

    w_glob = aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample, idxs_users) 
    
    for t in range(1, num_tiers+1):
       
        for k in w_glob_client_tier[t].keys():
            if k in w_glob_server_tier[t].keys():  
                if w_locals_client_tier[t] != []:
                    w_glob_client_tier[t][k] = FedAvg(w_locals_client_tier[t])[k]
                    continue
                else:
                    continue 
            
            w_glob_client_tier[t][k] = w_glob[k]

        for k in w_glob_server_tier[t].keys():
            w_glob_server_tier[t][k] = w_glob[k]
            
        net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
        net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
 
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect}) 
file_name = program+".xlsx"    


with pd.ExcelWriter(program+".xlsx") as writer:
    df.to_excel(writer, sheet_name= "Accuracy", index = False)
    client_times.to_excel(writer, sheet_name='Client Training Time', index = False)
    df_delay.to_excel(writer, sheet_name='Iteration Training Time', index = False)
     

#=============================================================================
#                         Program Completed
#=============================================================================