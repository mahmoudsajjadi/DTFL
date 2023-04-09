  #============================================================================
# multi-tier Splitfed
'''
change server model : each tier one model,, previously, each client has one model
change model tier and adapt it
'''
# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
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
from model.resnet_client import resnet18_SFL_tier
from model.resnet_client import resnet18_SFL_local_tier
from model.resnet56 import resnet56_SFL_local_tier
from model.resnet110_7t import resnet56_SFL_local_tier_7
#from model.resnet56_7t import resnet56_SFL_local_tier_7
from model.resnet110_7t import resnet56_SFL_tier_7
from model.resnet110_7t import resnet56_SFL_fedavg_base
from model.resnet_pretrained import resnet56_pretrained
# from model.resnet101 import resnet101_local_tier
from utils.loss import loss_dcor
from utils.loss import distance_corr
from utils.loss import dis_corr
from utils.fedavg import multi_fedavg
from utils.fedavg import aggregated_fedavg

from utils.dynamic_tier import dynamic_tier9
from utils.dynamic_tier import dqn_agent
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from multiprocessing import Process

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
program = "Multi-Tier Splitfed Local Loss with KD"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

''' Mahmoud Init '''

def add_args(parser):
    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--running_name', default="default", type=str)
    parser.add_argument('--KD_beta_init', default=1.0, type=float)    # (1-alpha) CE + alpha DCOR + beta KD(client-side)
    parser.add_argument('--KD_increase_factor', default=500.0, type=float)  # if more, it become like KD-init faster
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_patience', default=1000, type=float)
    parser.add_argument('--lr_min', default=0, type=float)
    parser.add_argument('--whether_dynamic_lr_client', default=1, type=int)
    parser.add_argument('--rounds', default=10000, type=int)
    parser.add_argument('--whether_distillation_on_the_server', default=0, type=int)
    parser.add_argument('--whether_distillation_on_clients', default=True, type=bool)
    parser.add_argument('--whether_GKT_local_loss', default=False, type=bool)
    parser.add_argument('--whether_local_loss', default=True, type=bool)
    parser.add_argument('--whether_local_loss_v2', default=False, type=bool)
    parser.add_argument('--whether_FedAVG_base', default=False, type=bool) # this is for base line of fedavg
    parser.add_argument('--whether_dcor', default=True, type=bool)
    parser.add_argument('--whether_multi_tier', default=True, type=bool)
    parser.add_argument('--whether_federation_at_clients', default=True, type=bool)
    parser.add_argument('--whether_aggregated_federation', default=1, type=int)
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--intra_tier_fedavg', default=True, type=bool)
    parser.add_argument('--inter_tier_fedavg', default=True, type=bool)
    parser.add_argument('--dcor_coefficient', default=0.5, type=float)  # same as alpha in paper
    parser.add_argument('--tier', default=5, type=int)
    parser.add_argument('--client_epoch', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--whether_pretrained_on_client', default=0, type=int) # from fedgkt
    parser.add_argument('--whether_pretrained', default=0, type=int)  # from https://github.com/chenyaofo/pytorch-cifar-models
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
    
    parser.add_argument('--dataset', type=str, default='HAM10000', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=1000000, metavar='PA',
                        help='partition alpha (default: 0.5)')
    
    parser.add_argument('--model', type=str, default='resnet56_7', metavar='N',
                        help='neural network used in training')
    
    parser.add_argument('--wheter_agg_tiers_on_server', type=int, default=0 , metavar='N',
                        help='tier aggregation on server')
    parser.add_argument('--version', type=int, default=1 , metavar='N',
                        help='version of aggregation')
    parser.add_argument('--global_model', type=int, default=0 , metavar='N',
                        help='global model for testing the method')
    parser.add_argument('--test_before_train', type=int, default=1 , metavar='N',  # by this we can check the accuracy of global model
                        help='test before train')  
    
    args = parser.parse_args()
    return args

lr_threshold = 0.0001
frac = 0.1        # participation of clients; if 1 then 100% clients participate in SFLV1

parser = argparse.ArgumentParser()
args = add_args(parser)
logging.info(args)
# wandb.init(mode="online")
# os.environ["WANDB_MODE"] = "online"
    
wandb.init(mode="disabled")
#wandb.init(mode="offline")
    
wandb.init(
    project="Multi-Tier_Splitfed",
    name="LocalSplitFed-Multi-tier",# + str(args.tier),
    config=args,
    tags="Tier1_5",
    # group="ResNet18",
)


SFL_tier = resnet18_SFL_tier
SFL_local_tier = resnet56_SFL_local_tier_7

#SFL_local_tier = resnet56_SFL_fedavg_base
#global_model = SFL_local_tier(classes=10, tier=1, fedavg_base = True)
#print(global_model.state_dict().keys())

if args.whether_FedAVG_base:
    SFL_local_tier = resnet56_SFL_fedavg_base
    num_tiers = 1
    args.tier = 1
    #net_glob_client_tier = SFL_local_tier(classes=10,tier=1, fedavg_base = True)
    #print(net_glob_client_tier.state_dict().keys())
elif args.model == 'resnet56_5' and args.whether_local_loss:
    SFL_local_tier = resnet56_SFL_local_tier
    num_tiers = 5
elif args.model == 'resnet56_7' and args.whether_local_loss:
    SFL_local_tier = resnet56_SFL_local_tier_7
    num_tiers = 7
elif args.model == 'resnet56_7' and not args.whether_local_loss:
    SFL_local_tier = resnet56_SFL_tier_7
    num_tiers = 7


whether_GKT_local_loss = args.whether_GKT_local_loss
whether_local_loss = args.whether_local_loss
whether_dcor = args.whether_dcor
dcor_coefficient = args.dcor_coefficient
tier = args.tier
client_epoch = args.client_epoch
client_epoch = np.ones(args.client_number,dtype=int) * client_epoch
whether_multi_tier = args.whether_multi_tier
whether_federation_at_clients = args.whether_federation_at_clients
intra_tier_fedavg = args.intra_tier_fedavg
inter_tier_fedavg = args.inter_tier_fedavg
whether_distillation_on_clients = args.whether_distillation_on_clients
client_type_percent = [0.0, 0.0, 0.0, 0.0, 1.0]
if args.whether_FedAVG_base:
    client_type_percent = [1.0]

elif num_tiers == 7:
    client_type_percent = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    tier = 1

# test to remove tiers
# num_tiers = 6
# client_type_percent = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    
# client_type_percent = [0.2, 0.2, 0.2, 0.2, 0.2]
# client_type_percent[args.tier-1] = 1
client_number_tier = (np.dot(args.client_number , client_type_percent))
net_speed_list = [10 * 1024**10000 , 10 * 1024**10000]
net_speed = random.choices(net_speed_list, weights = [80 , 20], k=args.client_number) # MB/S : speed for transmit data
delay_coefficient_list = [1000,2000,2500,3000,10000]
delay_coefficient = random.choices(delay_coefficient_list, k=args.client_number)
#delay_coefficient = [50,60,70,80,90,100,120,140,160,180,200,250,300,400,500,1000]
#delay_coefficient = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
#delay_coefficient = [100,50,400,50,500,50,50,650,50,50,50,750,50,850,50,1000]
#delay_coefficient = [100,50,400,50,500,50,50,650,50,50,50,750,50,850,50,1000]
delay_coefficient = [16,20,32,72,256] * 100

delay_coefficient = list(np.array(delay_coefficient)/10)

delay_coefficient_list = [16,20,32,72,256]
delay_coefficient_list = list(np.array(delay_coefficient_list)/10)
# delay_coefficient = random.choices(delay_coefficient_list, weights=(20, 20, 20, 20, 20), k=args.client_number)

# delay_coefficient = [100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,1500]
#delay_coefficient = [50,100,120,160,200,250,300,400,500,1000]
# delay_coefficient = 10 


total_time = 0 
# global avg_tier_time_list
avg_tier_time_list = []
max_time_list = pd.DataFrame({'time' : []})
# max_time_list.loc[0] = 0
    
client_delay_computing = 0.1
client_delay_net = 0.1
# client_delay_mu = [0.1, 0.1, 0.1, 2, 3]
# client_delay_c = [0.0,0.0,0.0,1.5,20.0]


# tier = 1
#===================================================================
# No. of users
num_users = args.client_number
epochs = args.rounds
lr = args.lr

# data transmmission
global data_transmit
data_transmited_fl = 0
data_transmited_sl = 0

whether_distillation_on_the_server =args.whether_distillation_on_the_server

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
        print(f'Client {i} :', dict(Counter(dataset_test[i].dataset.target)))
        print(f'Client {i} :', dict(Counter(dataset_train[i].dataset.target)))
        sataset_size[i] = sum(dict(Counter(dataset_train[i].dataset.target)).keys())
    avg_dataset = sum(sataset_size.values()) / len(sataset_size)
    


## global model
init_glob_model = resnet56_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)

## DQN agent




''' RL agent , source code from : #https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char01%20DQN/DQN.py

DQN structure

'''
# hyper-parameters
BATCH_SIZE = 50 # 128 # first 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.8
MEMORY_CAPACITY = 50 # first 2000
# https://medium.com/intro-to-artificial-intelligence/deep-q-network-dqn-applying-neural-network-as-a-functional-approximation-in-q-learning-6ffe3b0a9062
Q_NETWORK_ITERATION = 5  # original program has 2 net, combine to 1 by change 100 to 1
NUM_STATES = args.client_number
NUM_ACTIONS = args.client_number * num_tiers # clients * tiers
ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.num_tiers = 5
        self.num_clients = 2
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        # self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 1 + self.num_clients))
        self.memory = np.zeros((MEMORY_CAPACITY, args.client_number * 3 + 1))
        for i in range(0,args.client_number):
            self.memory[:,args.client_number+i] = np.ones((MEMORY_CAPACITY))*(num_tiers-1+i*num_tiers)
        self.memory[:,args.client_number:args.client_number*2] = np.ones((MEMORY_CAPACITY,args.client_number))*(num_tiers-1) # to have a better start point
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        


    def choose_action(self, state, client_tier):
        # client_tier = np.ones(self.num_clients,dtype=int)
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            # action = torch.max(action_value, 1)[1].data.numpy()
            action = torch.topk(action_value, self.num_clients)[1].data.numpy() # 2 change with number of clients
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            # action = np.random.randint(0,NUM_ACTIONS)
            action = np.random.randint(2,NUM_ACTIONS, size = self.num_clients) # 2 change with number of clients
            action_value = torch.from_numpy(np.random.rand(1, NUM_ACTIONS)) # Mahmoud
            # action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
            
        # change output to client-tier pair
        for i in range(0,self.num_clients): # 2 change with number of clients
            action = torch.max(action_value[0, self.num_tiers * i: self.num_tiers * (i+1)], 0)[1].data.numpy() # 5 change with number of tiers
            client_tier[i] = action + 1
        
        
        return client_tier


    def store_transition(self, state, action, reward, next_state):
        # transition = np.hstack((state, [action, reward], next_state))
        transition = np.hstack((state, action, reward, next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # batch_memory = self.memory[sample_index, :]
        batch_memory = self.memory # eleminate sample
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+self.num_clients].astype(int))
        # batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+self.num_clients:NUM_STATES+self.num_clients+1])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_next_2 = q_eval
        
        for i in range(0,BATCH_SIZE):   # mahmoud: next state value: max q next
            for j in range(0,self.num_clients):
                q_next_2[i][j] = q_next[i][self.num_tiers * j: self.num_tiers * (j+1)].max()
        
        # q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_reward + GAMMA * q_next_2
        
        loss = self.loss_func(q_eval, q_target)   #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 10) # first it was 50
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(10,20) # first 30,50
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(20,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob
        

dqn = DQN()

def dqn_agent1(client_tier, client_times, client_epoch, reward, iter):
    # dqn = DQN()
    present_time = client_times.ewm(com=0.5).mean()[-1:]
    state = present_time.to_numpy()[0]
    next_state = state
    if iter == 0 :
        state = next_state * 0
    else:
        state = client_times[:-1].ewm(com=0.5).mean()[-1:].to_numpy()[0]
    action = client_tier
    # learning phase
    # dqn.store_transition(state, list(action.values()), reward, next_state)  # not sure always dict to array is sorted 
    dqn.store_transition(state, list(list(action.values()) - np.array([1-5*x for x in range(0,args.client_number)])), reward, next_state)  # not sure always dict to array is sorted 
    
    dqn.learn()
    
    # selection phase
    action = dqn.choose_action(state, client_tier)
    client_tier = action
    # dqn.learn()
    # print('dqn') 
    
    return client_tier, client_epoch





#####
#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class Baseblock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output
 
    
     
class ResNet18_client_side(nn.Module):
    # def __init__(self):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_client_side, self).__init__()
        self.input_planes = 64
        self.layer1 = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
        
        # Aux network  fedgkt
        
        self.layer2 = self._layer(block, 16, 1) # layers[0] =1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(16 * 1, classes )  # block.expansion = 1 , classes

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
        
        
    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))   # here from fedgkt code extracted_features = x without maxpool
        
        # Aux Network output
        # extracted_features = resudial1

        x = self.layer2(resudial1)  # B x 16 x 32 x 32
        # x = self.layer2(x)  # B x 32 x 16 x 16
        # x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        extracted_features = self.fc(x_f)  # B x num_classes

        return extracted_features, resudial1
    
# init_glob_model = resnet56_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)
net_glob_client_tier = {}
if whether_local_loss == False and False:
    # net_glob_client = ResNet18_client_side(Baseblock, [2], class_num) # for test duration
    net_glob_client,_ = resnet18_SFL_tier(classes=class_num,tier=tier) # it has backpropagation
    net_glob_client_tier[1],_ = resnet18_SFL_tier(classes=class_num,tier=1)
    net_glob_client_tier[2],_ = resnet18_SFL_tier(classes=class_num,tier=2)
    net_glob_client_tier[3],_ = resnet18_SFL_tier(classes=class_num,tier=3)
    net_glob_client_tier[4],_ = resnet18_SFL_tier(classes=class_num,tier=4)
    net_glob_client_tier[5],_ = resnet18_SFL_tier(classes=class_num,tier=5)
elif args.whether_FedAVG_base:
    net_glob_client_tier[1] = copy.deepcopy(init_glob_model)
    # net_glob_client_tier[1] = SFL_local_tier(classes=class_num,tier=1, fedavg_base = True)
    net_glob_client = SFL_local_tier(classes=class_num,tier=tier)
elif args.whether_local_loss_v2:
    for i in range(1,num_tiers+1):
        net_glob_client_tier[i],_ = SFL_local_tier(classes=class_num,tier=i, local_v2 = args.whether_local_loss_v2)
    net_glob_client,_ = SFL_local_tier(classes=class_num,tier=tier)
else:
    net_glob_client_tier[1],_ = SFL_local_tier(classes=class_num,tier=5)
    net_glob_client,_ = SFL_local_tier(classes=class_num,tier=tier)
    # net_glob_client_tier[1],_ = resnet101_local_tier(classes=class_num,tier=1)
    for i in range(1,num_tiers+1):
        net_glob_client_tier[i],_ = SFL_local_tier(classes=class_num,tier=i)

    
# net_glob_client = ResNet18_client_side(Baseblock, [2], class_num)
    
## pretrainted like fedgkt

"""
    Note that we only initialize the client feature extractor to mitigate the difficulty of alternating optimization
"""
if args.whether_pretrained_on_client == 1 and False:
        
    if args.dataset == "cifar10" or args.dataset == "CIFAR10":
        resumePath = "./model/pretrained/CIFAR10/resnet56/best.pth"
    elif args.dataset == "cifar100" or args.dataset == "CIFAR100":
        resumePath = "./model/pretrained/CIFAR100/resnet56/best.pth"
    elif args.dataset == "cinic10" or args.dataset == "CINIC10":
        resumePath = "./model/pretrained/CINIC10/resnet56/best.pth"
    # else:
    #     resumePath = "./../../../fedml_api/model/cv/pretrained/CIFAR10/resnet56/best.pth"
    if args.dataset == "cifar10" or args.dataset != "HAM10000":
        pretrained_model1 = resnet56_pretrained(class_num, pretrained=True, path=resumePath)
        logging.info("########pretrained model#################")
        logging.info(pretrained_model1)
    
        # copy pretrained parameters to client models
        params_featrue_extractor = dict()
        for name, param in pretrained_model1.named_parameters():
            if name.startswith("conv1") or name.startswith("bn1") or True: #or name.startswith("layer1"):
                logging.info(name)
                params_featrue_extractor[name] = param
        
        # client_model = resnet8_56(n_classes)  -> net_glob_client_tier[1]
        
        logging.info("pretrained:")
        for i in range(1, num_tiers+1):
            for name, param in net_glob_client_tier[i].named_parameters():
                if name.startswith("conv1"):
                    param.data = params_featrue_extractor[name]
                    param.data.requires_grad_(True) # Mahmoud
                    # if args.whether_training_on_client == 0:
                    #     param.requires_grad = False
                elif name.startswith("bn1") and not name.startswith("fc"):
                    param.data = params_featrue_extractor[name]
                    param.data.requires_grad_(True) # Mahmoud
                    # if args.whether_training_on_client == 0:
                    #     param.requires_grad = False
                # elif name.startswith("layer1"):
                #     param.data = params_featrue_extractor[name]
                    # if args.whether_training_on_client == 0:
                    #     param.requires_grad = False
            logging.info(net_glob_client_tier[i])


#torch.distributed.init_process_group("NCCL", world_size = 4, )
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    #net_glob_client = nn.parallel.DistributedDataParallel(net_glob_client)
    net_glob_client = nn.DataParallel(net_glob_client, device_ids=list(range(torch.cuda.device_count())))  
    for i in range(1, num_tiers+1):
        net_glob_client_tier[i] = nn.DataParallel(net_glob_client_tier[i], device_ids=list(range(torch.cuda.device_count())))

for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)

# pre trained model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True) and from fedgkt
if args.whether_pretrained == 1 and args.dataset == 'cifar10':
    
    if args.dataset == "cifar10" or args.dataset == "CIFAR10":
        resumePath = "./model/pretrained/CIFAR10/resnet56/best.pth"
        
    pretrained_model1 = resnet56_pretrained(class_num, pretrained=True, path=resumePath)
    logging.info("########pretrained model#################")
    #logging.info(pretrained_model1)
    
    # pretrained_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True) # it does not have layera.conv3
    params_featrue_extractor = dict()
    for name, param in pretrained_model1.named_parameters():
        if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1"):
            logging.info(name)
            params_featrue_extractor['module.'+name] = param
    for i in range(1, num_tiers+1):
        for name, param in net_glob_client_tier[i].named_parameters():
            #print(name)
            #logging.info(name)
            #logging.info("tier")
            #logging.info(i)
            
            if name.startswith("module.conv1"):
                #logging.info(param.data)
                #logging.info(params_featrue_extractor[name])
                param.data = params_featrue_extractor[name]
                #logging.info(param.data)
                if args.whether_training_on_client == 0:
                    param.requires_grad = False
            elif name.startswith("module.bn1"):
                param.data = params_featrue_extractor[name]
                if args.whether_training_on_client == 0:
                    param.requires_grad = False
            elif name.startswith("module.layer1"):
                logging.info(name)
                param.data = params_featrue_extractor[name]
                if args.whether_training_on_client == 0:
                    param.requires_grad = False

    
net_glob_client.to(device)
print(net_glob_client)     

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side
class Baseblock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output


class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64

        self.layer3 = self._layer(block, 64, num_layers[0])
        self.layer4 = self._layer(block, 128, num_layers[1], stride = 2)
        self.layer5 = self._layer(block, 256, num_layers[2], stride = 2)
        self.layer6 = self._layer(block, 512, num_layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
        
    
    def forward(self, x):

        
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        
        x7 = self.avgpool(x6)
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        
        return y_hat  



# _, net_glob_server = resnet18_SFL_tier(classes=class_num,tier=tier) # it has backpropagation
net_glob_server = ResNet18_server_side(Baseblock, [2,2,2,2], class_num) # for test duration

net_glob_server_tier = {}
if whether_local_loss == False and False:
    _, net_glob_server = resnet18_SFL_tier(classes=class_num,tier=tier) # it has backpropagation normal SplitFed
    _, net_glob_server_tier[1] = resnet18_SFL_tier(classes=class_num,tier=1)
    _, net_glob_server_tier[2] = resnet18_SFL_tier(classes=class_num,tier=2)
    _, net_glob_server_tier[3] = resnet18_SFL_tier(classes=class_num,tier=3)
    _, net_glob_server_tier[4] = resnet18_SFL_tier(classes=class_num,tier=4)
    _, net_glob_server_tier[5] = resnet18_SFL_tier(classes=class_num,tier=5)
    # net_glob_server = ResNet18_server_side(Baseblock, [2,2,2,2], class_num) # for test duration
elif args.whether_FedAVG_base:
    net_glob_server_tier[1] = copy.deepcopy(init_glob_model)
    # net_glob_server_tier[1] = SFL_local_tier(classes=class_num,tier=1, fedavg_base = True)
    net_glob_server = SFL_local_tier(classes=class_num,tier=tier)
else:
    _, net_glob_server = SFL_local_tier(classes=class_num,tier=tier) # local loss SplitFed
    for i in range(1,num_tiers+1):
        _, net_glob_server_tier[i] = SFL_local_tier(classes=class_num,tier=i)
        # _, global_model = SFL_local_tier(classes=class_num,tier=i)
    # _, net_glob_server_tier[1] = SFL_local_tier(classes=class_num,tier=1)
    # _, net_glob_server_tier[2] = SFL_local_tier(classes=class_num,tier=2)
    # _, net_glob_server_tier[3] = SFL_local_tier(classes=class_num,tier=3)
    # _, net_glob_server_tier[4] = SFL_local_tier(classes=class_num,tier=4)
    # _, net_glob_server_tier[5] = SFL_local_tier(classes=class_num,tier=5)
    # _, net_glob_server_tier[6] = SFL_local_tier(classes=class_num,tier=6)
    # _, net_glob_server_tier[7] = SFL_local_tier(classes=class_num,tier=7)
    
    
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
#    net_glob_server = nn.DistributedDataParallel(net_glob_server)   # to use the multiple GPUs 
 #   for i in range(1, num_tiers+1):
  #      net_glob_server_tier[i] = nn.parallel.DistributedDataParallel(net_glob_server_tier[i])
    net_glob_server = nn.DataParallel(net_glob_server, device_ids=list(range(torch.cuda.device_count())))   # to use the multiple GPUs 
    for i in range(1, num_tiers+1):
        net_glob_server_tier[i] = nn.DataParallel(net_glob_server_tier[i], device_ids=list(range(torch.cuda.device_count())))
        
        
for i in range(1, num_tiers+1):
    net_glob_server_tier[i].to(device)

net_glob_server.to(device)
print(net_glob_server)   
# wandb.watch(net_glob_server)
   

#global_model = resnet56_SFL_fedavg_base(classes=class_num, tier=1, fedavg_base = True)
#global_model = resnet56_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)
#print(global_model.state_dict().keys())

#global_model = SFL_local_tier(classes=class_num,tier=1, fedavg_base = True)

    
# print(net_glob_server_tier[i].state_dict().keys())

#resnet56_SFL_fedavg_base

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

# criterion for KD
KD_beta = args.KD_beta_init
def criterion_KL(outputs, teacher_outputs):
    # KD_beta = KD_beta
    T = 1
    criterion_KL = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                              # F.softmax(teacher_outputs/T, dim=1)) * (args.KD_beta * T * T)
                              F.softmax(teacher_outputs/T, dim=1) + 10 ** (-7)) * (args.KD_beta * T * T)
    return criterion_KL
    
    # code from KD paper
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
    #                      F.softmax(teacher_outputs/T, dim=1)) * (KD_beta * T * T) + F.cross_entropy(outputs, labels) * (1. - KD_beta)
    
    # code from gkt paper
        # output_batch = F.log_softmax(output_batch / self.T, dim=1)
        #     teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        #     loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

#====================================================================================================
#                                  Server Side Program
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

def FedAvg_wighted(w, client_sample):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # w_avg[k] += w[i][k] * client_sample[i]  # to solve long error
            w_avg[k] += w[i][k] * client_sample[i].to(w_avg[k].dtype)  # maybe other method can be used
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# original fedavg
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))
    # return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
best_acc = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# w_glob_server = {}
# for k in init_glob_model.state_dict():
#     k1 = k
#     if not k.startswith('module'):
#         k1 = 'module.'+k
#         # k1 = k1[7:]
#     w_glob_server[k1] = init_glob_model.state_dict()[k]
w_glob_server = net_glob_server.state_dict()
# w_glob_server = init_glob_model.state_dict()
w_glob_server_tier ={}
net_glob_server_tier[tier].load_state_dict(w_glob_server)
for i in range(1, num_tiers+1):
   w_glob_server_tier[i] = net_glob_server_tier[i].state_dict()
w_locals_server = []
w_locals_server_tier = {}
for i in range(1,num_tiers+1):
    w_locals_server_tier[i]=[]

# net_glob_client_tier[tier].load_state_dict(w_glob_client)
# w_glob_client_tier[tier] = net_glob_client_tier[tier].state_dict()

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server_tier = {}
net_model_client_tier = {}
client_tier = {}
for i in range (0, num_users):
    client_tier[i] = num_tiers
k = 0
net_model_server = [net_glob_server for i in range(num_users)]
if args.version == 1:
    for i in range(len(client_number_tier)):
        for j in range(int(client_number_tier[i])):
            net_model_server_tier[k] = net_glob_server_tier[i+1]
            # net_model_client_tier[k] = net_glob_client_tier[i+1]
            client_tier[k] = i+1
            k +=1
    net_server = copy.deepcopy(net_model_server[0]).to(device)
    net_server = copy.deepcopy(net_model_server_tier[0]).to(device)
elif args.version == 2:
    for i in range(1, num_tiers+1):
        net_model_server_tier[i] = net_glob_server_tier[i]
        net_model_server_tier[i].train()
        w_glob_server_tier[i] = net_glob_server_tier[i].state_dict()
    net_server = copy.deepcopy(net_model_server[0]).to(device)
    net_server = copy.deepcopy(net_model_server_tier[client_tier[0]]).to(device)
        
# for i in range(1, num_tiers+1): # chenge to only one model for each server tier type
#     net_model_server_tier[i] = net_glob_server_tier[i]

# for t in range(1,num_tiers):  # check number of model in server and client
#     net_model_client_tier[t] = net_glob_client_tier[t]
# net_model_server_tier[] = [net_glob_server for i in range(num_users)]

#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
optimizer_server_glob =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
scheduler_server = ReduceLROnPlateau(optimizer_server_glob, 'max', factor=0.8, patience=0, threshold=0.0000001)
patience = args.lr_patience
factor= args.lr_factor
wait=0
new_lr = lr
min_lr = args.lr_min

times_in_server = []
# if args.optimizer == "Adam":
#     optimizer_server =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
# elif args.optimizer == "SGD":
#     optimizer_server =  torch.optim.SGD(net_server.parameters(), lr=lr, momentum=0.9,
#                                           nesterov=True,
#                                           weight_decay=args.wd)
# scheduler_server = ReduceLROnPlateau(optimizer_server, 'max', factor=0.7, patience=-1, threshold=0.0000001)
# scheduler_server = ReduceLROnPlateau(optimizer_server, 'max')
        
        
# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, extracted_features):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server, time_train_server_train, time_train_server_train_all, w_glob_server_tier, w_locals_server_tier, w_locals_tier
    global loss_train_collect_user, acc_train_collect_user, lr, total_time, times_in_server, new_lr
    batch_logits = extracted_features
    time_train_server_s = time.time()
    
    if args.version == 1:
        if not whether_multi_tier:
            net_server = copy.deepcopy(net_model_server[idx]).to(device)
        else:
            net_server = copy.deepcopy(net_model_server_tier[idx]).to(device)
    elif args.version == 2:
        if not whether_multi_tier:
            net_server = copy.deepcopy(net_model_server[client_tier[idx]]).to(device)
        else:
            net_server = copy.deepcopy(net_model_server_tier[client_tier[idx]]).to(device)
        
    net_server.train()
    # optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
    lr = new_lr
    if args.optimizer == "Adam":# and False:
        optimizer_server =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
    elif args.optimizer == "SGD":
        optimizer_server =  torch.optim.SGD(net_server.parameters(), lr=lr, momentum=0.9,
                                              nesterov=True,
                                              weight_decay=args.wd)
    scheduler_server = ReduceLROnPlateau(optimizer_server, 'max', factor=0.8, patience=1, threshold=0.0000001)
    
    time_train_server_copy = time.time() - time_train_server_s
    
    time_train_server_s = time.time()
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    # loss = criterion(fx_server, y)
    if args.dataset != 'HAM10000':
        y = y.to(torch.long)
        # y.int()
    loss = criterion(fx_server, y) # to solve change dataset
    
    # if whether_dcor:
    #     loss += dcor_coefficient * dis_corr(y,fx_server)
    
    # KD loss
    
    if whether_distillation_on_the_server == 1:
        # loss_kd = self.criterion_KL(output_batch, batch_logits).to(self.device)
        # loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
        # loss = loss_kd + self.args.KD_beta * loss_true
        loss_kd = criterion_KL(fx_server, batch_logits)
        loss = loss_kd + (1 - args.KD_beta) * loss

        
                    
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    if whether_GKT_local_loss:# or whether_distillation_on_clients:
        # loss.backward(retain_graph=True) # to solve error of 2 backward
        loss.backward(retain_graph=True, inputs=list(net_server.parameters())) # final for split local loss
    else:
        loss.backward()  #original
        dfx_client = fx_client.grad.clone().detach()
    # dfx_client = fx_client.grad.clone().detach()
    dfx_server = fx_server.clone().detach()
    optimizer_server.step()
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    # scheduler_server.step(best_acc)#, epoch=l_epoch_count) #from fedgkt
    
    # Update the server-side model for the current batch
    if args.version == 1:
        net_model_server[idx] = copy.deepcopy(net_server)
        net_model_server_tier[idx] = copy.deepcopy(net_server)
    elif args.version == 2:
        # net_model_server[client_tier[idx]] = copy.deepcopy(net_server)
        net_model_server_tier[client_tier[idx]] = copy.deepcopy(net_server)
    time_train_server_train += time.time() - time_train_server_s
    # count1: to track the completion of the local batch associated with one client
    # like count1 , aggregate time_train_server_train
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        wandb.log({"Client{}_Training_Time_in_Server".format(idx): time_train_server_train, "epoch": l_epoch_count}, commit=False)
        times_in_server.append(time_train_server_train)
        time_train_server_train_all += time_train_server_train
        total_time += time_train_server_train
        time_train_server_train = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        # wandb.log({"Client{}_Training_Accuracy".format(idx): acc_avg_train, "epoch": l_epoch_count}, commit=False)
        # wandb.log({"Client_Training_Accuracy": acc_avg_train, "epoch": l_epoch_count}, commit=True)

        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            w_locals_server_tier[client_tier[idx]].append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        # if len(idx_collect) == num_users:
        if len(idx_collect) == m:  # federation after evfery epoch not when all clients complete thier process like splitfed
            fed_check = True 
                                                             # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display 
            if not whether_GKT_local_loss:                                   
                if not args.whether_aggregated_federation == 1 :
                    w_glob_server = FedAvg(w_locals_server) # check with w_locals_server[1]['fc.bias'] and w_locals_server_tier[5][0]['fc.bias']
                    # for t in range(1,num_tiers+1):
                    #     if w_locals_server_tier[t] != []:
                    #         w_glob_server_tier[t] = FedAvg(w_locals_server_tier[t])
                    # for i in range(1, num_tiers+1):
                    #     w_glob_server_tier[i] = FedAvg(w_locals_server_tier)
                
                # server-side global model update and distribute that model to all clients ------------------------------
                    if not whether_multi_tier:
                        net_glob_server.load_state_dict(w_glob_server) 
                        net_model_server = [net_glob_server for i in range(num_users)]
                    else:
                        if intra_tier_fedavg:
                            for t in range(1,num_tiers+1):
                                if w_locals_server_tier[t] != []:
                                    w_glob_server_tier[t] = FedAvg(w_locals_server_tier[t])
                        if inter_tier_fedavg:
                            w_glob_server_tier = multi_fedavg(w_glob_server_tier, num_tiers, client_number_tier, client_tier, idx_collect, agent = 'server')
    
                        for t in range(1, num_tiers+1):
                            net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])  
                        # k = 0
                        # for i in range(len(client_number_tier)):
                        #     for j in range(int(client_number_tier[i])):
                        #         net_model_server_tier[k] = net_glob_server_tier[i+1]
                        #         k +=1
                        if args.version == 1:
                            for i in client_tier.keys():  # assign each server-side to its tier model  # move it to the end of client program
                                net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]
                    # k=0
                    # for i in range(len(client_number_tier)):
                    #     for j in range(int(client_number_tier[i])):
                    #         net_glob_server.load_state_dict(w_glob_server_tier[k])
                    #         net_model_server_tier[k] = net_glob_server_tier[i+1]
                    #         k +=1
                elif args.whether_aggregated_federation == 1:
                    w_locals_tier = w_locals_server
                w_locals_server = []
                w_locals_server_tier = {}
                for i in range(1,num_tiers+1):
                    w_locals_server_tier[i]=[]
                # w_glob_server_tier = {}
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
            wandb.log({"Server_Training_Time": time_train_server_train_all, "epoch": l_epoch_count}, commit=False)
            # time_train_server_train_all = 0
            # scheduler_server.step(best_acc, epoch=l_epoch_count) #from fedgkt
            print("Server LR: ", optimizer_server.param_groups[0]['lr'])
            new_lr = optimizer_server.param_groups[0]['lr']
            wandb.log({"Server_LR": optimizer_server.param_groups[0]['lr'], "epoch": l_epoch_count}, commit=False)
            
    
    # print(time_train_server_copy, time_train_server_train)
    # send gradients to the client               
    # return dfx_client
    if whether_GKT_local_loss:
        return fx_server  # output of server model gkt
    elif whether_distillation_on_clients:
        return dfx_server, dfx_client
    else:
        return dfx_client  # output of server 

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server, net_glob_server_tier 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check, w_glob_server_tier
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, acc_avg_all_user, loss_avg_all_user_train, best_acc
    global wait, new_lr
    
    # if iter == 5:
    #     # idx = 1
    #     print(iter)
    if args.version == 1:
        if whether_multi_tier:
            net = copy.deepcopy(net_model_server_tier[idx]).to(device)
        else:
            net = copy.deepcopy(net_model_server[idx]).to(device)
    elif args.version == 2:
        if whether_multi_tier:
            net = copy.deepcopy(net_model_server_tier[client_tier[idx]]).to(device)
        else:
            net = copy.deepcopy(net_model_server[client_tier[idx]]).to(device)
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
        # if whether_dcor:
        #     loss += dcor_coefficient * dis_corr(y,fx_server)
        # calculate accuracy
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
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            wandb.log({"Client{}_Test_Accuracy".format(idx): acc_avg_test, "epoch": 22}, commit=False)

            if loss_avg_test > 100:
                print(loss_avg_test)
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
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                
                
                if (acc_avg_all_user/100) > best_acc  * ( 1 + lr_threshold ):
                    print("- Found better accuracy")
                    best_acc = (acc_avg_all_user/100)
                    wait = 0
                else:
                     wait += 1 
                     print('wait', wait)
                if wait > patience:   #https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
                    # factor = 0.8
                    new_lr = max(float(optimizer_server.param_groups[0]['lr']) * factor, min_lr)
                    wait = 0
                    # optimizer_server.param_groups[0]['lr'] = new_lr
                    
                    
                              
                print("====================== SERVER V1==========================")
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
            
        # if args.optimizer == "Adam":
        #     self.optimizer_client =  torch.optim.Adam(net_client_model.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
        # elif args.optimizer == "SGD":
        #     self.optimizer_client =  torch.optim.SGD(net_client_model.parameters(), lr=lr, momentum=0.9,
        #                                               nesterov=True,
        #                                               weight_decay=args.wd)
        # self.scheduler_client = ReduceLROnPlateau(self.optimizer_client, 'max', factor=0.5, patience=1, threshold=0.0000001)
        # self.best_acc = 0.0
                
        

    def train(self, net):
        net.train()
        self.lr , lr = new_lr, new_lr
        # optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        # optimizer_client = optimizer_client_tier[client_tier[idx]]
        
        ## server-side initializaion 
        # if args.version == 1:
        #     if not whether_multi_tier:
        #         net_server = copy.deepcopy(net_model_server[idx]).to(device)
        #     else:
        #         net_server = copy.deepcopy(net_model_server_tier[idx]).to(device)
        # elif args.version == 2:
        #     if not whether_multi_tier:
        #         net_server = copy.deepcopy(net_model_server[client_tier[idx]]).to(device)
        #     else:
        #         net_server = copy.deepcopy(net_model_server_tier[client_tier[idx]]).to(device)
            
        # net_server.train()
        
        # global optimizer_server
        # if args.optimizer == "Adam":# and False:
        #     optimizer_server =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
        # elif args.optimizer == "SGD":
        #     optimizer_server =  torch.optim.SGD(net_server.parameters(), lr=lr, momentum=0.9,
        #                                           nesterov=True,
        #                                           weight_decay=args.wd)


        if args.optimizer == "Adam":
            optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
        elif args.optimizer == "SGD":
            optimizer_client =  torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                                      nesterov=True,
                                                      weight_decay=args.wd)
        # optimizer_client = optimizer_client_tier[idx]
        scheduler_client = ReduceLROnPlateau(optimizer_client, 'max', factor=0.8, patience=-1, threshold=0.0000001)
        # print('client LR: ', optimizer_client.param_groups[0]['lr'])
        
        
        # optimizer_client = self.optimizer_client

        
        time_client=0
        data_transmited_sl_client = 0
        batch_size = args.batch_size
        CEloss_client_train = []
        Dcorloss_client_train = []
        KDloss_client_train = []
        if args.whether_FedAVG_base:
            epoch_acc = []
            epoch_loss = []
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            if args.whether_FedAVG_base:
                 batch_acc = []
                 batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                if args.optimizer == "Adam":
                    optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
                elif args.optimizer == "SGD":
                    optimizer_client =  torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                                              nesterov=True,
                                                              weight_decay=args.wd)
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                
                if args.whether_FedAVG_base:
                    fx = net(images)
                    if args.dataset != 'HAM10000':
                        labels = labels.to(torch.long)
                    loss = criterion(fx, labels)
                    acc = calculate_accuracy(fx, labels)
                    loss.backward()
                    optimizer_client.step()
                    time_client += time.time() - time_s
                    batch_loss.append(loss.item())
                    batch_acc.append(acc.item())
                    
                else:
                    
                    #---------forward prop-------------
                    if whether_GKT_local_loss or whether_local_loss:
                        extracted_features, fx = net(images)
                    else:
                        fx = net(images)
                    client_fx = fx.clone().detach().requires_grad_(True)
                    # fx.backward(fx) # backpropagation
                    # hidden.detach_()
                    # Sending activations to server and receiving gradients from server
                    time_client += time.time() - time_s
                    if whether_GKT_local_loss:
                        dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, extracted_features)
                    if whether_distillation_on_clients:
                        fx_server, dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, _)
                    else:
                        dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, _)
                    
                    
                    #--------backward prop -------------
                    time_s = time.time()
                    if whether_local_loss:
                        
                        if args.dataset != 'HAM10000':
                            labels = labels.to(torch.long)
                        loss = criterion(extracted_features, labels) # to solve change dataset)
                        CEloss_client_train.append(((1 - dcor_coefficient)*loss.item()))    
                        
                        
                        if whether_distillation_on_clients:
                            batch_logits = extracted_features
                            loss_kd = criterion_KL(batch_logits, fx_server)
                            KDloss_client_train.append(loss_kd.item())                    
                            # loss = args.KD_beta * loss_kd + (1 - dcor_coefficient) * loss # change coeficient 20220716
                            loss = loss_kd + (1 - dcor_coefficient) * loss
                            
                        if whether_dcor:
                            Dcor_value = dis_corr(images,fx)
                            loss = (1 - dcor_coefficient) * loss + dcor_coefficient * Dcor_value
                            Dcorloss_client_train.append(((dcor_coefficient) * Dcor_value))   

                        loss.backward()
    
                    elif not whether_GKT_local_loss:
                        fx.backward(dfx) # backpropagation
    
                    else: # for gkt
                                       
                        # calculate loss
                        loss_c = criterion(extracted_features, labels)
                        # KD loss
                        fx_server = dfx
                        batch_logits = extracted_features
                        if whether_distillation_on_the_server == 1:
                            # loss_kd = self.criterion_KL(output_batch, batch_logits).to(self.device)
                            # loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                            # loss = loss_kd + self.args.KD_beta * loss_true
                            loss_kd = criterion_KL(batch_logits, fx_server)
                            loss_c = loss_kd + (1 - args.KD_beta) * loss_c
                            
                        optimizer_client.zero_grad()
                        # loss_c.backward(inputs=list(dec.parameters()))
                        loss_c.backward(inputs=list(net.parameters()))  # for fedgkt
                        # loss.backward(extracted_features)
                    optimizer_client.step()
                    time_client += time.time() - time_s
                    
                    
                    if whether_GKT_local_loss:
                        data_transmited_sl_client += (sys.getsizeof(client_fx.storage()) + 
                                              sys.getsizeof(extracted_features.storage()) + sys.getsizeof(labels.storage()) + 
                                              sys.getsizeof(dfx.storage()))
                    elif whether_local_loss:
                        data_transmited_sl_client += (sys.getsizeof(client_fx.storage()) + 
                                              sys.getsizeof(labels.storage()))
                        if whether_distillation_on_clients:
                            data_transmited_sl_client += sys.getsizeof(fx_server.storage())
                    else:
                        data_transmited_sl_client += (sys.getsizeof(client_fx.storage()) + 
                                              sys.getsizeof(labels.storage()) + sys.getsizeof(dfx.storage()))
            if args.whether_FedAVG_base:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_acc.append(sum(batch_acc)/len(batch_acc))
                prRed('Client{} Train => Local Epoch: {}  \tAcc: {:.3f} \tLoss: {:.4f}'
                      .format(self.idx,iter, epoch_acc[-1], epoch_loss[-1]))

            
            # self.scheduler_client.step(best_acc)
            # scheduler_client.step(best_acc)
            # print("client LR:",optimizer_client.param_groups[0]['lr'])
        global data_transmited_sl
        data_transmited_sl += data_transmited_sl_client          
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
        # wandb.log({"Client{}_CELoss".format(idx): sum(CEloss_client_train)/len(CEloss_client_train), "epoch": iter}, commit=False)
        # wandb.log({"Client{}_KDLoss".format(idx): sum(KDloss_client_train)/len(KDloss_client_train), "epoch": iter}, commit=False)            
        
        # clients log
        # wandb.log({"Client{}_CELoss".format(idx): float(sum(CEloss_client_train)), "epoch": iter}, commit=False)
        wandb.log({"Client{}_DcorLoss".format(idx): float(sum(Dcorloss_client_train)), "epoch": iter}, commit=False)
        # wandb.log({"Client{}_KDLoss".format(idx): float(sum(KDloss_client_train)), "epoch": iter}, commit=False)            
        wandb.log({"Client{}_Training_Duration (s)".format(idx): time_client, "epoch": iter}, commit=False)
        print(f"Client{idx}_Training_Duration: {time_client:,.3f} (s)")
        
        if args.whether_FedAVG_base:
            return net.state_dict(), time_client, data_transmited_sl_client, sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc) 
        return net.state_dict(), time_client, data_transmited_sl_client 
    
    def evaluate(self, net, ell):
        net.eval()
        if args.whether_FedAVG_base:
            epoch_acc = []
            epoch_loss = []
           
        with torch.no_grad():
            if args.whether_FedAVG_base:
                batch_acc = []
                batch_loss = []
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                if args.whether_FedAVG_base:
                    fx = net(images)
                    if args.dataset != 'HAM10000':
                        labels = labels.to(torch.long)
                    loss = criterion(fx, labels)
                    acc = calculate_accuracy(fx, labels)
                    batch_loss.append(loss.item())
                    batch_acc.append(acc.item())

                elif whether_GKT_local_loss or whether_local_loss :
                    extracted_features, fx = net(images)
                # Sending activations to server 
                    evaluate_server(fx, labels, self.idx, len_batch, ell)

                else:
                    fx = net(images)
                # Sending activations to server 
                    evaluate_server(fx, labels, self.idx, len_batch, ell)
                
            
            if args.whether_FedAVG_base:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_acc.append(sum(batch_acc)/len(batch_acc))
                prGreen('Client{} Test =>                     \tAcc: {:.3f} \tLoss: {:.4f}'
                        .format(self.idx, epoch_acc[-1], epoch_loss[-1])) 
                
                return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return 

    def evaluate_glob(self, net, ell):
        net.eval()
        epoch_acc = []
        epoch_loss = []
           
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            len_batch = len(self.ldr_test)
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
            prGreen('Client{} Test =>                     \tAcc: {:.3f} \tLoss: {:.4f}'
                    .format(self.idx, epoch_acc[-1], epoch_loss[-1])) 
                
            return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
                
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users):
    
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
    df = pd.read_csv('data/HAM10000_metadata.csv')
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
    
    
    #print("path---------------------------------------", imageid_path.get)
    df['path'] = df['image_id'].map(imageid_path.get)
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes
    print(df['cell_type'].value_counts())
    print(df['target'].value_counts())

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
    
    df = df[1:10015:20]
    train, test = train_test_split(df, test_size = 0.2)
    
    train = train.reset_index()
    test = test.reset_index()

#=============================================================================
#                         Data preprocessing
#=============================================================================  
# Data preprocessing: Transformation
if args.dataset == "HAM10000": 
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
data_transmitelist_sl =[]
data_transmitelist_fl =[]
client_tier_all = []
client_tier_all.append(copy.deepcopy(client_tier))
total_training_time = 0
time_train_server_train_all_list = []

client_sample = np.ones(num_users)


for i in range(0, num_users):
    wandb.log({"Client{}_Tier".format(i): client_tier[i], "epoch": -1}, commit=False)

#------------ Training And Testing  -----------------
net_glob_client.train()
w_glob_client_tier ={}

#copy weights
for i in range(1, num_tiers+1):
    w_glob_client_tier[i] = net_glob_client_tier[i].state_dict()

# net_glob_client_tier[tier].load_state_dict(w_glob_client)
w_glob_client_tier[tier] = net_glob_client_tier[tier].state_dict()

client_sample = np.ones(num_tiers)
# to start with same weigths 
for i in range(1, num_tiers+1):
    net_glob_client_tier[i].to(device)
    
if args.whether_aggregated_federation == 1 and not args.whether_FedAVG_base:
    # w_locals_tier, w_locals_client = [], []
    # client_sample = np.ones(num_tiers)
    
    # for t in range(1, num_tiers+1):
    #     w = net_glob_server_tier[t].state_dict()
    #     w_locals_tier.append(copy.deepcopy(w))
    #     w = net_glob_client_tier[t].state_dict()
    #     w_locals_client.append(copy.deepcopy(w))
        
    # w_glob = aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample) # w_locals_tier is for server-side
    #print('w_glob',w_glob.keys())
    # print('init',init_glob_model.state_dict().keys())
    w_glob = copy.deepcopy(init_glob_model.state_dict())
    
    for t in range(1, num_tiers+1):
        if whether_federation_at_clients:
            for k in w_glob_client_tier[t].keys():
                k1 = k
                if k.startswith('module'):
                    #k1 = 'module'+k
                    k1 = k1[7:]
                
                if (k == 'fc.bias' or k == 'fc.weight'):
                # if (k == 'module.fc.bias' or k == 'module.fc.weight'):
                    continue 
                
                w_glob_client_tier[t][k] = w_glob[k1]
        if args.version == 1 or args.wheter_agg_tiers_on_server == 1:
            for k in w_glob_server_tier[t].keys():
                k1 = k
                if k.startswith('module'):
                    #k1 = 'module'+k
                    k1 = k1[7:]
                w_glob_server_tier[t][k] = w_glob[k1]
            
        net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
        net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
        
    w_locals_tier, w_locals_client, w_locals_server = [], [], []


# w_glob_client = init_glob_model.state_dict() # copy weights
# net_glob_client_tier[tier].load_state_dict(w_glob_client) # copy weights
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
        optimizer_client_tier[i] =  torch.optim.Adam(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
    elif args.optimizer == "SGD":
        optimizer_client_tier[i] =  torch.optim.SGD(net_glob_client_tier[client_tier[i]].parameters(), lr=lr, momentum=0.9,
                                                          nesterov=True,
                                                          weight_decay=args.wd)


# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds

df_delay = pd.DataFrame()
start_time = time.time() 

client_times = pd.DataFrame()
# client_times = client_times.append(pd.DataFrame(np.zeros(num_users)).T, ignore_index = True)
torch.manual_seed(SEED)
delay_actual= np.zeros(num_users)
# delay_actual= np.empty(num_users)
# delay_actual[:] = np.NaN

for i in range(0, num_users):
    data_server_to_client = 0
    for k in w_glob_client_tier[client_tier[i]]:
        data_server_to_client += sys.getsizeof(w_glob_client_tier[client_tier[i]][k].storage())
    delay_actual[i] = data_server_to_client / net_speed[i]

# for iter in range(epochs):
#     if iter == int(epochs / 4) and True:
        # for c in range(0, num_users):
        #     # if c % 5 == 0 :
        #     #     delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(60, 25, 10, 5, 0))[0]
        #     # elif c % 5 == 1:
        #     #     delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(15, 60, 15, 10, 0))[0]
        #     # elif c % 5 == 2:
        #     #     delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(5, 15, 60, 15, 5))[0]
        #     # elif c % 5 == 3:
        #     #     delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(0, 10, 15, 60, 15))[0]
        #     # elif c % 5 == 4:
        #     #     delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(0, 5, 10, 25, 60))[0]
                
        #     if c % 5 == 0 :
        #         delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(20, 20, 20, 20, 20))[0]
        #     elif c % 5 == 1:
        #         delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(15, 60, 15, 10, 0))[0]
        #     elif c % 5 == 2:
        #         delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(5, 15, 60, 15, 5))[0]
        #     elif c % 5 == 3:
        #         delay_coefficient[c] = random.choices(delay_coefficient_list, weights=(0, 10, 15, 60, 15))[0]
        #     elif c % 5 == 4:
for iter in range(epochs):
    if iter == int(10) and False:
        delay_coefficient[0] = delay_coefficient_list[2]
        delay_coefficient[1] = delay_coefficient_list[3]
        delay_coefficient[2] = delay_coefficient_list[4]
        delay_coefficient[3] = delay_coefficient_list[0]
        delay_coefficient[4] = delay_coefficient_list[0]
        delay_coefficient[5] = delay_coefficient_list[4]
        delay_coefficient[6] = delay_coefficient_list[4]
        delay_coefficient[7] = delay_coefficient_list[0]
        delay_coefficient[8] = delay_coefficient_list[1]
        delay_coefficient[9] = delay_coefficient_list[1]
    elif iter == int(20) and False:
        delay_coefficient[0] = delay_coefficient_list[4]
        delay_coefficient[1] = delay_coefficient_list[3]
        delay_coefficient[2] = delay_coefficient_list[2]
        delay_coefficient[3] = delay_coefficient_list[3]
        delay_coefficient[4] = delay_coefficient_list[2]
        delay_coefficient[5] = delay_coefficient_list[3]
        delay_coefficient[6] = delay_coefficient_list[3]
        delay_coefficient[7] = delay_coefficient_list[2]
        delay_coefficient[8] = delay_coefficient_list[0]
        delay_coefficient[9] = delay_coefficient_list[1]
    elif iter == int(30) and False:
        delay_coefficient[0] = delay_coefficient_list[4]
        delay_coefficient[1] = delay_coefficient_list[2]
        delay_coefficient[2] = delay_coefficient_list[4]
        delay_coefficient[3] = delay_coefficient_list[3]
        delay_coefficient[4] = delay_coefficient_list[2]
        delay_coefficient[5] = delay_coefficient_list[1]
        delay_coefficient[6] = delay_coefficient_list[3]
        delay_coefficient[7] = delay_coefficient_list[0]
        delay_coefficient[8] = delay_coefficient_list[1]
        delay_coefficient[9] = delay_coefficient_list[2]
                
        # delay_coefficient = list(np.roll(delay_coefficient,1))
    if args.whether_FedAVG_base:
        loss_locals_train, acc_locals_train, loss_locals_test, acc_locals_test = [], [], [], []
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []
    w_locals_client_tier = {}
    for i in range(1,num_tiers+1):
        w_locals_client_tier[i]=[]
    client_time = np.zeros(num_users)
    # delay_actual= np.zeros(num_users)
    for i in range(0, num_users):
        wandb.log({"Client{}_Tier".format(i): client_tier[i], "epoch": -1}, commit=False)
    if args.dataset != "HAM10000": 
        client_sample = []
    # client_tier_all.append(copy.deepcopy(client_tier)) # check deepcopy problem
    args.KD_beta = args.KD_beta_init + (-args.KD_beta_init) * math.exp(-1. * args.KD_increase_factor * iter / args.rounds )     #    Eps = Eps_end + (Eps_start - Eps_end) * \math.exp(-1. * step / Eps_Decay)
    wandb.log({"KD_beta": args.KD_beta, "epoch": iter}, commit=False)
    print('KD_beta',args.KD_beta)
      
    for idx in idxs_users:
        data_transmited_fl_client = 0
        time_train_test_s = time.time()
        if whether_multi_tier:
            net_glob_client = net_model_client_tier[client_tier[idx]]
            w_glob_client_tier[client_tier[idx]] = net_glob_client_tier[client_tier[idx]].state_dict() # may be I can eliminate this line
            # net_glob_client.train()
        if args.dataset == "HAM10000":
        #     dataset_train = dataset_train[idx]
        #     dataset_test = dataset_test[idx]
            local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # print('dataset_train: ',len(dataset_train))
        else:
            local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = [], idxs_test = [])
            # print('dataset_train: ',train_data_local_num_dict[idx]) #num batch * batch size
            
        if args.test_before_train == 1:# and iter % 20 == 0:
            if args.whether_FedAVG_base and idx == idxs_users[0]:
            #if args.whether_FedAVG_base and idx == idxs_users[0] and (iter % int(args.rounds / 1) == 0):
                loss_test, acc_test = local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
                loss_locals_test.append(copy.deepcopy(loss_test))
                acc_locals_test.append(copy.deepcopy(acc_test))
            #elif idx == idxs_users[0] and (iter % int(args.rounds / 1) == 0):
            elif idx == idxs_users[0]:
                local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        # Training ------------------
        if args.whether_FedAVG_base:
            [w_client, duration, data_transmited_sl_client, loss_train, acc_train] = local.train(net = copy.deepcopy(net_glob_client).to(device))    
            loss_locals_train.append(copy.deepcopy(loss_train))
            acc_locals_train.append(copy.deepcopy(acc_train))
        else:
            # process = Process(target = local.train, args = [ copy.deepcopy(net_glob_client).to(device)])
            # process.start()
            
            [w_client, duration, data_transmited_sl_client] = local.train(net = copy.deepcopy(net_glob_client).to(device))
        if args.dataset != "HAM10000": 
            # client_sample[idx] = train_data_local_num_dict[idx] / sum(train_data_local_num_dict.values()) * num_users
            client_sample.append(train_data_local_num_dict[idx] / sum(train_data_local_num_dict.values()) * num_users)
            #client_sample.append(1)
            # print('the ratio of samples: ',client_sample[-1])
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_client_tier[client_tier[idx]].append(copy.deepcopy(w_client))
        
        
        # Testing -------------------  # why for testing do not use last weight update of that client?? # at this point it use last upate of weights
        if args.test_before_train == 0:
            net = copy.deepcopy(net_glob_client)
            w_previous = copy.deepcopy(net.state_dict())  # to test for updated model
            net.load_state_dict(w_client)
            net.to(device)
            
            if args.whether_FedAVG_base:
                loss_test, acc_test = local.evaluate(net, ell= iter)
                loss_locals_test.append(copy.deepcopy(loss_test))
                acc_locals_test.append(copy.deepcopy(acc_test))
            else:
                local.evaluate(net, ell= iter)
            net.load_state_dict(w_previous) # to return to previous state for other clients
            
        client_time[idx] = duration
        # wandb.log({"Client{}_Training_Duration_new (s)".format(idx): duration, "epoch": iter}, commit=False)
        if not whether_GKT_local_loss:   # this is sum of model size of all clients sent to server
            for k in w_client:
                data_transmited_fl_client = data_transmited_fl_client + sys.getsizeof(w_client[k].storage())
        data_transmited_fl += data_transmited_fl_client         
        # delay_actual[idx] = (time.time() - time_train_test_s)
        data_transmited_client = data_transmited_sl_client + data_transmited_fl_client
        
        # if iter == 0 and idx == idxs_users[0]:
        #     duration = 0.03 * duration
        
        # delay_actual[idx] = (delay_coefficient[idx] * duration + data_transmited_client / net_speed 
        #                      + client_delay_mu[idx] * np.random.rand() + client_delay_c[idx])
        delay_actual[idx] += ((delay_coefficient[idx] * duration * (1 + np.random.rand()* client_delay_computing) 
                              + data_transmited_client / net_speed[idx] *  (1 + np.random.rand() * client_delay_net))/client_epoch[idx]) # this is per epoch
        total_time += delay_actual[idx]
        # wandb.log({"Client{}_Training_Duration (s)".format(idx): delay_actual[idx], "epoch": iter}, commit=False)
        wandb.log({"Client{}_Actual_Delay".format(idx): delay_actual[idx], "epoch": iter}, commit=False)
        wandb.log({"Client{}_Data_Transmission(MB)".format(idx): data_transmited_client/1024**2, "epoch": iter}, commit=False)
    server_wait_time = (max(delay_actual * client_epoch) - min(delay_actual * client_epoch))
    wandb.log({"Server_wait_time": server_wait_time, "epoch": iter}, commit=False)
    wandb.log({"Total_training_time": total_time, "epoch": iter}, commit=False)
    if args.whether_FedAVG_base:
        training_time = (max(delay_actual))
    else:
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
    if not args.whether_FedAVG_base:
        
        [client_tier, client_epoch, avg_tier_time_list, max_time_list, client_times] = dynamic_tier9(client_tier_all[:], df_delay, 
                                                    num_tiers, server_wait_time, client_epoch,
                                                    time_train_server_train_all_list, num_users, iter,
                                                    sataset_size = sataset_size, avg_tier_time_list = avg_tier_time_list,
                                                    max_time_list = max_time_list, idxs_users = idxs_users) # assign next tier and model
        #print(max_time_list)
        #print(max_time_list[-1])                                            
        wandb.log({"max_time": float(max_time_list.loc[len(max_time_list)-1]), "epoch": iter}, commit=False)
                                                    
    # reward = (pow(2,(acc_avg_all_user/100)) - 1 ) * 100 / training_time * first_training_time # self.config.fl.target_accuracy = 0
    # reward = (pow(2,(100/100)) - 1 ) * 100 / training_time * first_training_time # 
    # reward = (first_training_time - training_time) / first_training_time * 100 # 
    # print ('reward' , reward)
    # [client_tier, client_epoch] = dqn_agent1(client_tier, df_delay, client_epoch, reward, iter)
    client_tier_all.append(copy.deepcopy(client_tier))
    
    delay_actual= np.zeros(num_users)
    for i in range(0, num_users):
        data_server_to_client = 0
        for k in w_glob_client_tier[client_tier[i]]:
            data_server_to_client += sys.getsizeof(w_glob_client_tier[client_tier[i]][k].storage())
        delay_actual[i] = data_server_to_client / net_speed[i]
        # wandb.log({"Client{}_Tier".format(i): client_tier[i], "epoch": iter}, commit=False)
    
    if args.version == 1:
        for i in client_tier.keys():  # assign each server-side to its tier model
            net_model_server_tier[i] = net_glob_server_tier[client_tier[i]]
    
    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    if args.whether_FedAVG_base:
        
        w_glob = FedAvg(w_locals_client)
        # w_glob = FedAvg_wighted(w_locals_client, torch.tensor(client_sample))
        # w_glob = aggregated_fedavg([], w_locals_client, num_tiers, num_users, whether_local_loss, client_sample) # to solve weighted problem of FedAvg
        # w_glob = aggregated_fedavg(w_locals_client, w_locals_client, num_tiers, num_users, whether_local_loss, np.divide(client_sample,2), idxs_users)# to solve weighted problem of FedAvg
        net_glob_client_tier[1].load_state_dict(w_glob)
        # Train/Test accuracy
        acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
        acc_train_collect.append(acc_avg_train)
        
        #if (iter % int(args.rounds / 1) == 0):
        if True:# and iter % 20 == 0:
            acc_avg_test = sum(acc_locals_test) / len(acc_locals_test)
            #acc_avg_test = sum(acc_locals_test) / 1
            # print(acc_locals_test)
            acc_test_collect.append(acc_avg_test)
            loss_avg_test = sum(loss_locals_test) / len(loss_locals_test)
            #loss_avg_test = sum(loss_locals_test) / 1
            # print(loss_locals_test)
            loss_test_collect.append(loss_avg_test)
            
        else:
            acc_avg_test = 0
            loss_avg_test = 0
            
            acc_test_collect.append(acc_test_collect[-1])
            loss_test_collect.append(loss_test_collect[-1])
        
        # Train/Test loss
        loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
        loss_train_collect.append(loss_avg_train)
        
        print('------------------- SERVER ----------------------------------------------')
        print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_train, loss_avg_train))
        print('Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_test, loss_avg_test))
        print('-------------------------------------------------------------------------')
        
        wandb.log({"Server_Training_Accuracy": acc_avg_train, "epoch": iter}, commit=False)
        wandb.log({"Server_Test_Accuracy": acc_avg_test, "epoch": iter}, commit=False)
        wandb.log({"Clients_LR": lr, "epoch": iter}, commit=False)
        
        if (acc_avg_test/100) > best_acc  * ( 1 + lr_threshold ):
        #if (acc_avg_train/100) > best_acc  * ( 1 + lr_threshold ):
            print("- Found better accuracy")
            #best_acc = (acc_avg_train/100)
            best_acc = (acc_avg_test/100)
            wait = 0
        else:
             wait += 1 
             print('wait', wait)
        if args.whether_dynamic_lr_client == 1:
            
            if wait > patience:   #https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
                # factor = 0.8
                new_lr = max(lr * factor, min_lr)
                lr = new_lr
                wait = 0
                # optimizer_server.param_groups[0]['lr'] = new_lr

            
    elif not whether_GKT_local_loss: # and whether_federation_at_clients:  
        if args.whether_aggregated_federation == 1:
            if args.wheter_agg_tiers_on_server == 1:  # this part help to have between tier aggregation with last update of clients
                for n in list(idxs_users)[::-1]:
                    for m in [k for k,v in client_tier_all[-2].items() if v == client_tier_all[-2][n]]:
                        #w_locals_tier[list(idxs_users).index(m)] = []
                        # aa = (len(w_locals_tier[m]))
                        #w_locals_tier[client_tier_all[-2][m]] = w_locals_tier[client_tier_all[-2][n]]
                        w_locals_tier[list(idxs_users).index(m)] = copy.deepcopy(w_locals_tier[list(idxs_users).index(n)])
                        # print(len(w_locals_tier[m]))
                        # if aa != len(w_locals_tier[m]):
                            # print('error')
            # client_sample = np.ones(num_users) # for test
            if args.whether_local_loss_v2:
                w_glob = aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample, idxs_users, local_v2 = args.whether_local_loss_v2) # w_locals_tier is for server-side
            else:
                w_glob = aggregated_fedavg(w_locals_tier, w_locals_client, num_tiers, num_users, whether_local_loss, client_sample, idxs_users) # w_locals_tier is for server-side
            
            for t in range(1, num_tiers+1):
                if whether_federation_at_clients:
                    for k in w_glob_client_tier[t].keys():
                        # if (k == 'fc.bias' or k == 'fc.weight'):# or args.whether_local_loss_v2:  # this part avg on same tiers on fc layer, so in v2 local we do not need this
                        if k in w_glob_server_tier[t].keys():  # This is local updading  // another method can be updating and supoose its similar to global model
                        # if (k == 'module.fc.bias' or k == 'module.fc.weight'):
                            if w_locals_client_tier[t] != []:
                                w_glob_client_tier[t][k] = FedAvg(w_locals_client_tier[t])[k]
                                continue
                            else:
                                continue 
                        
                        w_glob_client_tier[t][k] = w_glob[k]
                if args.version == 1 or args.wheter_agg_tiers_on_server == 1:
                    for k in w_glob_server_tier[t].keys():
                        w_glob_server_tier[t][k] = w_glob[k]
                    
                net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])
                net_glob_server_tier[t].load_state_dict(w_glob_server_tier[t])
            #print(w_glob_server_tier[t].keys(),w_glob.keys())
            #global_model.load_state_dict(w_glob)
 
    # Update client-side global model 
        

                # for t in range(1,num_tiers+1):
                #     if w_locals_server_tier[t] != []:
                #         w_glob_server_tier[t] = FedAvg(w_locals_server_tier[t])
                # for i in range(1, num_tiers+1):
                #     w_glob_server_tier[i] = FedAvg(w_locals_server_tier)
            
            # server-side global model update and distribute that model to all clients ------------------------------
        elif not whether_multi_tier:
            w_glob_client = FedAvg(w_locals_client) # fedavg at client side  
            net_glob_client.load_state_dict(w_glob_client)  
            for k in w_client:      # this is sum of model size of all clients server sent to each clients
                data_transmited_fl = data_transmited_fl + len(w_locals_client) * sys.getsizeof(w_glob_client[k].storage())
            
        else:
            if intra_tier_fedavg:
                for t in range(1,num_tiers+1):
                    if w_locals_client_tier[t] != []:
                        w_glob_client_tier[t] = FedAvg(w_locals_client_tier[t])
            if inter_tier_fedavg:
                w_glob_client_tier = multi_fedavg(w_glob_client_tier, num_tiers, client_number_tier, client_tier, idx_collect, agent = 'client')
            for t in range(1, num_tiers+1):
                net_glob_client_tier[t].load_state_dict(w_glob_client_tier[t])  
                
            for t in range(1,num_tiers+1):
                for k in w_glob_client_tier[t]:      # this is sum of model size of all clients server sent to each clients
                    data_transmited_fl = data_transmited_fl + client_number_tier[t-1] * sys.getsizeof(w_glob_client_tier[t][k].storage())
            
            # k = 0
            # for i in range(len(client_number_tier)):
            #     for j in range(int(client_number_tier[i])):
            #         net_model_client_tier[k] = net_glob_client_tier[i+1]
            #         k +=1
                            
                # check for data for federation
        # for k in w_client:
            # data_transmited_fl = data_transmited_fl + sys.getsizeof(w_glob_client[k].storage())
    # data_transmitelist.append(f'{(data_transmited/1024**2):,.2f}')
    # if args.global_model == 1:
        
        
    # this part for test of the global model
    # local.evaluate(net = copy.deepcopy(global_model).to(device))
    # if args.dataset == "HAM10000":
    #     local = Client(global_model, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
    # else:
    #     local = Client(global_model, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = [], idxs_test = [])
    # loss_test, acc_test = local.evaluate_glob(net = copy.deepcopy(global_model).to(device), ell= iter)
    # print('global accuracy: ', acc_test)
    # global_model.load_state_dict(w_glob)
    

                    
                    
    
    data_transmitelist_fl.append(data_transmited_fl/1024**2)
    data_transmitelist_sl.append(data_transmited_sl/1024**2)
    print(f'Total Data Transferred FL {(data_transmited_fl/1024**2):,.2f} Mega Byte')
    print(f'Total Data Transferred SL {(data_transmited_sl/1024**2):,.2f} Mega Byte')

    wandb.log({"Model_Parameter_Data_Transmission(MB) ": data_transmited_fl/1024**2, "epoch": iter}, commit=False)
    wandb.log({"Intermediate_Data_Transmission(MB) ": data_transmited_sl/1024**2, "epoch": iter}, commit=True)
    
    
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')
    
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
#round_process = [i for i in range(1, len(acc_train_collect)+1)]
#df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect,
#                'Data Transmitted(GB) for federation process':data_transmitelist_fl,
#                'Data Transmitted(GB) for split learning process':data_transmitelist_sl,
#                'total program time': f'Total Training Time: {elapsed:.2f} min'}) 
#file_name = program+".xlsx"    
# df.to_excel(file_name, sheet_name= "v1_test", index = False)
# client_times.to_excel(file_name, sheet_name= "Client Training Time", index = False)

#with pd.ExcelWriter(program+".xlsx") as writer:
#    df.to_excel(writer, sheet_name= "Accuracy", index = False)
#    client_times.to_excel(writer, sheet_name='Client Training Time', index = False)
#    df_delay.to_excel(writer, sheet_name='Iteration Training Time', index = False)
     

#=============================================================================
#                         Program Completed
#=============================================================================