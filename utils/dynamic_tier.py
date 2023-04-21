import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

def dynamic_tier(client_tier, client_times, num_tiers, server_wait_time, client_epoch):
    present_time = client_times.ewm(com=0.5).mean()[-1:]
    client_tier_last = client_tier
    min_time = min(client_times.iloc[-1])
    # min_time = min(present_time.iloc[-1])
    for c in client_tier.keys():
        if present_time[c].item() < min_time + (0.2 * server_wait_time):  # change it based on network time need
            client_tier[c] = 1
        elif present_time[c].item() < min_time + (0.4 * server_wait_time):
            client_tier[c] = 2
        elif present_time[c].item() < min_time + (0.6 * server_wait_time):
            client_tier[c] = 3
        elif present_time[c].item() < min_time + (0.8 * server_wait_time):
            client_tier[c] = 4
        else:
            client_tier[c] = 5
           
            
        # adapt client epoch
        max_increase_training_time = (0.9 * server_wait_time)
        if (client_tier[c] == 1) and (present_time[c].item() < max_increase_training_time):
            #client_epoch[c] += max_increase_training_time // present_time[c].item()
            client_epoch[c] = 1 + max_increase_training_time // present_time[c].item()
        else:
            client_epoch[c] = 1
            
            
        # if (client_tier_last [c] - 1) <= client_tier[c] <= (client_tier_last [c] - 1):  # prevent jumping
        #     client_tier[c] = client_tier_last[c]
    
    return client_tier, client_epoch


def dynamic_tier2(client_tier, client_times, num_tiers, server_wait_time, client_epoch):
    present_time = client_times.ewm(com=0.5).mean()[-1:]
    client_tier_last = client_tier.copy()
    min_time = min(client_times.iloc[-1])
    # num_param_tier_model  = [11_189_127, 2_791_047, 689_159, 162_503, 9_991 ] #proportional to the number of parameters
    num_param_tier_model  = [4, 4, 4, 4, 2] #proportional to the number of layers in clients
    prob_interval = np.cumsum(num_param_tier_model)/np.sum(num_param_tier_model)
    # prob_interval = [0.1, 0.3, 0.5, 0.7]
    # prob_interval = num_param_tier_model / sum(num_param_tier_model)
    # min_time = min(present_time.iloc[-1])
    for c in client_tier.keys():
        if (present_time[c].item()/1) < min_time + (prob_interval[0] * server_wait_time):  # change it based on network time need
            client_tier[c] = 1
        elif (present_time[c].item()/1) < min_time + (prob_interval[1] * server_wait_time):
            client_tier[c] = 2
        elif (present_time[c].item()/1) < min_time + (prob_interval[2] * server_wait_time):
            client_tier[c] = 3
        elif (present_time[c].item()/1) < min_time + (prob_interval[3] * server_wait_time):
            client_tier[c] = 4
        else:
            client_tier[c] = 5
           
            
        # adapt client epoch
        max_increase_training_time = (0.8 * server_wait_time) - 0
        if ((client_tier[c] == 1) and (client_tier_last[c] == 1) and 
            (present_time[c].item() < max_increase_training_time) and len(client_times) > 3):
            #client_epoch[c] += max_increase_training_time // present_time[c].item()
            client_epoch[c] = 1 + max_increase_training_time // present_time[c].item()
        else:
            client_epoch[c] = 1
            
            
        if (client_tier_last [c] + 1) < client_tier[c]:
            client_tier[c] = client_tier_last [c] + 1
        elif (client_tier_last [c] - 1) > client_tier[c]:
            client_tier[c] = client_tier_last [c] - 1
            # <= (client_tier_last [c] - 1):  # prevent jumping
        #     client_tier[c] = client_tier_last[c]
    
    return client_tier, client_epoch

def dynamic_tier3(client_tier, client_times, num_tiers, server_wait_time, client_epoch, time_train_server):
    present_time = client_times.ewm(com=0.5).mean()[-1:]
    present_server_time = pd.DataFrame(time_train_server).ewm(com=0.5).mean()[-1:]

    client_tier_last = client_tier[-1].copy()
    min_time = min(client_times.iloc[-1])  # it can be based on smooth data from start point
    max_time = max(client_times.iloc[-1])
    # num_param_tier_model  = [11_189_127, 2_791_047, 689_159, 162_503, 9_991 ] #proportional to the number of parameters
    num_param_tier_model  = [4, 4, 4, 4, 2] #proportional to the number of layers in clients
    prob_interval = np.cumsum(num_param_tier_model)/np.sum(num_param_tier_model)
    # prob_interval = [0.1, 0.3, 0.5, 0.7]
    # prob_interval = num_param_tier_model / sum(num_param_tier_model)
    # min_time = min(present_time.iloc[-1])
    temp_time = []
    if len(client_tier) > 6:
        print(1)
    for i in range(len(client_tier)):
        for j in range(1,num_tiers+1):
            if client_tier[i] == j:
                temp_time.append(client_times[i][j])
                
    client_tier = client_tier[-1].copy()  
    
    for c in client_tier.keys():
        if (present_time[c].item()/1) < min_time + (prob_interval[0] * server_wait_time):  # change it based on network time need
            client_tier[c] = 1
        elif (present_time[c].item()/1) < min_time + (prob_interval[1] * server_wait_time):
            client_tier[c] = 2
        elif (present_time[c].item()/1) < min_time + (prob_interval[2] * server_wait_time):
            client_tier[c] = 3
        elif (present_time[c].item()/1) < min_time + (prob_interval[3] * server_wait_time):
            client_tier[c] = 4
        else:
            client_tier[c] = 5
           
            
        # adapt client epoch
        max_increase_training_time = (0.8 * server_wait_time) - 10000
        if ((client_tier[c] == 1) and (client_tier_last[c] == 1) and 
            (present_time[c].item() < max_increase_training_time) and len(client_times) > 3):
            #client_epoch[c] += max_increase_training_time // present_time[c].item()
            client_epoch[c] = 1 + max_increase_training_time // present_time[c].item()
        else:
            client_epoch[c] = 1
            
            
        if (client_tier_last [c] + 1) < client_tier[c]:
            client_tier[c] = client_tier_last [c] + 1
        elif (client_tier_last [c] - 1) > client_tier[c]:
            client_tier[c] = client_tier_last [c] - 1
            # <= (client_tier_last [c] - 1):  # prevent jumping
        #     client_tier[c] = client_tier_last[c]
    
    return client_tier, client_epoch


def dynamic_tier4(client_tier, client_times, num_tiers, server_wait_time, client_epoch, time_train_server, num_users, step):
    present_time = client_times.ewm(com=0.5).mean()[-1:]
    present_server_time = pd.DataFrame(time_train_server).ewm(com=0.5).mean()[-1:]
    avg_tier_client_time_serie=pd.Series()
    avg_tier_client_time_serie_list = []
    Eps_start = 0#1#.2
    Eps_end = 0#.05#.01
    Eps_Decay = 50
    Eps = Eps_end + (Eps_start - Eps_end) * \
        math.exp(-1. * step / Eps_Decay)
        
    delay_co = 10
    if num_tiers == 5 :
        delay_co = 10
    elif num_tiers == 7 :
        delay_co = 5

    client_tier_last = client_tier[-1].copy()
    min_time = min(client_times.iloc[-1])  # it can be based on smooth data from start point
    max_time = max(client_times.iloc[-1])
    # num_param_tier_model  = [11_189_127, 2_791_047, 689_159, 162_503, 9_991 ] #proportional to the number of parameters
    num_param_tier_model  = [4, 4, 4, 4, 2] #proportional to the number of layers in clients
    
    avg_tier_client_time = np.ones((num_users,num_tiers)) * 0# max_time
    # avg_tier_client_time = np.append(avg_tier_client_time,np.ones((num_users,1)) * (np.inf), axis=1)
    avg_tier_client_time = np.append(np.ones((num_users,1)) * (np.inf), avg_tier_client_time, axis=1)
    for i in range(0,num_users):
        avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            c = 0
            for t in range(0, len(client_tier)):
                if client_tier[t][i] == j:
                    avg_tier_client_time[i][j] += client_times[i][t]
                    c +=1
            if c>0 :
                avg_tier_client_time[i][j] = avg_tier_client_time[i][j] / c
            if c == 0:
                avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(np.nan,index=[j])]) #avg_tier_client_time_serie.append(pd.Series(np.nan,index=[j]))
            else:
                avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series([avg_tier_client_time[i][j]],index=[j])])
        # avg_tier_client_time_serie_list.append(avg_tier_client_time_serie.interpolate())
        if len(avg_tier_client_time_serie.value_counts()) == 1:
            avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(avg_tier_client_time_serie.dropna().max() * delay_co,index=[0])])
        else:
            avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(np.nan,index=[0])])
        avg_tier_client_time[i] = avg_tier_client_time_serie.sort_index().interpolate(method = "spline", order = 1, limit_direction = "both").tolist()
                
            # elif c == 0 :
            #     avg_tier_client_time[i][j] = (avg_tier_client_time[i][j-1] + avg_tier_client_time[i][j+1]) /2
                
    
    client_tier = client_tier[-1].copy()
    max_time = max(avg_tier_client_time[:,num_tiers]) # T_max is max of tier 5 of all clients
    max_increase_training_time = (0.9 * max_time) #- 10000
    
    for c in client_tier.keys():
        
        # if avg_tier_client_time[c][num_tiers] >= max(avg_tier_client_time[:,num_tiers]):
            # client_tier[c] = num_tiers
        if random.random() >= Eps:# greedy policy

            for t in range(0,num_tiers):
                # if (np.mean(avg_tier_client_time[:,num_tiers-t]) <= avg_tier_client_time[c][num_tiers-t]
                #     < np.mean(avg_tier_client_time[:,num_tiers-t+1])): # <= ubstead of < to help not stup in tier 5
                #     client_tier[c] = max(1,num_tiers -t)
                # if max(avg_tier_client_time[:,num_tiers-t-1]) <= avg_tier_client_time[c][num_tiers-t] < max(avg_tier_client_time[:,num_tiers-t]): # <= ubstead of < to help not stup in tier 5
                #     client_tier[c] = max(1,num_tiers -t)
                if avg_tier_client_time[c][num_tiers-t] <= max_time:
                    client_tier[c] = max(1,num_tiers -t)
                    
        else: # random policy
            client_tier[c] += random.choice([-1,1])
            if client_tier[c] > num_tiers:
                client_tier[c] = num_tiers
            if client_tier[c] < num_tiers:
                client_tier[c] = 1
                
        # client_tier[c] = 5 # always tier 5
                
            
        if client_tier[c] == 1 and 0 < avg_tier_client_time[c][client_tier[c]] < (max_time/2) and client_tier_last[c] == 1 and False:
            client_epoch[c] = max(1, max_increase_training_time // avg_tier_client_time[c][client_tier[c]])
        else:
            client_epoch[c] = 1
            
        if (client_tier_last [c] + 1) < client_tier[c]:
            client_tier[c] = client_tier_last [c] + 1
        elif (client_tier_last [c] - 1) > client_tier[c]:
            client_tier[c] = client_tier_last [c] - 1
            # <= (client_tier_last [c] - 1):  # prevent jumping
        #     client_tier[c] = client_tier_last[c]
    
    #manual_tier = math.ceil((step+1) / 143)
    
    manual_tier = 1
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 1,
         2: 2,
         3: 2,
         5: 3,
         6: 3,
         4: 4,
         7: 4,
         8: 4,
         9: 4,
         10: 6,
         11: 7,
         12: 6,
         13: 6,
         14: 7,
         15: 7}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % num_tiers) + 1
    elif False:
         client_tier = {0: manual_tier,
         1: manual_tier,
         2: manual_tier,
         3: manual_tier,
         5: manual_tier,
         6: manual_tier,
         4: manual_tier,
         7: manual_tier,
         8: manual_tier,
         9: manual_tier,
         10: manual_tier,
         11: manual_tier,
         12: manual_tier,
         13: manual_tier,
         14: manual_tier,
         15: manual_tier}
    elif False:
         client_tier = {0: 1,
         1: 1,
         2: 1,
         3: 1,
         5: 1,
         6: 1,
         4: 1,
         7: 1,
         8: 1,
         9: 1,
         10: 1,
         11: 1,
         12: 1,
         13: 1,
         14: 1,
         15: 1}
            
            
    print("client tier:", client_tier, "local epoch:", client_epoch, "T_max:", max_time)
            
    '''
    prob_interval = np.cumsum(num_param_tier_model)/np.sum(num_param_tier_model)
    # prob_interval = [0.1, 0.3, 0.5, 0.7]
    # prob_interval = num_param_tier_model / sum(num_param_tier_model)
    # min_time = min(present_time.iloc[-1])
    temp_time = []
    if len(client_tier) > 6:
        print(1)
    for i in range(len(client_tier)):
        for j in range(1,num_tiers+1):
            if client_tier[i] == j:
                temp_time.append(client_times[i][j])
                
    client_tier = client_tier[-1].copy()  
    
    for c in client_tier.keys():
        if (present_time[c].item()/1) < min_time + (prob_interval[0] * server_wait_time):  # change it based on network time need
            client_tier[c] = 1
        elif (present_time[c].item()/1) < min_time + (prob_interval[1] * server_wait_time):
            client_tier[c] = 2
        elif (present_time[c].item()/1) < min_time + (prob_interval[2] * server_wait_time):
            client_tier[c] = 3
        elif (present_time[c].item()/1) < min_time + (prob_interval[3] * server_wait_time):
            client_tier[c] = 4
        else:
            client_tier[c] = 5
           
            
        # adapt client epoch
        max_increase_training_time = (0.8 * server_wait_time) - 10000
        if ((client_tier[c] == 1) and (client_tier_last[c] == 1) and 
            (present_time[c].item() < max_increase_training_time) and len(client_times) > 3):
            #client_epoch[c] += max_increase_training_time // present_time[c].item()
            client_epoch[c] = 1 + max_increase_training_time // present_time[c].item()
        else:
            client_epoch[c] = 1
            
            
        if (client_tier_last [c] + 1) < client_tier[c]:
            client_tier[c] = client_tier_last [c] + 1
        elif (client_tier_last [c] - 1) > client_tier[c]:
            client_tier[c] = client_tier_last [c] - 1
            # <= (client_tier_last [c] - 1):  # prevent jumping
        #     client_tier[c] = client_tier_last[c]
    '''
    return client_tier, client_epoch

def dynamic_tier5(client_tier, client_times, num_tiers, server_wait_time, client_epoch, time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 5 # how many previous experiments look at for each tier in  one client
    avg_tier_client_time_serie=pd.Series()
    avg_tier_client_time_serie_list = []
    Eps_start = 0#1#.2
    Eps_end = 0#.05#.01
    Eps_Decay = 50
    Eps = Eps_end + (Eps_start - Eps_end) * \
        math.exp(-1. * step / Eps_Decay)
        
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
    avg_dataset = sum(sataset_size.values()) / len(sataset_size)
    
    delay_co = 10
    if num_tiers == 5 :
        delay_co = 10
    elif num_tiers == 7 :
        delay_co = 20

    client_tier_last = client_tier[-1].copy()
    max_time = max(client_times.iloc[-1])
    
    avg_tier_client_time = np.ones((num_users,num_tiers)) * 0# max_time
    # avg_tier_client_time = np.append(avg_tier_client_time,np.ones((num_users,1)) * (np.inf), axis=1)
    avg_tier_client_time = np.append(np.ones((num_users,1)) * (np.inf), avg_tier_client_time, axis=1)
    avg_tier_client_time_serie_list = []
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            c = 0
            for t in range(max(0,len(client_tier) - 5 * memory_size), len(client_tier)):
                if client_tier[t][i] == j and c < memory_size:
                    avg_tier_client_time[i][j] += client_times[i][t]
                    c +=1
            if c>0 :
                avg_tier_client_time[i][j] = avg_tier_client_time[i][j] / c
            if c == 0:
                avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(np.nan,index=[j])]) #avg_tier_client_time_serie.append(pd.Series(np.nan,index=[j]))
            else:
                avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series([avg_tier_client_time[i][j]],index=[j])])
        # avg_tier_client_time_serie_list.append(avg_tier_client_time_serie.interpolate())
        if len(avg_tier_client_time_serie.value_counts()) == 1:
            avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(avg_tier_client_time_serie.dropna().max() * delay_co,index=[0])])
        else:
            avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(np.nan,index=[0])])
        avg_tier_client_time_serie_list.append(avg_tier_client_time_serie)
           
    
    avg_tier_client_time_serie_avg = []
    for i in range(0,num_users):  # make normalized / new comment: this help to consider dataset size for each client, because only consider the ratios in each client
        #avg_tier_client_time_serie_avg[i].append(avg_tier_client_time_serie_list[i] / avg_tier_client_time_serie_list[i].abs().max())
        avg_tier_client_time_serie_avg.append(avg_tier_client_time_serie_list[i] / avg_tier_client_time_serie_list[i][num_tiers])
    # avg_tier_client_time_serie_avg = sum(avg_tier_client_time_serie_list)/len(avg_tier_client_time_serie_list)        
    # avg_tier_client_time_serie_avg = sum(avg_tier_client_time_serie_avg)/len(avg_tier_client_time_serie_avg)
    

    for j in range(0, num_tiers+1):  # global ratio of tiers / should change based on samles in each client / it solved by normalizing in previous loop
        c = 0
        mean_time = 0
        for i in range(0,num_users):
            if not np.isnan(avg_tier_client_time_serie_avg[i][j]):
                mean_time += avg_tier_client_time_serie_avg[i][j]
                # mean_time += avg_tier_client_time_serie_avg[i][j] * avg_dataset / sataset_size[i]  # make avg time proportional to dataset size of each client
                c += 1
        if c == 0:
            avg_tier_time[j] = np.nan
        else:
            avg_tier_time[j] = ( mean_time / c)
    # print('avg_tier_time_ratio: ', avg_tier_time)
    avg_tier_time_list.append(list(avg_tier_time.values()))
    avg_tier_time = list(np.nanmean(avg_tier_time_list,axis=0))
    avg_tier_time = list(pd.DataFrame(avg_tier_time_list).ewm(com=0.1).mean().iloc[-1])  # make avg_tier_time ewm
    # np.nanmean(pd.DataFrame(avg_tier_time_list).ewm(com=2).mean(),axis=0)
            
        
    # avg_tier_client_time_serie_avg = avg_tier_client_time_serie_avg / avg_tier_client_time_serie_avg[num_tiers]  # make it normal to last tier
    for i in range(0,num_users):
        for j in range(1, num_tiers+1):
            if not np.isnan(avg_tier_time[j]):
                if np.isnan(avg_tier_client_time_serie_list[i][j]):
                    for k in range(1, num_tiers+1): # maybe consider continue when see a match
                        if not np.isnan(avg_tier_client_time_serie_list[i][k]):
                            # avg_tier_client_time_serie_list[i][j] = avg_tier_time[j] * avg_tier_client_time_serie_list[i][num_tiers]  # later all availble number can be used and not only tier 7
                            # avg_tier_client_time_serie_list[i][j] = avg_tier_time[j] * avg_tier_client_time_serie_list[i][num_tiers] * sataset_size[i] / avg_dataset # consider size of dataset in this client
                            # avg_tier_client_time_serie_list[i][j] = avg_tier_time[j] * avg_tier_client_time_serie_list[i][k] # * sataset_size[i] / avg_dataset # consider size of dataset in this client
                            avg_tier_client_time_serie_list[i][j] = avg_tier_time[j] / avg_tier_time[k] * avg_tier_client_time_serie_list[i][k] # * sataset_size[i] / avg_dataset # consider size of dataset in this client
                            continue
            
    for i in range(0,num_users):
        if avg_tier_client_time_serie_list[i].isna().sum() != (len(avg_tier_client_time_serie_list[i]) - 2): # remove tier 0 if there is more than 1 values
            avg_tier_client_time_serie_list[i][0] = 'nan'
        
        # avg_tier_client_time_serie_list[1].fillna(np.nanmax(avg_tier_client_time_serie_list[1][0:-1].values))
        avg_tier_client_time_serie_list[i][0:-1] = avg_tier_client_time_serie_list[i][0:-1].fillna(method="bfill") # fill backward
        
        # avg_tier_client_time[i] = avg_tier_client_time_serie_list[i].sort_index().interpolate(method = "spline", order = 1, limit_direction = "both").tolist()
                
            # elif c == 0 :
            #     avg_tier_client_time[i][j] = (avg_tier_client_time[i][j-1] + avg_tier_client_time[i][j+1]) /2
                
    
    client_tier = client_tier[-1].copy()
    max_time = max(avg_tier_client_time[:,num_tiers]) # T_max is max of tier 5 of all clients
    max_time_list.loc[len(max_time_list)] = max_time
    max_time = float(pd.DataFrame(max_time_list).ewm(com=0.5).mean().iloc[-1])
    max_increase_training_time = (0.9 * max_time) #- 10000
    max_increase_training_time = (1.1 * max_time)
    smooth_param = 0.05
    
    for c in client_tier.keys():
        # print('avg_tier_client_time', avg_tier_client_time[c])
        print('avg_tier_client_time_serie_list', avg_tier_client_time_serie_list[c])
        
        # if avg_tier_client_time[c][num_tiers] >= max(avg_tier_client_time[:,num_tiers]):
            # client_tier[c] = num_tiers
        if random.random() >= Eps:# greedy policy

            for t in range(0,num_tiers):
                # if (np.mean(avg_tier_client_time[:,num_tiers-t]) <= avg_tier_client_time[c][num_tiers-t]
                #     < np.mean(avg_tier_client_time[:,num_tiers-t+1])): # <= ubstead of < to help not stup in tier 5
                #     client_tier[c] = max(1,num_tiers -t)
                # if max(avg_tier_client_time[:,num_tiers-t-1]) <= avg_tier_client_time[c][num_tiers-t] < max(avg_tier_client_time[:,num_tiers-t]): # <= ubstead of < to help not stup in tier 5
                #     client_tier[c] = max(1,num_tiers -t)
                # if avg_tier_client_time[c][num_tiers-t] <= max_time:
                # if avg_tier_client_time[c][num_tiers-t] <= max_increase_training_time:
                # if avg_tier_client_time_serie_list[c][num_tiers-t] <= max_increase_training_time:
                if not (avg_tier_client_time_serie_list[c][num_tiers-t] > max_increase_training_time):
                    client_tier[c] = max(1,num_tiers -t)
                else:
                    continue
                    
        else: # random policy
            client_tier[c] += random.choice([-1,1])
            if client_tier[c] > num_tiers:
                client_tier[c] = num_tiers
            if client_tier[c] < num_tiers:
                client_tier[c] = 1
            
        if client_tier[c] == 1 and 0 < avg_tier_client_time[c][client_tier[c]] < (max_time/2) and client_tier_last[c] == 1 and False:
            client_epoch[c] = max(1, max_increase_training_time // avg_tier_client_time[c][client_tier[c]])
        else:
            client_epoch[c] = 1
            
        if (client_tier_last [c] + 1) < client_tier[c]:
            client_tier[c] = client_tier_last [c] + 1
        elif (client_tier_last [c] - 1) > client_tier[c]:
            client_tier[c] = client_tier_last [c] - 1
            # <= (client_tier_last [c] - 1):  # prevent jumping
        #     client_tier[c] = client_tier_last[c]
        
        if abs(avg_tier_client_time[c][client_tier[c]] - avg_tier_client_time[c][client_tier_last[c]]) < smooth_param * avg_tier_client_time[c][client_tier_last[c]]:
            client_tier[c] = client_tier_last[c]
            
    client_tier[np.argmax(avg_tier_client_time[:,num_tiers])] = num_tiers # asign slowet client always to tier num_tiers
    
    manual_tier = 1
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 1,
         2: 2,
         3: 2,
         5: 3,
         6: 3,
         4: 4,
         7: 4,
         8: 4,
         9: 4,
         10: 6,
         11: 7,
         12: 6,
         13: 6,
         14: 7,
         15: 7}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % num_tiers) + 1
    elif False:
         client_tier = {0: manual_tier,
         1: manual_tier,
         2: manual_tier,
         3: manual_tier,
         5: manual_tier,
         6: manual_tier,
         4: manual_tier,
         7: manual_tier,
         8: manual_tier,
         9: manual_tier,
         10: manual_tier,
         11: manual_tier,
         12: manual_tier,
         13: manual_tier,
         14: manual_tier,
         15: manual_tier}

            
            
    print("client tier:", client_tier, "local epoch:", client_epoch, "T_max:", max_time, "tier_ratio", avg_tier_time)
    
    return client_tier, client_epoch, avg_tier_time_list, max_time_list

def dynamic_tier6(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 5 # how many previous experiments look at for each tier in  one client
    avg_tier_client_time_serie=pd.Series()
    avg_tier_client_time_serie_list = []
    Eps_start = 0#1#.2
    Eps_end = 0#.05#.01
    Eps_Decay = 50
    Eps = Eps_end + (Eps_start - Eps_end) * \
        math.exp(-1. * step / Eps_Decay)
        
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
        idxs_users = kwargs['idxs_users']
    avg_dataset = sum(sataset_size.values()) / len(sataset_size)
    
    delay_co = 10
    if num_tiers == 5 :
        delay_co = 10
    elif num_tiers == 7 :
        delay_co = 20

    client_tier_last = client_tier[-1].copy()
    # max_time = max(client_times.iloc[-1])
    
    avg_tier_client_time = np.ones((num_users,num_tiers)) * 0# max_time
    # avg_tier_client_time = np.append(avg_tier_client_time,np.ones((num_users,1)) * (np.inf), axis=1)
    avg_tier_client_time = np.append(np.ones((num_users,1)) * (np.inf), avg_tier_client_time, axis=1)
    avg_tier_client_time_serie_list = []
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            c = 0
            for t in range(max(0,len(client_tier) - 5 * memory_size), len(client_tier)):
                if client_tier[t][i] == j and c < memory_size and not np.isnan(client_times[i][t]):
                    avg_tier_client_time[i][j] += client_times[i][t]
                    c +=1
            if c>0 :
                avg_tier_client_time[i][j] = avg_tier_client_time[i][j] / c
            if c == 0:
                avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(np.nan,index=[j])]) #avg_tier_client_time_serie.append(pd.Series(np.nan,index=[j]))
            else:
                avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series([avg_tier_client_time[i][j]],index=[j])])
        # avg_tier_client_time_serie_list.append(avg_tier_client_time_serie.interpolate())
        if len(avg_tier_client_time_serie.value_counts()) == 1:
            avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(avg_tier_client_time_serie.dropna().max() * delay_co,index=[0])])
        else:
            avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(np.nan,index=[0])])
        avg_tier_client_time_serie_list.append(avg_tier_client_time_serie)
           
    
    avg_tier_client_time_serie_avg = []
    for i in range(0,num_users):  # make normalized / new comment: this help to consider dataset size for each client, because only consider the ratios in each client
        #avg_tier_client_time_serie_avg[i].append(avg_tier_client_time_serie_list[i] / avg_tier_client_time_serie_list[i].abs().max())
        avg_tier_client_time_serie_avg.append(avg_tier_client_time_serie_list[i] / avg_tier_client_time_serie_list[i][num_tiers])
    # avg_tier_client_time_serie_avg = sum(avg_tier_client_time_serie_list)/len(avg_tier_client_time_serie_list)        
    # avg_tier_client_time_serie_avg = sum(avg_tier_client_time_serie_avg)/len(avg_tier_client_time_serie_avg)
    

    for j in range(0, num_tiers+1):  # global ratio of tiers / should change based on samles in each client / it solved by normalizing in previous loop
        c = 0
        mean_time = 0
        for i in range(0,num_users):
            if not np.isnan(avg_tier_client_time_serie_avg[i][j]):
                mean_time += avg_tier_client_time_serie_avg[i][j]
                # mean_time += avg_tier_client_time_serie_avg[i][j] * avg_dataset / sataset_size[i]  # make avg time proportional to dataset size of each client
                c += 1
        if c == 0:
            avg_tier_time[j] = np.nan
        else:
            avg_tier_time[j] = ( mean_time / c)
    # print('avg_tier_time_ratio: ', avg_tier_time)
    avg_tier_time_list.append(list(avg_tier_time.values()))
    avg_tier_time = list(np.nanmean(avg_tier_time_list,axis=0))
    print('ratios', avg_tier_time)
    avg_tier_time = list(pd.DataFrame(avg_tier_time_list).ewm(com=0.1).mean().iloc[-1])  # make avg_tier_time ewm
            
    for i in range(0,num_users):
        if avg_tier_client_time_serie_list[i].isna().sum() != (len(avg_tier_client_time_serie_list[i]) - 2): # remove tier 0 if there is more than 1 values
            avg_tier_client_time_serie_list[i][0] = 'nan'
        
        # avg_tier_client_time_serie_list[1].fillna(np.nanmax(avg_tier_client_time_serie_list[1][0:-1].values))
        # avg_tier_client_time_serie_list[i][0:-1] = avg_tier_client_time_serie_list[i][0:-1].fillna(method="bfill") # fill backward
                
    
    client_tier = client_tier[-1].copy()
    max_time = max(avg_tier_client_time[:,num_tiers]) # T_max is max of tier 5 of all clients
    max_time_list.loc[len(max_time_list)] = max_time
    max_time = float(pd.DataFrame(max_time_list).ewm(com=0.5).mean().iloc[-1])
    max_increase_training_time = (0.9 * max_time) #- 10000
    # max_increase_training_time = (1.1 * max_time)
    smooth_param = 0.05
    
    for c in client_tier.keys():
        if c in idxs_users:
            print('avg_tier_client_time_serie_list', avg_tier_client_time_serie_list[c])
            
            if random.random() >= Eps:# greedy policy
    
                for t in range(0,num_tiers):
                    if not (avg_tier_client_time_serie_list[c][num_tiers-t] > max_increase_training_time): # should check again
                        client_tier[c] = max(1,num_tiers -t)
                    else:
                        continue
                        
            else: # random policy
                client_tier[c] += random.choice([-1,1])
                if client_tier[c] > num_tiers:
                    client_tier[c] = num_tiers
                if client_tier[c] < num_tiers:
                    client_tier[c] = 1
                
            if client_tier[c] == 1 and 0 < avg_tier_client_time[c][client_tier[c]] < (max_time/2) and client_tier_last[c] == 1 and False:
                client_epoch[c] = max(1, max_increase_training_time // avg_tier_client_time[c][client_tier[c]])
            else:
                client_epoch[c] = 1
                
            if (client_tier_last [c] + 1) < client_tier[c]:
                client_tier[c] = client_tier_last [c] + 1
            elif (client_tier_last [c] - 1) > client_tier[c]:
                client_tier[c] = client_tier_last [c] - 1
                # <= (client_tier_last [c] - 1):  # prevent jumping
            #     client_tier[c] = client_tier_last[c]
            
            if abs(avg_tier_client_time[c][client_tier[c]] - avg_tier_client_time[c][client_tier_last[c]]) < smooth_param * avg_tier_client_time[c][client_tier_last[c]]:
                client_tier[c] = client_tier_last[c]
        else:
            client_tier[c] = client_tier_last[c]
            
    # client_tier[np.argmax(avg_tier_client_time[:,num_tiers])] = num_tiers # asign slowet client always to tier num_tiers
    client_tier[np.nanargmax(avg_tier_client_time[:,num_tiers])] = num_tiers # asign slowet client always to tier num_tiers
    
    manual_tier = 1
    if num_users == 16 or True:
         client_tier = {0: 7,
         1: 1,
         2: 2,
         3: 3,
         5: 4,
         6: 5,
         4: 6,
         7: 7,
         8: 1,
         9: 2,
         10: 3,
         11: 4,
         12: 5,
         13: 6,
         14: 7,
         15: 1}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % num_tiers) + 1
    elif False:
         client_tier = {0: manual_tier,
         1: manual_tier,
         2: manual_tier,
         3: manual_tier,
         5: manual_tier,
         6: manual_tier,
         4: manual_tier,
         7: manual_tier,
         8: manual_tier,
         9: manual_tier,
         10: manual_tier,
         11: manual_tier,
         12: manual_tier,
         13: manual_tier,
         14: manual_tier,
         15: manual_tier}

            
            
    print("client tier:", client_tier, "local epoch:", client_epoch, "T_max:", max_time, "tier_ratio", avg_tier_time)
    
    return client_tier, client_epoch, avg_tier_time_list, max_time_list

def dynamic_tier7(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 20 # how many previous experiments look at for each tier in  one client
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
        idxs_users = kwargs['idxs_users']
    
    client_tier_last = client_tier[-1].copy()
    # max_time = max(client_times.iloc[-1])
    
    # avg_tier_client_time_serie_list = []
    avg_tier_client_time = {}
    client_tier_time = np.empty((num_users,num_tiers,memory_size))
    client_tier_time[:] = np.NaN
    client_tier_time = dict()
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        # avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        count = 0
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            avg_tier_client_time[i] = []
            client_tier_time[i,j] = []
            # for t in range(max(0,len(client_tier) - 5 * memory_size), len(client_tier)):
            for t in range(len(client_tier)-1,-1,-1):
                if not count > memory_size:
                    if client_tier[t][i] == j and not np.isnan(client_times[i][t]):
                        client_tier_time[i,j].append(client_times[i][t])
                        count += 1
                    # if j == num_tiers:
                    #     avg_tier_client_time[i].append(client_times[i][t])
         # avg_tier_client_time_serie_list.append(avg_tier_client_time_serie)
            
        
    client_tier = client_tier[-1].copy()
    # max_client_list = [np.nanmean(avg_tier_client_time[i]) for i in avg_tier_client_time.keys()]
    max_client_list = [np.nanmean(client_tier_time[i,num_tiers]) for i in range(0,num_users)]
    # max_time = np.nanmax(max_client_list) # T_max is max of tier max of all clients
    max_time = float(np.nanmax(max_client_list) if not np.isnan(np.nanmax(max_client_list)) else max_time_list.iloc[-1]) # if not tier7 avilable
    # slow_index = max_client_list.index(max_time) # index of slowest client
    slow_index = int(max_client_list.index(max_time) if not np.isnan(np.nanmax(max_client_list)) else 1) # if not tier7 avilable
    max_time_list.loc[len(max_time_list)] = max_time
    # max_time = float(pd.DataFrame(max_time_list).ewm(com=0.5).mean().iloc[-1])
    # max_increase_training_time = (0.8 * max_time) #- 10000
    # max_time = max(client_tier_time[slow_index,num_tiers]) can be used for std of tmax
    smooth_param = 0.5
    outliers = 3
    
    tier_ratios = {1:9.1, 2:6.3, 3:5.1, 4:4.6, 5:3.3, 6:2.5, 7:1.0}
    
    print('client_times:', client_times.iloc[-1])

    
    for c in client_tier.keys():
        if c in idxs_users:
            client_tier[c] = client_tier_last[c]
            
            mean = np.mean(client_tier_time[c,client_tier_last[c]])
            
            if len(client_tier_time[c,client_tier_last[c]]) != 1:
                
                std = np.std(client_tier_time[c,client_tier_last[c]]) # will be zero when only one sample
            
                # if ((mean + outliers * std) > max_time or 
                # ((client_times[c].iloc[-1] - mean) / std) > outliers): #(smooth_param * max_time): # compare to tmax # if sample is far at least 2 std
                #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                # elif (((mean) < smooth_param * max_time) or 
                #       ((-client_times[c].iloc[-1] + mean) / std) <(-outliers)):
                #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                
                # if ((client_times[c].iloc[-1]) > max_time or 
                # ((client_times[c].iloc[-1] - mean) / std) > outliers): #(smooth_param * max_time): # compare to tmax # if sample is far at least 2 std
                #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                # elif ((client_times[c].iloc[-1] < smooth_param * max_time) or 
                #       ((-client_times[c].iloc[-1] + mean) / std) <(-outliers)):
                #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                    
                if (mean - outliers * std) < client_times[c].iloc[-1] < (mean + outliers * std):
                    if ((mean + outliers * std) < max_time / tier_ratios[max(client_tier_last[c]-1,1)] * tier_ratios[client_tier_last[c]]):
                        client_tier[c] = max(client_tier_last[c] - 1, 1)
                        if len(client_tier_time[c,client_tier[c]]) >= 1:
                            if len(client_tier_time[c,client_tier[c]]) == 1:
                                std = 0
                            else:
                                std = np.std(client_tier_time[c,client_tier[c]])
                            mean = np.mean(client_tier_time[c,client_tier[c]])
                            if (mean + outliers * std) > max_time: # next iteration if higher than max, it fluctuate and this prevent fluctuate
                                client_tier[c] = client_tier_last[c]                        
                    elif (mean + outliers * std) > max_time:
                        client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                else: # if this is far from previous delete it
                    client_times[c] = np.NaN
                    # print(c,'NaN')
                
            else:   # significant change, del previous measurments
            
                
                if ((mean) >=  max_time):  # compare to tmax
                    client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                elif ((mean * tier_ratios[max(client_tier_last[c]-1,1)] / tier_ratios[client_tier_last[c]]) < max_time):
                    client_tier[c] = max(client_tier_last[c] - 1, 1)
                
                            
                        
                        

        else:
            client_tier[c] = client_tier_last[c]
            
    client_tier[slow_index] = num_tiers
    
    manual_tier = 1
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 1,
         2: 2,
         3: 2,
         5: 3,
         6: 3,
         4: 4,
         7: 4,
         8: 4,
         9: 4,
         10: 6,
         11: 7,
         12: 6,
         13: 6,
         14: 7,
         15: 7}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % num_tiers) + 1
    elif True:
         client_tier = {0: manual_tier,
         1: manual_tier,
         2: manual_tier,
         3: manual_tier,
         5: manual_tier,
         6: manual_tier,
         4: manual_tier,
         7: manual_tier,
         8: manual_tier,
         9: manual_tier,
         10: manual_tier,
         11: manual_tier,
         12: manual_tier,
         13: manual_tier,
         14: manual_tier,
         15: manual_tier}

            
            
    print("client tier:", client_tier, "local epoch:", client_epoch, "T_max:", max_time, "tier_ratio", avg_tier_time)
    
    return client_tier, client_epoch, avg_tier_time_list, max_time_list, client_times

def dynamic_tier8(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 20 # how many previous experiments look at for each tier in  one client
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
        idxs_users = kwargs['idxs_users']
    
    client_tier_last = client_tier[-1].copy()
    # max_time = max(client_times.iloc[-1])
    
    # avg_tier_client_time_serie_list = []
    avg_tier_client_time = {}
    client_tier_time = np.empty((num_users,num_tiers,memory_size))
    client_tier_time[:] = np.NaN
    client_tier_time = dict()
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        # avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        count = 0
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            avg_tier_client_time[i] = []
            client_tier_time[i,j] = []
            # for t in range(max(0,len(client_tier) - 5 * memory_size), len(client_tier)):
            for t in range(len(client_tier)-1,-1,-1):
                if not count > memory_size:
                    if client_tier[t][i] == j and not np.isnan(client_times[i][t]):
                        client_tier_time[i,j].append(client_times[i][t])
                        count += 1
                    # if j == num_tiers:
                    #     avg_tier_client_time[i].append(client_times[i][t])
         # avg_tier_client_time_serie_list.append(avg_tier_client_time_serie)
            
        
    #print(client_tier_time)
    client_tier = client_tier[-1].copy()
    # max_client_list = [np.nanmean(avg_tier_client_time[i]) for i in avg_tier_client_time.keys()]
    max_client_list = [np.nanmean(client_tier_time[i,num_tiers]) for i in range(0,num_users)]
    # max_time = np.nanmax(max_client_list) # T_max is max of tier max of all clients
    max_time = float(np.nanmax(max_client_list) if not np.isnan(np.nanmax(max_client_list)) else max_time_list.iloc[-1]) # if not tier7 avilable
    # slow_index = max_client_list.index(max_time) # index of slowest client
    slow_index = int(max_client_list.index(max_time) if not np.isnan(np.nanmax(max_client_list)) else 1) # if not tier7 avilable
    max_time_list.loc[len(max_time_list)] = max_time
    # max_time = float(pd.DataFrame(max_time_list).ewm(com=0.5).mean().iloc[-1])
    # max_increase_training_time = (0.8 * max_time) #- 10000
    # max_time = max(client_tier_time[slow_index,num_tiers]) can be used for std of tmax
    smooth_param = 0.5
    outliers = 3
    
    # tier_ratios = {1:9.1, 2:6.3, 3:5.1, 4:4.6, 5:3.3, 6:2.5, 7:1.0}
    tier_ratios = {1:11.48, 2:10.22, 3:8.39, 4:6.62, 5:4.94, 6:2.92, 7:1.0}
    
    print('client_times:', client_times.iloc[-1])

    
    for c in client_tier.keys():
        if c in idxs_users:
            client_tier[c] = client_tier_last[c]
            
            mean = np.mean(client_tier_time[c,client_tier_last[c]])
            
            if len(client_tier_time[c,client_tier_last[c]]) <= 3:
                if ((mean) >=  max_time):  # compare to tmax
                    client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                elif ((mean * tier_ratios[max(client_tier_last[c]-1,1)] / tier_ratios[client_tier_last[c]]) < max_time):
                    client_tier[c] = max(client_tier_last[c] - 1, 1)
                
            else:   # significant change, del previous measurments
                # std = np.std(client_tier_time[c,client_tier_last[c]][:-1]) # will be zero when only one sample
                std = np.std(client_tier_time[c,client_tier_last[c]][1:]) # list indexing from end
                
                # mean = np.mean(client_tier_time[c,client_tier_last[c]][:-1]) # mean over previous 
                mean = np.mean(client_tier_time[c,client_tier_last[c]][1:])
                
                min_interval = mean - outliers * std
                max_interval = mean + outliers * std
                print('time range client', c, 'min_interval',min_interval,'current time',client_times[c].iloc[-1],'max_interval',max_interval)
            
                # if ((mean + outliers * std) > max_time or 
                # ((client_times[c].iloc[-1] - mean) / std) > outliers): #(smooth_param * max_time): # compare to tmax # if sample is far at least 2 std
                #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                # elif (((mean) < smooth_param * max_time) or 
                #       ((-client_times[c].iloc[-1] + mean) / std) <(-outliers)):
                #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                
                # if ((client_times[c].iloc[-1]) > max_time or 
                # ((client_times[c].iloc[-1] - mean) / std) > outliers): #(smooth_param * max_time): # compare to tmax # if sample is far at least 2 std
                #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                # elif ((client_times[c].iloc[-1] < smooth_param * max_time) or 
                #       ((-client_times[c].iloc[-1] + mean) / std) <(-outliers)):
                #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                    
                if (min_interval) < client_times[c].iloc[-1] < (max_interval):
                    if ((max_interval) < max_time / tier_ratios[max(client_tier_last[c]-1,1)] * tier_ratios[client_tier_last[c]]):
                        client_tier[c] = max(client_tier_last[c] - 1, 1)
                        if len(client_tier_time[c,client_tier[c]]) >= 1:
                            if len(client_tier_time[c,client_tier[c]]) != 1: # to see if next tier time is more than tmax
                                std = np.std(client_tier_time[c,client_tier[c]])
                                mean = np.mean(client_tier_time[c,client_tier[c]])
                        #        print('next tier time and max_time',(mean + outliers * std),'max_time',max_time
                         #             ,'min_interval',min_interval,'max_interval',max_interval)
                                if (mean + outliers * std) > max_time: # next iteration if higher than max, it fluctuate and this prevent fluctuate
                          #          print('next tier more than max_time',(mean + outliers * std),max_time )
                                    client_tier[c] = client_tier_last[c]
                                if (min_interval) < (mean - outliers * std) < (max_interval):# next iteration if training time is in current tier distribution
                                    client_tier[c] = client_tier_last[c]
                    elif (mean + outliers * std) > max_time:
                        client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                else: # if this is far from previous delete it
                    # client_times[c] = np.NaN
                    client_times[c][0:len(client_times[c])-1]  = np.NaN # delete only previous measurements
                    print('change in client', c)
                    mean = client_times[c].iloc[-1]
                    
                    # compare current time to assign tier
                    if ((mean) >=  max_time):  # compare to tmax
                        client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                    elif ((mean * tier_ratios[max(client_tier_last[c]-1,1)] / tier_ratios[client_tier_last[c]]) < max_time):
                        client_tier[c] = max(client_tier_last[c] - 1, 1)
                            
                        
                        

        else:
            client_tier[c] = client_tier_last[c]
            
    client_tier[slow_index] = num_tiers
    print('slow_index',slow_index)
    
    manual_tier = 6
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 1,
         2: 2,
         3: 2,
         5: 3,
         6: 3,
         4: 4,
         7: 4,
         8: 4,
         9: 4,
         10: 6,
         11: 7,
         12: 6,
         13: 6,
         14: 7,
         15: 7}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % num_tiers) + 1
    elif False:
         for i in range(0,100):    
             client_tier[i] = manual_tier
        
         client_tier = {0: manual_tier,
         1: manual_tier,
         2: manual_tier,
         3: manual_tier,
         5: manual_tier,
         6: manual_tier,
         4: manual_tier,
         7: manual_tier,
         8: manual_tier,
         9: manual_tier,
         10: manual_tier,
         11: manual_tier,
         12: manual_tier,
         13: manual_tier,
         14: manual_tier,
         15: manual_tier}
         for i in range(0,100):    
             client_tier[i] = manual_tier

            
            
    print("client tier:", client_tier, "local epoch:", client_epoch, "T_max:", max_time, "tier_ratio", avg_tier_time)
    
    return client_tier, client_epoch, avg_tier_time_list, max_time_list, client_times
    

def dynamic_tier9(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 20 # how many previous experiments look at for each tier in  one client
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
        idxs_users = kwargs['idxs_users']
    
    client_tier_last = client_tier[-1].copy()
    avg_tier_client_time = {}
    client_tier_time = np.empty((num_users,num_tiers,memory_size))
    client_tier_time[:] = np.NaN
    client_tier_time = dict()
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        # avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        count = 0
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            avg_tier_client_time[i] = []
            client_tier_time[i,j] = []
            # for t in range(max(0,len(client_tier) - 5 * memory_size), len(client_tier)):
            for t in range(len(client_tier)-1,-1,-1):
                if not count > memory_size:
                    if client_tier[t][i] == j and not np.isnan(client_times[i][t]):
                        client_tier_time[i,j].append(client_times[i][t])
                        count += 1
        
    #print(client_tier_time)
    client_tier = client_tier[-1].copy()
    # max_client_list = [np.nanmean(avg_tier_client_time[i]) for i in avg_tier_client_time.keys()]
    max_client_list = [np.nanmean(client_tier_time[i,num_tiers]) for i in range(0,num_users)]
    # max_time = np.nanmax(max_client_list) # T_max is max of tier max of all clients
    max_time = float(np.nanmax(max_client_list) if not np.isnan(np.nanmax(max_client_list)) else max_time_list.iloc[-1]) # if not tier7 avilable
    # slow_index = max_client_list.index(max_time) # index of slowest client
    slow_index = int(max_client_list.index(max_time) if not np.isnan(np.nanmax(max_client_list)) else 1) # if not tier7 avilable
    max_time_list.loc[len(max_time_list)] = max_time
    # max_time = float(pd.DataFrame(max_time_list).ewm(com=0.5).mean().iloc[-1])
    # max_increase_training_time = (0.8 * max_time) #- 10000
    # max_time = max(client_tier_time[slow_index,num_tiers]) can be used for std of tmax
    smooth_param = 0.5
    outliers = 3
    
    # tier_ratios = {1:9.1, 2:6.3, 3:5.1, 4:4.6, 5:3.3, 6:2.5, 7:1.0}
    tier_ratios = {1:11.48, 2:10.22, 3:8.39, 4:6.62, 5:4.94, 6:2.92, 7:1.0}
    
    print('client_times:', client_times.iloc[-1])

    
    for c in client_tier.keys():
        if c in idxs_users:
            client_tier[c] = client_tier_last[c]
            
            mean = np.mean(client_tier_time[c,client_tier_last[c]])
            
            if len(client_tier_time[c,client_tier_last[c]]) <= 3:
                client_tier[c] = client_tier_last[c]
                
            else:   # significant change, del previous measurments
                # std = np.std(client_tier_time[c,client_tier_last[c]][:-1]) # will be zero when only one sample
                std = np.std(client_tier_time[c,client_tier_last[c]][1:]) # list indexing from end
                
                # mean = np.mean(client_tier_time[c,client_tier_last[c]][:-1]) # mean over previous 
                mean = np.mean(client_tier_time[c,client_tier_last[c]][1:])
                
                min_interval = mean - outliers * std
                max_interval = mean + outliers * std
                print('time range client', c, 'min_interval',min_interval,'current time',client_times[c].iloc[-1],'max_interval',max_interval)
            
                # if ((mean + outliers * std) > max_time or 
                # ((client_times[c].iloc[-1] - mean) / std) > outliers): #(smooth_param * max_time): # compare to tmax # if sample is far at least 2 std
                #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                # elif (((mean) < smooth_param * max_time) or 
                #       ((-client_times[c].iloc[-1] + mean) / std) <(-outliers)):
                #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                
                # if ((client_times[c].iloc[-1]) > max_time or 
                # ((client_times[c].iloc[-1] - mean) / std) > outliers): #(smooth_param * max_time): # compare to tmax # if sample is far at least 2 std
                #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                # elif ((client_times[c].iloc[-1] < smooth_param * max_time) or 
                #       ((-client_times[c].iloc[-1] + mean) / std) <(-outliers)):
                #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                    
                if not ((min_interval) < client_times[c].iloc[-1] < (max_interval)): # if this is far from previous delete it
                    # client_times[c] = np.NaN
                    client_times[c][0:len(client_times[c])-1]  = np.NaN # delete only previous measurements
                    print('change in client', c)
                    mean = client_times[c].iloc[-1]
                    

                else :
                    if ((max_interval) < max_time / tier_ratios[max(client_tier_last[c]-1,1)] * tier_ratios[client_tier_last[c]]):
                        client_tier[c] = max(client_tier_last[c] - 1, 1)
                        if len(client_tier_time[c,client_tier[c]]) >= 1:
                            if len(client_tier_time[c,client_tier[c]]) != 1: # to see if next tier time is more than tmax
                                std = np.std(client_tier_time[c,client_tier[c]])
                                mean = np.mean(client_tier_time[c,client_tier[c]])
                        #        print('next tier time and max_time',(mean + outliers * std),'max_time',max_time
                         #             ,'min_interval',min_interval,'max_interval',max_interval)
                                if (mean + outliers * std) > max_time: # next iteration if higher than max, it fluctuate and this prevent fluctuate
                          #          print('next tier more than max_time',(mean + outliers * std),max_time )
                                    client_tier[c] = client_tier_last[c]
                                if (min_interval) < (mean - outliers * std) < (max_interval):# next iteration if training time is in current tier distribution
                                    client_tier[c] = client_tier_last[c]
                    elif (mean + outliers * std) > max_time: # tier def. is diff. from the paper
                        client_tier[c] = min(client_tier_last[c] + 1, num_tiers)

                            
                                            # # compare current time to assign tier
                                            # if ((mean) >=  max_time):  # compare to tmax
                                            #     client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                                            # elif ((mean * tier_ratios[max(client_tier_last[c]-1,1)] / tier_ratios[client_tier_last[c]]) < max_time):
                                            #     client_tier[c] = max(client_tier_last[c] - 1, 1)
                        

        else:
            client_tier[c] = client_tier_last[c]
            
    client_tier[slow_index] = num_tiers
    print('slow_index',slow_index)
    
    manual_tier = 6
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 1,
         2: 2,
         3: 2,
         5: 3,
         6: 3,
         4: 4,
         7: 4,
         8: 4,
         9: 4,
         10: 6,
         11: 7,
         12: 6,
         13: 6,
         14: 7,
         15: 7}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % num_tiers) + 1
    elif False:
         for i in range(0,100):    
             client_tier[i] = manual_tier
        
         client_tier = {0: manual_tier,
         1: manual_tier,
         2: manual_tier,
         3: manual_tier,
         5: manual_tier,
         6: manual_tier,
         4: manual_tier,
         7: manual_tier,
         8: manual_tier,
         9: manual_tier,
         10: manual_tier,
         11: manual_tier,
         12: manual_tier,
         13: manual_tier,
         14: manual_tier,
         15: manual_tier}
         for i in range(0,100):    
             client_tier[i] = manual_tier

            
            
    print("client tier:", client_tier, "local epoch:", client_epoch, "T_max:", max_time, "tier_ratio", avg_tier_time)
    
    return client_tier, client_epoch, avg_tier_time_list, max_time_list, client_times

''' RL agent , source code from : #https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char01%20DQN/DQN.py

DQN structure

'''
# hyper-parameters
BATCH_SIZE = 5 # 128 # first 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.99
MEMORY_CAPACITY = 5 # first 2000
Q_NETWORK_ITERATION = 100
NUM_STATES = 2
NUM_ACTIONS = 2 * 5 # clients * tiers
ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape
num_clients = 2

global memory
memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 1 + num_clients))


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
        self.memory = memory
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
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+self.num_clients].astype(int))
        # batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+self.num_clients:NUM_STATES+self.num_clients+1])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

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
        
        
def dqn_agent(client_tier, client_times, client_epoch, reward, iter):
    if iter == 0 :
        dqn = DQN()
    present_time = client_times.ewm(com=0.5).mean()[-1:]
    state = present_time.to_numpy()[0]
    next_state = state
    if iter == 0 :
        state = next_state * 0
    else:
        state = client_times[:-1].ewm(com=0.5).mean()[-1:].to_numpy()[0]
    action = client_tier
    
    # learning phase
    dqn.store_transition(state, list(action.values()), reward, next_state)  # not sure always dict to array is sorted 
    
    dqn.learn()
    
    # selection phase
    action = dqn.choose_action(state, client_tier)
    client_tier = action
    # dqn.learn()
    # print('dqn') 
    
    return client_tier, client_epoch