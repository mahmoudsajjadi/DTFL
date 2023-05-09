import numpy as np
import pandas as pd
import random
import math


def index_of_greatest_smaller(lst, max_time):
    # Create a new list of all elements in lst that are greater than xx
    greater_lst = [num for num in lst if num <= max_time]
    
    # If greater_lst is empty, return None
    if not greater_lst:
        return None
    
    # Find the smallest number in greater_lst and return its index in lst
    # smallest = max(greater_lst)
    
    # find the one has most layers at the client side
    smallest = greater_lst[0]
    return list(lst).index(smallest) + 1, smallest

def find_straggler(client_tier_time : list, num_users: int, client_tier_last: list,
                   net_speed: list, data_transmitted_client_all: list, tier_ratios: dict,
                   tier_intermediate_data_profile: dict, idxs_users: list) -> int:
    '''
    
    Parameters
    ----------
    client_tier_time : list
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.
        index of slowest client

    '''
    
    clients_min_estimation_time = []
    for c in idxs_users:
        # check if this client has history
        if len(client_tier_time[c,client_tier_last[c]]) < 1:
            continue


        # Compute total time, transmission time, and client computation time
        total_time = np.mean(client_tier_time[c,client_tier_last[c]])
        transmission_time = data_transmitted_client_all[c] / net_speed[c]
        clinet_computaion_time = total_time - transmission_time
        
        # Compute computation and communication estimation for each tier
        tier_computation_estimation = np.array([value for value in tier_ratios.values()]) * clinet_computaion_time / tier_ratios[client_tier_last[c]]
        tier_communication_estimation = np.array([value for value in tier_intermediate_data_profile.values()]) / net_speed[c]
        client_estimation_time_all_tiers = list(tier_computation_estimation + tier_communication_estimation)


        
        # client_estimation_time_all_tiers = (list(np.array([value for value in tier_ratios.values()]) * clinet_computaion_time / tier_ratios[client_tier_last[c]]) # computation estimation
        #                                     + np.array([value for value in tier_intermediate_data_profile.values()]) / net_speed[c] ) # commmunication estimation
        
        clients_min_estimation_time.append(min(client_estimation_time_all_tiers))
        print(client_estimation_time_all_tiers)
    
    # check if all new 
    if len(clients_min_estimation_time) < 1:
        return -1, -1
    
    max_estimation_time = max(clients_min_estimation_time)
    
    return clients_min_estimation_time.index(max_estimation_time), max_estimation_time
        
        
        


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



   

def dynamic_tier9(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 10 # how many previous experiments look at for each tier in  one client
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
        idxs_users = kwargs['idxs_users']
    
    client_tier_last = client_tier[-1].copy()
    # avg_tier_client_time = {}
    client_tier_time = np.empty((num_users,num_tiers,memory_size))
    client_tier_time[:] = np.NaN
    client_tier_time = dict()
    
    # I should revise this , and list of training time of each client in each tier
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        # avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        count = 0
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            # avg_tier_client_time[i] = []
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
    # smooth_param = 0.5
    outliers = 3
    
    # tier_ratios = {1:9.1, 2:6.3, 3:5.1, 4:4.6, 5:3.3, 6:2.5, 7:1.0}
    tier_ratios = {1:11.48, 2:10.22, 3:8.39, 4:6.62, 5:4.94, 6:2.92, 7:1.0}
    
    print('mean_client_times:\n', client_times.ewm(com=0.5).mean()[-1:].to_string(float_format='%.2f'))
        
    for c in client_tier.keys():
        if c in idxs_users:
            client_tier[c] = client_tier_last[c]
            
            mean = np.mean(client_tier_time[c,client_tier_last[c]])
            
            if len(client_tier_time[c,client_tier_last[c]]) <= 2:
                client_tier[c] = client_tier_last[c]
                
            else:   # significant change, del previous measurments
                # std = np.std(client_tier_time[c,client_tier_last[c]][:-1]) # will be zero when only one sample
                std = np.std(client_tier_time[c,client_tier_last[c]][:]) # list indexing from end
                
                # mean = np.mean(client_tier_time[c,client_tier_last[c]][:-1]) # mean over previous 
                mean = np.mean(client_tier_time[c,client_tier_last[c]][:])
                
                min_interval = mean - outliers * std
                max_interval = mean + outliers * std
                print('time range client', c, 'min_interval',min_interval,'current time',client_times[c].iloc[-1],'max_interval',max_interval)
                # print('client time in this tier', client_tier_time[c,client_tier_last[c]])
            
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
                        print(f'assign client{c} to a larger model - tier')
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
                        if client_tier[c] != client_tier_last[c]:
                            print(f'assign client{c} to a smaller model - tier')

                            
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
    # for i in range(num_users):
    #     client_tier[i] = i % 7 + 1
    
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 2,
         2: 3,
         3: 4,
         4: 5,
         5: 6,
         6: 7,
         7: 1,
         8: 2,
         9: 3,
         10: 4,
         11: 5,
         12: 6,
         13: 7,
         14: 1,
         15: 2}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % (num_tiers - 2 )) + 1 + 2
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


def tier_scheduler(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    # global avg_tier_time_list
    avg_tier_time = {}
    memory_size = 10 # how many previous experiments look at for each tier in  one client
    
    if kwargs:
        sataset_size = kwargs['sataset_size']
        avg_tier_time_list = kwargs['avg_tier_time_list']
        max_time_list = kwargs['max_time_list']
        idxs_users = kwargs['idxs_users']
        data_transmitted_client_all = kwargs['data_transmitted_client_all']
        net_speed = kwargs['net_speed']
        # data_transmitted_client_all =  = kwargs['data_transmitted_client_all']
        
    
    
    client_tier_last = client_tier[-1].copy()
    # avg_tier_client_time = {}
    client_tier_time = np.empty((num_users,num_tiers,memory_size))
    client_tier_time[:] = np.NaN
    client_tier_time = dict()
    
    # I should revise this , and list of training time of each client in each tier
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        # avg_tier_client_time_serie=pd.Series()
        # avg_tier_client_time_serie = pd.concat([avg_tier_client_time_serie, pd.Series(max_time * 10,index=[0])])
        count = 0
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            # avg_tier_client_time[i] = []
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

    outliers = 3
    # Define a threshold for time difference as a fraction of the estimated time
    time_diff_threshold = 0.05
    
    # tier_ratios = {1:9.1, 2:6.3, 3:5.1, 4:4.6, 5:3.3, 6:2.5, 7:1.0}
    tier_ratios = {1:11.48, 2:10.22, 3:8.39, 4:6.62, 5:4.94, 6:2.92, 7:1.0} # should update for each experiemtns
    MB = 1024 ** 2
    tier_intermediate_data_profile = {1:314.5 * MB, 2:313.9 * MB, 3:625.6 * MB, 4:625.2 * MB, 5:1250.1 * MB, 6:1250.3 * MB, 7:312.6 * MB}
    # tier_intermediate_data = data_transmitted_client_all
    
    print('mean_client_times:\n', client_times.ewm(com=0.5).mean()[-1:].to_string(float_format='%.2f'))
    
    # check the straggler in the next epoch, for next epoch clients
    # print('net_speed' , net_speed)
    # print('data_transmitted_client_all', data_transmitted_client_all)
    straggler_index, max_estimation_time = find_straggler(client_tier_time, num_users, client_tier_last,
                       net_speed, data_transmitted_client_all, tier_ratios,
                       tier_intermediate_data_profile, idxs_users)
    max_time = max_estimation_time
    max_time_list.loc[len(max_time_list)] = max_time
        
    for c in client_tier.keys(): # I can change it and ignore straggler index, and assign that to best available
        if c in idxs_users:
            client_tier[c] = client_tier_last[c]
            
            mean = np.mean(client_tier_time[c,client_tier_last[c]])
            
            if len(client_tier_time[c,client_tier_last[c]]) <= 2:
                client_tier[c] = client_tier_last[c]
                
            else:   # significant change, del previous measurments
                # std = np.std(client_tier_time[c,client_tier_last[c]][:-1]) # will be zero when only one sample
                std = np.std(client_tier_time[c,client_tier_last[c]][:]) # list indexing from end
                
                # mean = np.mean(client_tier_time[c,client_tier_last[c]][:-1]) # mean over previous 
                mean = np.mean(client_tier_time[c,client_tier_last[c]][:])
                
                min_interval = mean - outliers * std
                max_interval = mean + outliers * std
                # print('time range client', c, 'min_interval',min_interval,'current time',client_times[c].iloc[-1],'max_interval',max_interval)

                    
                if not ((min_interval) < client_times[c].iloc[-1] < (max_interval)): # if this is far from previous delete it
                    # client_times[c] = np.NaN
                    client_times[c][0:len(client_times[c])-1]  = np.NaN # delete only previous measurements
                    print('dynamic change in client:', c)
                    mean = client_times[c].iloc[-1]
                    

                else:
                    clinet_computaion_time = np.mean(client_tier_time[c,client_tier_last[c]]) - data_transmitted_client_all[c] / net_speed[c]
                    client_estimation_time_all_tiers = (list(np.array([value for value in tier_ratios.values()]) * clinet_computaion_time / tier_ratios[client_tier_last[c]]) # computation estimation
                                                        + np.array([value for value in tier_intermediate_data_profile.values()]) / net_speed[c] ) # commmunication estimation
                    client_tier[c], estimated_time = index_of_greatest_smaller(client_estimation_time_all_tiers, max_time)
                    
                    # not change if time dif is not large
                    if abs(estimated_time - np.mean(client_tier_time[c,client_tier_last[c]])) < estimated_time / 10:
                        client_tier[c] = client_tier_last[c]
                        
                    
                    # Check if the time difference is large enough to trigger a change in client tier
                    estimated_time_diff = abs(estimated_time - np.mean(client_tier_time[c, client_tier_last[c]]))
                    if estimated_time_diff < estimated_time * time_diff_threshold:
                        # Keep the current client tier if the time difference is not large enough
                        client_tier[c] = client_tier_last[c]
                        
                    
                    # print(f'assign client {c} to tier {7 - client_tier[c] + 1}')
                    
                    
                       

        else:
            client_tier[c] = client_tier_last[c]
            
    client_tier[straggler_index] = num_tiers
    print('slow_index',straggler_index)
    
    manual_tier = 6
    # for i in range(num_users):
    #     client_tier[i] = i % 7 + 1
    
    if num_users == 16 and False:
         client_tier = {0: 1,
         1: 2,
         2: 3,
         3: 4,
         4: 5,
         5: 6,
         6: 7,
         7: 1,
         8: 2,
         9: 3,
         10: 4,
         11: 5,
         12: 6,
         13: 7,
         14: 1,
         15: 2}
    elif num_users == 16 and False:
        for i in range(0,num_users):
            client_tier[i] = (((step + i * 10 )//10) % (num_tiers - 2 )) + 1 + 2
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
