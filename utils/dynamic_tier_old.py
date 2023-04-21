import numpy as np


def dynamic_tier8(client_tier, client_times, num_tiers, server_wait_time, client_epoch, 
                  time_train_server, num_users, step, **kwargs):
    memory_size = 20 
    
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
        count = 0
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            avg_tier_client_time[i] = []
            client_tier_time[i,j] = []
            for t in range(len(client_tier)-1,-1,-1):
                if not count > memory_size:
                    if client_tier[t][i] == j and not np.isnan(client_times[i][t]):
                        client_tier_time[i,j].append(client_times[i][t])
                        count += 1
        
    client_tier = client_tier[-1].copy()
    max_client_list = [np.nanmean(client_tier_time[i,num_tiers]) for i in range(0,num_users)]
    max_time = float(np.nanmax(max_client_list) if not np.isnan(np.nanmax(max_client_list)) else max_time_list.iloc[-1]) # if not tier7 avilable
    slow_index = int(max_client_list.index(max_time) if not np.isnan(np.nanmax(max_client_list)) else 1) # if not tier7 avilable
    max_time_list.loc[len(max_time_list)] = max_time
    outliers = 3
    
    tier_ratios = {1:11.48, 2:10.22, 3:8.39, 4:6.62, 5:4.94, 6:2.92, 7:1.0} # need update for new environment
    
    print('client_times:', client_times.iloc[-1])

    
    for c in client_tier.keys():
        if c in idxs_users:
            client_tier[c] = client_tier_last[c]
            
            mean = np.mean(client_tier_time[c,client_tier_last[c]])
            
            if len(client_tier_time[c,client_tier_last[c]]) <= 2:
                if ((mean) >=  max_time):  # compare to tmax
                    client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                elif ((mean * tier_ratios[max(client_tier_last[c]-1,1)] / tier_ratios[client_tier_last[c]]) < max_time):
                    client_tier[c] = max(client_tier_last[c] - 1, 1)
                
            else:   # significant change, del previous measurments
                std = np.std(client_tier_time[c,client_tier_last[c]][1:]) # list indexing from end
                
                mean = np.mean(client_tier_time[c,client_tier_last[c]][1:])
                
                min_interval = mean - outliers * std
                max_interval = mean + outliers * std
                    
                if (min_interval) < client_times[c].iloc[-1] < (max_interval):
                    if ((max_interval) < max_time / tier_ratios[max(client_tier_last[c]-1,1)] * tier_ratios[client_tier_last[c]]):
                        client_tier[c] = max(client_tier_last[c] - 1, 1)
                        if len(client_tier_time[c,client_tier[c]]) >= 1:
                            if len(client_tier_time[c,client_tier[c]]) != 1: # to see if next tier time is more than tmax
                                std = np.std(client_tier_time[c,client_tier[c]])
                                mean = np.mean(client_tier_time[c,client_tier[c]])
                                if (mean + outliers * std) > max_time: # next iteration if higher than max, it fluctuate and this prevent fluctuate
                                    client_tier[c] = client_tier_last[c]
                                if (min_interval) < (mean - outliers * std) < (max_interval):# next iteration if training time is in current tier distribution
                                    client_tier[c] = client_tier_last[c]
                    elif (mean + outliers * std) > max_time:
                        client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                else: # if this is far from previous delete it
                    client_times[c][0:len(client_times[c])-1]  = np.NaN # delete only previous measurements
                    mean = client_times[c].iloc[-1]
                    
                    # compare current time to assign tier
                    if ((mean) >=  max_time):  # compare to tmax
                        client_tier[c] = min(client_tier_last[c] + 1, num_tiers)
                    elif ((mean * tier_ratios[max(client_tier_last[c]-1,1)] / tier_ratios[client_tier_last[c]]) < max_time):
                        client_tier[c] = max(client_tier_last[c] - 1, 1)
                            
                        
                        

        else:
            client_tier[c] = client_tier_last[c]
            
    client_tier[slow_index] = num_tiers
    
    return client_tier, client_epoch, avg_tier_time_list, max_time_list, client_times