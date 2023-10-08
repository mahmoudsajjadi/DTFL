import numpy as np
import pandas as pd

# Define the smoothing factor (0 < smoothing_factor < 1)
smoothing_factor = 0.2
SMALLEST_TIER = 6

def index_of_greatest_smaller(lst, T_max):
    # Create a new list of all elements in lst that are greater than xx
    greater_lst = [num for num in lst if num <= T_max]
    
    # If greater_lst is empty, return None
    if not greater_lst:
        # return None
        return SMALLEST_TIER, lst[SMALLEST_TIER-1]
    
    # Find the smallest number in greater_lst and return its index in lst
    # smallest = max(greater_lst)
    
    # find the one has most layers at the client side
    smallest = greater_lst[0]
    return list(lst).index(smallest) + 1, smallest

# index_of_greatest_smaller([1,2,3,4,5] , 4)

def client_time_tier(computation_time_clients, client_tier, num_users, num_tiers):
    client_tier_time = {}
    for i in range(0,num_users):   # this part calculate avg time of each tier each client in window
        for j in range(1, num_tiers+1): #range(1, num_tiers+1) range(num_tiers,0,-1)
            client_tier_time[i,j] = []
            for t in range(len(client_tier)):
                if client_tier[t][i] == j and not np.isnan(computation_time_clients[i][t]):
                    client_tier_time[i,j].append(computation_time_clients[i][t])
                    
    return client_tier_time
                    
                        
                        
                        

def TierScheduler(computation_time_clients, T_max, **kwargs):

                      
    if kwargs:
        client_tier = kwargs['client_tier_all']
        delay_history = kwargs['delay_history']
        num_tiers = kwargs['num_tiers']
        num_users = kwargs['num_users']
        dataset_size = kwargs['dataset_size']
        batch_size = kwargs['batch_size']
        net_speed = kwargs['net_speed']

    # T_max = max(T_r)
    
    
    #### profiling
    MB = 1024 ** 2
    total_data_size_tier = {1:0.629 * MB, 2:313.9 * MB, 3:625.6 * MB, 4:625.2 * MB, 5:1250.1 * MB, 6:1250.3 * MB, 7:312.6 * MB}
    
    total_data_size_tier = {
        1: 0.629 * MB,
        2: 0.6278 * MB,
        3: 1.2512 * MB,
        4: 1.2504 * MB,
        5: 2.5002 * MB,
        6: 2.5006 * MB,
    }
        
    profile_client_side = {
        6: 0.160,
        5: 0.118,
        4: 0.065,
        3: 0.060,
        2: 0.037,
        1: 0.019
    }
    
    profile_server_side = {
        6: 0.005,
        5: 0.026,
        4: 0.063,
        3: 0.098,
        2: 0.105,
        1: 0.133
    }
    
    # as the tier naming is in opsite order of the paper
    total_data_size_tier = {
        6: 0.6278 * MB,
        5: 0.629 * MB,
        4: 1.2504 * MB,
        3: 1.2512 * MB,
        2: 2.5006 * MB,
        1: 2.5002 * MB,
    }
        
    profile_client_side = {
        1: 0.160,
        2: 0.118,
        3: 0.065,
        4: 0.060,
        5: 0.037,
        6: 0.019
    }
    
    profile_server_side = {
        1: 0.005,
        2: 0.026,
        3: 0.063,
        4: 0.098,
        5: 0.105,
        6: 0.133
    }
    
    
    batch_num_clients = {key: value / batch_size for key, value in dataset_size.items()}
    # transfer_data_size_client = 
    for k in range(num_users):
        transfer_data_size_client = batch_num_clients[k] * total_data_size_tier[client_tier[-1][k]]
        communication_time_clients = transfer_data_size_client / net_speed[k]
        
        computation_time_clients[k].append(delay_history[k].iloc[-1] - communication_time_clients)
    
    
    time_estimation_client = {}
    client_tier_next = {}
    client_times_tier = client_time_tier(computation_time_clients, client_tier, num_users, num_tiers)
    
    for k in range(num_users):
        
        times_last_tier = client_times_tier[k,client_tier[-1][k]]
        if times_last_tier:
            # current_comp_estimation_time = pd.DataFrame({"Times":computation_time_clients[k]})['Times'].ewm(span=10, adjust=False).mean().iloc[-1]
            current_comp_estimation_time = pd.DataFrame({"Times":times_last_tier})['Times'].ewm(span=2, adjust=False).mean().iloc[-1]
            
            time_estimation_sever_side = {}
            time_estimation_client_side = {}
            time_estimation = {}
            
            for m in range(1,num_tiers):
                
                # estimate time for each tier
                
                time_estimation_client_side[m] = (profile_client_side[m] / profile_client_side[client_tier[-1][k]] * current_comp_estimation_time
                                                 + total_data_size_tier[m] * batch_num_clients[k] / net_speed[k])
                
                time_estimation_sever_side[m] = (total_data_size_tier[m] / net_speed[k] + profile_server_side[m]) * batch_num_clients[k]
                time_estimation[m] = max(time_estimation_sever_side[m], time_estimation_client_side[m])
                
            print(f'client {k}', [f'{time:.2f}' for time in time_estimation.values()])
            time_estimation_list = [time_estimation[key] for key in sorted(time_estimation.keys())]
            time_estimation_client[k] = min(time_estimation_list)
            
            client_tier_next[k], _ = index_of_greatest_smaller(time_estimation_list, T_max)
        else:
            print([f'client {k} not participate yet'])
            client_tier_next[k] = SMALLEST_TIER
        
    T_max = max([time_estimation_client[key] for key in sorted(time_estimation_client.keys())])
    print('T_max: ', T_max)
    
    return client_tier_next, T_max, computation_time_clients
