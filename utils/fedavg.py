import torch
import copy

def aggregated_fedavg(w_locals_server_tier, w_locals_client_tier, num_tiers, num_users, whether_local_loss, client_sample, idxs_users, **kwargs):  
    local_v2 = False
    if kwargs:
        local_v2 = kwargs['local_v2']
    if local_v2:
        for t in range(0, len(w_locals_client_tier)):
            for k in w_locals_server_tier[t].keys():
                if k in w_locals_client_tier[t].keys():
                    del w_locals_client_tier[t][k]
    largest_client, largest_server = 0, 0
    for i in range(0,len(w_locals_client_tier)): # largest model in server-side
        if whether_local_loss and not local_v2:
            del w_locals_client_tier[i]['module.fc.bias']
            del w_locals_client_tier[i]['module.fc.weight']
        if len(w_locals_client_tier[i]) > largest_client:
            largest_client = len(w_locals_client_tier[i])
            id_largest_client = i
        if len(w_locals_server_tier[i]) > largest_server:
            largest_server = len(w_locals_server_tier[i])
            id_largest_server = i
            
        
                    
    w_avg = copy.deepcopy(w_locals_server_tier[id_largest_server]) # largest model weight (suppose last tier in server is the biggest)
    
    for k in w_locals_client_tier[id_largest_client]:
        if k not in w_avg.keys():
            w_avg[k] = 0
    for k in w_avg.keys():
        w_avg[k] = 0
        c = 0
        for i in range(0, len(w_locals_client_tier)):
            if k in w_locals_client_tier[i]:

                w_avg[k] += w_locals_client_tier[i][k] * client_sample[i]
                c += 1
        for i in range(0, len(w_locals_server_tier)):
            if k in w_locals_server_tier[i]:

                w_avg[k] += w_locals_server_tier[i][k] * client_sample[i]
                c += 1
        w_avg[k] = torch.div(w_avg[k], sum(client_sample))  # devide by number of involved clients
        
    
    return w_avg

