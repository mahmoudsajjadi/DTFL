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
        # if len(w_locals_server_tier[1]) > len(w_locals_server_tier[num_tiers]):
        #     tier_max = 1
        # for j in range(0, len(w_locals_client_tier[i])):
        if whether_local_loss and not local_v2:
            # if 'fc.bias' in w_locals_client_tier[i]:
            del w_locals_client_tier[i]['fc.bias']
            # if 'fc.wight' in w_locals_client_tier[i]:
            del w_locals_client_tier[i]['fc.weight']
            #del w_locals_client_tier[i]['module.fc.bias']
            #del w_locals_client_tier[i]['module.fc.weight']
            
            if 'module.fc.bias' in w_locals_client_tier[i]:
                del w_locals_client_tier[i]['module.fc.bias']
            if 'module.fc.weight' in w_locals_client_tier[i]:
                del w_locals_client_tier[i]['module.fc.weight']
        # if 
        # del w_locals_client_tier[i]['linear_2.bias']
        # del w_locals_client_tier[i]['linear_2.weight']
        if len(w_locals_client_tier[i]) > largest_client:
            largest_client = len(w_locals_client_tier[i])
            id_largest_client = i
        if len(w_locals_server_tier[i]) > largest_server:
            largest_server = len(w_locals_server_tier[i])
            id_largest_server = i
            
        
                    
    w_avg = copy.deepcopy(w_locals_server_tier[id_largest_server]) # largest model weight (suppose last tier in server is the biggest)
    
    # for k in w_locals_client_tier[num_tiers]:
    for k in w_locals_client_tier[id_largest_client]:
        if k not in w_avg.keys():
            w_avg[k] = 0
    for k in w_avg.keys():
        w_avg[k] = 0
        c = 0
        for i in range(0, len(w_locals_client_tier)):
            if k in w_locals_client_tier[i]:
                if k == 'fc.bias':
                    print(k)
                w_avg[k] += w_locals_client_tier[i][k] * client_sample[i]
                c += 1
        for i in range(0, len(w_locals_server_tier)):
            if k in w_locals_server_tier[i]:
                # if k == 'fc.bias':
                #     print(k)
                w_avg[k] += w_locals_server_tier[i][k] * client_sample[i]
                # print(k)
                c += 1
        # if c != num_users:# and False:
        #     print(k, c)            
        # w_avg[k] = torch.div(w_avg[k], num_users)
        #w_avg[k] = torch.div(w_avg[k], len(w_locals_server_tier))  # devide by number of involved clients
        w_avg[k] = torch.div(w_avg[k], sum(client_sample))  # devide by number of involved clients
        
                
        
    
    # for i in range(1,num_tiers+1):
        
        
    
    
    return w_avg

# multi_fedavg(w, num_tiers)