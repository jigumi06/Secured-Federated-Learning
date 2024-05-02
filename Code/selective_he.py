import torch
from torch.utils.data import DataLoader, Dataset
import client

def get_k_most_sensitivie(map, p):
    #print(map.device)
    map_list = map.view(-1)
    top_k_indices = torch.topk(map_list, int(len(map_list) * p), largest=True).indices
    Mask = torch.zeros(map_list.shape)
    Mask[top_k_indices] = 1
    Mask = torch.tensor(Mask).view(map.shape).to(map.device)
    #print(Mask)
    return Mask

def calculate_mask(args, global_model, dataset_train):
    model = global_model.nn
    dataset_index = global_model.dict_users
    sensitivity_maps = client.calculate_sensitivity_map(args, model, dataset_train, dataset_index)

    data_num = 0
    for k in range(args.K):
        data_num += len(global_model.dict_users[k])

    encryption_mask = []
    aggregated_map = [torch.zeros(layer.shape) for layer in model.parameters()]



    for k in range(args.K):
        weight = len(global_model.dict_users[k])/data_num
        weighted_map = [(layer * weight) for layer in sensitivity_maps[k]]
        aggregated_map = [x.to(args.device)+y for x,y in zip(aggregated_map, weighted_map)]

    

    for map in aggregated_map:
        encryption_mask.append(get_k_most_sensitivie(map, 0.01))

    # for layer in encryption_mask:
    #     print("layer: ",layer.shape)

    return encryption_mask