#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from torch import nn
import torch.nn.functional as F
import copy 
import numpy as np

 
import copy
import sys
sys.path.append("..") 

from utils.global_test import globalmodel_test_on_specifdataset




def mutiListToFlatten(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(mutiListToFlatten(element))  # Recursion call
        else:
            flat_list.append(element)
    return flat_list

 
def add_list_and_list(list1, list2):
    
    if isinstance(list1, list) and isinstance(list2, list):
        return [add_list_and_list(sub1, sub2) for sub1, sub2 in zip(list1, list2)]
    else:
        return list1 + list2

def multiply_nested_list(nested_list, multiplier):

    if isinstance(nested_list, list):
        return [multiply_nested_list(sublist, multiplier) for sublist in nested_list]
    else:
        return nested_list * multiplier

def safe_aggregation(client_index, dict_users, global_layer_list, clients_layer_list):
    # print("before global layer: ", global_layer_list)
    # flat_global_layer_list = mutiListToFlatten(global_layer_list)
    # print("after global layer: ", len(flat_global_layer_list))
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #local dataset size
    
    # j => client index
    for i, j in enumerate(client_index):
        net_para = clients_layer_list[j]
        if i == 0:
            global_layer_list = multiply_nested_list(net_para, len(dict_users[j]) / s)

        else:
            global_layer_list = add_list_and_list(global_layer_list, multiply_nested_list(net_para, len(dict_users[j]) / s))

    # print("after aggregation global layer: ",global_layer_list)
    return global_layer_list

def aggregation(client_index, global_model, client_models, dict_users, fedbn = False):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #local dataset size

    global_w = global_model.state_dict()
    
    if fedbn:
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()
            if i == 0:
                for key in net_para:
                    if 'bn' not in key:
                        global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para:
                    if 'bn' not in key:
                        global_w[key] += net_para[key] * ( len(dict_users[j]) / s)
    else:
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()
            if i == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)

    global_model.load_state_dict(global_w)
    
        


    
 


