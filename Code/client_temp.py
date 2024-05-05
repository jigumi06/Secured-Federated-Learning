import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
import threading
import queue
import pickle

import utils.optimizer as op
import tenseal as ts
import utils.ckks as ckks

def calculate_sensitivity_map(args, global_model, dataset, dataset_index):
    sensitivity_maps = []
    
    for k in range(args.K):
        Dtr = DataLoader(op.DatasetSplit(dataset, dataset_index[k]), batch_size=args.B, shuffle=True)
        loss_function = torch.nn.CrossEntropyLoss().to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = global_model(imgs)

        loss = loss_function(y_preds, labels)
        loss.backward()
        gradients = [torch.abs(param.grad.data) for param in global_model.parameters()]
        sensitivity_maps.append(gradients)

    return sensitivity_maps

def decryption_to_plain(layer):
    #ckks_count = 0
    if isinstance(layer, list):
        return [decryption_to_plain(sublayer) for sublayer in layer]
    elif isinstance(layer, ts.CKKSVector):
        #ckks_count += 1
        # print("CKKSVector")
        return layer.decrypt()[0]
    else:
        #print("count of CKKSVector: ", ckks_count)
        return layer


def client_fedfa_cl_secured(args, k, global_model, global_round, anchorloss_funcs, dataset_train, dict_users, masks, layer_list, shape_list, results_queue):  # update nn

    client_model = decrypt_by_para(args, global_round, layer_list, shape_list, k, global_model)
    anchorloss_func, client_model, loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_model, global_model, global_round, dataset_train, dict_users[k])
    layer_list = encrypt_by_para(args, layer_list, client_model, k, masks)

    results_queue.put(k, client_model, anchorloss_func, loss, layer_list)

def start_clients(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict, masks, layer_list, shape_list, isAuthentic):
    if isAuthentic and global_round != 0:
        # Decryption
        #decrypted_model = decrypt_by_para(args, global_round, layer_list[client_index[0]], shape_list, global_model)
        decryption_threads = []
        decryption_queue = queue.Queue()
        for k in client_index:
            thread = threading.Thread(target=decrypt_by_para, args=(args, global_round, layer_list[k], shape_list, global_model, client_models[k]))
            decryption_threads.append(thread)
            thread.start()
            
        for thread in decryption_threads:
            thread.join()
        '''
        while not decryption_queue.empty():
            k, client_model_bytes = decryption_queue.get()
            print(k)
            client_models[k] = pickle.loads(client_model_bytes)
        '''
        
        # Local training
        loss_list = [None] * args.K
        for k in client_index:
            #client_models[k] = decrypted_model
            #print(client_models[k])
            anchorloss_funcs[k], client_models[k], loss_list[k] = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
            loss_dict[k].extend(loss_list[k])
            index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
            for j in index_nonselect:
                loss_list[k] = [loss_dict[j][-1]]*args.E 
                loss_dict[j].extend(loss_list[k])
        
        #print("Encryption")
        # Encryption
        encryption_threads = []
        encryption_queue = queue.Queue()
        for k in client_index:
            thread = threading.Thread(target=encrypt_by_para, args=(args, layer_list[k], client_models[k], k, masks, encryption_queue))
            encryption_threads.append(thread)
            thread.start()
            
        for thread in encryption_threads:
            thread.join()    

    else:
        for k in client_index:
            anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], global_model, global_model, global_round, dataset_train, dict_users[k])
            
            loss_dict[k].extend(loss)
            index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
            for j in index_nonselect:
                loss = [loss_dict[j][-1]]*args.E 
                loss_dict[j].extend(loss)
                
    return anchorloss_funcs, client_models, loss_dict, layer_list

def decrypt_by_para(args, global_round, layer_list, shape_list, global_model, client_model):
    #print("<Enter decrypt>")
   # if global_round != 0 :
    #decrypted_layers = []
    layer_count = 0
    for layer_encrypted, param in zip(layer_list, global_model.parameters()):
        layer_count += 1
        if (layer_count != 3):
            decrypted_layer = decryption_to_plain(layer_encrypted)
        else:
            decrypted_layer = layer_encrypted
            #decrypted_layer = decryption_to_plain(layer_encrypted)
            #decrypted_layers.append(decrypted_layer)
        tensor = torch.Tensor(decrypted_layer).detach().view(shape_list[layer_count-1]).to(args.device)
        param.data.copy_(torch.Tensor(tensor.tolist()))
    client_model = global_model
    #return global_model


def encrypt_by_para(args, layer_list, client_model, k, masks, encryption_queue):
    #print("<Enter encrypt>")
    layer_count = 0
    layer_list = []
    for layer, mask_layer in zip(client_model.parameters(), masks):
        layer_count += 1
        # print("layer size: ", layer.size())
        if (layer_count != 3):
            # layer_shape = layer.shape
            flat_tensor = layer.flatten()
            flat_tensor_list = flat_tensor.tolist()
            # print(" layer size: ", len(flat_tensor_list))
            flat_mask = mask_layer.flatten()
            flat_mask_list = flat_mask.tolist()
            # print(" mask size: ", len(flat_mask_list))
    
            layer = [ ckks.EncryptionManager().encrypt_vector(flat_tensor_list[i]) 
                                if flat_mask_list[i] == 1 else flat_tensor_list[i] for i in range(len(flat_mask_list))]
        
        layer_list.append(layer)
    #return layer_list
