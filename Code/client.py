import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    ckks_count = 0
    if isinstance(layer, list):
        return [decryption_to_plain(sublayer) for sublayer in layer]
    elif isinstance(layer, ts.CKKSVector):
        ckks_count += 1
        # print("CKKSVector")
        return layer.decrypt()[0]
    else:
        # print("count of CKKSVector: ", ckks_count)
        return layer


def client_fedfa_cl_secured(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict, masks, layer_list, global_list, shape_list, isAuthentic):  # update nn
    #client_models = [[] for i in range(args.K)]
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))
        
        if global_round == 0:
            client_models[k] = copy.deepcopy(global_model)

        if isAuthentic:
            if global_round != 0:
                #client_model = decryption_of_client_model(args, layer_list[k], global_model)
                # Decrypt by parameters
                client_models[k] = decrypt_by_para(args, global_round, layer_list, shape_list, k, global_model)
        else:
            client_models[k] = copy.deepcopy(global_model)
        #print("dataset_index: ", dataset_index)
        #print("len of dict_users: ", len(dict_users))
        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        
        loss_dict[k].extend(loss)
        index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        for j in index_nonselect:
            loss = [loss_dict[j][-1]]*args.E 
            loss_dict[j].extend(loss)  
    
        if isAuthentic:
            # Encryption on parameters
            temp = encrypt_by_para(args, layer_list[k], client_models[k], k, masks)
            layer_list[k] = temp
            
    return anchorloss_funcs, client_models, loss_dict, layer_list
            

def decrypt_by_para(args, global_round, layer_list, shape_list, k, model):
    #print("<Enter decrypt>")
    if global_round != 0 :
    
        decrypted_layers = []
        layer_count = 0
        for layer_encrypted in layer_list[k]:
            layer_count += 1
            if (layer_count != 3):
                decrypted_layer = decryption_to_plain(layer_encrypted)
            else:
                decrypted_layer = layer_encrypted
            #decrypted_layer = decryption_to_plain(layer_encrypted)
            decrypted_layers.append(decrypted_layer)
        
        reshaped_tensors = []
        for layer_data, shape in zip(decrypted_layers, shape_list):
            tensor = torch.Tensor(layer_data).detach().view(shape).to(args.device)
            reshaped_tensors.append(tensor)

    
        with torch.no_grad():  # Ensure we do not track these operations in the gradient computation
            for param, new_data in zip(model.parameters(), decrypted_layers):
                param.data.copy_(torch.Tensor(new_data.tolist()))

        # print("stop decryption")
        #print("<Finish decrypt>")
        return model


def encrypt_by_para(args, layer_list, client_model, k, masks):
    #print("<Enter encrypt>")
    layer_count = 0
    temp = []
    for layer, mask_layer in zip(layer_list, masks):
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
        
        temp.append(layer)
    #print("<Finish encrypt>")
    return temp

