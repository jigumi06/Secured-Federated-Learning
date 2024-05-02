import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np



import model
from client import *
from utils.aggregator import *
from utils.dispatchor import *
from utils.optimizer import *
# from utils.clusteror import *
from utils.global_test import *
from utils.local_test import *
from utils.sampling import *
from utils.AnchorLoss import *
from utils.ContrastiveLoss import *
from utils.CKA import linear_CKA, kernel_CKA


def seed_torch(seed, test = True):
    if test:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Server:

    def __init__(self, args, model, dataset, synthetic_dataset, dict_users, synthetic_dict_users):
        seed_torch(args.seed)
        self.args = args
        self.nn = copy.deepcopy(model)
        self.nns = [[] for i in range(self.args.K)]
        self.shape_list= []
        for layer in model.parameters():
            self.shape_list.append(layer.shape)

        self.global_layer_list = []
        for layer in model.parameters():
            self.global_layer_list.append(layer)

        self.global_model_state = model.state_dict()
        
        self.layer_list = [[] for i in range(self.args.K)]
        self.p_nns = []
        self.cls = []
        self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict =  dict((k, [0]) for k in key)
        #self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict =  dict((i, []) for i in range(args.r))
        self.dataset = dataset
        self.dict_users = dict_users
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_dict_users = synthetic_dict_users
        

        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2) 
            
        self.contrastiveloss = ContrastiveLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.contrastiveloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.contrals.append(temp2) 

    def fedfa_anchorloss(self, testset, dict_users_test, mask, similarity=False, test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        acc_list = []
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # Modification for alternative SHE
            isAuthentic = True if t%2 == 0 else False
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            #print("active clients: ", index)
            self.index_dict[t]= index
            
            # dispatch
            # Modification for alternative SHE
            if isAuthentic:
                # safe_dispatch(index, self.nn, self.nns)
                safe_dispatch(index, self.global_layer_list, self.layer_list)
            else:
                dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)
            
            if isAuthentic:
                print("CIFAR10 round")
                dict_users_used = self.dict_users
                dataset_used = self.dataset
            else:
                print("Synthetic round")
                dict_users_used = self.synthetic_dict_users
                dataset_used = self.synthetic_dataset
            
            self.cls, self.nns, self.loss_dict, self.layer_list = client_fedfa_cl_secured(self.args, index, self.cls, self.nns, self.nn, t, dataset_used, dict_users_used, self.loss_dict, mask, self.layer_list, self.global_layer_list, self.shape_list, isAuthentic)

            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)
            
            # aggregation
            # Modification for alternative SHE
            if isAuthentic:
                # aggregation(index, self.nn, self.nns, self.dict_users, fedbn=False)
                self.global_model_state = safe_aggregation(index, self.dict_users, self.global_model_state, self.layer_list)
            else:
                aggregation(index, self.nn, self.nns, self.dict_users)
            aggregation(index, self.anchorloss, self.cls, self.dict_users)

            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                print(acc)

            
        mean_CKA_dict = acc_list

        for k in range(self.args.K):
            path="results/Test/{} skew/{}/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E, self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)
        self.nns = [[] for i in range(self.args.K)]
        self.layer_list = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict

def compute_mean_feature_similarity(args, index, client_models, trainset, dict_users_train, testset, dict_users_test):
    pdist = nn.PairwiseDistance(p=2)
    dict_class_verify = {i: [] for i in range(args.num_classes)}
    for i in dict_users_test:
        for c in range(args.num_classes):
            if np.array(testset.targets)[i] == c:
                dict_class_verify[c].append(i)
    #dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}
    dict_clients_features = {k: [] for k in index}
    for k in index:
        # labels = np.array(trainset.targets)[list(dict_users_train[k])]
        # labels_class = set(labels.tolist())
        #for c in labels_class:
        for c in range(args.num_classes):
            features_oneclass = verify_feature_consistency(args, client_models[k], testset,
                                                                     dict_class_verify[c])
            features_oneclass = features_oneclass.view(1,features_oneclass.size()[0],
                                                        features_oneclass.size()[1])
            if c ==0:
                dict_clients_features[k] = features_oneclass
            else:
                dict_clients_features[k] = torch.cat([dict_clients_features[k],features_oneclass])
            
    cos_sim_matrix = torch.zeros(len(index),len(index))
    for p, k in enumerate(index):
        for q, j in enumerate(index):
            for c in range(args.num_classes):
                cos_sim0 = pdist(dict_clients_features[k][c],
                                  dict_clients_features[j][c])
                # cos_sim0 = torch.cosine_similarity(dict_clients_features[k][c],
                #                                   dict_clients_features[j][c])
                # cos_sim0 = get_cos_similarity_postive_pairs(dict_clients_features[k][c],
                #                                    dict_clients_features[j][c])
                if c ==0:
                    cos_sim = cos_sim0
                else:
                    cos_sim = torch.cat([cos_sim,cos_sim0])
            cos_sim_matrix[p][q] = torch.mean(cos_sim)
    mean_feature_similarity = torch.mean(cos_sim_matrix)

    return mean_feature_similarity

def get_cos_similarity_postive_pairs(target, behaviored):
    attention_distribution_mean = []
    for j in range(target.size(0)):
        attention_distribution = []
        for i in range(behaviored.size(0)):
            attention_score = torch.cosine_similarity(target[j], behaviored[i].view(1, -1))  
            attention_distribution.append(attention_score)
        attention_distribution = torch.Tensor(attention_distribution)
        mean = torch.mean(attention_distribution)
        attention_distribution_mean.append(mean)
    attention_distribution_mean = torch.Tensor(attention_distribution_mean)
    return attention_distribution_mean
