import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import random, os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from PIL import Image
import generate_synthetic_dataset as synthetic

import pandas as pd
import copy

from fedlab.utils.dataset import FMNISTPartitioner,CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

import selective_he
from args_cifar10_c2 import args_parser
import server_se1 as server
import model

from utils.global_test import test_on_globaldataset, globalmodel_test_on_localdataset,globalmodel_test_on_specifdataset
from utils.local_test import test_on_localdataset
from utils.training_loss import train_loss_show,train_localacc_show,train_globalacc_show

from utils.sampling import testset_sampling, trainset_sampling, trainset_sampling_label
from utils.tSNE import FeatureVisualize

args = args_parser()

def seed_torch(seed=args.seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()
GLOBAL_SEED = 1
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

C = "2CNN_2" 

specf_model = model.Client_Model(args, name='cifar10').to(args.device)

trans_cifar10 =  transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.491, 0.482, 0.447], 
                                                        std=[0.247, 0.243, 0.262])])
root = "data/CIFAR10/"
trainset = torchvision.datasets.CIFAR10(root=root,train=True, download=True, transform=trans_cifar10)
testset = torchvision.datasets.CIFAR10(root=root,train=False, download=True, transform=trans_cifar10)


num_classes = args.num_classes
num_clients = args.K
number_perclass = args.num_perclass
 

col_names = [f"class{i}" for i in range(num_classes)]
print(col_names)
hist_color = '#4169E1'
plt.rcParams['figure.facecolor'] = 'white'


# perform partition
noniid_labeldir_part = CIFAR10Partitioner(trainset.targets, 
                                num_clients=num_clients,
                                balance=None, 
                                partition="shards",
                                num_shards=200,
                                seed=1)
# generate partition report
csv_file = "data/CIFAR10/cifar10_noniid_labeldir_clients_10.csv"
partition_report(trainset.targets, noniid_labeldir_part.client_dict, 
                 class_num=num_classes, 
                 verbose=False, file=csv_file)

noniid_labeldir_part_df = pd.read_csv(csv_file,header=1)
noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')
for col in col_names:
    noniid_labeldir_part_df[col] = (noniid_labeldir_part_df[col] * noniid_labeldir_part_df['Amount']).astype(int)

# select first 10 clients for bar plot
noniid_labeldir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  

trainset_sample_rate = args.trainset_sample_rate
rare_class_nums = 0
dict_users_train = trainset_sampling_label(args, trainset, trainset_sample_rate,rare_class_nums, noniid_labeldir_part) 
dict_users_test = testset_sampling(args, testset, number_perclass, noniid_labeldir_part_df)

# Generate synthetic dataset
synthetic_dataset = synthetic.SyntheticDataset(num_classes=num_classes, train = True)
synthetic_noniid_labeldir_part = CIFAR10Partitioner(synthetic_dataset.targets, 
                                num_clients=num_clients,
                                balance=None, 
                                partition="shards",
                                num_shards=200,
                                seed=1)
synthetic_dict_users = trainset_sampling_label(args, synthetic_dataset, trainset_sample_rate,rare_class_nums, synthetic_noniid_labeldir_part)

# initiate the server with defined model and dataset
serverz = server.Server(args, specf_model, trainset, synthetic_dataset.dataset_split, dict_users_train, synthetic_dict_users)
print("global_model: ", serverz.nn.state_dict)
total_params = sum(p.numel() for p in serverz.nn.parameters())
print("Total number of parameters:", total_params)

def run_FedFA():
    print("Enter FedFA!")
    server_feature = copy.deepcopy(serverz)
    # Selective HE
    init_model = copy.deepcopy(serverz)
    encryption_mask = selective_he.calculate_mask(args, init_model, trainset)

    print("Start server")
    # 
    global_modelfa, similarity_dictfa, client_modelsfa, loss_dictfa, clients_indexfa, acc_listfa = server_feature.fedfa_anchorloss(testset, dict_users_test, encryption_mask, similarity = False, test_global_model_accuracy = True)
    print("End Server")
    
    torch.save(acc_listfa,"results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed,args.E,C))
    path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed,args.E,C)
    torch.save(global_modelfa.state_dict(), path_fedfa)

    print("Start drawing")
    train_loss_show(args, loss_dictfa, clients_indexfa)
    train_globalacc_show(args, acc_listfa)

if __name__ == "__main__":
    print("setting: ", int(args.C*args.K), "/", args.K)
    run_FedFA()
