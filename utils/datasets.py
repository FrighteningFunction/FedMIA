import numpy as np
import json
import errno
import os
import sys
import random
import logging

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS


logger = logging.getLogger(__name__)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def _make_binn_synthetic_dataset(n_samples, input_size, seed, signal_strength=2.0):
    generator = torch.Generator().manual_seed(int(seed))
    x = torch.randn(int(n_samples), int(input_size), generator=generator)

    # Sparse, nonlinear signal spread over gene groups. The generated labels are
    # intentionally simple but not linearly trivial, so FL smoke tests exercise
    # the BINN hierarchy without needing protected medical data.
    informative = min(int(input_size), 24)
    weights = torch.zeros(int(input_size), 1)
    weights[:informative] = torch.randn(informative, 1, generator=generator)
    group_term = torch.sin(x[:, :informative]).matmul(weights[:informative]).view(-1)
    if input_size >= 8:
        interaction = x[:, 0] * x[:, 1] - 0.5 * x[:, 2] * x[:, 3]
    else:
        interaction = x[:, 0]
    logits = float(signal_strength) * (group_term + interaction)
    threshold = torch.median(logits)
    y = (logits > threshold).float().view(-1, 1)
    return torch.utils.data.TensorDataset(x.float(), y)


def get_data(dataset, data_root, iid, num_users, data_aug, noniid_beta, args=None):
    ds = dataset 
    
    if ds == 'cifar10':
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])  
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])

        train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    
    if ds == 'cifar100':
        if data_aug :
            print("data_aug:",data_aug)
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),#
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(45),
                                                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])  
            transform_test = transforms.Compose([transforms.CenterCrop(32),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])
            
            transform_train_mia = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test_mia = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            
            transform_train_mia=transform_train
            transform_test_mia=transform_test

        train_set_mia = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train_mia
                                               )
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test_mia
                                                )

        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
    if ds == 'dermnet':
        data=torch.load(data_root+"/dermnet_ts.pt")

        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        print(total_set[0].shape) # 19559, 3, 64, 64
        print(total_set[1].shape) # 19559
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:15000],total_set[1][0:15000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-4000:],total_set[1][-4000:] )
        train_set_mia = train_set
        test_set_mia = test_set
    if ds == 'oct':
        data=torch.load(data_root+"/oct_ts.pt")
        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:20000],total_set[1][0:20000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-2000:],total_set[1][-2000:] )
        train_set_mia = train_set
        test_set_mia = test_set

    if ds == 'binn_synthetic':
        if args is None:
            raise ValueError("BINN synthetic data requires args for sizing and seed.")
        train_set = _make_binn_synthetic_dataset(
            args.binn_synthetic_samples,
            args.binn_input_size,
            args.seed,
            args.binn_synthetic_signal,
        )
        test_set = _make_binn_synthetic_dataset(
            args.binn_synthetic_test_samples,
            args.binn_input_size,
            args.seed + 1009,
            args.binn_synthetic_signal,
        )
        train_set_mia = train_set
        test_set_mia = test_set
        logger.info(
            "Generated BINN synthetic dataset: train=%s, test=%s, features=%s",
            len(train_set),
            len(test_set),
            args.binn_input_size,
        )

    if iid:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA(train_set, num_users)
    elif ds == 'binn_synthetic':
        dict_users, train_idxs, val_idxs = label_beta(train_set, noniid_beta, num_users)
    else:
        dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)

    return train_set, test_set, train_set_mia, test_set_mia, dict_users, train_idxs, val_idxs

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        image, label = self.dataset[self.idxs[item]]
        return image, label

class WMDataset(Dataset):
    def __init__(self, root, labelpath, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.labelpath = labelpath
        self.labels = np.loadtxt(self.labelpath)
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)

def prepare_wm(datapath='/trigger/pics/', num_back=1, shuffle=True):
    
    triggerroot = datapath
    labelpath = '/home/lbw/Data/trigger/labels-cifar.txt'

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ]

    transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset(triggerroot, labelpath, wm_transform)
    
    dict_users_back = wm_iid(dataset, num_back, 100)

    return dataset, dict_users_back

def prepare_wm_indistribution(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    triggerroot = datapath
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.ToTensor()
    ]

    #transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset_indistribution(triggerroot, wm_transform)
    
    num_all = num_trigger * num_back 

    dataset = DatasetSplit(dataset, np.arange(0, num_all))
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back

def prepare_wm_new(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    wm_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(datapath, wm_transform)
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back



class WMDataset_indistribution(Dataset):
    def __init__(self, root, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = 5
        if index in self.cache:
            img = self.cache[index]
        else:
        
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)
