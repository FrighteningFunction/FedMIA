import numpy as np
from numpy import random
import numpy as np
import torch
from torch.utils.data import Subset


np.random.seed(1)

def wm_iid(dataset, num_users, num_back):
    """
    Sample I.I.D. client data from watermark dataset
    """
    num_items = min(num_back, int(len(dataset)/num_users))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid_MIA(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idx0=all_idxs
    train_idxs=[]
    val_idxs=[]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        train_idxs.append(list(dict_users[i] ))
        all_idxs = list(set(all_idxs) - dict_users[i])
        val_idxs.append(list(set(all_idx0)-dict_users[i]))
    return dict_users, train_idxs, val_idxs


def cifar_beta(dataset, beta, n_clients):
     #beta = 0.1, n_clients = 10
    print("The dataset is splited with non-iid param ", beta)
    label_distributions = []
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    #print("labels:",labels)
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}
    #print("classes:",dataset.dataset.classes)
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_y_idx = np.where(labels == y)[0] # [93   107   199   554   633   639 ... 54222]
        label_y_size = len(label_y_idx)
        #print(label_y_idx[0:100])
        
        sample_size = (label_distributions[y]*label_y_size).astype(np.int32)
        #print(sample_size)
        sample_size[n_clients-1] += label_y_size - np.sum(sample_size)
        #print(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i>0 else 0):sample_interval[i]]

    train_idxs=[]
    val_idxs=[]    
    client_datasets = []
    all_idxs=[i for i in range(len(dataset))]
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)
        # save the idxs for attack
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs)-set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs


def label_beta(dataset, beta, n_clients):
    """
    Non-IID split for tensor/tabular datasets with integer or binary labels.

    This mirrors cifar_beta but does not assume torchvision's ``classes`` or
    ``targets`` attributes, which lets BINN synthetic/medical tensors use the
    same federated-learning path.
    """
    print("The dataset is splited with non-iid param ", beta)
    labels = []
    for idx in range(len(dataset)):
        label = dataset[idx][1]
        if torch.is_tensor(label):
            label = label.detach().cpu().view(-1)[0].item()
        labels.append(int(label))
    labels = np.array(labels).astype(np.int32)
    classes = np.unique(labels)

    label_distributions = {
        y: np.random.dirichlet(np.repeat(beta, n_clients))
        for y in classes
    }
    client_idx_map = {i: [] for i in range(n_clients)}

    for y in classes:
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)
        sample_size = (label_distributions[y] * label_y_size).astype(np.int32)
        sample_size[n_clients - 1] += label_y_size - np.sum(sample_size)

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            start = sample_interval[i - 1] if i > 0 else 0
            client_idx_map[i].extend(label_y_idx[start:sample_interval[i]].tolist())

    train_idxs = []
    val_idxs = []
    client_datasets = []
    all_idxs = [i for i in range(len(dataset))]
    for i in range(n_clients):
        client_i_idx = np.array(client_idx_map[i], dtype=np.int64)
        np.random.shuffle(client_i_idx)
        client_datasets.append(Subset(dataset, client_i_idx))
        train_idxs.append(client_i_idx.tolist())
        val_idxs.append(list(set(all_idxs) - set(client_i_idx.tolist())))

    return client_datasets, train_idxs, val_idxs

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

