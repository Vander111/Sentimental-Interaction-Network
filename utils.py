import numpy as np
import scipy.sparse as sp
import torch
import json
import os
import re

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        dic = json.load(f)
    return dic

def save_json(path,data):
    json_str = json.dumps(data, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_str)

def get_project_root():
    cur_path = re.split('pro',os.getcwd())[0] + '/'
    return cur_path