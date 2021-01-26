import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import re
import cv2
from utils import *
import os
from PIL import Image
from torchvision import transforms

def load_light_pred(path):
    f = open(path,'r')
    imgs = {}
    for item in f.readlines():
        item = item.strip('\n')
        file = re.split(' ', item)[0]
        pred = int(re.split(' ', item)[1])
        imgs[file]= {"pred":pred}
    return imgs

def opencvloader(imgpath,resizeH,resizeW):
    image = cv2.imread(imgpath)
    image = cv2.resize(image,(resizeH,resizeW),interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image,(2,1,0))
    image = torch.from_numpy(image)
    return image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_data(feature_path):
    idx_features_labels = np.loadtxt(feature_path)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])


    return features, labels


def load_nodedata(feature_path):

    idx_features_labels = np.loadtxt(feature_path)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    return features


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class dataloader(Dataset):
    def __init__(self,split_txt,img_root,graph_root,label_dict,transform):
        self.images = []
        self.transfrom = transform
        f = open(split_txt)
        imglist = []
        for line in f.readlines():
            line = line.strip('\n')
            imglist.append(line)
        if 'hsi' in img_root[1]:
            hsi_flag = True
        else:
            hsi_flag = False

        for file_name in imglist:
            file = re.split(' ',file_name)[0]
            emotion = re.split(' ',file_name)[1]
            img_label = label_dict[emotion]
            img_path = img_root + file
            graph_feature = graph_root + re.split('.jpg',file)[0] + '.txt'
            self.images.append((img_path,graph_feature,img_label))

    def __getitem__(self, item):
        image,node_feature_path,label = self.images[item]
        img = pil_loader(image)
        if self.transfrom is not None:
            img = self.transfrom(img)

        node_feature= load_nodedata(node_feature_path)
        data = (img, node_feature)
        return data,label

    def __len__(self):
        return len(self.images)