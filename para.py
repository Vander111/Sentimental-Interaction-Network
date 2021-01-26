import torch.nn as nn
from torch.utils.data import DataLoader

from models.backbone.resnet_feature import resnet101fm, resnet101
from models.backbone.vgg_feature import vgg16_feature
from models.backbone.generate_graph import graph_creat
from models.model import resgraphfullimg_fusion
from models.loss import CEloss
from models.optimizer import get_opt
from models.weight_init import fc_weight_init
from tansform import train_transform, query_transform
from dataloader import dataloader

from utils import *
from logger import get_best_record

project_root = get_project_root()

def get_data_info(data_name, edge_name, node_name,train_mode, class_num):
    dataset = data_info[data_name]
    img_ori_path = dataset['img']
    imgs = img_ori_path

    if node_name:
        node_feature_path = dataset['node_feature'][node_name]
    else:
        node_feature_path = None
    if edge_name:
        edge_path = data_info['edges'][edge_name]
    else:
        edge_path = None
    train_splits = dataset["train_txt"][train_mode]

    if class_num == 2:
        label_dict = dataset["sentdict"]
    else:
        label_dict = dataset["emodict"]

    best_acc_log = get_best_record(data_info,data_name,class_num,train_mode)

    return (imgs,node_feature_path),edge_path,train_splits,label_dict,best_acc_log

def mk_graph(edge_path):
    edges = load_json(edge_path)
    g = graph_creat(edges)
    return g

def get_dataloader(label_paths,data_paths,transform_flag,batch_size,val_flag,label_dict):
    train_txt,test_txt,val_txt = label_paths['train'],label_paths['test'],label_paths['val']
    img,gra_feature = data_paths
    if transform_flag:
        train_trans = train_transform()
        test_trans = query_transform()
    else:
        train_trans = None
        test_trans = None

    train_data = dataloader_gcn(train_txt, img, gra_feature, label_dict, train_trans)
    val_data = dataloader_gcn(val_txt, img, gra_feature, label_dict, test_trans)
    test_data = dataloader_gcn(test_txt, img, gra_feature, label_dict, test_trans)
    train_num, test_num, val_num = len(train_data),len(test_data),len(val_data)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=3)
    if val_flag:
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=3)
    else:
        val_loader = None

    return train_loader,test_loader,val_loader,(train_num, test_num,val_num)

def load_backbone(backbone):
    if backbone == 'vgg16':
        backbone = vgg16_feature()
        pretrain_path = data_info['prerained model']['vgg16']
        backbone.load_state_dict(torch.load(pretrain_path))
        backbone.classifier[-1] = nn.Linear(4096, 2048)

    elif backbone == 'resnet101':
        backbone = resnet101()
        pretrain_path = data_info['prerained model']['resnet101']
        backbone.load_state_dict(torch.load(pretrain_path))
    else:
        print('backbone not found')
    return backbone

def get_model(mode,backbone,fc_init_flag,batch_size, node_inputsize,node_size,calss_num,device):
    model = load_backbone(backbone)
    model = resgraphfullimg_fusion(model, batch_size, calss_num, node_inputsize,node_size, device)

    if fc_init_flag:
        print('fc initing...')
        model.apply(fc_weight_init)
    return model

def get_loss(loss_name):
    if loss_name == 'CEloss':
        loss = CEloss()
    else:
        print('loss not found')
    return loss