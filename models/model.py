import dgl
import torch.nn as nn
from utils import *
from models.backbone.gcn import GCN
from models.backbone.generate_graph import graph_creat

def agg_graph(graph, num):
    temp = [graph for i in range(num)]
    return dgl.batch(temp)

def agg_nodes(node_features,inchannel,device,batchsize):
    graph_num = len(node_features)
    node_num = len(node_features[0])
    zero_arr = torch.zeros(node_num, inchannel).to(device)
    if graph_num < batchsize:
        flag = True
    else:
        flag = False

    for i in range(graph_num):
        if i == 0:
            row = node_features[i].to(device)
        else:
            row = zero_arr
        for j in range(1, graph_num):
            if j == i:
                row = torch.cat((row, node_features[i]), 1)
            elif flag and i > (graph_num - 1):
                row = torch.cat((row, zero_arr), 1)
            else:
                row = torch.cat((row, zero_arr), 1)
        if i == 0:
            feature = row.to(device)
        else:
            feature = torch.cat((feature, row), 0)
    return feature


def resgraphfullimg_fusion(back, batchsize, num_class, weights, node_inputsize,node_size, cuda):
    model = ResGraphFullImgFS(back, batchsize, num_class, weights, node_inputsize,node_size, cuda)
    return model

class ResGraphFullImgFS(torch.nn.Module):
    def __init__(self, back, batchsize, num_classes, weights=None, node_inputsize=512, node_size=128, cuda='cpu'):
        super(ResGraphFullImgFS, self).__init__()
        self.back = back
        self.device = cuda

        self.node_size = node_size
        self.node_inputsize = node_inputsize
        self.batchsize = batchsize

        self.GCN = GCN3(self.node_size * self.batchsize, 2048)
        self.fc_class = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, num_classes))

        self.fc_node = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.node_inputsize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.node_size))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, batch_img, batch_u, g):
        
        batch_img = batch_img.to(self.device)
        feature = self.back(batch_img)

        g = agg_graph(g, batch_u.shape[0])
        batch_u = self.fc_node(batch_u)
        batch_u = agg_nodes(batch_u,self.node_size,self.device,self.batchsize)
        feature_g = self.GCN(g, batch_u)

        y = feature_g
        y = torch.cat((feature,y), 1)
        y = self.relu(y)
        y = self.fc_class(y)
        return y